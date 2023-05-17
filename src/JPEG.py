
# import
import numpy as np
import cv2 as cv
import math
import os
import sys
import heapq
from collections import Counter
import struct
import pickle
import bitarray

# Quantization Tables
Luminance_Quantization_Table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                        [12, 12, 14, 19, 26, 58, 60, 55],
                                        [14, 13, 16, 24, 40, 57, 69, 56],
                                        [14, 17, 22, 29, 51, 87, 80, 62],
                                        [18, 22, 37, 56, 68, 109, 103, 77],
                                        [24, 35, 55, 64, 81, 104, 113, 92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103, 99]])

Chrominance_Quantization_Table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                        [18, 21, 26, 66, 99, 99, 99, 99],
                                        [24, 26, 56, 99, 99, 99, 99, 99],
                                        [47, 66, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99]])

# Compression
def compress(filename):
    # Read the image to be compressed
    filename_without_ext = os.path.splitext(filename)[0]
    imgBGR = cv.imread(filename)
    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB) # OpenCV stores images by default in BGR format, and we need to convert it to RGB
    img_height, img_width = imgRGB.shape[:2]
    # Transform RGB to YCbCr
    imgYCbCr = rgb2ycbcr(imgRGB)
    # Separate the three channels and perform 4:2:0 color subsampling
    imgY, imgCb, imgCr = downSample(imgYCbCr)
    # Pad the height and width of Y, Cb, Cr components to a multiple of 8    
    imgY_padded = pad(imgY)
    imgCb_padded = pad(imgCb)
    imgCr_padded = pad(imgCr)
    # Divide the image into small blocks of 8x8
    blocksY = split2blocks(imgY_padded)
    blocksCb = split2blocks(imgCb_padded)
    blocksCr = split2blocks(imgCr_padded)
    # Perform DCT on image blocks
    dct_matrix = getDctMatrix()
    dctY = dct2D(blocksY, dct_matrix)
    dctCb = dct2D(blocksCb, dct_matrix)
    dctCr = dct2D(blocksCr, dct_matrix)
    # Apply Quantization
    qY = quantization(dctY, Luminance_Quantization_Table)
    qCb = quantization(dctCb, Chrominance_Quantization_Table)
    qCr = quantization(dctCr, Chrominance_Quantization_Table)
    # Zigzag Ordering
    zigY = zigzag(qY) # (num_block, 64)
    zigCb = zigzag(qCb)
    zigCr = zigzag(qCr)
    # DPCM on DC coefficients, e(0) = DC(0), e(i) = DC(i) - DC(i-1)   
    dpcmY = dpcm(zigY) # numpy NDArray[float64]
    dpcmCb = dpcm(zigCb)
    dpcmCr = dpcm(zigCr)
    # RLC on AC coefficients
    rlcY = rlc(zigY) # list
    rlcCb = rlc(zigCb)
    rlcCr = rlc(zigCr)
    # Perform Entropy Coding(VLC, Huffman Coding) 
    dcY_encoding_table, dcY_encoded_data = huffman_encode(dpcmY)
    dcCb_encoding_table, dcCb_encoded_data = huffman_encode(dpcmCb)
    dcCr_encoding_table, dcCr_encoded_data = huffman_encode(dpcmCr)
    acY_encoding_table, acY_encoded_data = huffman_encode(rlcY)
    acCb_encoding_table, acCb_encoded_data = huffman_encode(rlcCb)
    acCr_encoding_table, acCr_encoded_data = huffman_encode(rlcCr)
    # Save the compressed data to a binary file
    huffman_tables = [dcY_encoding_table, dcCb_encoding_table, dcCr_encoding_table, acY_encoding_table, acCb_encoding_table, acCr_encoding_table]
    encoded_datas = [dcY_encoded_data, dcCb_encoded_data, dcCr_encoded_data, acY_encoded_data, acCb_encoded_data, acCr_encoded_data]
    write_binary_file(huffman_tables, encoded_datas, img_height, img_width, filename_without_ext+".myjpeg")
    return filename_without_ext+".myjpeg"

# Transform RGB to YCbCr
def rgb2ycbcr(rgb):
    # Define the transformation matrix
    xform_matrix = np.array([[0.299, 0.587, 0.114],
                           [-0.168736, -0.331264, 0.5],
                           [0.5, -0.418688, -0.081312]])
    ycbcr = np.dot(rgb, xform_matrix.T)
    ycbcr[:, :, [1, 2]] += 128.0
    return ycbcr

# Separate the three channels and perform 4:2:0 color subsampling
def downSample(ycbcr):
    # Separate the three channels
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    # Perform 4:2:0 color subsampling, take the upper-left corner of the 2x2 square
    cb = cb[::2, ::2]
    cr = cr[::2, ::2]
    return y, cb, cr

# Pad the height and width of Y, Cb, Cr components to a multiple of 8
def pad(img): # (heignt, width)
    h, w = img.shape[:2]
    h_padding = 0
    w_padding = 0
    if h % 8 != 0:
        h_padding = 8 - (h % 8)
    if w % 8 != 0:
        w_padding = 8 - (w % 8)
    img_padded = cv.copyMakeBorder(img, 0, h_padding, 0, w_padding, cv.BORDER_REPLICATE)
    return img_padded # (height_padded, width_padded)

# Divide the image into small blocks of 8x8
def split2blocks(img): # (height_padded, width_padded)
    h, w = img.shape[:2]
    blocks = np.zeros(((h//8) * (w//8), 8, 8), dtype=np.float32)
    for i in range(h//8):
        for j in range(w//8):
            yoff, xoff = i*8, j*8
            blocks[i*(w//8)+j, :, :] = img[yoff:yoff+8, xoff:xoff+8]
    return blocks # (h//8 * w//8, 8, 8)

# Perform DCT on image blocks
def getDctMatrix():
    dct_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                dct_matrix[i][j] = 1 / (2*np.sqrt(2))
            else:
                dct_matrix[i][j] = (1.0/2) * np.cos((2*j+1)*i*np.pi/16)
    return dct_matrix # T

def dct2D(blocks, dct_matrix): # (num_block, 8, 8)
    dct_blocks = np.zeros_like(blocks) # (num_block, 8, 8)
    for block_index in range(dct_blocks.shape[0]):
        for u in range(8):
            for v in range(8):
                # F = T · f · T.T
                dct_blocks[block_index] = dct_matrix @ blocks[block_index] @ dct_matrix.T
    return dct_blocks # (num_block, 8, 8)

def quantization(blocks, quantization_table): # (num_block, 8, 8)
    return np.round(blocks / quantization_table) # (num_block, 8, 8)

# Zigzag Ordering
def zigzag(blocks): # (num_block, 8, 8)
    zigzag_table = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                            [2, 4, 7, 13, 16, 26, 29, 42],
                            [3, 8, 12, 17, 25, 30, 41, 43],
                            [9, 11, 18, 24, 31, 40, 44, 53],
                            [10, 19, 23, 32, 39, 45, 52, 54],
                            [20, 22, 33, 38, 46, 51, 55, 60],
                            [21, 34, 37, 47, 50, 56, 59, 61],
                            [35, 36, 48, 49, 57, 58, 62, 63]])
    
    linearized_block = np.zeros(64)
    blocks = blocks.reshape((-1, 64))
    for x in range(blocks.shape[0]):
        for i in range(8):
            for j in range(8):
                linearized_block[zigzag_table[i][j]] = blocks[x][i*8+j]
        blocks[x] = linearized_block
        
    return blocks # (num_block, 64)

# DPCM on DC coefficients, e(0) = DC(0), e(i) = DC(i) - DC(i-1)   
def dpcm(blocks): # (num_block, 64)
    dc = blocks[:, 0]
    dpcm_list = np.zeros(dc.shape[0])

    for i in range(dc.shape[0]): # num_block
        if (i == 0):
            dpcm_list[i] = dc[i]
        else:
            dpcm_list[i] = dc[i] - dc[i-1]
    
    return dpcm_list # (num_block,)

# RLC on AC coefficients
def rlc(blocks): # (num_block, 64)
    ac = blocks[:, 1:]
    rlc_list = []

    for i in range(ac.shape[0]):
        zero_cnt = 0
        for j in range(ac.shape[1]):
            if (ac[i][j] == 0):
                zero_cnt = zero_cnt + 1
            else:
                rlc_list.append((zero_cnt, ac[i][j])) # (skip, value)
                zero_cnt = 0
        rlc_list.append((0, 0)) # the end of a block

    return rlc_list # (num_block * ?), element: (skip, value)

# Define the node class of the Huffman tree
class HuffmanNode():
    def __init__(self, char, freq, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    counter = Counter(data)
    heap = [HuffmanNode(char, freq) for char, freq in counter.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, parent)

    return heap[0]

def create_encoding_table(root, encoding_table=None, code=''):
    if encoding_table is None:
        encoding_table = {}

    if root.char is not None:
        encoding_table[root.char] = code
        return encoding_table

    create_encoding_table(root.left, encoding_table, code + '0')
    create_encoding_table(root.right, encoding_table, code + '1')

    return encoding_table

def huffman_encode(data):
    root = build_huffman_tree(data)
    encoding_table = create_encoding_table(root)
    encoded_data = ''.join(encoding_table[char] for char in data)

    return encoding_table, encoded_data

def write_binary_file(huffman_tables, encoded_datas, img_height, img_width, filename):
    with open(filename, 'wb') as f:
        # Write the height and width
        f.write(struct.pack('ii', img_height, img_width))

        # Write the number of Huffman coding tables
        num_tables = len(huffman_tables)
        f.write(struct.pack('B', num_tables))

        for i in range(num_tables):
            # Serialize coding table
            serialized_table = pickle.dumps(huffman_tables[i])
            
            # Write the length of the coding table
            f.write(struct.pack('I', len(serialized_table)))

            # Write the serialized coding table
            f.write(serialized_table)

            # Convert the encoded data to a bitarray object
            data_bits = bitarray.bitarray(encoded_datas[i])

            # Write the length of the encoded data in bits
            f.write(struct.pack('I', len(data_bits)))

            # Write the encoded data
            data_bits.tofile(f)

# Decompression
def decompress(filename):
    # Read compressed data from a binary file
    filename_without_ext = os.path.splitext(filename)[0]
    huffman_tables, encoded_datas, img_height, img_width = read_binary_file(filename)
    dcY_encoding_table, dcCb_encoding_table, dcCr_encoding_table, acY_encoding_table, acCb_encoding_table, acCr_encoding_table = huffman_tables
    dcY_encoded_data, dcCb_encoded_data, dcCr_encoded_data, acY_encoded_data, acCb_encoded_data, acCr_encoded_data = encoded_datas
    # Perform Entropy Decoding
    dpcmY = np.array(huffman_decode(dcY_encoding_table, dcY_encoded_data))
    dpcmCb = np.array(huffman_decode(dcCb_encoding_table, dcCb_encoded_data))
    dpcmCr = np.array(huffman_decode(dcCr_encoding_table, dcCr_encoded_data))
    rlcY = huffman_decode(acY_encoding_table, acY_encoded_data)
    rlcCb = huffman_decode(acCb_encoding_table, acCb_encoded_data)
    rlcCr = huffman_decode(acCr_encoding_table, acCr_encoded_data)
    # Compute num_block, create zig_blocks(num_block, 64)
    num_blockY = dpcmY.shape[0]
    num_blockCb = dpcmCb.shape[0]
    num_blockCr = dpcmCr.shape[0]
    zigY = np.zeros((num_blockY, 64))
    zigCb = np.zeros((num_blockCb, 64))
    zigCr = np.zeros((num_blockCr, 64))
    # DPCM Decoding
    dpcm_decode(dpcmY, zigY)
    dpcm_decode(dpcmCb, zigCb)
    dpcm_decode(dpcmCr, zigCr)
    # RLC Decoding
    rlc_decode(rlcY, zigY)
    rlc_decode(rlcCb, zigCb)
    rlc_decode(rlcCr, zigCr)
    # Inverse Zigzag
    qY = izigzag(zigY)
    qCb = izigzag(zigCb)
    qCr = izigzag(zigCr)
    # Inverse Quantization
    dctY = iquantization(qY, Luminance_Quantization_Table)
    dctCb = iquantization(qCb, Chrominance_Quantization_Table)
    dctCr = iquantization(qCr, Chrominance_Quantization_Table)
    # Perform IDCT on image blocks
    dct_matrix = getDctMatrix()
    blocksY = idct(dctY, dct_matrix)
    blocksCb = idct(dctCb, dct_matrix)
    blocksCr = idct(dctCr, dct_matrix)
    # Calculate the size of padded Y, Cb, Cr components based on the size of the original image
    imgY_padded = np.zeros(getPaddedSize(img_height, img_width))
    imgCb_padded = np.zeros(getPaddedSize(img_height/2, img_width/2))
    imgCr_padded = np.zeros(getPaddedSize(img_height/2, img_width/2))
    # Merge the 8x8 small blocks into the image
    mergeBlocks(blocksY, imgY_padded)
    mergeBlocks(blocksCb, imgCb_padded)
    mergeBlocks(blocksCr, imgCr_padded)
    # Crop the padded image back to the original size
    imgY = imgY_padded[:img_height, :img_width]
    imgCb = imgCb_padded[:img_height//2, :img_width//2]
    imgCr = imgCr_padded[:img_height//2, :img_width//2]
    # Perform color upsampling interpolation and merge the three channels
    imgYCbCr = upSample(imgY, imgCb, imgCr, img_height, img_width)
    # Transform YCbCr to RGB
    imgRGB = ycbcr2rgb(imgYCbCr)
    imgRGB = np.clip(imgRGB, 0, 255) # https://www.cnblogs.com/sunny-li/p/10265755.html
    imgRGB = imgRGB.astype(np.uint8)
    # Show and save the decompressed image
    imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)
    cv.imshow("result", imgBGR)
    cv.waitKey(0)
    cv.imwrite(filename_without_ext+"_decompressed"+".bmp", imgBGR)

# Read compressed data from a binary file
def read_binary_file(filename):
    with open(filename, 'rb') as f:
        # Read the height and width
        height, width = struct.unpack('ii', f.read(8))

        # Read the number of Huffman coding tables
        num_tables = struct.unpack('B', f.read(1))[0]

        huffman_tables = []
        encoded_datas = []

        for _ in range(num_tables):
            # Read the length of the coding table
            table_length = struct.unpack('I', f.read(4))[0]

            # Read the serialized coding table and perform deserialization
            serialized_table = f.read(table_length)
            huffman_tables.append(pickle.loads(serialized_table))

            # Read the length of the encoded data in bits
            data_length = struct.unpack('I', f.read(4))[0]
            # Convert the units from bits to bytes
            data_length_in_bytes = math.ceil(data_length/8)

            # Read the encoded data
            data = bitarray.bitarray()
            data.fromfile(f, data_length_in_bytes)
            data = data[:data_length]

            # Convert the bitarray object to a 01 string
            data = data.to01()
            encoded_datas.append(data)

    return huffman_tables, encoded_datas, height, width

# Perform Entropy Decoding
def huffman_decode(encoding_table, encoded_data):
    decoded_data = []
    reversed_encoding_table = {v: k for k, v in encoding_table.items()}
    current_code = ''

    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_encoding_table:
            char = reversed_encoding_table[current_code]
            decoded_data.append(char)
            current_code = ''

    return decoded_data

# DPCM Decoding
def dpcm_decode(dpcm_table, zig_blocks): # (num_block,), (num_block, 64)
    for i in range(zig_blocks.shape[0]):
        if i == 0:
            zig_blocks[i][0] = dpcm_table[i]
        else:
            zig_blocks[i][0] = zig_blocks[i-1][0] + dpcm_table[i]

# RLC Decoding
def rlc_decode(rlc_table, zig_blocks): # (num_block * ?), (num_block, 64)
    i = 0 # block_index
    j = 1 # block_element_index
    for k in range(len(rlc_table)):
        if rlc_table[k] == (0, 0): # the end of this block
            i = i + 1 # turn to next block
            j = 1
        else: # (skip, value)
            j = j + rlc_table[k][0]
            zig_blocks[i][j] = rlc_table[k][1]
            j = j + 1

# Inverse Zigzag
def izigzag(linearized_blocks): # (num_block, 64)
    zigzag_table = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                            [2, 4, 7, 13, 16, 26, 29, 42],
                            [3, 8, 12, 17, 25, 30, 41, 43],
                            [9, 11, 18, 24, 31, 40, 44, 53],
                            [10, 19, 23, 32, 39, 45, 52, 54],
                            [20, 22, 33, 38, 46, 51, 55, 60],
                            [21, 34, 37, 47, 50, 56, 59, 61],
                            [35, 36, 48, 49, 57, 58, 62, 63]])

    blocks = np.zeros((linearized_blocks.shape[0], 8, 8))
    for x in range(blocks.shape[0]):
        for i in range(8):
            for j in range(8):
                blocks[x][i][j] = linearized_blocks[x][zigzag_table[i][j]]

    return blocks # (num_block, 8, 8)

# Inverse Quantization
def iquantization(blocks, quantization_table): # (num_block, 8, 8)
    return blocks * quantization_table # (num_block, 8, 8)

# Perform IDCT on image blocks
def idct(dct_blocks, dct_matrix): # (num_block, 8, 8)
    blocks = np.zeros_like(dct_blocks)
    for block_index in range(blocks.shape[0]):
        for u in range(8):
            for v in range(8):
                blocks[block_index] = dct_matrix.T @ dct_blocks[block_index] @ dct_matrix

    return blocks # (num_block, 8, 8)

# Calculate the size of padded Y, Cb, Cr components based on the size of the original image
def getPaddedSize(h, w): # (heignt, width)
    h = round(h)
    w = round(w)
    h_padding = 0
    w_padding = 0
    if h % 8 != 0:
        h_padding = 8 - (h % 8)
    if w % 8 != 0:
        w_padding = 8 - (w % 8)
    return (h+h_padding, w+w_padding) # (height_padded, width_padded)

# Merge the 8x8 small blocks into the image
def mergeBlocks(blocks, img): # (num_block, 8, 8), (height_padded, width_padded)
    h, w = img.shape[:2]
    for i in range(h//8):
        for j in range(w//8):
            yoff, xoff = i*8, j*8
            img[yoff:yoff+8, xoff:xoff+8] = blocks[i*(w//8)+j, :, :]

# Perform color upsampling interpolation and merge the three channels
def upSample(y, cb, cr, img_height, img_width): # (height, width) Y, Cb, Cr
    ycbcr = np.zeros((img_height, img_width, 3))
    for i in range(cb.shape[0]):
        for j in range(cb.shape[1]):
            ycbcr[i*2][j*2][1] = ycbcr[i*2+1][j*2][1] = ycbcr[i*2][j*2+1][1] = ycbcr[i*2+1][j*2+1][1] = cb[i][j]
            ycbcr[i*2][j*2][2] = ycbcr[i*2+1][j*2][2] = ycbcr[i*2][j*2+1][2] = ycbcr[i*2+1][j*2+1][2] = cr[i][j]
    
    ycbcr[:, :, 0] = y
    return ycbcr # (height, width, 3) RGB

# Transform YCbCr to RGB
def ycbcr2rgb(imgYCbCr): # (height, width)
    # Define the transformation matrix
    xform_matrix = np.array([[1, 0, 1.402],
                           [1, -0.344136, -0.714136],
                           [1, 1.772, 0]])
    imgYCbCr[:, :, [1, 2]] -= 128.0
    rgb = np.dot(imgYCbCr, xform_matrix.T)
    return rgb # (height, width)

# main
filename = sys.argv[1]
compressed_file = compress(filename)
decompress(compressed_file)