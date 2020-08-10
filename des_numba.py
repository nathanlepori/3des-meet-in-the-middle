import numpy as np
from numba import cuda
import time

PC_1 = np.array([
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
], dtype=np.uint8)


@cuda.jit(device=True)
def read_bit(val, ix):
    byte_off = ix // 8
    off = 7 - ix % 8

    return (val[byte_off] & (1 << off)) >> off


@cuda.jit(device=True)
def write_bit(val, ix, bit_val):
    byte_off = ix // 8
    off = 7 - ix % 8

    m = 1 << off

    # Write back to the original array
    val[byte_off] = (val[byte_off] & ~m) | ((bit_val << off) & m)


@cuda.jit(device=True)
def swap_bits(val, ix1, ix2):
    tmp = read_bit(val, ix1)

    # Swap bits
    return write_bit(write_bit(val, ix1, read_bit(val, ix2)), ix2, tmp)


@cuda.jit(device=True)
def permute(val, p_val, perm_table):
    ixp = cuda.threadIdx.x
    write_bit(p_val, ixp, read_bit(val, perm_table[ixp] - 1))
    # for ixp, ix in enumerate(perm_table):
    #     write_bit(p_val, ixp, read_bit(val, ix - 1))


@cuda.jit('uint8[:], uint8[:], uint8[:]')
def kernel_main(val, p_val, perm_table):
    permute(val, p_val, perm_table)


@cuda.jit
def des_encrypt():
    pass


if __name__ == '__main__':
    # Allocate starting array
    val = cuda.device_array(8, np.uint8)
    # Copy to GPU memory
    val.copy_to_device(np.frombuffer(0b0001001100110100010101110111100110011011101111001101111111110001.to_bytes(8, 'big'), np.uint8))

    # Allocate result array
    p_val = cuda.device_array(len(val) - 1, np.uint8)

    print('Starting GPU kernel')
    # Start kernel on 56 threads
    start_t = time.time_ns()
    kernel_main[1, 56](cuda.to_device(val), p_val, PC_1)
    time = time.time_ns() - start_t

    print('Execution time: {} ns\n'.format(time))

    # Get result from GPU memory
    res = p_val.copy_to_host()

    expected = 0b11110000110011001010101011110101010101100110011110001111
    actual = int.from_bytes(res, 'big')

    print('Expected:\t{}'.format(bin(expected)))
    print('Actual:\t\t{}'.format(bin(actual)))
    print('Successful: {}'.format(expected == actual))
