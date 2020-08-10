from Crypto.Cipher import DES3
from Crypto.Cipher import DES
from Crypto.Util import Padding

from des_numba import create_lt_entry_numba

from Crypto.Random import get_random_bytes

import multiprocessing as mp
from multiprocessing import Pool
from functools import reduce

from numba import jit, cuda

from tqdm import tqdm, trange


def keys_from_value(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]


def generate_key():
    while True:
        try:
            key = DES3.adjust_key_parity(get_random_bytes(24))
            break
        except ValueError:
            pass
    return key


def encrypt(plaintext, key):
    des3 = DES3.new(key, DES3.MODE_ECB)

    return des3.encrypt(Padding.pad(plaintext, 8))


def decrypt(crypto, key):
    des3 = DES3.new(key, DES3.MODE_ECB)

    return Padding.unpad(des3.decrypt(crypto), 8)


def test_3des(plaintext, key):
    cipher1 = encrypt(plaintext, key)

    k1 = key[:8]
    k2 = key[8:16]
    k3 = key[16:24]

    des1 = DES.new(k1, DES.MODE_ECB)
    des2 = DES.new(k2, DES.MODE_ECB)
    des3 = DES.new(k3, DES.MODE_ECB)

    cipher2 = des3.encrypt(des2.decrypt(des1.encrypt(Padding.pad(plaintext, 8))))

    print('3DES vs. concatenation of DES leads to same cipher: {}'.format('passed' if cipher1 == cipher2 else 'failed'))


def attack_single_threaded(plaintext, cipher, key_length=8):
    max_key = 2 ** (8 * key_length)

    total_keys = max_key ** 2
    progress = tqdm(desc='Key 1 and 2', total=total_keys, unit='keys')

    left_table = {}
    for k1 in range(max_key):
        for k2 in range(max_key):
            kb1 = k1.to_bytes(8, byteorder='big')
            kb2 = k2.to_bytes(8, byteorder='big')

            des1 = DES.new(kb1, DES.MODE_ECB)
            des2 = DES.new(kb2, DES.MODE_ECB)

            c1 = des2.decrypt(des1.encrypt(Padding.pad(plaintext, 8)))

            left_table[kb1, kb2] = c1
            progress.update()

    total_keys = max_key
    progress = tqdm(desc='Key 3', total=total_keys, unit='keys')

    right_table = {}
    for k3 in range(max_key):
        kb3 = k3.to_bytes(8, byteorder='big')

        des3 = DES.new(kb3, DES.MODE_ECB)

        p1 = des3.decrypt(cipher)

        right_table[kb3] = p1
        progress.update()

    common = list(set(right_table.values()).intersection(left_table.values()))

    k1k2 = keys_from_value(left_table, common[0])
    k3 = keys_from_value(right_table, common[0])

    return k1k2, k3


# Numba

def attack_numba(plaintext, cipher, key_length=8):
    max_key = 2 ** (8 * key_length)

    total_keys = max_key ** 2
    # progress = tqdm(desc='Key 1 and 2', total=total_keys, unit='keys')

    left_table = {}
    for k1 in range(max_key):
        for k2 in range(max_key):
            kb1 = k1.to_bytes(8, byteorder='big')
            kb2 = k2.to_bytes(8, byteorder='big')

            des1 = DES.new(kb1, DES.MODE_ECB)
            des2 = DES.new(kb2, DES.MODE_ECB)

            c1 = create_lt_entry_numba(plaintext, kb1, kb2)

            left_table[kb1, kb2] = c1
            # progress.update()

    total_keys = max_key
    # progress = tqdm(desc='Key 3', total=total_keys, unit='keys')

    right_table = {}
    for k3 in range(max_key):
        kb3 = k3.to_bytes(8, byteorder='big')

        des3 = DES.new(kb3, DES.MODE_ECB)

        p1 = des3.decrypt(cipher)

        right_table[kb3] = p1
        # progress.update()

    common = list(set(right_table.values()).intersection(left_table.values()))

    k1k2 = keys_from_value(left_table, common[0])
    k3 = keys_from_value(right_table, common[0])

    return k1k2, k3


@cuda.jit
def _create_partial_left_table(plaintext, from_key, to_key, key_length=8, show_progress=False):
    max_key = 2 ** (8 * key_length)

    total_keys = (to_key - from_key) * max_key
    progress = tqdm(desc='Key 1 and 2', total=total_keys, disable=not show_progress, unit='keys')

    left_table = {}
    # Calculates requested keys
    for k1 in range(from_key, to_key):
        for k2 in range(max_key):
            kb1 = k1.to_bytes(8, byteorder='big')
            kb2 = k2.to_bytes(8, byteorder='big')

            des1 = DES.new(kb1, DES.MODE_ECB)
            des2 = DES.new(kb2, DES.MODE_ECB)

            c1 = des2.decrypt(des1.encrypt(Padding.pad(plaintext, 8)))

            left_table[kb1, kb2] = c1
            progress.update()

    return left_table


def _create_partial_right_table(cipher, from_key, to_key, show_progress=True):
    total_keys = to_key - from_key
    progress = tqdm(desc='Key 3', total=total_keys, disable=not show_progress, unit='keys')

    right_table = {}
    for k3 in range(from_key, to_key):
        kb3 = k3.to_bytes(8, byteorder='big')

        des3 = DES.new(kb3, DES.MODE_ECB)

        p1 = des3.decrypt(cipher)

        right_table[kb3] = p1
        progress.update()

    return right_table


def attack(plaintext, cipher, key_length=8):
    max_key = 2 ** (8 * key_length)

    processes_count = mp.cpu_count()
    keys_per_process = int(max_key / processes_count)

    key_ranges = []
    for i in range(processes_count):
        key_ranges.append((i * keys_per_process, (i + 1) * keys_per_process))

    left_table_args = [[plaintext, *k, key_length] for k in key_ranges]
    right_table_args = [[cipher, *k] for k in key_ranges]

    # Show progress for first process only
    left_table_args[0].append(True)
    right_table_args[0].append(True)

    with Pool(processes_count) as p:
        res = p.starmap_async(_create_partial_left_table, left_table_args)
        left_tables = res.get()

        res = p.starmap_async(_create_partial_right_table, right_table_args)
        right_tables = res.get()

        # Merge dictionaries from all processes
    left_table = reduce((lambda d1, d2: {**d1, **d2}), left_tables)
    right_table = reduce((lambda d1, d2: {**d1, **d2}), right_tables)

    common = list(set(right_table.values()).intersection(left_table.values()))

    candidates_k1k2 = keys_from_value(left_table, common[0])
    candidates_k3 = keys_from_value(right_table, common[0])

    keys = []
    for k1, k2 in candidates_k1k2:
        for k3 in candidates_k3:
            keys.append(k1 + k2 + k3)
    return keys

# Main
if __name__ == '__main__':
    p = b'Secret secret'
    k = b'\x01\x10'.rjust(8, b'\x00') + \
        b'\xaa\x0f'.rjust(8, b'\x00') + \
        b'\x2d\x01'.rjust(8, b'\x00')

    c = encrypt(p, k)

    p1 = decrypt(c, k)

    k1 = attack(p, c, 2)
    print('Key was {}'.format('found!' if k in k1 else 'not found...'))
