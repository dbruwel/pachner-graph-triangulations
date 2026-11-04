import os
from multiprocessing import Pool

import h5py
from tqdm import tqdm

FILE_PATH = "/g/data/io00/js1886/pachner-graph-triangulations/data/input_data/dehydration/processed/spheres_15_1b.hdf5"
DATASET = "isos"
BLOCK_SIZE = 500_000  # entries per block
WORKERS = os.cpu_count()


sphere_list_path = "/g/data/io00/db4467/generated_spheres_dehydration.txt"
with open(sphere_list_path) as f:
    generated_sigs = f.readlines()
    generated_sigs = [s.strip() for s in generated_sigs]


def check(sig):
    # isosig = Triangulation3.rehydrate(sig).isoSig()
    return sig in generated_sigs


def count_block(bounds):
    start, stop = bounds
    with h5py.File(FILE_PATH, "r") as f:
        block = f[DATASET][start:stop]

    res = 0
    for x in block:
        if check(x.decode()):
            res = res + 1

    return res


if __name__ == "__main__":
    with h5py.File(FILE_PATH, "r") as f:
        n = f[DATASET].shape[0]

    blocks = [(s, min(s + BLOCK_SIZE, n)) for s in range(0, n, BLOCK_SIZE)]
    print(f"There are {len(blocks)} blocks")

    with Pool(WORKERS) as pool:
        results = pool.imap_unordered(count_block, blocks)
        total = sum(tqdm(results, total=len(blocks), unit="block"))

    print(f"{total:,} / {n:,} entries returned True")
