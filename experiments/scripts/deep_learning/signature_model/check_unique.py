import os
from multiprocessing import Pool

import h5py
from regina.engine import Triangulation3
from tqdm import tqdm

FILE_PATH = "/g/data/io00/js1886/pachner-graph-triangulations/data/input_data/dehydration/processed/spheres_15_1b.hdf5"
DATASET = "isos"
BLOCK_SIZE = 5_000_000  # entries per block
WORKERS = os.cpu_count()

print(f"There are {WORKERS} workers.")

sphere_list_path = "/g/data/io00/db4467/generated_spheres_iso.txt"
with open(sphere_list_path) as f:
    generated_isosigs = f.readlines()
    generated_isosigs = [s.strip() for s in generated_isosigs]

print("Complete setup.")


def check(sig):
    isosig = Triangulation3.rehydrate(sig).isoSig()
    return isosig in generated_isosigs


def count_block(bounds):
    start, stop = bounds
    with h5py.File(FILE_PATH, "r") as f:
        block = f[DATASET][start:stop]
    return sum(1 for x in tqdm(block) if check(x.decode()))


if __name__ == "__main__":
    with h5py.File(FILE_PATH, "r") as f:
        n = f[DATASET].shape[0]

    blocks = [(s, min(s + BLOCK_SIZE, n)) for s in range(0, n, BLOCK_SIZE)]

    with Pool(WORKERS) as pool:
        results = pool.imap_unordered(count_block, blocks)
        total = sum(tqdm(results, total=len(blocks), unit="block"))

    print(f"{total:,} / {n:,} entries returned True")
