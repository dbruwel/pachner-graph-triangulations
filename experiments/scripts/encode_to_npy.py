import pathlib
import numpy as np
from regina import Triangulation3
from pachner_traversal.data_io_dehydration import Dataset
from pachner_traversal.glue_encoding import encode
from pachner_traversal.utils import data_path, results_path

if __name__ == "__main__":
    processed_data_path = data_path / "input_data" / "dehydration" / "processed"
    file_path = processed_data_path / "d_training_spheres_13.hdf5"
    save_path = data_path / "input_data" / "dit"
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(file_path, num_test_samps=1000)
    batch_size = 1000
    total = len(dataset)
    out_shape = (total, 144, 1296)
    mmap_path = save_path / "d_training_spheres_13.npy"
    mmap = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float32, shape=out_shape
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset.read_lines(np.arange(start, end))
        for i, sig in enumerate(batch):
            t = Triangulation3.fromIsoSig(sig)
            encoding = encode(t).astype(np.float32)
            mmap[start + i] = encoding
        print(f"Encoded {end} / {total}")
    mmap.flush()
    print(f"Saved {mmap.shape} to {mmap_path}")
