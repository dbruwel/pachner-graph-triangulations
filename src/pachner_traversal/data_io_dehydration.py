import logging
import pathlib
from pathlib import Path

import h5py
import numpy as np

from pachner_traversal.utils import get_data_root

logger = logging.getLogger(__name__)
data_root = get_data_root()


def get_max_len_txt(input_text_file):
    max_len = 0

    with open(input_text_file) as f:
        line = f.readline()
        while line:
            max_len = max(max_len, len(line))
            line = f.readline()

    return max_len


def convert_to_hdf5(input_text_file, output_hdf5_file, max_len=None):
    i = 1
    if max_len is None:
        max_len = get_max_len_txt(input_text_file)
    with open(input_text_file, "r", encoding="utf-8") as f:
        with h5py.File(output_hdf5_file, "w") as hf:
            dt = h5py.string_dtype(encoding="utf-8", length=max_len)
            dset = hf.create_dataset(
                "isos",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                chunks=(100_000,),
            )

            chunk_size = 1_000_000
            lines_buffer = []

            for line in f:
                line = line.strip()
                lines_buffer.append(line)

                if len(lines_buffer) >= chunk_size:
                    logger.info(f"Writing Chunk {i}.")
                    dset.resize(dset.shape[0] + len(lines_buffer), axis=0)
                    dset[-len(lines_buffer) :] = lines_buffer
                    lines_buffer = []
                    i += 1

            if lines_buffer:
                dset.resize(dset.shape[0] + len(lines_buffer), axis=0)
                dset[-len(lines_buffer) :] = lines_buffer


def split_to_encoding(signature: str) -> list[str]:
    lead_char = signature[0]
    N = ord(lead_char) - 97
    signature = signature[1:]  # Remove the first character
    new_tet_chars = signature[: len(signature) - 2 * (N + 1)]
    where_chars = signature[len(signature) - 2 * (N + 1) : -(N + 1)]
    how_chars = signature[-(N + 1) :]

    new_tet_split_enc = ["N" + c for c in new_tet_chars]
    where_split_enc = ["W" + c for c in where_chars]
    how_split_enc = ["H" + c for c in how_chars]

    return [lead_char] + new_tet_split_enc + where_split_enc + how_split_enc


def merge_pairs(split_enc: list[str]) -> str:
    signature = "".join(split_enc)
    signature = signature.replace("N", "")
    signature = signature.replace("W", "")
    signature = signature.replace("H", "")

    return signature


class Dataset:
    def __init__(
        self,
        hdf5_file: pathlib.Path,
        num_test_samps: int | None = None,
        chars: list | None = None,
        max_len: int | None = None,
        store_in_memory: bool = False,
        test_rng_seed: int = 42,
    ):
        self.hdf5_file = hdf5_file
        self.num_test_samps = num_test_samps
        self.store_in_memory = store_in_memory
        self.test_rng_seed = test_rng_seed

        logger.debug("Getting data size")
        self.data_size = self.get_data_size()

        if (chars is None) or (max_len is None):
            logger.info("Getting chars and max length.")
            self.chars, self.max_len = self.get_chars_and_max_len()
        else:
            self.chars = chars
            self.max_len = max_len

        if self.store_in_memory:
            self.read_into_memory()

        if self.num_test_samps is not None:
            self.setup_train_test()

    def __len__(self):
        return self.data_size

    def __contains__(self, item):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]

            chunk_size = 10_000

            for i in range(0, len(self), chunk_size):
                end_index = min(i + chunk_size, len(self))
                chunk = dset[i:end_index]  # type: ignore
                for s in chunk:  # type: ignore
                    if s.decode("utf-8") == item:
                        return True
        return False

    def get_chars_and_max_len(self):
        chars = set()
        max_len = 0
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]
            chunk_size = 10_000

            for i in range(0, len(self), chunk_size):
                end_index = min(i + chunk_size, len(self))
                chunk = dset[i:end_index]  # type: ignore
                for s in chunk:  # type: ignore
                    string = s.decode("utf-8")
                    pairs = split_to_encoding(string)
                    chars.update(pairs)
                    max_len = max(max_len, len(pairs))

        chars = sorted(list(chars))

        return chars, max_len

    def get_data_size(self, dset_name="isos"):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf[dset_name]
            data_size = dset.shape[0]  # type: ignore

        return data_size

    def read_lines(self, indices, dset_name="isos"):
        logger.debug("Setting up mapping.")
        unique_indices, inverse_map = np.unique(indices, return_inverse=True)
        sorted_indices = np.sort(unique_indices)

        if self.store_in_memory:
            logger.debug("Reading from memory.")
            unique_lines = self.all_lines[sorted_indices]
            restored_lines = unique_lines[inverse_map]
            if dset_name == "isos":
                res = np.array([line.decode("utf-8") for line in restored_lines])
            else:
                res = restored_lines
            return res

        else:
            with h5py.File(self.hdf5_file, "r", rdcc_nbytes=4 * (1024**3)) as hf:
                dset = hf[dset_name]
                unique_lines = np.array(dset[sorted_indices])  # type: ignore
                restored_lines = unique_lines[inverse_map]
                if dset_name == "isos":
                    res = np.array([line.decode("utf-8") for line in restored_lines])
                else:
                    res = restored_lines
                return res

    def read_into_memory(self, dset_name="isos"):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf[dset_name]
            all_lines = dset[:]  # type: ignore
            all_lines = np.array(all_lines)
            self.all_lines = all_lines

    def setup_train_test(self):
        assert self.num_test_samps is not None, "`num_test_samps` must be defined."

        rng = np.random.default_rng(seed=self.test_rng_seed)
        self.test_idx = rng.choice(len(self), size=self.num_test_samps, replace=False)
        self.test_idx = np.sort(self.test_idx)
        self.test_data = self.read_lines(self.test_idx)

    def read_all_data(self, dset_name="isos"):
        return self.read_lines(np.arange(len(self)), dset_name=dset_name)


class Encoder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.setup_mappings()

    def setup_mappings(self):
        self.char_to_id = {ch: i for i, ch in enumerate(self.dataset.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.dataset.chars)}

        self.char_to_id["[BOS]"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "[BOS]"
        self.char_to_id["[EOS]"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "[EOS]"
        self.char_to_id["[PAD]"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "[PAD]"

    @staticmethod
    def strip_excess(x_str):
        return x_str == "[PAD]" or x_str == "[BOS]" or x_str == "[EOS]"

    def encode_in(self, x):
        pad_len = self.dataset.max_len - len(x)
        res = [self.char_to_id["[BOS]"]]
        res = res + [self.char_to_id[c] for c in x]
        res = res + [self.char_to_id["[PAD]"]] * pad_len

        return res

    def encode_out(self, x):
        pad_len = self.dataset.max_len - len(x)
        res = [self.char_to_id[c] for c in x]
        res = res + [self.char_to_id["[EOS]"]]
        res = res + [self.char_to_id["[PAD]"]] * pad_len
        return res

    def decode_in(self, x):
        return [
            self.id_to_char[i] for i in x if not self.strip_excess(self.id_to_char[i])
        ]

    def encode(self, batch):
        batch_input = [self.encode_in(split_to_encoding(x)) for x in batch]
        batch_label = [self.encode_out(split_to_encoding(x)) for x in batch]

        return np.array(batch_input), np.array(batch_label)

    def decode(self, batch):
        return [merge_pairs(self.decode_in(x)) for x in batch]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting.")

    dehydration_root = data_root / "input_data" / "dehydration"

    input_path = Path.home() / "final_unique.txt"
    hdf5_file = dehydration_root / "processed" / "spheres_15_1b.hdf5"

    convert_to_hdf5(input_path, hdf5_file, max_len=41)
