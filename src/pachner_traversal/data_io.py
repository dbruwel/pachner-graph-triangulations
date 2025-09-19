import h5py
import numpy as np


def get_max_len_txt(input_text_file):
    max_len = 0

    with open(input_text_file) as f:
        line = f.readline()
        while line:
            max_len = max(max_len, len(line))
            line = f.readline()

    return max_len


def convert_to_hdf5(input_text_file, output_hdf5_file):
    max_len = get_max_len_txt(input_text_file)
    with open(input_text_file, "r", encoding="utf-8") as f:
        with h5py.File(output_hdf5_file, "w") as hf:
            dt = h5py.string_dtype(encoding="utf-8", length=max_len)
            dset = hf.create_dataset("isos", shape=(0,), maxshape=(None,), dtype=dt)

            chunk_size = 10000
            lines_buffer = []

            for line in f:
                line = line.strip()
                lines_buffer.append(line)

                if len(lines_buffer) >= chunk_size:
                    dset.resize(dset.shape[0] + len(lines_buffer), axis=0)
                    dset[-len(lines_buffer) :] = lines_buffer
                    lines_buffer = []

            if lines_buffer:
                dset.resize(dset.shape[0] + len(lines_buffer), axis=0)
                dset[-len(lines_buffer) :] = lines_buffer


class Dataset:
    def __init__(
        self,
        hdf5_file: str,
        num_test_samps: int = 1_000,
    ):
        self.hdf5_file = hdf5_file
        self.num_test_samps = num_test_samps

        self.data_size = self.get_data_size()
        self.chars = self.get_chars()
        self.max_len = self.get_max_len()

        self.setup_train_test()

    def __len__(self):
        return self.data_size

    def __contains__(self, item):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]

            chunk_size = 10_000

            for i in range(0, len(self), chunk_size):
                end_index = min(i + chunk_size, len(self))
                chunk = dset[i:end_index]
                for s in chunk:
                    if s.decode("utf-8") == item:
                        return True
        return False

    def get_chars(self):
        chars = set()
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]
            chunk_size = 10_000

            for i in range(0, len(self), chunk_size):
                end_index = min(i + chunk_size, len(self))
                chunk = dset[i:end_index]
                for s in chunk:
                    chars.update(s.decode("utf-8"))

        chars = sorted(list(chars))

        return chars

    def get_max_len(self):
        max_len = 0
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]
            total_strings = dset.shape[0]
            chunk_size = 10_000

            for i in range(0, total_strings, chunk_size):
                end_index = min(i + chunk_size, total_strings)
                chunk = dset[i:end_index]
                for s in chunk:
                    max_len = max(max_len, len(s.decode("utf-8")))

        return max_len

    def get_data_size(self):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]
            data_size = dset.shape[0]

        return data_size

    def read_lines(self, indices):
        with h5py.File(self.hdf5_file, "r") as hf:
            dset = hf["isos"]
            lines = dset[indices]
            return [line.decode("utf-8") for line in lines]

    def setup_train_test(self):
        assert self.num_test_samps < 0.2 * len(self)

        self.test_idx = np.random.choice(
            len(self), size=self.num_test_samps, replace=False
        )
        self.test_idx = np.sort(self.test_idx)
        idx_map_dict = {
            t: len(self) - self.num_test_samps + i for i, t in enumerate(self.test_idx)
        }
        self.idx_map = np.vectorize(lambda x: idx_map_dict.get(x, x))
        self.test_data = self.read_lines(self.test_idx)

    def samp_batch_idx(self, batch_size: int = 32):
        train_idx = np.random.choice(
            len(self) - self.num_test_samps, size=batch_size, replace=True
        )
        remap_idx = np.intersect1d(train_idx, self.test_idx)
        if len(remap_idx) > 0:
            remap_idx = self.idx_map(remap_idx)
        valid_idx = np.setdiff1d(train_idx, self.test_idx)

        batch_idx = np.concatenate([valid_idx, remap_idx])
        batch_idx = np.sort(batch_idx)

        return batch_idx

    def samp_batch(self, batch_size: int = 32):
        batch_idx = self.samp_batch_idx(batch_size)
        batch_data = self.read_lines(batch_idx)

        return batch_data


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

    def encode(self, batch):
        encode_in = (
            lambda x: [self.char_to_id["[BOS]"]]
            + [self.char_to_id[c] for c in x]
            + [self.char_to_id["[PAD]"]] * (self.dataset.max_len - len(x))
        )
        encode_out = (
            lambda x: [self.char_to_id[c] for c in x]
            + [self.char_to_id["[EOS]"]]
            + [self.char_to_id["[PAD]"]] * (self.dataset.max_len - len(x))
        )

        batch_input = [encode_in(x) for x in batch]
        batch_label = [encode_out(x) for x in batch]

        return np.array(batch_input), np.array(batch_label)

    def decode(self, batch):
        strip_excess = (
            lambda x_str: x_str.replace("[BOS]", "")
            .replace("[EOS]", "")
            .replace("[PAD]", "")
        )
        decode_in = lambda x: "".join([self.id_to_char[i] for i in x])

        return [strip_excess(decode_in(x)) for x in batch]
