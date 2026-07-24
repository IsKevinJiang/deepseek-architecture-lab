import numpy as np
from pathlib import Path
import torch
#Dataset Sharding so data doesn't have to all sit in one file.
class ShardWriter:
    def __init__(self, shard_size, output_dir, split):
        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")

        self.shard_size = shard_size
        self.position = 0
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.shard_index = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split

    def add(self, tokens):
        token_position = 0
        total = len(tokens)
        while token_position < total:
            available_space = self.shard_size - self.position
            tokens_remaining = total - token_position
            amount_to_copy = min(available_space, tokens_remaining)

            #Filling up buffer 
            self.buffer[self.position:self.position + amount_to_copy] = tokens[token_position:token_position + amount_to_copy]

            self.position += amount_to_copy
            token_position += amount_to_copy

            #Creates a new .bin since the buffer is full
            if self.position == self.shard_size:
                self.flush()

    def flush(self):
        if self.position == 0:
            return

        header = np.zeros(256, np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = self.position

        filename = self.output_dir / f"dataset_{self.split}_{self.shard_index:06d}.bin"
        with open(filename, "wb") as file:
            file.write(header.tobytes())
            file.write(self.buffer[:self.position].tobytes())
        self.shard_index += 1
        self.position = 0

    def finish(self):
        self.flush()

class ShardDataLoader:
    def __init__(self, data_dir, batch_size, sequence_length, split, seed=32):
        self.position = 0
        self.current_shard = 0
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.file_path = Path(data_dir)
        self.epoch = 0
        self.rng = np.random.default_rng(seed)

        if (split not in ["train", "val"]):
            raise ValueError("Split needs to be train or val")
        self.split = split

        self.shards = []
        files = self.file_path.glob(f"dataset_{split}_*.bin")
        for file in files:
            self.shards.append(file)
        self.shards.sort()

        if not self.shards:
            raise ValueError("No shards found")
        #Reshuffling helps with training so model doesn't see the same stuff in the same order
        if self.split == "train":
            self.rng.shuffle(self.shards)
        self._load_shard(0)

    def _load_shard(self, shard_index):
        shard_file = self.shards[shard_index]
        with open(shard_file, "rb") as file:
            header = np.fromfile(file, dtype=np.int32, count=256)

            if (header.size != 256 or header[0] != 20240520 or header[1] != 1):
                raise ValueError("File not right format")
            if (header[2] <= 0):
                raise ValueError("Token count not positive")
            token_count= header[2]

            self.tokens = np.memmap(shard_file, dtype=np.uint16, offset=1024, shape=(int(token_count),), mode="r")
        self.current_shard = shard_index
        self.position = 0

    def next_batch(self):
        batch = self.batch_size * self.sequence_length + 1
        shards_tried = 0
        while (len(self.tokens) - self.position < batch):
            if shards_tried >= len(self.shards):
                raise ValueError(f"No shard contains enough tokens for a batch of size {batch}")
            next_shard = (self.current_shard + 1) % len(self.shards)
            if (next_shard == 0):
                self.epoch += 1
                if (self.split == "train"):
                    self.rng.shuffle(self.shards)
            self._load_shard(next_shard)
            shards_tried += 1

        window = self.tokens[self.position: self.position + batch]
        inputs = torch.tensor(window[:len(window)-1], dtype=torch.long).reshape(self.batch_size, self.sequence_length)
        outputs = torch.tensor(window[1:], dtype=torch.long).reshape(self.batch_size, self.sequence_length)
        self.position += batch - 1

        return inputs, outputs

    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "current_shard": self.current_shard,
            "token_position": self.position,
            "rng_state": self.rng.bit_generator.state,
            "split": self.split,
            "batch_size": self.batch_size,
            "sequence_length" : self.sequence_length,
            "shard_names": [shard.name for shard in self.shards]
        }

        return state_dict

    def load_state_dict(self, state):
        if (self.split != state["split"] or self.batch_size != state["batch_size"] or self.sequence_length != state["sequence_length"]):
            raise ValueError("Split, batch size, or sequence length seems to mismatch this loader")
        
        self.shards =  [self.file_path / name for name in state["shard_names"]]
        if not self.shards:
            raise ValueError("Checkpoint has no shards")
        missing_shards = [path for path in self.shards if not path.is_file()]
        if missing_shards:
            raise FileNotFoundError(f"Missing shard files: {missing_shards}")

        current_shard = state["current_shard"]
        if not 0 <= current_shard < len(self.shards):
            raise ValueError("Saved current shard index is invalid")
        self._load_shard(current_shard)

        token_position = state["token_position"]
        if not 0 <= token_position <= len(self.tokens):
            raise ValueError("Saved token position is invalid")
        self.position = token_position

        self.epoch = state["epoch"]
        self.rng.bit_generator.state = state["rng_state"]


