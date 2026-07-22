import numpy as np
from pathlib import Path

#Dataset Sharding so data doesn't have to all sit in one file.
class ShardWriter:
    def __init__(self, shard_size, output_dir):
        self.shard_size = shard_size
        self.position = 0
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.shard_index = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        if self.shard_index == 0:
            split = "val"
        else:
            split = "train"

        header = np.zeros(256, np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = self.position

        filename = self.output_dir / f"dataset_{split}_{self.shard_index:06d}.bin"
        with open(filename, "wb") as file:
            file.write(header.tobytes())
            file.write(self.buffer[:self.position].tobytes())
        self.shard_index += 1
        self.position = 0

    def finish(self):
        self.flush()

class ShardDataLoader:
    def __init__(self, data_dir, batch_size, sequence_length, split):
        self.position = 0
        self.current_shard = 0
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.file_path = Path(data_dir)

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
