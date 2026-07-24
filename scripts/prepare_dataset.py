import tiktoken
import numpy as np
from modelforge.data import ShardWriter
from datasets import load_dataset
from tqdm import tqdm
import hashlib
from pathlib import Path
import json

enc = tiktoken.get_encoding("gpt2")
OUTPUT_DIR = Path("data/fineweb_60m")
TRAIN_TOKEN_TARGET = 60_000_000
VAL_TOKEN_TARGET = 2_000_000
SHARD_SIZE = 1_000_000
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

def tokenize_document(text):
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))

    if (max(tokens) > np.iinfo(np.uint16).max):
        raise ValueError("Some tokens can't be represented as uint16")
    tokens = np.array(tokens, dtype=np.uint16)

    return tokens


def assign_split(document):
    text = document["text"]
    encoded = text.encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 32
    if bucket == 0:
        return "val"
    else:
        return "train"
#Preparing the fineweb-edu dataset(Stops after reaching both token budgets)
def main():
    if (MANIFEST_PATH.exists()):
        print("Dataset already exists")
        return
    if (OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())):
        raise RuntimeError("Incomplete dataset already exists")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    train_writer = ShardWriter(SHARD_SIZE, OUTPUT_DIR, "train")
    val_writer = ShardWriter(SHARD_SIZE, OUTPUT_DIR, "val")
    train_tokens, val_tokens = 0, 0

    for document in tqdm(dataset):
        split = assign_split(document)
        if split == "train":
            if train_tokens >= TRAIN_TOKEN_TARGET:
                continue
            else:
                tokens = tokenize_document(document["text"])
                train_writer.add(tokens)
                train_tokens += len(tokens)

        elif split == "val":
            if( val_tokens >= VAL_TOKEN_TARGET):
                continue
            else:
                tokens = tokenize_document(document["text"])
                val_writer.add(tokens)
                val_tokens += len(tokens)

        if train_tokens >= TRAIN_TOKEN_TARGET and val_tokens >= VAL_TOKEN_TARGET:
            break
    if train_tokens < TRAIN_TOKEN_TARGET or val_tokens < VAL_TOKEN_TARGET:
        raise RuntimeError(f"Target tokens not hit, Train Tokens: {train_tokens}, Validation Tokens: {val_tokens}")

    train_writer.finish()
    val_writer.finish()

    manifest = {
        "source_dataset": "HuggingFaceFW/fineweb-edu",
        "source_subset": "sample-10BT",
        "tokenizer": "gpt2",
        "split_method": "sha256(text), bucket 0 of 32 is validation",
        "train_token_target": TRAIN_TOKEN_TARGET,
        "train_tokens": train_tokens,
        "validation_token_target": VAL_TOKEN_TARGET,
        "validation_tokens": val_tokens,
        "shard_size": SHARD_SIZE,
        "train_shards": train_writer.shard_index,
        "validation_shards": val_writer.shard_index,
    }

    temp_manifest_path = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    with temp_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    temp_manifest_path.replace(MANIFEST_PATH)

    print(
        f"Dataset complete: {train_tokens:,} training tokens, "
        f"{val_tokens:,} validation tokens"
    )

if __name__ == "__main__":
    main()
