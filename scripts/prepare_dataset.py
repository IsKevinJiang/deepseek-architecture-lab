import tiktoken
import numpy as np
from modelforge.data import ShardWriter
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

def tokenize_document(text):
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))

    if (max(tokens) > np.iinfo(np.uint16).max):
        raise ValueError("Some tokens can't be represented as uint16")
    tokens = np.array(tokens, dtype=np.uint16)

    return tokens

#Preparing the fineweb-edu dataset(Stops after 100 documents)
def main():
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    shard_writer = ShardWriter(10_000, "data/fineweb_smoke")
    documents = dataset.take(100)
    for document in tqdm(documents, total=100):
        tokens = tokenize_document(document["text"])
        shard_writer.add(tokens)
    shard_writer.finish()

if __name__ == "__main__":
    main()
