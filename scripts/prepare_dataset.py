import tiktoken
import numpy as np
from deepseek_lab.data import ShardWriter

def tokenize_document(text):
    enc = tiktoken.get_encoding("gpt2")
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))

    if (max(tokens) > np.iinfo(np.uint16).max):
        raise ValueError("Some tokens can't be represented as uint16")
    tokens = np.array(tokens, dtype=np.uint16)

    return tokens

def main():
    text = "Modern Watermelons originated in Sudan, where it was first cultivated.[2][3] Kordofan melons from Sudan are the closest relatives and may be progenitors of modern, cultivated watermelons.[4] Wild watermelon seeds were found in Uan Muhuggiag, a prehistoric site in Libya that dates to approximately 3500 BC.[5] In 2022, a study was released that traced 6,000-year-old watermelon seeds found in the Libyan desert to the Egusi seeds of Nigeria, West Africa.[6] Watermelons were domesticated in Sudan and cultivated in Egypt by 2000 BC; an image of an apparent one on a tray has been found in an Egyptian tomb dating at least to 4000 years ago. Those were not of the sweet modern variety, but Hebrew texts from early in the Christian era place watermelon with other sweet table fruits, and dessert watermelons spread across the Mediterranean world during Roman antiquity.[7]"
    shard_writer = ShardWriter(10, "data")
    tokens = tokenize_document(text)
    shard_writer.add(tokens)
    shard_writer.finish()

if __name__ == "__main__":
    main()