import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

def tokenize_document(text):
    tokens = enc.encode_ordinary(text)
    tokens.append(enc.eot_token)

    if (max(tokens) > np.iinfo(np.uint16).max):
        raise ValueError("Some tokens can't be represented as uint16")
    tokens = np.array(tokens, dtype=np.uint16)

    return tokens
