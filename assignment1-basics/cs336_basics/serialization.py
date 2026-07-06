import yaml
import base64

def bytes_to_b64(x):
    return base64.b64encode(x).decode("ascii")


def save_tokenizer_yaml(vocab, merges, vocab_filename, merges_filename):

    vocab_serializable = {
        int(k): bytes_to_b64(v)
        for k, v in vocab.items()
    }

    merges_serializable = [
        (
            bytes_to_b64(a),
            bytes_to_b64(b)
        )
        for a, b in merges
    ]

    with open(vocab_filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "vocab": vocab_serializable
            },
            f,
            allow_unicode=True,
            sort_keys=False
        )
    with open(merges_filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "merges": merges_serializable
            },
            f,
            allow_unicode=True,
            sort_keys=False
        )
                
           

def b64_to_bytes(s):
    return base64.b64decode(s.encode("ascii"))


def load_tokenizer_yaml(vocab_filename, merges_filename):

    with open(vocab_filename, "r", encoding="utf-8") as f:
        data_vocab = yaml.safe_load(f)

    with open(merges_filename, "r", encoding="utf-8") as f:
        data_merges = yaml.safe_load(f)

    vocab = {
        int(k): b64_to_bytes(v)
        for k, v in data_vocab["vocab"].items()
    }

    merges = [
        (
            b64_to_bytes(a),
            b64_to_bytes(b)
        )
        for a, b in data_merges["merges"]
    ]

    return vocab, merges