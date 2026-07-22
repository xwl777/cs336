from cs336_basics.bpe import *
from cs336_basics.serialization import *

DATASET = "TinyStoriesV2-GPT4-train"
vocab, merges = train_bpe(input_path=f"./data/{DATASET}.txt",
                          vocab_size=10000,
                          special_tokens=["<|endoftext|>"])

save_tokenizer_yaml(vocab, merges, f"vocab_{DATASET}.yaml", f"merges_{DATASET}.yaml",)