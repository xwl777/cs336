from cs336_basics.bpe import *
vocab, merges = train_bpe(input_path="./data/TinyStoriesV2-GPT4-valid.txt",
                          vocab_size=10000,
                          special_tokens=["<|endoftext|>"])