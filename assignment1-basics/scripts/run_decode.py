from cs336_basics.decoding import *
from cs336_basics.checkpoint import *
from cs336_basics.train import load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--vocab", type=str, help="vocab path of the tokenizer")
    parser.add_argument("--merges", type=str, help="merges path of the tokenizer")
    parser.add_argument("--config", type=str, help="config file of the model")
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = config.device
    dtype = getattr(torch, config.dtype.split(".")[-1])
    model = Transformer(**config.model, device=torch.device(device), dtype=dtype)
    model.to(device=torch.device(device), dtype=dtype)
    load_model(args.model, model)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])

    model.eval()
    print(decode(model=model,
                 tokenizer=tokenizer,
                 prompt=args.prompt,
                 context_length=config.model.context_length,
                 max_new_tokens=512,
                 temperature=1,
                 top_p=0.9,
                 end_of_text="<|endoftext|>"))

