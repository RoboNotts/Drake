import torch
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('filename')
    args.add_argument('outfile')
    a = args.parse_args()

    model = torch.load(a.filename)
    torch.save(model.state_dict(), a.outfile)