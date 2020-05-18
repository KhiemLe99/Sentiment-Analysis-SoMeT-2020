# arguments
# --mode                ["train", "test"]
# --pretrained          ["fasttext"]
# --head_model          ["fc", "lstm", "gru", "lstm-attn", "gru-attn", "lstm-cnn", "gru-cnn", "cnn", "transformer"]                      
# --dataset             ["aivivn", "tiki"]
# --use_dataset         [-1]

# example for train: python run.py --mode="train" --pretrained="fasttext" --head_model="fc" --dataset="aivivn" --use_dataset=-1 --num_epochs=30 --batch_size=256 --learning_rate=1e-3
# example for test : python run.py --mode="test" --pretrained="fasttext" --head_model="fc" --weights="fastText/logs/aivivn/full/fasttext_fc_full.pth" --dataset="aivivn"

import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--head_model', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--use_dataset', default=-1, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)

args = parser.parse_args()

if args.mode == "train":
    if args.pretrained == "fasttext":
        from fastText.main import train
        train(args.head_model, args.dataset, args.use_dataset, args.num_epochs, args.batch_size, args.learning_rate)
elif args.mode == "test" and args.weights is not None:
    if args.pretrained == "fasttext":
        from fastText.main import test
        test(args.head_model, args.weights, args.dataset)