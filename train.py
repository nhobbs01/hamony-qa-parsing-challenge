import json
import pandas as pd

from intervaltree import Interval, IntervalTree
from transformers import AutoTokenizer

from dotenv import load_dotenv

load_dotenv()


with open("./data/train_raw.txt", encoding="utf8") as f:
    raw_train = f.read()

with open("data/train_clean.txt", encoding="utf8") as g:
    clean_train = g.read()

with open("data/train_labels.json", encoding="utf8") as h:
    labels_train = json.load(h)

tree_q = IntervalTree(
    Interval(start, end) for start, end in labels_train["q"] if start != end
)

tree_a = IntervalTree(
    Interval(start, end) for start, end in labels_train["a"] if start != end
)


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Implement training script

# Handle command line arguments and provide default training args

# Fetch train and eval dataset from disk

# Tokenize datasets

# Setup training arguments
# Setup trainer
# Start training
# Save model