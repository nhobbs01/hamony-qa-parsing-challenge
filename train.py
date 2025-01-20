from intervaltree import Interval, IntervalTree
import json
from transformers import AutoTokenizer
from datasets import Dataset
import evaluate
import numpy as np
from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from transformers import TrainingArguments, Trainer



MAX_LENGTH = 512
STRIDE = 32

class QATrainer():

    def __init__(self, debug = 1):
        self.label_list = ["other", "question", "answer"]
        self.seqeval = evaluate.load("seqeval")
        self.debug = debug

    def train(self):
        if self.debug == 1: 
            print("Reading test files")  
        # Read test file
        with open("data/train_raw.txt", encoding="utf8") as f:
            if self.debug == 1: 
                print("- Reading raw train")  
            raw_train = f.read()

        with open("data/train_clean.txt", encoding="utf8") as g:
            if self.debug == 1: 
                print("- Clean train")  
            clean_train = g.read()

        with open("data/train_labels.json", encoding="utf8") as h:
            if self.debug == 1: 
                print("- Labels train")  
            labels_train = json.load(h)


        # For faster labelling of data
        tree_q = IntervalTree(
            Interval(start, end) for start, end in labels_train["q"] if start != end
        )

        tree_a = IntervalTree(
            Interval(start, end) for start, end in labels_train["a"] if start != end
        )

        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

        label_list = ["other", "question", "answer"]

        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}

        
        tokenized_dataset = self.tokenize(clean_train, tokenizer, tree_q, tree_a, label2id)

        dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_dataset["input_ids"],
                "attention_mask": tokenized_dataset["attention_mask"],
                "labels": tokenized_dataset["labels"],
            }
        )

        # No suffle, always select the same data for training and test
        dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
        training_dataset = dataset['train']
        test_dataset = dataset['test']

        if(self.debug == 1):
            print("Training dataset size: ", len(training_dataset))
            print("Test dataset size: ", len(test_dataset))

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        model = AutoModelForTokenClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

        if(self.debug == 1):
            print("Model Name: ", model.name_or_path)
            print("Model size: ", model.num_parameters())



        training_args = TrainingArguments(
            output_dir="checkpoints",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            do_eval=True,
            save_strategy="epoch",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.evaluate()
        # END

    def tokenize(self, text, tokenizer, tree_q, tree_a, label2id):
        encodings = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            add_special_tokens=True,  # Includes the [CLS] and [SEP] tokens
        )

        all_token_labels = []
        for batch_index, (input_ids, offsets) in enumerate(
            zip(encodings["input_ids"], encodings["offset_mapping"])
        ):
            word_ids = encodings.word_ids(batch_index=batch_index)

            token_labels = []
            current_word_idx = None

            for word_id, (start, end) in zip(word_ids, offsets):
                if word_id is None:  # Special tokens like [CLS] or [SEP]
                    token_labels.append(-100)
                elif word_id != current_word_idx:  # New word
                    if len(tree_q.overlap(start, end)) > 0:
                        label = "question"
                    elif len(tree_a.overlap(start, end)) > 0:
                        label = "answer"
                    else:
                        label = "other"

                    token_labels.append(label2id[label])
                    current_word_idx = word_id
                else:  # Subword token
                    token_labels.append(-100)

            all_token_labels.append(token_labels)

        encodings["labels"] = all_token_labels

        return encodings

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    

if __name__ == "__main__":
    print("Starting trainer")
    trainer = QATrainer()
    trainer.train()