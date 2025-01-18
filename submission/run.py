import pathlib
import sys
from typing import Any, Generator, Literal, Tuple

directory = pathlib.Path(__file__).parent
sys.path.insert(0, str(directory.resolve()))


import torch
from competition import BaseEvaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer

#################################################################################
#                                                                               #
#   This file gets run when you submit your work for evaluation on the DOXA     #
#   AI platform. Modify the predict() method to implement your own strategy!    #
#                                                                               #
#################################################################################


class Evaluator(BaseEvaluator):
    def predict(
        self, text: str
    ) -> Generator[Tuple[int, int, Literal["Q", "A"]], Any, None]:
        """Write all the code you need to generate predictions for the test set here!

        You only need to classify sections of text as being questions ("Q") or potential
        question answers ("A"). We will assume everything else does not match. See the
        competition page for more on how your submission is evaluated.

        Args:
            text (str): This is the plain-text test set

        Yields:
            Tuple[int, int, Literal["Q", "A"]: A starting index (inclusive), an ending index (exclusive)
                                                and a categorisation ("Q" or "A").
        """

        # Implement your own strategy here!

        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        model = AutoModelForTokenClassification.from_pretrained(directory / "model")

        inputs = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            padding=True,
            max_length=512,
            stride=64,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(model.device)

        done = set()

        with torch.inference_mode():
            for input_ids, attention_mask, offsets in zip(inputs["input_ids"], inputs["attention_mask"], inputs["offset_mapping"]):  # type: ignore
                predictions = torch.argmax(
                    model(input_ids=input_ids, attention_mask=attention_mask).logits,
                    dim=2,
                )

                for t, (start, end) in zip(predictions[0], offsets):
                    if (start, end) in done or (start == 0 and end == 0):
                        continue

                    done.add((start, end))

                    predicted_token_class = model.config.id2label[t.item()]
                    if predicted_token_class == "question":
                        yield (start, end, "Q")
                    elif predicted_token_class == "answer":
                        yield (start, end, "A")


if __name__ == "__main__":
    Evaluator().run()
