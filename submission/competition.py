import os
from typing import Any, Generator, Literal, Tuple


class BaseEvaluator:
    def predict(
        self, text: str
    ) -> Generator[Tuple[int, int, Literal["Q", "A"]], Any, None]:
        raise NotImplementedError

    def run(self):
        stream_directory = os.environ.get("DOXA_STREAMS")

        in_file = f"{stream_directory}/in" if stream_directory else "train_clean.txt"
        out_file = f"{stream_directory}/out" if stream_directory else "predictions.csv"

        with (
            open(in_file, "r", encoding="utf8") as r,
            open(out_file, "w") as w,
        ):
            w.write(f"OK\n")
            w.flush()

            for start, end, category in self.predict(r.read()):
                w.write(f"{start},{end},{category}\n")
                w.flush()
