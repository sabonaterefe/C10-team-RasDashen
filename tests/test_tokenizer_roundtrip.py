import unittest
import json
from src.tokenizer import Tokenizer


class RoundTripTest(unittest.TestCase):
    def test_roundtrip_debug_corpus(self):
        with open("debug_corpus.txt", "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]
        # ensure model exists
        try:
            with open("data/tokenizer.json", "r", encoding="utf-8") as f:
                pass
        except FileNotFoundError:
            # train a small model
            from src.train_tokenizer import train
            corpus = "\n".join(lines)
            vocab_map = train(corpus, vocab_size=200)
            with open("data/tokenizer.json", "w", encoding="utf-8") as f:
                json.dump({"vocab": vocab_map}, f, ensure_ascii=False)

        t = Tokenizer("data/tokenizer.json")
        encoded = t.encode(lines)
        decoded = t.decode(encoded)
        self.assertEqual(decoded, lines)


if __name__ == "__main__":
    unittest.main()
