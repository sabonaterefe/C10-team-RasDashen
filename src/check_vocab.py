from src.tokenizer import Tokenizer

t = Tokenizer('data/tokenizer.json')
print('vocab size', len(t.vocab))
for ch in ['A', ' ', '[', ']', '!']:
    print(repr(ch), ch in t.vocab)
