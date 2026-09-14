from tokenizer import Tokenizer

t = Tokenizer('tokenizer.json')
print('vocab size', len(t.vocab))
for ch in ['A', ' ', '[', ']', '!']:
    print(repr(ch), ch in t.vocab)
