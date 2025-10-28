from dst.tokenizer import DSTTokenizer


def test_reversibility():
    corpus = [
        "GET /index.html HTTP/1.1",
        "Host: example.com",
        "image: nginx:1.21-alpine",
    ]
    tokenizer = DSTTokenizer.train(corpus, min_freq=1)
    assert tokenizer.verify(corpus), "Reversibility check failed."
