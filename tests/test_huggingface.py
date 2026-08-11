from kvbridge.huggingface import tokenizer_fingerprint


class FakeTokenizer:
    special_tokens_map = {"eos_token": "</s>"}

    def get_vocab(self):
        return {"hello": 1, "world": 2}

    def get_added_vocab(self):
        return {"</s>": 3}


def test_tokenizer_fingerprint_is_deterministic() -> None:
    assert tokenizer_fingerprint(FakeTokenizer()) == tokenizer_fingerprint(FakeTokenizer())
