#!/usr/bin/env python3
"""Small regression checks for transliteration training utilities."""

from pathlib import Path
import importlib.util
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_unicode_to_translit.py"
spec = importlib.util.spec_from_file_location("train_unicode_to_translit", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def assert_close(actual: float, expected: float, eps: float = 1e-9) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"expected {expected}, got {actual}")


def test_edit_metrics() -> None:
    assert module.edit_distance("abc", "abc") == 0
    assert module.edit_distance("abc", "adc") == 1
    assert_close(module.cer("abc", "adc"), 1 / 3)
    assert_close(module.wer("a b", "a c"), 1 / 2)
    assert_close(module.cer("", ""), 0.0)


def test_repetition_metrics() -> None:
    assert module.max_char_run("") == 0
    assert module.max_char_run("aaabbc") == 3
    assert module.max_char_run("abc") == 1
    assert module.repeated_bigram_rate("aaaa") > module.repeated_bigram_rate("abcd")


def test_vocab_stops_at_eos() -> None:
    vocab = module.Vocab(["a", "b"])
    ids = [vocab.stoi["a"], vocab.eos_idx, vocab.stoi["b"]]
    assert vocab.decode(ids) == "a"


if __name__ == "__main__":
    test_edit_metrics()
    test_repetition_metrics()
    test_vocab_stops_at_eos()
    print("training utility tests passed")
