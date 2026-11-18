"""Tests: one block per milestone in README.md. Everything here uses tiny models so it runs in seconds on CPU."""
import math

import pytest
import torch

from app import core

STORY = ("The captain looked at the map. The map showed the island. "
         "The crew sailed to the island and the captain found the gold. ") * 40


# Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
def test_strip_gutenberg_keeps_only_the_book():
    raw = ("Licence blah\n*** START OF THE PROJECT GUTENBERG EBOOK TREASURE ISLAND ***\n"
           "Squire Trelawney asked me to write.\n*** END OF THE PROJECT GUTENBERG EBOOK ***\nmore licence")
    assert core.strip_gutenberg(raw).strip() == "Squire Trelawney asked me to write."


def test_normalize_rejoins_wrapped_lines_and_cleans_typography():
    raw = "“Ahoy,” said\r\nthe _captain_.\r\n\r\n\r\n[Illustration: a ship]\r\nNext  para—here."
    assert core.normalize(raw) == "\"Ahoy,\" said the captain.\n\nNext para--here.\n"


def test_corpus_is_built_from_public_domain_books():
    text = core.load_corpus()
    assert len(text) > 500_000
    assert "gutenberg" not in text.lower()
    assert "Long John Silver" in text and "Quatermain" in text  # both novels made it in


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
@pytest.fixture(scope="module")
def tok():
    return core.BPETokenizer.train(STORY, vocab_size=300)


def test_bpe_round_trips_any_text(tok):
    for text in (STORY[:200], "Unseen words: zebra, 42 doubloons! Ünïcødé ☠", ""):
        assert tok.decode(tok.encode(text)) == text


def test_bpe_merges_frequent_pairs_and_compresses(tok):
    assert tok.vocab_size == 300
    assert tok.decode([256]) in {" t", "th", "he", "e "}  # the most frequent pair is merged first
    assert len(tok.encode(STORY)) < len(STORY.encode()) / 2
    assert tok.encode(" captain") == [tok.encode(" captain")[0]]  # a frequent word becomes one token


def test_bpe_save_and_load(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    assert core.BPETokenizer.load(tmp_path / "tok.json").encode(STORY[:100]) == tok.encode(STORY[:100])


def test_embeddings_add_token_and_position():
    emb = core.Embeddings(vocab_size=50, block_size=8, n_embd=16)
    out = emb(torch.tensor([[3, 3, 3]]))
    assert out.shape == (1, 3, 16)
    assert not torch.allclose(out[0, 0], out[0, 1])  # same token, different position -> different vector


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a text generation feature that creates coherent story continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

