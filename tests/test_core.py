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
def test_attention_cannot_see_the_future():
    torch.manual_seed(0)
    attn = core.CausalSelfAttention(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[0, 4:] = torch.randn(2, 16)  # change only positions 4 and 5
    a, b = attn(x), attn(changed)
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-6)  # earlier positions are unaffected
    assert not torch.allclose(a[0, 4:], b[0, 4:])


def test_block_keeps_shape_and_checks_heads():
    block = core.Block(n_embd=16, n_head=4, block_size=8)
    assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)
    with pytest.raises(ValueError):
        core.CausalSelfAttention(n_embd=10, n_head=4, block_size=8)


# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
TINY = dict(block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture(scope="module")
def trained(tok):
    config = core.GPTConfig(vocab_size=tok.vocab_size, **TINY)
    model, history = core.train_model(tok.encode(STORY), config, steps=150, batch_size=16,
                                      lr=3e-3, eval_every=50, log=lambda *_: None)
    return model, history


def test_untrained_loss_is_near_uniform(tok):
    torch.manual_seed(0)
    model = core.GPT(core.GPTConfig(vocab_size=tok.vocab_size, **TINY))
    x, y = core.get_batch(torch.tensor(tok.encode(STORY)), 32, 4)
    assert torch.equal(x[:, 1:], y[:, :-1])  # targets are the inputs shifted by one
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(tok.vocab_size)) < 0.5


def test_training_lowers_the_loss(trained):
    _, history = trained
    assert history[-1]["train_loss"] < history[0]["train_loss"] / 2
    assert history[-1]["val_loss"] < history[0]["val_loss"] / 2


def test_checkpoint_round_trip(trained, tok, tmp_path):
    model, history = trained
    core.save_checkpoint(tmp_path / "m.pt", model, tok, history)
    loaded, tok2, hist2 = core.load_checkpoint(tmp_path / "m.pt")
    x = torch.tensor([tok.encode("The captain")])
    assert torch.allclose(model(x)[0], loaded(x)[0])
    assert tok2.merges == tok.merges and hist2 == history


# Milestone 5: Implement a text generation feature that creates coherent story continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

