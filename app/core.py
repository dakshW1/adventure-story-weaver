"""Adventure Story Weaver: a GPT built from scratch and trained on public-domain adventure novels.

Built in six milestones (see README.md):
  1. corpus       - download public-domain novels, strip licence boilerplate, normalise text
  2. tokenizer    - byte-pair encoding from scratch + token/position embeddings
  3. transformer  - causal self-attention and the transformer block
  4. training     - the GPT model, batching, AdamW training loop, checkpoints
  5. generation   - greedy, temperature, top-k and nucleus (top-p) sampling
  6. CLI          - lives in app/cli.py and calls the functions below
"""
from __future__ import annotations

import json
import math
import re
import ssl
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "adventure_corpus.txt"
CHECKPOINT = ROOT / "models" / "story_gpt.pt"


# --- Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.

# Project Gutenberg ebook ids. Both books are in the public domain.
SOURCES = {
    "treasure_island": (120, "Treasure Island - Robert Louis Stevenson (1883)"),
    "king_solomons_mines": (2166, "King Solomon's Mines - H. Rider Haggard (1885)"),
}
GUTENBERG_URLS = (  # main site first, then the official PGLAF mirror
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://gutenberg.pglaf.org/{path}/{id}/{id}-0.txt",
)


def _mirror_path(book_id: int) -> str:
    """Gutenberg mirrors file book 2166 under 2/1/6/2166: one folder per digit except the last."""
    digits = str(book_id)
    return "/".join(digits[:-1]) if len(digits) > 1 else "0"


def strip_gutenberg(text: str) -> str:
    """Keep only the book: drop the Project Gutenberg header and licence footer."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def normalize(text: str) -> str:
    """Clean the text so the model spends its capacity on the story, not on formatting.

    - unify line endings and curly quotes/dashes to plain ASCII
    - remove _italic_ markers and [illustration] notes
    - re-join hard-wrapped lines into paragraphs; paragraphs are separated by one blank line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for a, b in {"‘": "'", "’": "'", "“": '"', "”": '"',
                 "—": "--", "–": "-", "…": "...", " ": " "}.items():
        text = text.replace(a, b)
    text = re.sub(r"\[(Illustration|Footnote)[^\]]*\]", "", text, flags=re.I)
    text = text.replace("_", "")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p) + "\n"


def build_corpus(raw_texts: dict[str, str], out_path: Path = CORPUS) -> str:
    """Strip + normalise each raw book and join them into one training file."""
    corpus = "\n".join(normalize(strip_gutenberg(raw)) for raw in raw_texts.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")
    return corpus


def fetch_books(sources: dict = SOURCES) -> dict[str, str]:
    """Download the raw books from Project Gutenberg."""
    try:  # use certifi's CA bundle when available (some Windows Pythons cannot verify HTTPS otherwise)
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    books = {}
    for name, (book_id, _title) in sources.items():
        last_error = None
        for url in GUTENBERG_URLS:
            req = urllib.request.Request(url.format(id=book_id, path=_mirror_path(book_id)),
                                         headers={"User-Agent": "adventure-story-weaver"})
            try:
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    books[name] = r.read().decode("utf-8-sig")
                break
            except OSError as e:  # HTTPError/URLError are OSErrors: try the next mirror
                last_error = e
        else:
            raise RuntimeError(f"could not download {name}: {last_error}")
    return books


def load_corpus(path: Path = CORPUS) -> str:
    return path.read_text(encoding="utf-8")


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
# Hint: Byte-Pair Encoding & Embeddings: Implement the BPE algorithm and create token embeddings from your corpus.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal Self-Attention & Transformer Block: Construct the causal self-attention layer and integrate it into a transformer block.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
# Hint: Training Loop: Implement the full training process, including forward/backward passes and optimization, for your GPT model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
# Hint: Text Generation: Develop the sampling strategy (e.g., greedy, temperature) to generate new sequences from your trained model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# Hint: Command-Line Interface: Build a simple CLI for user input of prompts and display of generated story segments.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
