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

# Pre-tokenisation: merges never cross word boundaries (a leading space stays with its word).
_PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level byte-pair encoding, built from scratch.

    Starts from the 256 possible bytes (so any text can be encoded, nothing is ever unknown)
    and repeatedly merges the most frequent adjacent pair into a new token.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 1024) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        word_freq = Counter(_PRETOKEN.findall(text))
        seqs = {w: list(w.encode("utf-8")) for w in word_freq}
        merges = []
        for new_id in range(256, vocab_size):
            pairs: Counter = Counter()
            for w, f in word_freq.items():
                s = seqs[w]
                for pair in zip(s, s[1:]):
                    pairs[pair] += f
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
            merges.append(best)
            for w, s in seqs.items():
                if len(s) > 1:
                    seqs[w] = _merge(s, best, new_id)
        return cls(merges)

    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, math.inf))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for word in _PRETOKEN.findall(text) for i in self._encode_word(word)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])


class Embeddings(nn.Module):
    """Token embedding (what the token is) + learned position embedding (where it is)."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(idx.shape[1], device=idx.device)
        return self.tok(idx) + self.pos(positions)


def nearest_tokens(model: "GPT", tok: BPETokenizer, text: str, k: int = 5) -> list[tuple[str, float]]:
    """Tokens whose learned embeddings are closest (cosine similarity) to the token for text."""
    ids = tok.encode(text)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens; pick something that is a single token")
    weights = F.normalize(model.emb.tok.weight.detach(), dim=1)
    sims = weights @ weights[ids[0]]
    best = sims.topk(k + 1).indices.tolist()
    return [(tok.decode([i]), round(sims[i].item(), 3)) for i in best if i != ids[0]][:k]


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
