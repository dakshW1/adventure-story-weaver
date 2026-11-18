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

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position may only look at itself and earlier positions."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))  # (B, h, T, hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                                 # (B, h, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))                        # hide the future
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)                        # re-join heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.

@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.emb = Embeddings(config.vocab_size, config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head, config.block_size, config.dropout)
                                      for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.emb.tok.weight  # weight tying: same matrix reads and writes tokens
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.blocks(self.drop(self.emb(idx)))
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of the token stream; targets are the inputs shifted one token left."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batch_size: int = 32, iters: int = 20,
                  seed: int = 0) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = [model(*get_batch(data, model.config.block_size, batch_size, gen))[1].item()
              for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def train_model(ids: list[int], config: GPTConfig, steps: int = 3000, batch_size: int = 32,
                lr: float = 3e-3, eval_every: int = 250, seed: int = 1337, log=print,
                model: GPT | None = None) -> tuple[GPT, list[dict]]:
    """AdamW + linear warm-up + cosine decay. 90/10 train/validation split. Returns model and loss history."""
    torch.manual_seed(seed)
    data = torch.tensor(ids, dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    model = model or GPT(config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    history, start = [], time.time()
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": round(estimate_loss(model, train_data), 4),
                   "val_loss": round(estimate_loss(model, val_data), 4),
                   "seconds": round(time.time() - start, 1)}
            history.append(row)
            log(f"step {step:>5} | train {row['train_loss']:.3f} | val {row['val_loss']:.3f} "
                f"| {row['seconds']}s")
        if step == steps:
            break
        x, y = get_batch(train_data, config.block_size, batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model, history


def save_checkpoint(path: Path, model: GPT, tok: BPETokenizer, history: list[dict] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict(),
                "merges": tok.merges, "history": history or []}, path)


def load_checkpoint(path: Path = CHECKPOINT) -> tuple[GPT, BPETokenizer, list[dict]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, BPETokenizer(ckpt["merges"]), ckpt.get("history", [])


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.

def next_token(logits: torch.Tensor, strategy: str = "sample", temperature: float = 0.8,
               top_k: int | None = None, top_p: float | None = None,
               gen: torch.Generator | None = None) -> int:
    """Pick the next token from a (vocab,) vector of logits.

    greedy: always the most likely token.
    sample: softmax(logits / temperature), optionally keeping only the top_k tokens
            and/or the smallest set whose probability adds up to top_p (nucleus sampling).
    """
    if strategy == "greedy":
        return int(logits.argmax())
    if strategy != "sample":
        raise ValueError("strategy must be 'greedy' or 'sample'")
    logits = logits / max(temperature, 1e-6)
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        drop = cum - F.softmax(sorted_logits, dim=-1) > top_p  # keep the token that crosses top_p
        logits = logits.masked_fill(torch.zeros_like(drop).scatter(0, order, drop), float("-inf"))
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen))


@torch.no_grad()
def generate(model: GPT, tok: BPETokenizer, prompt: str, max_new_tokens: int = 120,
             strategy: str = "sample", temperature: float = 0.8, top_k: int | None = 40,
             top_p: float | None = None, seed: int | None = None, whole_sentences: bool = True) -> str:
    """Continue prompt. Returns only the new text (the continuation)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = tok.encode(prompt) or tok.encode("\n")
    start = len(ids)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.config.block_size:]])
        logits, _ = model(context)
        ids.append(next_token(logits[0, -1], strategy, temperature, top_k, top_p, gen))
    text = tok.decode(ids[start:])
    if whole_sentences:  # cut a dangling half-sentence at the end
        cut = max(text.rfind(c) for c in ".!?")
        if cut > len(text) // 3:
            text = text[:cut + 1]
            text += '"' if text.count('"') % 2 else ""
    return text


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# The CLI lives in app/cli.py and calls the functions above.
