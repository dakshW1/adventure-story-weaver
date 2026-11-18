# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ypmztra32u/badge.svg)](https://ask.thili.ai/b/ypmztra32u)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A GPT built **from scratch** in PyTorch (byte-pair tokenizer, causal self-attention, transformer blocks,
training loop, sampling) and trained on two public-domain adventure novels. Give it the start of a scene and it
writes the next story segment for a text adventure. Everything runs on a laptop CPU; a trained 3.8 MB checkpoint
is included, so you can generate straight away.

```text
$ python -m app.cli generate "We landed on the island at dawn, and the captain" --seed 1 --segments 2
--- segment 2 ---
We landed on the island at dawn, and the captain's company to be sure of us.

But they stopped a little to the work of the captain's.

"And now," said the captain; "you're with a bit, and getting your own, and help soft to that, I'm off the
best man; and you will me. And now I'll get a word of nail for it."

"Well, Hawkins here?" said Morgan. "I'm not you."
```

It is a 0.94M-parameter model trained for 24 minutes on a CPU, so expect the *style* of an 1880s adventure
(dialogue, characters, sailors and treasure) rather than a coherent plot.

## Milestones

- [x] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**
      *Treasure Island* (Stevenson, 1883) and *King Solomon's Mines* (Haggard, 1885) from Project Gutenberg:
      licence header/footer stripped, typography normalised, hard-wrapped lines re-joined. 807k characters.
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.** Byte-level BPE
      written from scratch (1024 tokens) + learned token and position embeddings.
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.** Multi-head attention
      with a causal mask, MLP, pre-LayerNorm residual block.
- [x] **4. Set up the training loop and train the GPT model on the narrative data.** AdamW, warm-up + cosine
      schedule, gradient clipping, train/validation split, checkpoints.
- [x] **5. Implement a text generation feature that creates coherent story continuations.** Greedy, temperature,
      top-k and nucleus (top-p) sampling.
- [x] **6. Design a CLI where users can input a story prompt and receive generated segments.** `generate`
      (several alternative segments, length presets) and `play` (an interactive story you write together).

Stretch ideas done: length control (`--length short|medium|long` or `--tokens`), checkpoint save/load with
`train --resume` to continue training, and four generation strategies.

## Run it

You need Python 3.12+ and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m app.cli generate "The old map showed a cave beyond the mountains."   # uses models/story_gpt.pt
python -m app.cli generate "Silver looked at me and said," --top-p 0.9 --top-k 0 --segments 3
python -m app.cli play                                  # you write a line, the model continues, repeat
python -m app.cli info                                  # size, training curve, nearest embeddings
pytest

# rebuild everything from scratch (about 25 minutes on a laptop CPU)
python -m app.cli fetch                                 # download the two novels, build the corpus
python -m app.cli tokenizer                             # train BPE and show how it splits a sentence
python -m app.cli train --steps 3000
```

## How it works (`app/core.py`)

**1. Corpus.** `fetch_books` downloads the plain-text ebooks (main Gutenberg site, falling back to the official
PGLAF mirror). `strip_gutenberg` keeps only the text between the `*** START/END OF THE PROJECT GUTENBERG EBOOK`
markers. `normalize` turns curly quotes and dashes into ASCII, drops `_italics_` and `[Illustration]` notes, and
re-joins hard-wrapped lines so every paragraph is one line.

**2. Tokenizer and embeddings.** `BPETokenizer` starts from the 256 bytes, so any text can be encoded and
nothing is ever unknown. It then repeatedly merges the most frequent adjacent pair. Text is first split into
words with their leading space, so merges never cross word boundaries. The first merges it learns are
`' t'`, `'he'`, `' a'`, `' s'`, `' w'`, `' the'`. With 1024 tokens the corpus shrinks from 807k bytes to 293k
tokens (2.76 bytes per token), and `" captain"` and `" the"` become single tokens. `Embeddings` adds a learned
token vector to a learned position vector. After training, the nearest neighbours of `" captain"` in embedding
space are `" squire"`, `" doctor"`, `" Silver"` and `" John"`: the model learned that these play similar roles
in the story.

**3. Attention and the block.** `CausalSelfAttention` projects every position to query, key and value, splits
them into 4 heads, scores `q·k / sqrt(d_head)` and sets every score for a *later* position to `-inf` before the
softmax, so a token can only use earlier tokens. A test checks this: changing tokens 4-5 leaves the outputs at
positions 0-3 exactly unchanged. `Block` is `x + attention(LayerNorm(x))` then `x + MLP(LayerNorm(x))`.

**4. Training.** `GPT` = embeddings, 4 blocks, final LayerNorm and an output layer that shares its weights with
the token embedding. `get_batch` takes random 128-token windows, and the targets are the same windows shifted one
token left. `train_model` uses AdamW (weight decay 0.1), linear warm-up and cosine decay, clips gradients at 1.0,
and measures train and validation loss on a 90/10 split.

**5. Generation.** `next_token` implements greedy, temperature, top-k and nucleus sampling. `generate` feeds back
at most the last 128 tokens (the context window) and trims a dangling half-sentence at the end.

## Results

Config: 4 layers, 4 heads, 128-dim embeddings, 128-token context, 1024-token vocabulary, **940,800 parameters**.
3000 steps of 32 x 128 tokens, about 24 minutes on a laptop CPU.

| step | train loss | validation loss |
|---:|---:|---:|
| 0 | 6.972 | 6.973 |
| 500 | 3.886 | 4.122 |
| 1000 | 3.390 | 3.805 |
| 2000 | 2.879 | 3.626 |
| 3000 | 2.738 | **3.587** |

- An untrained model scores about `ln(1024) = 6.93`, i.e. it guesses uniformly. The final validation loss of 3.59
  is a perplexity of about 36: on average the model narrows the next token down to about 36 equally likely
  choices out of 1024.
- The validation set is the last 10% of the corpus, which is the end of *King Solomon's Mines*. That is a
  harder test than random windows, because the model has not seen that part of the plot.
- The gap between train (2.74) and validation (3.59) loss grows after about step 1500: with only 293k tokens,
  the model starts memorising. More text or more dropout would be the next steps.
- **Greedy decoding loops** ("the king, and the king, and the king..."), because it always takes the single
  most likely token and a small model keeps returning to the same high-probability phrase. Sampling with
  temperature 0.8 and top-k 40 (the default) gives varied text.

## Limitations

- At this size the model learns spelling, punctuation, dialogue format and the vocabulary of the books, but not
  a long-range plot. The same code with more layers, a bigger context and a GPU would go much further.
- Two books is a small corpus. Adding more public-domain adventures (Haggard's *She*, Verne, Kipling) is the
  cheapest improvement.

## Project layout

```text
app/core.py              corpus, BPE tokenizer, embeddings, attention, GPT, training, sampling
app/cli.py               fetch | tokenizer | train | generate | play | info
corpus/                  the cleaned training text (public domain)
models/story_gpt.pt      trained weights + tokenizer merges + loss history
tests/test_core.py       one block of tests per milestone, using tiny models so they run in seconds
```

Built for the askThili drill *Build GPT from Scratch*. Written with help from an AI coding assistant.
