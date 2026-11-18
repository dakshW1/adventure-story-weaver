"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import core

LENGTHS = {"short": 60, "medium": 120, "long": 240}  # tokens per segment


def _gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=Path, default=core.CHECKPOINT)
    p.add_argument("--length", choices=LENGTHS, default="medium", help="segment length")
    p.add_argument("--tokens", type=int, help="exact number of new tokens (overrides --length)")
    p.add_argument("--strategy", choices=["sample", "greedy"], default="sample")
    p.add_argument("--temperature", type=float, default=0.8, help="lower = safer, higher = wilder")
    p.add_argument("--top-k", type=int, default=40, help="sample only from the k likeliest tokens (0 = off)")
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling, e.g. 0.9")
    p.add_argument("--seed", type=int, default=None, help="fix the randomness for repeatable output")


def _gen_kwargs(args) -> dict:
    return {"max_new_tokens": args.tokens or LENGTHS[args.length], "strategy": args.strategy,
            "temperature": args.temperature, "top_k": args.top_k or None, "top_p": args.top_p}


def play(model, tok, kwargs: dict, inp=input, out=print, seed: int | None = None) -> str:
    """Interactive story: the user writes a line, the model continues; the story keeps growing."""
    out("Write the opening of your adventure, then what happens next after each segment. Empty line to stop.")
    story, turn = "", 0
    while True:
        try:
            line = inp("\nYou> ").strip()
        except EOFError:
            line = ""
        if not line:
            break
        story += ("\n\n" if story else "") + line
        cont = core.generate(model, tok, story, seed=None if seed is None else seed + turn, **kwargs)
        story += cont
        out(f"\nStory> {line}{cont}")
        turn += 1
    return story


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adventure-story-weaver", description="Game developers build a text adventure story segment generator, where a custom GPT, trained on existing adventure game narratives, creates dynamic story continuations.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the public-domain novels and build corpus/adventure_corpus.txt")

    p = sub.add_parser("tokenizer", help="train the BPE tokenizer and show how it splits a sentence")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--text", default="The captain drew his cutlass and stared at the treasure map.")

    p = sub.add_parser("train", help="train the GPT on the corpus and save a checkpoint")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--resume", action="store_true", help="continue training from --checkpoint")
    p.add_argument("--checkpoint", type=Path, default=core.CHECKPOINT)

    p = sub.add_parser("generate", help="continue a story prompt")
    p.add_argument("prompt", help='e.g. "We landed on the island at dawn,"')
    p.add_argument("--segments", type=int, default=1, help="how many alternative continuations")
    _gen_args(p)

    p = sub.add_parser("play", help="write an adventure together with the model, turn by turn")
    _gen_args(p)

    p = sub.add_parser("info", help="show the checkpoint: size, training curve, learned embeddings")
    p.add_argument("--checkpoint", type=Path, default=core.CHECKPOINT)

    args = parser.parse_args(argv)

    if args.command == "fetch":
        for name, (_, title) in core.SOURCES.items():
            print(f"downloading {title}")
        corpus = core.build_corpus(core.fetch_books())
        print(f"corpus: {len(corpus):,} characters, {len(corpus.split()):,} words -> "
              f"{core.CORPUS.relative_to(core.ROOT)}")

    elif args.command == "tokenizer":
        text = core.load_corpus()
        tok = core.BPETokenizer.train(text, args.vocab_size)
        ids = tok.encode(text)
        print(f"vocab {tok.vocab_size} | corpus {len(text):,} chars -> {len(ids):,} tokens "
              f"({len(text.encode()) / len(ids):.2f} bytes per token)")
        print("first merges:", [tok.decode([256 + i]) for i in range(10)])
        print("tokens:", " | ".join(tok.token_strings(tok.encode(args.text))))

    elif args.command == "train":
        text = core.load_corpus()
        if args.resume:
            model, tok, old = core.load_checkpoint(args.checkpoint)
            config = model.config
        else:
            print(f"training BPE tokenizer (vocab {args.vocab_size})...")
            tok, model, old = core.BPETokenizer.train(text, args.vocab_size), None, []
            config = core.GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                                    n_layer=args.layers, n_head=args.heads, n_embd=args.embd)
        ids = tok.encode(text)
        print(f"{len(ids):,} tokens | config {json.dumps(core.asdict(config))}")
        model, history = core.train_model(ids, config, steps=args.steps, batch_size=args.batch_size,
                                          lr=args.lr, model=model)
        print(f"{model.num_params():,} parameters")
        core.save_checkpoint(args.checkpoint, model, tok, old + history)
        print(f"saved {args.checkpoint}")

    elif args.command == "generate":
        model, tok, _ = core.load_checkpoint(args.checkpoint)
        for i in range(args.segments):
            seed = None if args.seed is None else args.seed + i
            cont = core.generate(model, tok, args.prompt, seed=seed, **_gen_kwargs(args))
            if args.segments > 1:
                print(f"--- segment {i + 1} ---")
            print(args.prompt + cont)

    elif args.command == "play":
        model, tok, _ = core.load_checkpoint(args.checkpoint)
        play(model, tok, _gen_kwargs(args), seed=args.seed)

    elif args.command == "info":
        model, tok, history = core.load_checkpoint(args.checkpoint)
        print(f"{model.num_params():,} parameters | config {json.dumps(core.asdict(model.config))}")
        for row in history:
            print(f"step {row['step']:>5}  train {row['train_loss']:.3f}  val {row['val_loss']:.3f}")
        for word in (" captain", " gold", " ship"):
            if len(tok.encode(word)) == 1:
                print(f"nearest to {word!r}:", core.nearest_tokens(model, tok, word))


if __name__ == "__main__":
    main()
