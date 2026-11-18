# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ypmztra32u/badge.svg)](https://ask.thili.ai/b/ypmztra32u)

This project empowers game developers to create dynamic text adventure experiences. You'll build a custom GPT from scratch, training it on existing narrative data to generate unique story continuations. The result is a powerful tool for crafting engaging and unpredictable text adventures.

## What you are building

**Goal.** You will build a text adventure story segment generator using a custom GPT, trained on adventure game narratives, to create dynamic story continuations.

**Done when.** The generator successfully produces coherent story continuations based on user prompts, demonstrating the application of a custom GPT trained on narrative data and accessible via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Corpus Preparation: Gather and preprocess your raw text data from public domain sources.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: Byte-Pair Encoding & Embeddings: Implement the BPE algorithm and create token embeddings from your corpus.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal Self-Attention & Transformer Block: Construct the causal self-attention layer and integrate it into a transformer block.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Training Loop: Implement the full training process, including forward/backward passes and optimization, for your GPT model.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Text Generation: Develop the sampling strategy (e.g., greedy, temperature) to generate new sequences from your trained model.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: Command-Line Interface: Build a simple CLI for user input of prompts and display of generated story segments.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify parameters like desired length or tone for generated segments.
- Implement a feature to save and load trained model checkpoints for continued training or deployment.
- Expand the CLI to offer multiple text generation strategies (e.g., top-k sampling, nucleus sampling).
