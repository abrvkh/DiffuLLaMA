#!/usr/bin/env python3
"""
Single-file entrypoint that:
1) Loads a pretrained AR model from HF and adds a mask token if missing.
2) Trains a diffusion objective in distributed mode (Accelerate).
3) Uses attention mask annealing (causal -> noncausal).
4) Provides optional tokenization and evals (Lambada, HellaSwag, Winogrande, PIQA, SIQA).

DDM training: 
Here’s how DDM (in this repo) does mask sampling and loss, based on LLaMA-Factory/src/llamafactory/train/ddm/trainer.py
  and model.py.

  Mask sampling (training)

  - It uses continuous time t ~ Uniform(ε, 1) where ε = 1e-3.
  - Defines:
      - sigma = t
      - dsigma = 1 / t
  - Masking rule (absorbing diffusion): for each token in the maskable region (~src_mask), it is replaced by [MASK] with
    probability sigma.
    Code: transition(x, sigma[:, None], maskable_mask=~src_mask)
    File: LLaMA-Factory/src/llamafactory/train/ddm/trainer.py

  So early time (small t) = few masks, late time (near 1) = heavy masking.

  Loss

  - Forward pass uses annealed attention mask (see below).
  - Loss is cross‑entropy only on masked positions:
      - loss_mask = x_t == mask_token_id
      - loss = CE(logits, x) with reduction="none"
      - loss = loss.masked_fill(~loss_mask, 0)
  - Final loss is weighted by dsigma:
      - final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()
        This weights earlier timesteps more (since 1/t is larger).
        File: LLaMA-Factory/src/llamafactory/train/ddm/trainer.py

  Optional shift

  - If shift=True, it drops the first token and shifts labels left by 1, same as AR loss.
    File: LLaMA-Factory/src/llamafactory/train/ddm/trainer.py

  Attention mask annealing

  - It gradually removes the causal mask:
      - attn_mask_ratio = min(1.0, (global_step + 1) / anneal_steps)
      - Builds a 4D mask where causal mask OR random mask is allowed.
        So early training is mostly causal; later, more noncausal attention.
        File: model.py (function get_anneal_attn_mask) and trainer.py (use in inner_forward)

  Eval-time loss (eval_forward)

  - Instead of sampling continuous t, it iterates over discrete timesteps t = T..1.
  - At each step it sets rate = t / T, uses sigma = rate, then re‑masks and computes the same masked‑token loss.
    File: LLaMA-Factory/src/llamafactory/train/ddm/trainer.py

  Sampling/generation (in repo)

  - In model.py, generate_samples does iterative unmasking:
      1. Start with all non‑source tokens masked.
      2. Predict logits, sample x0.
      3. At each step t, unmask a fraction p_to_x0 = 1/(t+1) of remaining masked tokens.
      4. Repeat until no masks remain.
         File: model.py
"""

import argparse
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_patch import replace_attention_mask

replace_attention_mask()

# Optional imports used when running with packed datasets / seq-parallel.
from packed_dataset import CombinedDataset, PackedDataset
from easy_context import (
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    prepare_seq_parallel_inputs,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch,
)

apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()


def get_anneal_attn_mask(seq_len: int, bsz: int, dtype, device, attn_mask_ratio: float) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)

    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)
    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def transition(x_0: torch.Tensor, sigma: torch.Tensor, maskable_mask: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    return torch.where(move_indices, mask_token_id, x_0)


class DiffusionWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def vocab_size(self) -> int:
        return self.model.get_input_embeddings().weight.size(0)

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        return out.logits


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size
        self.num_blocks = len(token_ids) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.block_size
        end = start + self.block_size
        return torch.tensor(self.token_ids[start:end], dtype=torch.long)


def _noisy_mean_initialization(embed_weight: torch.Tensor, num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def resize_embedding_layer(model: torch.nn.Module, tokenizer) -> None:
    current_embedding_size = model.get_input_embeddings().weight.size(0)
    if len(tokenizer) > current_embedding_size:
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = new_embedding_size - current_embedding_size
        _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
        _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)


def ensure_mask_token(tokenizer, model, mask_token: str) -> None:
    added_tokens = 0
    if tokenizer.mask_token is None:
        added_tokens += tokenizer.add_special_tokens({"mask_token": mask_token})

    if tokenizer.pad_token is None:
        pad = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.mask_token
        added_tokens += tokenizer.add_special_tokens({"pad_token": pad})

    if added_tokens > 0:
        resize_embedding_layer(model, tokenizer)


def build_packed_dataloader(
    packed_dir: Path,
    prefixes: List[str],
    weights: List[float],
    batch_size: int,
    block_size: int,
    accelerator: Accelerator,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    import glob

    datasets = []
    for prefix in prefixes:
        filenames = sorted(glob.glob(str(packed_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        dataset = PackedDataset(
            filenames,
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + accelerator.process_index,
            num_processes=accelerator.num_processes,
            process_rank=accelerator.process_index,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No packed data found at {packed_dir} with prefixes {prefixes}")

    if not weights:
        weights = [1.0 / len(datasets)] * len(datasets)
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def build_tokenized_dataloader(
    tokenizer,
    dataset_name: str,
    split: str,
    text_field: Optional[str],
    batch_size: int,
    block_size: int,
    shuffle: bool,
    max_samples: Optional[int],
    seed: int,
) -> DataLoader:
    from datasets import load_dataset

    if os.path.exists(dataset_name):
        ds = load_dataset("text", data_files=dataset_name, split=split)
        field = "text"
    else:
        ds = load_dataset(dataset_name, split=split)
        field = text_field
        if field is None:
            field = "text" if "text" in ds.column_names else ds.column_names[0]

    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    def tokenize_fn(examples):
        return tokenizer(examples[field], add_special_tokens=False)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    all_ids: List[int] = []
    for item in tokenized:
        all_ids.extend(item["input_ids"])

    dataset = TokenBlockDataset(all_ids, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def diffusion_step(
    model: DiffusionWrapper,
    input_ids: torch.Tensor,
    mask_token_id: int,
    shift: bool,
    global_step: int,
    anneal_steps: int,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    src_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

    sampling_eps = 1e-3
    t = (1 - sampling_eps) * torch.rand(batch_size, device=input_ids.device) + sampling_eps
    sigma = t
    dsigma = torch.reciprocal(t)

    x_t = transition(input_ids, sigma[:, None], maskable_mask=~src_mask, mask_token_id=mask_token_id)

    attn_mask_ratio = 1.0
    if anneal_steps > 0:
        attn_mask_ratio = min(1.0, float(global_step + 1) / float(anneal_steps))

    x_embed = model.get_embeds(input_ids)
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=input_ids.device, attn_mask_ratio=attn_mask_ratio)
    logits = model(x_t, attention_mask=attention_mask)

    loss_mask = x_t == mask_token_id
    targets = input_ids

    if shift:
        logits = logits[:, :-1]
        loss_mask = loss_mask[:, 1:]
        targets = input_ids[:, 1:]

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(batch_size, -1)
    loss = loss.masked_fill(~loss_mask, 0)
    loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()

    return loss


def eval_forward(
    model: DiffusionWrapper,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    mask_token_id: int,
    diffusion_steps: int,
    shift: bool,
) -> torch.Tensor:
    model.eval()
    x = input_ids
    batch_size, seq_len = x.shape
    total_unw_loss = torch.tensor(0.0, device=x.device)

    for t in range(diffusion_steps, 0, -1):
        rate = t / diffusion_steps
        tt = torch.tensor([rate] * batch_size, device=x.device)

        sigma = tt
        dsigma = torch.reciprocal(tt)

        x_t = transition(x, sigma[:, None], maskable_mask=~src_mask, mask_token_id=mask_token_id)

        x_embed = model.get_embeds(x)
        attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0)
        logits = model(x_t, attention_mask=attention_mask)

        loss_mask = x_t == mask_token_id
        labels = x

        if shift:
            logits = logits[:, :-1]
            loss_mask = loss_mask[:, 1:]
            labels = x[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape(batch_size, -1)
        loss = loss.masked_fill(~loss_mask, 0)
        if loss_mask.sum() == 0:
            continue
        unw_loss = loss.sum() / loss_mask.sum()
        total_unw_loss += unw_loss

    return total_unw_loss / diffusion_steps


def generate_samples(model: DiffusionWrapper, tokenizer, input_ids: torch.Tensor, src_mask: torch.Tensor, diffusion_steps: int, shift: bool) -> torch.Tensor:
    model.eval()
    x = input_ids.to(model.device)
    src_mask = src_mask.to(model.device)

    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0)

    maskable_mask = ~src_mask
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

    logits = model(xt, attention_mask=attention_mask)
    scores = torch.log_softmax(logits, dim=-1)
    x0 = torch.distributions.Categorical(logits=scores).sample()

    if shift:
        x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)

    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

    for t in range(diffusion_steps - 1, 0, -1):
        with torch.no_grad():
            p_to_x0 = 1 / (t + 1)
            masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
            xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)

            logits = model(xt, attention_mask=attention_mask)
            scores = torch.log_softmax(logits, dim=-1)
            x0 = torch.distributions.Categorical(logits=scores).sample()

            if shift:
                x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)

            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

    if shift:
        x0 = x0[:, 1:]

    return x0


def eval_lambada(model: DiffusionWrapper, tokenizer, shift: bool) -> None:
    total_cnt = 0
    correct = 0
    path = Path("evaluation/evaluation/lambada_test_plain_text.txt")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_cnt += 1
            x0 = tokenizer.encode(line, add_special_tokens=False)
            prefix = tokenizer.encode(" ".join(line.split()[:-1]), add_special_tokens=False)
            masked_nums = len(x0) - len(prefix)
            src_mask = [1] * len(prefix) + [0] * masked_nums
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            res = generate_samples(model, tokenizer, inputs, src_mask, diffusion_steps=masked_nums, shift=shift)
            pred = tokenizer.decode(res.tolist()[0][-masked_nums:]).strip()

            if pred == line.split()[-1].strip():
                correct += 1
    print("lambada_acc:", correct / max(1, total_cnt))


def eval_hellaswag(model: DiffusionWrapper, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> None:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    total_cnt = 0
    correct = 0

    for doc in ds:
        total_cnt += 1
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = f"{doc['activity_label']}: {ctx}"
        choices = [ending for ending in doc["endings"]]
        gold = int(doc["label"])

        scores = []
        prefix = tokenizer.encode(query, add_special_tokens=False)
        for choice in choices:
            x0 = prefix + tokenizer.encode(choice, add_special_tokens=False)
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([[1] * len(prefix) + [0] * (len(x0) - len(prefix))], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps=diffusion_steps, shift=shift)
            scores.append(score.item())

        pred = int(torch.tensor(scores).argmin().item())
        if pred == gold:
            correct += 1

    print("hellaswag_acc:", correct / max(1, total_cnt))


def eval_winogrande(model: DiffusionWrapper, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> None:
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    total_cnt = 0
    correct = 0

    for doc in ds:
        total_cnt += 1
        idx = doc["sentence"].index("_")
        options = [doc["option1"], doc["option2"]]
        gold = 0 if doc["answer"] == "1" else 1

        scores = []
        for opt in options:
            prefix = doc["sentence"][:idx]
            suffix = doc["sentence"][idx + 1 :].strip()
            x0 = tokenizer.encode(prefix, add_special_tokens=False) + tokenizer.encode(opt, add_special_tokens=False) + tokenizer.encode(suffix, add_special_tokens=False)
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([[1] * len(prefix) + [0] * (len(x0) - len(prefix))], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps=diffusion_steps, shift=shift)
            scores.append(score.item())

        pred = int(torch.tensor(scores).argmin().item())
        if pred == gold:
            correct += 1

    print("winogrande_acc:", correct / max(1, total_cnt))


def eval_piqa(model: DiffusionWrapper, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> None:
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    total_cnt = 0
    correct = 0

    for doc in ds:
        total_cnt += 1
        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        gold = int(doc["label"])

        scores = []
        prefix = tokenizer.encode(query, add_special_tokens=False)
        for choice in choices:
            x0 = prefix + tokenizer.encode(" " + choice, add_special_tokens=False)
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([[1] * len(prefix) + [0] * (len(x0) - len(prefix))], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps=diffusion_steps, shift=shift)
            scores.append(score.item())

        pred = int(torch.tensor(scores).argmin().item())
        if pred == gold:
            correct += 1

    print("piqa_acc:", correct / max(1, total_cnt))


def eval_siqa(model: DiffusionWrapper, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> None:
    from datasets import load_dataset
    ds = load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    total_cnt = 0
    correct = 0

    for doc in ds:
        total_cnt += 1
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
        gold = int(doc["label"]) - 1

        scores = []
        prefix = tokenizer.encode(query, add_special_tokens=False)
        for choice in choices:
            x0 = prefix + tokenizer.encode(choice, add_special_tokens=False)
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([[1] * len(prefix) + [0] * (len(x0) - len(prefix))], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps=diffusion_steps, shift=shift)
            scores.append(score.item())

        pred = int(torch.tensor(scores).argmin().item())
        if pred == gold:
            correct += 1

    print("siqa_acc:", correct / max(1, total_cnt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-every", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--anneal-steps", type=int, default=1000)
    parser.add_argument("--mask-token", type=str, default="[MASK]")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--parallel-mode", type=str, choices=["dist_flash_attn", "ulysses_attn", "data_parallel"], default="data_parallel")
    parser.add_argument("--diffusion-steps", type=int, default=64)

    # Dataset options
    parser.add_argument("--packed-data-dir", type=str, default=None)
    parser.add_argument("--packed-prefixes", type=str, default=None)
    parser.add_argument("--packed-weights", type=str, default=None)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    # Eval
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--eval-tasks", type=str, default="hellaswag,winogrande,piqa,siqa")
    parser.add_argument("--eval-max-samples", type=int, default=None)

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        kwargs_handlers=[timeout],
    )

    accelerator.print(f"Total processes: {accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation=args.attn_impl,
    )

    ensure_mask_token(tokenizer, model, args.mask_token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Remove causal mask for GPT2-like blocks when using non-causal attention.
    if getattr(model.config, "model_type", None) == "gpt2":
        for block in model.transformer.h:
            block.attn.bias.fill_(True)

    model_type = "llama" if getattr(model.config, "model_type", None) == "llama" else "mistral"
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    diff_model = DiffusionWrapper(model)

    if args.do_train:
        if args.packed_data_dir is None and args.dataset is None:
            raise ValueError("Provide --packed-data-dir or --dataset for training.")

        effective_block_size = args.seq_length + (1 if args.shift else 0)

        if args.packed_data_dir is not None:
            prefixes = [p.strip() for p in (args.packed_prefixes or "train").split(",")]
            weights = []
            if args.packed_weights:
                weights = [float(w) for w in args.packed_weights.split(",")]
            train_loader = build_packed_dataloader(
                Path(args.packed_data_dir),
                prefixes,
                weights,
                batch_size=args.batch_size,
                block_size=effective_block_size,
                accelerator=accelerator,
                shuffle=True,
                seed=args.seed,
            )
        else:
            train_loader = build_tokenized_dataloader(
                tokenizer,
                dataset_name=args.dataset,
                split=args.dataset_split,
                text_field=args.text_field,
                batch_size=args.batch_size,
                block_size=effective_block_size,
                shuffle=True,
                max_samples=args.max_samples,
                seed=args.seed,
            )

        optimizer = torch.optim.AdamW(diff_model.parameters(), lr=args.learning_rate)
        diff_model, optimizer, train_loader = accelerator.prepare(diff_model, optimizer, train_loader)
        train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)

        diff_model.train()
        global_step = 0
        for step, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if args.parallel_mode != "data_parallel":
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                prepared = prepare_seq_parallel_inputs(
                    args.parallel_mode,
                    input_ids,
                    position_ids,
                    input_ids,
                    accelerator.process_index,
                    accelerator.num_processes,
                    accelerator.device,
                )
                input_ids = prepared["local_input_ids"]

            with accelerator.accumulate(diff_model):
                loss = diffusion_step(
                    diff_model,
                    input_ids,
                    tokenizer.mask_token_id,
                    shift=args.shift,
                    global_step=global_step,
                    anneal_steps=args.anneal_steps,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    accelerator.print(f"step {global_step} | loss {loss.item():.4f}")
                    global_step += 1

                if global_step >= args.max_train_steps:
                    break

        accelerator.print("Training finished")

        if args.output_dir is not None and accelerator.is_main_process:
            unwrap = accelerator.unwrap_model(diff_model)
            unwrap.model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and accelerator.is_main_process:
        eval_model = accelerator.unwrap_model(diff_model)
        eval_model.eval()
        tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        for task in tasks:
            if task == "lambada":
                eval_lambada(eval_model, tokenizer, shift=args.shift)
            elif task == "hellaswag":
                eval_hellaswag(eval_model, tokenizer, diffusion_steps=args.diffusion_steps, shift=args.shift, max_samples=args.eval_max_samples)
            elif task == "winogrande":
                eval_winogrande(eval_model, tokenizer, diffusion_steps=args.diffusion_steps, shift=args.shift, max_samples=args.eval_max_samples)
            elif task == "piqa":
                eval_piqa(eval_model, tokenizer, diffusion_steps=args.diffusion_steps, shift=args.shift, max_samples=args.eval_max_samples)
            elif task == "siqa":
                eval_siqa(eval_model, tokenizer, diffusion_steps=args.diffusion_steps, shift=args.shift, max_samples=args.eval_max_samples)
            else:
                raise ValueError(f"Unknown eval task: {task}")


if __name__ == "__main__":
    main()
