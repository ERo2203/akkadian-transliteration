#!/usr/bin/env python3
"""Train a small Unicode cuneiform -> transliteration transformer.

The defaults are intentionally conservative for an Apple Silicon machine with
8 GB RAM. The script saves resumable checkpoints after every epoch and keeps the
best validation checkpoint separately.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


@dataclass
class TrainConfig:
    data_path: str = "data/raw/mtm24_transliterated.csv"
    output_dir: str = "checkpoints/unicode_to_translit"
    seed: int = 42
    max_src_len: int = 200
    max_tgt_len: int = 300
    max_rows: int = 0
    val_size: float = 0.05
    test_size: float = 0.05
    batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 50
    patience: int = 7
    min_delta: float = 0.0005
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eos_loss_weight: float = 4.0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 0
    decode_samples: int = 128
    decode_max_ratio: float = 2.2
    decode_max_extra: int = 40
    decode_min_len: int = 8
    show_predictions: int = 0
    eval_only: bool = False
    resume: str = ""
    device: str = "auto"


class Vocab:
    def __init__(self, tokens: Iterable[str]) -> None:
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok in tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def sos_idx(self) -> int:
        return self.stoi["<sos>"]

    @property
    def eos_idx(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, text: str, add_bounds: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_idx) for ch in text]
        if add_bounds:
            return [self.sos_idx, *ids, self.eos_idx]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        chars: List[str] = []
        for idx in ids:
            tok = self.itos.get(int(idx), "<unk>")
            if tok == "<eos>":
                break
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            chars.append(tok)
        return "".join(chars)

    def to_dict(self) -> Dict[str, int]:
        return self.stoi

    @classmethod
    def from_dict(cls, mapping: Dict[str, int]) -> "Vocab":
        vocab = cls([])
        vocab.stoi = dict(mapping)
        vocab.itos = {i: tok for tok, i in mapping.items()}
        return vocab


class CuneiformDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        self.frame = frame.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        src = torch.tensor(self.src_vocab.encode(row["src"]), dtype=torch.long)
        tgt = torch.tensor(self.tgt_vocab.encode(row["tgt"], add_bounds=True), dtype=torch.long)
        return src, tgt


def make_collate(src_pad_idx: int, tgt_pad_idx: int):
    def collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = zip(*batch)
        return (
            pad_sequence(src, batch_first=True, padding_value=src_pad_idx),
            pad_sequence(tgt, batch_first=True, padding_value=tgt_pad_idx),
        )

    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class UnicodeToTranslitTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        cfg: TrainConfig,
    ) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, cfg.d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, cfg.d_model, padding_idx=tgt_pad_idx)
        self.positional = PositionalEncoding(cfg.d_model, max_len=max(cfg.max_src_len, cfg.max_tgt_len) + 8)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.encoder_layers,
            num_decoder_layers=cfg.decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fc_out = nn.Linear(cfg.d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src.eq(self.src_pad_idx)
        tgt_key_padding_mask = tgt_in.eq(self.tgt_pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)

        src_emb = self.positional(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.data_path, usecols=["original_cuneiform", "transliteration"]).dropna()
    df = df.rename(columns={"original_cuneiform": "src", "transliteration": "tgt"})
    df["src"] = df["src"].astype(str)
    df["tgt"] = df["tgt"].astype(str)
    df = df[(df["src"].str.len() <= cfg.max_src_len) & (df["tgt"].str.len() <= cfg.max_tgt_len)]

    if cfg.max_rows and len(df) > cfg.max_rows:
        df = df.sample(cfg.max_rows, random_state=cfg.seed)

    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    val_count = int(len(df) * cfg.val_size)
    test_count = int(len(df) * cfg.test_size)
    train_end = len(df) - val_count - test_count
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end : train_end + val_count]
    test_df = df.iloc[train_end + val_count :]
    return train_df, val_df, test_df


def build_vocabs(train_df: pd.DataFrame) -> Tuple[Vocab, Vocab]:
    src_chars = sorted({ch for text in train_df["src"] for ch in text})
    tgt_chars = sorted({ch for text in train_df["tgt"] for ch in text})
    return Vocab(src_chars), Vocab(tgt_chars)


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(pred: str, ref: str) -> float:
    return edit_distance(pred, ref) / max(1, len(ref))


def wer(pred: str, ref: str) -> float:
    return edit_distance(pred.split(), ref.split()) / max(1, len(ref.split()))


def max_char_run(text: str) -> int:
    if not text:
        return 0
    longest = 1
    current = 1
    previous = text[0]
    for ch in text[1:]:
        if ch == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
            previous = ch
    return max(longest, current)


def repeated_bigram_rate(text: str) -> float:
    if len(text) < 4:
        return 0.0
    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    repeats = sum(1 for a, b in zip(bigrams, bigrams[1:]) if a == b)
    return repeats / max(1, len(bigrams) - 1)


@torch.no_grad()
def greedy_decode(
    model: UnicodeToTranslitTransformer,
    src: torch.Tensor,
    tgt_vocab: Vocab,
    max_len: int,
    device: torch.device,
    max_len_by_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    if max_len_by_sample is not None:
        max_len_by_sample = max_len_by_sample.to(device)
    ys = torch.full((src.size(0), 1), tgt_vocab.sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model(src, ys)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        if max_len_by_sample is not None:
            over_limit = ys.size(1) >= max_len_by_sample
            next_tok = torch.where(over_limit & ~finished, torch.full_like(next_tok, tgt_vocab.eos_idx), next_tok)
        next_tok = torch.where(finished, torch.full_like(next_tok, tgt_vocab.pad_idx), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok.eq(tgt_vocab.eos_idx)
        if finished.all():
            break
    return ys[:, 1:]


def run_epoch(
    model: UnicodeToTranslitTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in tqdm(loader, leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tokens = tgt_out.ne(0).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_decoding(
    model: UnicodeToTranslitTransformer,
    loader: DataLoader,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    sample_limit: int,
    src_pad_idx: int,
    max_ratio: float,
    max_extra: int,
    min_len: int,
) -> Dict[str, float]:
    model.eval()
    cers: List[float] = []
    wers: List[float] = []
    pred_lens: List[int] = []
    ref_lens: List[int] = []
    eos_hits = 0
    forced_eos_hits = 0
    reached_global_max = 0
    max_runs: List[int] = []
    bigram_rates: List[float] = []
    count = 0
    for src, tgt in tqdm(loader, leave=False):
        src_lens = src.ne(src_pad_idx).sum(dim=1)
        dynamic_max = torch.clamp(
            (src_lens.float() * max_ratio).ceil().long() + max_extra,
            min=min_len,
            max=max_len,
        )
        preds = greedy_decode(
            model,
            src,
            tgt_vocab,
            max_len=max_len,
            device=device,
            max_len_by_sample=dynamic_max,
        ).cpu()
        refs = tgt[:, 1:]
        for pred_ids, ref_ids, limit in zip(preds, refs, dynamic_max):
            pred_list = pred_ids.tolist()
            eos_in_pred = tgt_vocab.eos_idx in pred_list
            if eos_in_pred:
                eos_hits += 1
            trimmed_pred_len = pred_list.index(tgt_vocab.eos_idx) if eos_in_pred else len(pred_list)
            if eos_in_pred and trimmed_pred_len >= int(limit.item()) - 1:
                forced_eos_hits += 1
            if not eos_in_pred and len(pred_list) >= max_len:
                reached_global_max += 1
            pred = tgt_vocab.decode(pred_ids.tolist())
            ref = tgt_vocab.decode(ref_ids.tolist())
            cers.append(cer(pred, ref))
            wers.append(wer(pred, ref))
            pred_lens.append(len(pred))
            ref_lens.append(len(ref))
            max_runs.append(max_char_run(pred))
            bigram_rates.append(repeated_bigram_rate(pred))
            count += 1
            if count >= sample_limit:
                break
        if count >= sample_limit:
            break

    n = max(1, len(cers))
    return {
        "cer": sum(cers) / n,
        "wer": sum(wers) / n,
        "eos_rate": eos_hits / n,
        "forced_eos_rate": forced_eos_hits / n,
        "global_max_rate": reached_global_max / n,
        "avg_pred_len": sum(pred_lens) / max(1, len(pred_lens)),
        "avg_ref_len": sum(ref_lens) / max(1, len(ref_lens)),
        "pred_ref_len_ratio": (sum(pred_lens) / max(1, len(pred_lens))) / max(1.0, (sum(ref_lens) / max(1, len(ref_lens)))),
        "avg_max_char_run": sum(max_runs) / max(1, len(max_runs)),
        "avg_repeated_bigram_rate": sum(bigram_rates) / max(1, len(bigram_rates)),
    }


@torch.no_grad()
def show_predictions(
    model: UnicodeToTranslitTransformer,
    loader: DataLoader,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    n: int,
    src_pad_idx: int,
    max_ratio: float,
    max_extra: int,
    min_len: int,
) -> List[Dict[str, str]]:
    model.eval()
    src, tgt = next(iter(loader))
    n = min(n, src.size(0))
    src_lens = src[:n].ne(src_pad_idx).sum(dim=1)
    dynamic_max = torch.clamp(
        (src_lens.float() * max_ratio).ceil().long() + max_extra,
        min=min_len,
        max=max_len,
    )
    preds = greedy_decode(
        model,
        src[:n],
        tgt_vocab,
        max_len=max_len,
        device=device,
        max_len_by_sample=dynamic_max,
    ).cpu()

    rows: List[Dict[str, str]] = []
    for i in range(n):
        pred_list = preds[i].tolist()
        eos_in_pred = tgt_vocab.eos_idx in pred_list
        pred_steps = pred_list.index(tgt_vocab.eos_idx) if eos_in_pred else len(pred_list)
        pred = tgt_vocab.decode(preds[i].tolist())
        ref = tgt_vocab.decode(tgt[i][1:].tolist())
        rows.append(
            {
                "ref": ref,
                "pred": pred,
                "ref_len": str(len(ref)),
                "pred_len": str(len(pred)),
                "decode_limit": str(int(dynamic_max[i].item())),
                "eos_hit": str(eos_in_pred),
                "pred_steps": str(pred_steps),
                "max_char_run": str(max_char_run(pred)),
                "repeated_bigram_rate": f"{repeated_bigram_rate(pred):.4f}",
                "cer": f"{cer(pred, ref):.4f}",
                "wer": f"{wer(pred, ref):.4f}",
            }
        )
    return rows


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: UnicodeToTranslitTransformer,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    history: List[Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in asdict(TrainConfig()).items():
        arg = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true")
        else:
            parser.add_argument(arg, type=type(value), default=value)
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))

    checkpoint = None
    if cfg.resume:
        requested_eval_only = cfg.eval_only
        requested_show_predictions = cfg.show_predictions
        requested_decode_samples = cfg.decode_samples
        requested_decode_max_ratio = cfg.decode_max_ratio
        requested_decode_max_extra = cfg.decode_max_extra
        requested_decode_min_len = cfg.decode_min_len
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        saved_cfg = TrainConfig(**checkpoint["config"])
        saved_cfg.resume = cfg.resume
        saved_cfg.eval_only = requested_eval_only
        saved_cfg.show_predictions = requested_show_predictions
        saved_cfg.decode_samples = requested_decode_samples
        saved_cfg.decode_max_ratio = requested_decode_max_ratio
        saved_cfg.decode_max_extra = requested_decode_max_extra
        saved_cfg.decode_min_len = requested_decode_min_len
        if cfg.device != "auto":
            saved_cfg.device = cfg.device
        cfg = saved_cfg

    set_seed(cfg.seed)
    device = select_device(cfg.device)
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data(cfg)
    if checkpoint:
        src_vocab = Vocab.from_dict(checkpoint["src_vocab"])
        tgt_vocab = Vocab.from_dict(checkpoint["tgt_vocab"])
    else:
        src_vocab, tgt_vocab = build_vocabs(train_df)

    collate = make_collate(src_vocab.pad_idx, tgt_vocab.pad_idx)
    train_loader = DataLoader(
        CuneiformDataset(train_df, src_vocab, tgt_vocab),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        CuneiformDataset(val_df, src_vocab, tgt_vocab),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        CuneiformDataset(test_df, src_vocab, tgt_vocab),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )

    model = UnicodeToTranslitTransformer(
        len(src_vocab.stoi),
        len(tgt_vocab.stoi),
        src_vocab.pad_idx,
        tgt_vocab.pad_idx,
        cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_weights = torch.ones(len(tgt_vocab.stoi), device=device)
    loss_weights[tgt_vocab.eos_idx] = cfg.eos_loss_weight
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        weight=loss_weights,
        label_smoothing=cfg.label_smoothing,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))

    if cfg.show_predictions:
        prediction_rows = show_predictions(
            model,
            test_loader,
            tgt_vocab,
            device,
            max_len=cfg.max_tgt_len,
            n=cfg.show_predictions,
            src_pad_idx=src_vocab.pad_idx,
            max_ratio=cfg.decode_max_ratio,
            max_extra=cfg.decode_max_extra,
            min_len=cfg.decode_min_len,
        )
        save_json(output_dir / "prediction_samples.json", prediction_rows)
        for row in prediction_rows:
            print("=" * 80)
            print(f"REF len={row['ref_len']}")
            print(row["ref"])
            print()
            print(
                f"PRED len={row['pred_len']} limit={row['decode_limit']} "
                f"eos={row['eos_hit']} steps={row['pred_steps']} CER={row['cer']} WER={row['wer']}"
            )
            print(row["pred"])

    if cfg.eval_only:
        test_scores = evaluate_decoding(
            model,
            test_loader,
            tgt_vocab,
            device,
            max_len=cfg.max_tgt_len,
            sample_limit=min(cfg.decode_samples, len(test_df)),
            src_pad_idx=src_vocab.pad_idx,
            max_ratio=cfg.decode_max_ratio,
            max_extra=cfg.decode_max_extra,
            min_len=cfg.decode_min_len,
        )
        save_json(output_dir / "eval_only_metrics.json", test_scores)
        print(
            f"eval_only_cer={test_scores['cer']:.4f} eval_only_wer={test_scores['wer']:.4f} "
            f"eos_rate={test_scores['eos_rate']:.4f} forced_eos_rate={test_scores['forced_eos_rate']:.4f} "
            f"global_max_rate={test_scores['global_max_rate']:.4f}"
        )
        return

    save_json(output_dir / "config.json", asdict(cfg))
    save_json(output_dir / "src_vocab.json", src_vocab.to_dict())
    save_json(output_dir / "tgt_vocab.json", tgt_vocab.to_dict())
    save_json(
        output_dir / "data_split_summary.json",
        {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "src_vocab_size": len(src_vocab.stoi),
            "tgt_vocab_size": len(tgt_vocab.stoi),
            "device": str(device),
        },
    )

    print(f"Device: {device}")
    print(f"Rows: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"Vocab: src={len(src_vocab.stoi):,} tgt={len(tgt_vocab.stoi):,}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, cfg.grad_clip)
        val_loss = run_epoch(model, val_loader, criterion, None, device, cfg.grad_clip)
        decode_scores = evaluate_decoding(
            model,
            val_loader,
            tgt_vocab,
            device,
            max_len=cfg.max_tgt_len,
            sample_limit=cfg.decode_samples,
            src_pad_idx=src_vocab.pad_idx,
            max_ratio=cfg.decode_max_ratio,
            max_extra=cfg.decode_max_extra,
            min_len=cfg.decode_min_len,
        )

        improved = val_loss < (best_val_loss - cfg.min_delta)
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": decode_scores["cer"],
            "val_wer": decode_scores["wer"],
            "val_eos_rate": decode_scores["eos_rate"],
            "val_forced_eos_rate": decode_scores["forced_eos_rate"],
            "val_global_max_rate": decode_scores["global_max_rate"],
            "val_avg_pred_len": decode_scores["avg_pred_len"],
            "val_avg_ref_len": decode_scores["avg_ref_len"],
            "val_pred_ref_len_ratio": decode_scores["pred_ref_len_ratio"],
            "val_avg_max_char_run": decode_scores["avg_max_char_run"],
            "val_avg_repeated_bigram_rate": decode_scores["avg_repeated_bigram_rate"],
            "best_val_loss": best_val_loss,
            "seconds": time.time() - started,
        }
        history.append(row)
        save_json(output_dir / "history.json", history)

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            cfg,
            src_vocab,
            tgt_vocab,
            epoch,
            best_val_loss,
            epochs_without_improvement,
            history,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                cfg,
                src_vocab,
                tgt_vocab,
                epoch,
                best_val_loss,
                epochs_without_improvement,
                history,
            )

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_cer={decode_scores['cer']:.4f} val_wer={decode_scores['wer']:.4f} "
            f"eos={decode_scores['eos_rate']:.3f} forced={decode_scores['forced_eos_rate']:.3f} "
            f"run={decode_scores['avg_max_char_run']:.1f} "
            f"best={best_val_loss:.4f} patience={epochs_without_improvement}/{cfg.patience}"
        )

        if epochs_without_improvement >= cfg.patience:
            print("Early stopping triggered.")
            break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_scores = evaluate_decoding(
        model,
        test_loader,
        tgt_vocab,
        device,
        max_len=cfg.max_tgt_len,
        sample_limit=min(cfg.decode_samples * 2, len(test_df)),
        src_pad_idx=src_vocab.pad_idx,
        max_ratio=cfg.decode_max_ratio,
        max_extra=cfg.decode_max_extra,
        min_len=cfg.decode_min_len,
    )
    save_json(output_dir / "test_metrics.json", test_scores)
    print(
        f"test_cer={test_scores['cer']:.4f} test_wer={test_scores['wer']:.4f} "
        f"eos_rate={test_scores['eos_rate']:.4f} forced_eos_rate={test_scores['forced_eos_rate']:.4f} "
        f"avg_run={test_scores['avg_max_char_run']:.2f}"
    )


if __name__ == "__main__":
    main()
