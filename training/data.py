# training/data.py
#
# Unified streaming data pipeline for MASSIF LLM training.
# Combines a modular dataset registry with memory-safe streaming.
#
# Design:
#   - DATASET_REGISTRY: add/remove domains without touching loop code
#   - All datasets stream via HuggingFace streaming=True: no full RAM load
#   - Curriculum stages: run Stage 1 -> 2 -> 3 by changing ACTIVE_STAGE
#   - Per-domain extractors handle schema differences cleanly
#   - Max 3 failures per domain before silent disable: no infinite loops
#
# Usage:
#   from training.data import build_dataloader, set_stage, list_domains
#   set_stage(1)                    # Daoist + Buddhist (linguistic foundation)
#   set_stage(2)                    # TCM core (domain mastery)
#   set_stage(3)                    # Ayurveda + cross-system integration
#   set_stage('all')                # Everything enabled
#   loader = build_dataloader(config, tokenizer)

import torch
import itertools
from typing import Iterator, Optional, List, Dict, Callable
from dataclasses import dataclass, field


# ===========================================================================
# Schema-aware extractors (one per dataset)
# ===========================================================================

def extract_tcm_shizhen(row: dict) -> str:
    return row.get("text", row.get("content", "")) or ""

def extract_tcm_chat(row: dict) -> str:
    instruction = row.get("instruction", "")
    output = row.get("output", "")
    if output:
        return f"{instruction}\n{output}".strip()
    return row.get("text", "") or ""

def extract_philosophy(row: dict) -> str:
    return row.get("text", row.get("content", "")) or ""

def extract_buddhist(row: dict) -> str:
    return row.get("text", row.get("content", "")) or ""

def extract_ayurveda(row: dict) -> str:
    """
    jaychedaa/Ayurveda-LLM-dataset uses 'instruction' and 'output' columns.
    Explicit mapping required — 'text' and 'content' keys do not exist.
    """
    instruction = row.get("instruction", "")
    output = row.get("output", "")
    if not instruction and not output:
        return row.get("text", row.get("content", "")) or ""
    return f"Medical Concept: {instruction}\nTherapy Context: {output}"


# ===========================================================================
# Modular dataset registry
# ===========================================================================
# To add a new domain: append one DatasetEntry to DATASET_REGISTRY.
# To remove: set enabled=False. Never delete entries.
#
# Curriculum stages:
#   Stage 1 — Linguistic foundation: classical Chinese, Buddhist canon
#   Stage 2 — Core domain mastery: TCM textbooks, research, case studies
#   Stage 3 — Cross-system integration: Ayurveda, Vedic, comparative medicine

@dataclass
class DatasetEntry:
    name:        str
    hf_path:     str
    extractor:   Callable
    stage:       int
    weight:      float = 1.0
    enabled:     bool  = True
    hf_name:     Optional[str] = None
    data_files:  Optional[str] = None
    hf_split:    str = "train"


DATASET_REGISTRY: List[DatasetEntry] = [

    # Stage 1: Linguistic foundation
    DatasetEntry(
        name="dao_confucian",
        hf_path="gujilab/chinese-classical-corpus",
        hf_name="corpus",
        extractor=extract_philosophy,
        stage=1, weight=2.0, enabled=True,
    ),
    DatasetEntry(
        name="buddhist_canon",
        hf_path="buddhist-nlp/daizhige",
        extractor=extract_buddhist,
        stage=1, weight=1.5, enabled=True,
    ),

    # Stage 2: Core domain mastery
    DatasetEntry(
        name="tcm_chat",
        hf_path="ZJUFanLab/TCMChat-dataset-600k",
        data_files="sft/train/knowledge.json",
        extractor=extract_tcm_chat,
        stage=2, weight=3.0, enabled=True,
    ),
    DatasetEntry(
        name="tcm_book",
        hf_path="FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT",
        hf_name="TCM_Book_Corpus (Text)",
        extractor=extract_tcm_shizhen,
        stage=2, weight=2.0, enabled=False,  # T4 16GB: keep off
    ),
    DatasetEntry(
        name="tcm_web",
        hf_path="FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT",
        hf_name="TCM_Web_Corpus (Text)",
        extractor=extract_tcm_shizhen,
        stage=2, weight=2.0, enabled=False,  # T4 16GB: keep off
    ),

    # Stage 3: Cross-system integration
    DatasetEntry(
        name="ayurveda",
        hf_path="jaychedaa/Ayurveda-LLM-dataset",
        extractor=extract_ayurveda,
        stage=3, weight=1.0, enabled=True,
    ),

    # Stage 0: Fallback baseline (disabled for domain curriculum)
    DatasetEntry(
        name="tinystories",
        hf_path="roneneldan/TinyStories",
        extractor=lambda r: r.get("text", "") or "",
        stage=0, weight=1.0, enabled=False,
    ),

    # Future domains: add here
    # DatasetEntry(
    #     name="tibetan_medicine",
    #     hf_path="user/tibetan-corpus",
    #     extractor=extract_tibetan,
    #     stage=3, weight=1.0, enabled=False,
    # ),
]


# ===========================================================================
# Curriculum stage control
# ===========================================================================

ACTIVE_STAGE: Optional[int] = None


def set_stage(stage):
    """
    Enable domains for a curriculum stage and disable all others.
    Stages are cumulative: stage 2 includes stage 1 domains.

    Args:
        stage: 1, 2, 3, or 'all'
    """
    global ACTIVE_STAGE
    ACTIVE_STAGE = stage
    for entry in DATASET_REGISTRY:
        if stage == 'all':
            entry.enabled = True
        elif stage == 0:
            entry.enabled = (entry.stage == 0)
        else:
            entry.enabled = (0 < entry.stage <= int(stage))
    print(f"Curriculum set to Stage {stage}:")
    list_domains()


def list_domains():
    print(f"\n{'Domain':<20} {'Stage':<8} {'Enabled':<10} {'Weight':<8} {'Path'}")
    print('-' * 75)
    for d in DATASET_REGISTRY:
        status = 'ON' if d.enabled else 'off'
        print(f"{d.name:<20} {d.stage:<8} {status:<10} {d.weight:<8} {d.hf_path}")
    print()


def enable_domain(name: str):
    for d in DATASET_REGISTRY:
        if d.name == name:
            d.enabled = True
            print(f"[data] Enabled: '{name}'")
            return
    print(f"[data] Not found: '{name}'")


def disable_domain(name: str):
    for d in DATASET_REGISTRY:
        if d.name == name:
            d.enabled = False
            print(f"[data] Disabled: '{name}'")
            return
    print(f"[data] Not found: '{name}'")


# ===========================================================================
# Streaming domain loader
# ===========================================================================

def _stream_domain(entry: DatasetEntry) -> Iterator[str]:
    try:
        from datasets import load_dataset
        kwargs = dict(path=entry.hf_path, split=entry.hf_split, streaming=True)
        if entry.hf_name:
            kwargs['name'] = entry.hf_name
        if entry.data_files:
            kwargs['data_files'] = entry.data_files
        dataset = load_dataset(**kwargs)
        for row in dataset:
            text = entry.extractor(row)
            if text and len(text.strip()) > 20:
                yield text.strip()
    except Exception as e:
        print(f"[data] Domain '{entry.name}' stream error: {e}")
        return


# ===========================================================================
# Weighted interleaved stream with failure guard
# ===========================================================================

def _build_interleaved_stream(entries: List[DatasetEntry]) -> Iterator[str]:
    MAX_FAILURES = 3
    schedule = []
    for e in entries:
        schedule.extend([e.name] * max(1, int(e.weight)))

    streams: Dict[str, Iterator[str]] = {}
    for e in entries:
        streams[e.name] = _stream_domain(e)
        print(f"[data] Streaming: '{e.name}' (stage={e.stage}, weight={e.weight})")

    failure_counts: Dict[str, int] = {e.name: 0 for e in entries}
    schedule_cycle = itertools.cycle(schedule)

    for domain_name in schedule_cycle:
        if domain_name not in streams:
            continue
        if failure_counts.get(domain_name, 0) >= MAX_FAILURES:
            continue
        try:
            text = next(streams[domain_name])
            if text:
                yield text
        except StopIteration:
            entry = next(e for e in entries if e.name == domain_name)
            streams[domain_name] = _stream_domain(entry)
            try:
                yield next(streams[domain_name])
            except StopIteration:
                failure_counts[domain_name] += 1
                if failure_counts[domain_name] >= MAX_FAILURES:
                    print(f"[data] '{domain_name}' failed {MAX_FAILURES}x. Disabling.")


# ===========================================================================
# Token buffer and batch builder
# ===========================================================================

def _token_batch_generator(text_stream, tokenizer, seq_len, batch_size,
                            max_steps) -> Iterator[Dict[str, torch.Tensor]]:
    token_buffer = []
    batch_accumulator = []
    steps_yielded = 0

    for text in text_stream:
        try:
            ids = tokenizer.encode(
                text, add_special_tokens=True,
                truncation=True, max_length=seq_len * 4,
            )
        except Exception:
            continue

        token_buffer.extend(ids)

        while len(token_buffer) >= seq_len + 1:
            chunk = token_buffer[:seq_len + 1]
            token_buffer = token_buffer[seq_len + 1:]
            batch_accumulator.append(
                torch.tensor(chunk[:seq_len], dtype=torch.long)
            )
            if len(batch_accumulator) == batch_size:
                yield {'input_ids': torch.stack(batch_accumulator)}
                batch_accumulator = []
                steps_yielded += 1
                if steps_yielded >= max_steps:
                    return

        if len(token_buffer) > seq_len * 8:
            token_buffer = token_buffer[-seq_len * 2:]


# ===========================================================================
# Public interface
# ===========================================================================

def build_dataloader(config, tokenizer, max_steps=None,
                     stage=None) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Build a memory-safe streaming dataloader.

    Args:
        config:    MASSIFConfig
        tokenizer: HuggingFace tokenizer
        max_steps: Stop after this many batches
        stage:     Use only domains up to this stage (1, 2, or 3)
    """
    steps = max_steps if max_steps is not None else config.total_steps

    if tokenizer is None:
        return _synthetic_dataloader(config, steps)

    if stage is not None:
        entries = [e for e in DATASET_REGISTRY
                   if e.enabled and 0 < e.stage <= stage]
    else:
        entries = [e for e in DATASET_REGISTRY if e.enabled]

    if not entries:
        print("[data] No enabled domains. Falling back to synthetic data.")
        return _synthetic_dataloader(config, steps)

    return _token_batch_generator(
        _build_interleaved_stream(entries), tokenizer,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        max_steps=steps,
    )


def _synthetic_dataloader(config, max_steps):
    print("[data] Synthetic mode.")
    for _ in range(max_steps):
        yield {'input_ids': torch.randint(
            0, config.vocab_size, (config.batch_size, config.max_seq_len)
        )}
