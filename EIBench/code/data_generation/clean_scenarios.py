#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format-clean generated EIBench scenarios.

This only checks and fixes the **section format** of each role profile. Every
profile's system_prompt must contain a fixed set of "### sections" in a fixed
order; records that do not match are sent to an LLM that re-orders / fills /
merges the sections back into the target layout without changing the meaning.

The LLM call is left empty -- implement call_llm() the same way as in
generate_scenarios.py (the paper uses Gemini-3.1-Pro). The Chinese prompt and
section names below are part of the benchmark and must not be translated.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kw):  # type: ignore
        return x


# -----------------------------------------------------------------------------
# LLM call -- IMPLEMENT THIS (same contract as generate_scenarios.py).
# Send `messages` to your model and return the text reply as a string.
# -----------------------------------------------------------------------------
def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.3,
    max_retry: int = 3,
) -> str:
    raise NotImplementedError(
        "Implement call_llm(): plug in your own LLM API "
        "(the paper uses Gemini-3.1-Pro)."
    )


# -----------------------------------------------------------------------------
# Target section format (the only thing this script enforces)
# -----------------------------------------------------------------------------
AGG_TARGET_SECTIONS = ["背景", "你的角色与立场", "说话风格与性格", "你知道的事实", "行为逻辑与偏好", "你的目的"]
DEF_TARGET_SECTIONS = ["背景", "你的角色与立场", "你知道的事实", "你的核心目标"]

# Human-readable role labels per scene, used in the fix prompt.
ROLE_DESC_MAP = {
    "support": ("求助者/倾诉者", "倾听者/安慰者"),
    "repair": ("生气方", "道歉方/修复方"),
    "defense": ("施压方/攻击者", "防守方"),
    "charm": ("被搭讪的目标", "搭讪者/魅力展示者"),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def strip_md(text: str) -> str:
    """Strip a leading/trailing ``` fenced code block, if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[1:end]).strip()
    return text


def get_sections(text: str) -> List[str]:
    """Extract the section names after '### '."""
    return [s.strip() for s in re.findall(r"###\s+(.+?)(?:\n|$)", text)]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# -----------------------------------------------------------------------------
# Step 1: rule-based section-format check
# -----------------------------------------------------------------------------
def check_record(rec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return {field: [issues]}. An empty dict means the section format is fine."""
    issues: Dict[str, List[str]] = {}
    for side, target, key in (
        ("aggressor_profile", AGG_TARGET_SECTIONS, "agg_structure"),
        ("defender_profile", DEF_TARGET_SECTIONS, "def_structure"),
    ):
        text = rec.get(side, {}).get("system_prompt", "")
        secs = get_sections(text)
        if secs == target:
            continue
        missing = [s for s in target if s not in secs]
        extra = [s for s in secs if s not in target]
        sub: List[str] = []
        if missing:
            sub.append(f"缺少: {missing}")
        if extra:
            sub.append(f"多余: {extra}")
        if secs and not missing and not extra:
            sub.append(f"顺序错误: {secs}")
        if sub:
            issues[key] = sub
    return issues


# -----------------------------------------------------------------------------
# Step 2: LLM re-clean of records whose section format is off
# -----------------------------------------------------------------------------
AGG_FIX_PROMPT = """\
你是一个严格的格式编辑器。以下角色设定的小节结构不符合规范，请修正。

## 场景类型: {scene_tag}
## 角色: {role_desc}

## 当前内容:
---
{current}
---

## 目标格式（必须按此顺序，不多不少）:
### 背景
### 你的角色与立场
### 说话风格与性格
### 你知道的事实
### 行为逻辑与偏好
### 你的目的

## 具体问题:
{problems}

## 要求:
- 重新排列小节为目标顺序
- 如果缺少某个小节，从现有内容中提取或简短补充
- 如果有多余小节，把内容合并到最相关的目标小节中
- 不要改变实际内容的含义
- 只输出修正后的完整内容（从 ### 背景 开始），不要任何解释"""

DEF_FIX_PROMPT = """\
你是一个严格的格式编辑器。以下角色设定的小节结构不符合规范，请修正。

## 场景类型: {scene_tag}
## 角色: {role_desc}

## 当前内容:
---
{current}
---

## 目标格式（必须按此顺序，不多不少）:
### 背景
### 你的角色与立场
### 你知道的事实
### 你的核心目标

## 具体问题:
{problems}

## 要求:
- 重新排列小节为目标顺序
- 如果缺少某个小节，从现有内容中提取或简短补充
- 如果有多余小节，把内容合并到最相关的目标小节中
- 不要改变实际内容的含义
- 只输出修正后的完整内容（从 ### 背景 开始），不要任何解释"""


def _llm_fix(prompt: str, model: str, max_retry: int) -> str:
    """Send a single-user-message fix prompt and return the cleaned text."""
    raw = call_llm([{"role": "user", "content": prompt}], model=model, max_retry=max_retry)
    raw = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", raw or "").strip()
    return strip_md(raw)


def fix_record(rec: Dict[str, Any], issues: Dict[str, List[str]], model: str,
               max_retry: int = 3) -> Tuple[Dict[str, Any], bool]:
    """Fix a record's section format (anchors/scores untouched). Returns (rec, success)."""
    rec = copy.deepcopy(rec)
    scene_tag = rec.get("scene_tag", "unknown")
    agg_desc, def_desc = ROLE_DESC_MAP.get(scene_tag, ("角色A", "角色B"))
    success = True

    if issues.get("agg_structure"):
        try:
            rec["aggressor_profile"]["system_prompt"] = _llm_fix(AGG_FIX_PROMPT.format(
                scene_tag=scene_tag, role_desc=agg_desc,
                current=rec["aggressor_profile"]["system_prompt"],
                problems="\n".join(f"- {p}" for p in issues["agg_structure"]),
            ), model, max_retry)
        except Exception:
            success = False

    if issues.get("def_structure"):
        try:
            rec["defender_profile"]["system_prompt"] = _llm_fix(DEF_FIX_PROMPT.format(
                scene_tag=scene_tag, role_desc=def_desc,
                current=rec["defender_profile"]["system_prompt"],
                problems="\n".join(f"- {p}" for p in issues["def_structure"]),
            ), model, max_retry)
        except Exception:
            success = False

    return rec, success


# -----------------------------------------------------------------------------
# Per-file pipeline
# -----------------------------------------------------------------------------
def process_file(path: str, model: str, workers: int, check_only: bool = False) -> Tuple[int, int, int]:
    """Return (total, failed_before, still_failed_after). Rewrites the file in place."""
    if not os.path.exists(path):
        print(f"  [SKIP] {path}")
        return 0, 0, 0

    data = _read_jsonl(path)
    failed = {i: iss for i, rec in enumerate(data) if (iss := check_record(rec))}
    print(f"  {os.path.basename(path)}: {len(data)} records, {len(failed)} with bad format")

    if not failed:
        return len(data), 0, 0

    issue_types: Counter = Counter()
    for iss in failed.values():
        issue_types.update(iss.keys())
    print(f"  issue types: {dict(issue_types)}")

    if check_only:
        return len(data), len(failed), len(failed)

    def _fix(idx: int) -> Tuple[int, Dict[str, Any], bool]:
        fixed, ok = fix_record(data[idx], failed[idx], model, max_retry=3)
        return idx, fixed, ok

    fixed_n = 0
    pbar = tqdm(total=len(failed), desc="  fixing", ncols=100, leave=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed({ex.submit(_fix, i): i for i in failed}):
            try:
                idx, fixed, ok = fut.result()
                data[idx] = fixed
                fixed_n += int(ok)
            except Exception:
                pass
            pbar.update(1)
    pbar.close()

    still_failed = sum(1 for i in failed if check_record(data[i]))
    print(f"  fixed {fixed_n}/{len(failed)}, still bad {still_failed}")

    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(data), len(failed), still_failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-based section-format check + LLM re-clean")
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more jsonl files (globs allowed) to check and clean in place")
    ap.add_argument("--model", default="gemini-3.1-pro", help="LLM used for the re-clean fix")
    ap.add_argument("--workers", type=int, default=20, help="Concurrency for the fix step")
    ap.add_argument("--check-only", action="store_true", help="Only report issues, do not call the LLM")
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)) or [pat])

    g_total = g_failed = g_still = 0
    for fp in files:
        t, fb, sf = process_file(fp, args.model, args.workers, check_only=args.check_only)
        g_total += t
        g_failed += fb
        g_still += sf

    print(f"\nTotal: {g_total} records, {g_failed} bad format, {g_still} still bad after clean")


if __name__ == "__main__":
    main()
