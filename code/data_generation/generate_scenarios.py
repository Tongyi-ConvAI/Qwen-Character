#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate EIBench training scenarios (dual role profiles + state anchors).

For each scenario we:
  1. sample one theme + one scenario keyword, plus one option from each
     modifier pool (relationship / place / intensity / personality);
  2. draw a few random same-scene examples from an existing pool as few-shot;
  3. ask an LLM to expand all of the above into a full scenario: the
     simulated_user (aggressor) and model (defender) profiles, an opening line,
     and the three state anchors (initial / target / worst-case).

The actual LLM call is left empty -- implement call_llm() with your own
provider (the paper uses Gemini-3.1-Pro). Output is written as jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------------------------------------------------------
# LLM call -- IMPLEMENT THIS.
#
# In the paper, scenarios are generated with Gemini-3.1-Pro. We leave the actual
# API call empty so you can plug in whatever model/SDK you use (Gemini, an
# OpenAI-compatible endpoint, etc.). Just return the model's text reply.
# -----------------------------------------------------------------------------
def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 1.0,
    max_retry: int = 3,
) -> str:
    """
    Send `messages` (OpenAI-style [{"role","content"}, ...]) to your LLM and
    return the text reply as a string.

    TODO: implement this with your own provider. For example, an
    OpenAI-compatible endpoint:

        import requests, os
        url = os.environ["GEN_BASE_URL"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['GEN_API_KEY']}"}
        for _ in range(max_retry):
            r = requests.post(url, headers=headers, json={
                "model": model, "messages": messages, "temperature": temperature,
            }, timeout=300)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
        return ""
    """
    raise NotImplementedError(
        "Implement call_llm(): plug in your own LLM API "
        "(the paper uses Gemini-3.1-Pro)."
    )


# -----------------------------------------------------------------------------
# JSONL / parsing helpers
# -----------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Tolerate {"row": {...}} wrappers
            if isinstance(obj, dict) and "row" in obj and isinstance(obj.get("row"), dict):
                rows.append(obj["row"])
            else:
                rows.append(obj)
    return rows


def _append_jsonl(path: str, obj: Dict[str, Any], lock: Optional[Lock] = None) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if lock:
        lock.acquire()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if lock:
            lock.release()


def _maybe_tqdm(it, total: Optional[int] = None, desc: str = "", unit: str = "it"):
    if tqdm is None:
        return it
    return tqdm(it, total=total, desc=desc, unit=unit)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = str(text).strip()
    try:
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
    except Exception:
        pass
    try:
        m = re.search(r"```json\s*(\{.*?\})\s*```", t, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(t[start : end + 1])
    except Exception:
        return {}
    return {}


def _clamp_int(x: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        v = int(float(x))
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def _round5(x: Any) -> int:
    """Round to the nearest multiple of 5 and clip to [0, 100]."""
    return _clamp_int(round(_clamp_int(x) / 5) * 5)


def _finalize_anchors(row: Dict[str, Any], margin: int = 5) -> None:
    """
    Rule-based post-processing of the three anchors (in place), following the
    paper's anchor-selection rules:
      - round every value to the nearest multiple of 5, clip to [0, 100];
      - enforce the required ordering
          a_succ < a_start < a_fail   (negative emotion goes down on success)
          t_fail < t_start < t_succ   (relational state goes up on success)
        by nudging the offending anchor by `margin` so the gap holds.
    Anchor field mapping:
      start = initial_calibration, succ = rub_goals.target_*, fail = worst_case.
    """
    init = row.get("initial_calibration") or {}
    rub = row.get("rub_goals") or {}
    worst = row.get("worst_case") or {}

    a_start, t_start = _round5(init.get("initial_anger", 55)), _round5(init.get("initial_trust", 35))
    a_succ,  t_succ  = _round5(rub.get("target_anger", 20)),   _round5(rub.get("target_trust", 60))
    a_fail,  t_fail  = _round5(worst.get("worst_anger_min", 90)), _round5(worst.get("worst_trust_min", 5))

    # Negative-emotion axis: success below start, failure above start.
    if a_succ >= a_start:
        a_succ = _clamp_int(a_start - margin)
    if a_fail <= a_start:
        a_fail = _clamp_int(a_start + margin)
    # Relational axis: failure below start, success above start.
    if t_fail >= t_start:
        t_fail = _clamp_int(t_start - margin)
    if t_succ <= t_start:
        t_succ = _clamp_int(t_start + margin)

    init["initial_anger"], init["initial_trust"] = a_start, t_start
    rub["target_anger"], rub["target_trust"] = a_succ, t_succ
    # Keep the target ranges consistent with the rounded target.
    ar = rub.get("target_anger_range")
    tr = rub.get("target_trust_range")
    rub["target_anger_range"] = [_round5(ar[0]), _round5(ar[1])] if isinstance(ar, list) and len(ar) == 2 else [_clamp_int(a_succ - 10), a_succ]
    rub["target_trust_range"] = [_round5(tr[0]), _round5(tr[1])] if isinstance(tr, list) and len(tr) == 2 else [t_succ, _clamp_int(t_succ + 10)]
    worst["worst_anger_min"], worst["worst_trust_min"] = a_fail, t_fail

    row["initial_calibration"], row["rub_goals"], row["worst_case"] = init, rub, worst


def _get_scene_pool(rows: List[Dict[str, Any]], scene_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    target_tag = scene_type.lower()
    for r in rows:
        tag = str(r.get("scene_tag", "")).strip().lower()
        if tag == target_tag:
            out.append(r)
    return out


# -----------------------------------------------------------------------------
# Modifier seed pools: combining dimensions increases generation diversity
# -----------------------------------------------------------------------------
# Modifier seed pools. One option from each pool is sampled per scenario,
# following the paper's seed-pool table:
#   relationship (7), emotional intensity (8), personality influence (10),
# plus a place/setting pool. Charm scenes use charm_intensity in place of
# emotional_intensity.
SEED_DIMENSIONS = {
    # Speaker relationship between the simulated user and the model (paper: 7).
    "relationship": [
        "陌生人 (Stranger)",
        "普通朋友/熟人 (Acquaintance)",
        "密友/挚友 (Close friend)",
        "伴侣/恋人 (Romantic partner)",
        "家人/亲属 (Family)",
        "同事/上下级 (Colleague or boss-subordinate)",
        "服务关系：客户与客服/商家 (Service: customer & agent)",
    ],
    # Physical setting of the conversation (the "setting" factor).
    "place": [
        "电话通话 (Phone call)",
        "视频通话 (Video call)",
        "当面：家中/私密空间 (In person, at home)",
        "当面：咖啡馆/餐厅 (In person, cafe/restaurant)",
        "职场：办公室/会议室 (Workplace/meeting room)",
        "服务场所：店铺/营业厅/医院 (Service venue)",
        "户外/活动现场 (Outdoor/event)",
        "车内：开车或乘车途中 (In a car)",
        "旅途中：机场/车站/酒店 (Traveling)",
        "聚会场合：饭局/派对/婚礼 (Gathering/party)",
        "校园：教室/宿舍 (Campus)",
    ],
    # Strength of the user's emotional state (paper: 8).
    "emotional_intensity": [
        "轻微不满 (Mild Discontent)",
        "中等冲突 (Moderate Conflict)",
        "强烈对抗 (Intense Confrontation)",
        "情绪崩溃 (Emotional Breakdown)",
        "理性讨论 (Rational Discussion)",
        "温和交流 (Gentle Communication)",
        "激动宣泄 (Passionate Venting)",
        "冷静协商 (Calm Negotiation)"
    ],
    # Charm-specific interaction intensity (charm scenes use this instead).
    "charm_intensity": [
        "轻松调侃 (Playful Teasing)",
        "暧昧试探 (Flirty Probing)",
        "深度共鸣 (Deep Rapport)",
        "幽默互动 (Humorous Exchange)",
        "自信展示 (Confident Display)",
        "温和吸引 (Gentle Attraction)",
        "张力制造 (Tension Building)",
        "愉悦对话 (Pleasant Conversation)"
    ],
    # User's personality / speaking tendency (paper: 10).
    "personality_influence": [
        "内向保守型 (Introverted Conservative)",
        "外向激进型 (Extroverted Aggressive)",
        "理性分析型 (Rational Analytical)",
        "感性冲动型 (Emotional Impulsive)",
        "传统保守型 (Traditional Conservative)",
        "现代开放型 (Modern Open-Minded)",
        "强势主导型 (Dominant Assertive)",
        "温和妥协型 (Gentle Compromising)",
        "完美主义者 (Perfectionist)",
        "实用主义者 (Pragmatist)"
    ]
}

# -----------------------------------------------------------------------------
# Prompt: generate the dual profiles for each scene type
# -----------------------------------------------------------------------------
# Per-scene theme directions and scenario keywords. One theme + one keyword
# under it is sampled per scenario. The conversation goal is decided by the
# model from the generation prompt, so there is no separate goal pool here.
SCENE_KEYWORDS = {
    "support": {
        "scene": [
            {
                "category": "丧失与哀伤 (Grief & Loss)",
                "description": "失去重要的人、物或关系，用户处于悲痛状态",
                "specific_scenes": [
                    "亲人离世后的哀恸 (Loss of family)",
                    "宠物离世带来的空虚 (Pet loss)",
                    "断崖式分手的痛苦 (Breakup withdrawal)",
                    "确诊慢性病或重病的绝望 (Diagnosis of illness)",
                    "重要纪念物品遗失的懊恼 (Loss of sentimental item)",
                    "流产或生育失败的悲痛 (Miscarriage/Infertility)",
                    "离家在外的思乡与想家 (Homesickness)"
                ]
            },
            {
                "category": "挫折与自我价值危机 (Failure & Self-Worth)",
                "description": "遭遇失败，产生强烈的自我怀疑和羞耻感",
                "specific_scenes": [
                    "重要考试或面试失败的挫败 (Exam/Interview failure)",
                    "创业失败或投资亏损的崩溃 (Business failure)",
                    "长期努力却无回报的无力感 (Effort-reward imbalance)",
                    "冒充者综合症，觉得自己不配 (Imposter syndrome)",
                    "对外貌/身材的极度焦虑 (Body image anxiety)",
                    "被裁员/失业后的自我否定 (Job loss & self-doubt)",
                    "觉得自己一事无成的空心感 (Existential emptiness)"
                ]
            },
            {
                "category": "焦虑与恐惧 (Anxiety & Fear)",
                "description": "对未来的不确定性感到恐慌，需要安抚",
                "specific_scenes": [
                    "对演讲/公开表现的惊恐 (Performance anxiety)",
                    "对裁员流言的深夜焦虑 (Layoff anxiety)",
                    "等待体检结果的疑病恐慌 (Health anxiety)",
                    "社恐面临必须社交的场合 (Social anxiety)",
                    "产前/产后的抑郁与焦虑 (Perinatal anxiety)",
                    "对未来不确定性的弥漫性恐慌 (Generalized worry)",
                    "经济压力/还债的窒息感 (Financial stress)"
                ]
            },
            {
                "category": "第三方冲突与吐槽 (Venting about a Third Party)",
                "description": "对生活或职场中的第三方（非待测方）感到愤怒或委屈",
                "specific_scenes": [
                    "控诉领导不公或瞎指挥 (Unfair boss)",
                    "抱怨同事甩锅或抢功 (Toxic colleague)",
                    "吐槽客户无理取闹 (Difficult client)",
                    "控诉伴侣的冷暴力或坏习惯 (Complaining about partner)",
                    "抱怨父母/公婆的过度干涉 (In-law conflict)",
                    "吐槽室友的生活习惯 (Roommate conflict)",
                    "被朋友背刺或借钱不还的委屈 (Friendship betrayal)"
                ]
            },
            {
                "category": "迷茫与人生抉择 (Confusion & Life Decisions)",
                "description": "面临两难选择或失去方向，需要理清思路而非简单安慰",
                "specific_scenes": [
                    "在两个 offer 或城市间犹豫不决 (Decision paralysis)",
                    "纠结是否要结束一段关系 (Relationship dilemma)",
                    "对转行/裸辞等迈出舒适区的恐惧 (Fear of change)",
                    "面临赡养/生育等家庭责任的两难 (Family duty dilemma)",
                    "对人生意义与方向的迷失 (Loss of direction)",
                    "面临道德或价值观冲突的纠结 (Moral dilemma)"
                ]
            },
            {
                "category": "孤独、倦怠与慢性消耗 (Loneliness & Burnout)",
                "description": "没有突发事件，但长期的孤独或耗竭让用户低落",
                "specific_scenes": [
                    "深夜独处时的强烈孤独感 (Late-night loneliness)",
                    "长期高压工作后的职业倦怠 (Burnout)",
                    "异乡打拼缺乏归属感 (No sense of belonging)",
                    "在亲密关系中却感到孤独 (Lonely within a relationship)",
                    "长期照护家人后的身心耗竭 (Caregiver fatigue)",
                    "找不到人倾诉的情绪积压 (No one to talk to)"
                ]
            }
        ]
    },
    "defense": {
        "scene": [
            {
                "category": "无理要求与越界 (Unreasonable Demands & Boundary-Crossing)",
                "description": "用户提出超出规则或职责范围的要求，模型需既守住边界又不激化",
                "specific_scenes": [
                    "要求超出规则的退款或赔偿 (Out-of-policy refund)",
                    "要求免费追加大量工作量 (Free scope creep)",
                    "要求泄露他人隐私或保密信息 (Demand for confidential info)",
                    "要求破例走后门 (Demand for an exception)",
                    "强行索要私人联系方式或行程 (Pushing for private info)",
                    "要求承担本不属于自己的责任 (Offloading responsibility)",
                    "临时加价或单方面变更已定条款 (Last-minute term change)"
                ]
            },
            {
                "category": "指责甩锅与情绪攻击 (Blame-Shifting & Verbal Aggression)",
                "description": "用户用指责、贬低或攻击施压，模型需冷静且不被带节奏",
                "specific_scenes": [
                    "把自己的错甩锅给对方 (Shifting blame)",
                    "用大嗓门或人身攻击施压 (Verbal aggression)",
                    "翻旧账连环指责 (Dredging up the past)",
                    "阴阳怪气贬低/嘲讽 (Passive-aggressive belittling)",
                    "用‘别人都行你为什么不行’对比施压 (Unfair comparison)",
                    "迁怒：把无关的火气撒在对方身上 (Displaced anger)",
                    "当众让对方下不来台 (Public humiliation)"
                ]
            },
            {
                "category": "威胁施压与最后通牒 (Threats & Ultimatums)",
                "description": "用户用威胁或升级手段逼迫让步",
                "specific_scenes": [
                    "威胁投诉、差评或公开曝光 (Threatening to escalate)",
                    "威胁分手/断交/辞职相要挟 (Threatening to walk away)",
                    "下最后通牒‘不答应就……’ (Ultimatum)",
                    "扬言找关系/找媒体施压 (Threatening to pull strings)",
                    "以举报或法律手段恐吓 (Intimidation)",
                    "用‘我再也不……’进行情绪要挟 (Emotional ultimatum)"
                ]
            },
            {
                "category": "情感勒索与道德绑架 (Emotional Blackmail & Guilt-Tripping)",
                "description": "用户用愧疚、亏欠或人情来绑住对方，模型需温和而坚定地拒绝",
                "specific_scenes": [
                    "道德绑架‘你不帮就是没良心’ (Moral coercion)",
                    "卖惨或装可怜博取让步 (Playing the victim)",
                    "用‘我为你付出这么多’索取回报 (Invoking past sacrifice)",
                    "用辈分/资历压人 (Seniority pressure)",
                    "用‘为你好’之名强加安排 (Imposing under good intentions)",
                    "拿亲情/友情/爱情做筹码 (Weaponizing the relationship)"
                ]
            },
            {
                "category": "关系内的索取与控制 (Demands & Control within a Relationship)",
                "description": "在亲近关系中被过度索取或控制，模型需守住自我边界",
                "specific_scenes": [
                    "伴侣过度查岗与控制 (Controlling partner)",
                    "家人逼婚/逼生/干涉人生选择 (Family pressure on life choices)",
                    "朋友反复借钱不还还理直气壮 (Entitled borrowing)",
                    "同事或上级越界派活 (Boundary-crossing work demands)",
                    "被要求随叫随到/无限付出 (Demand for constant availability)",
                    "亲友过度干涉你的隐私与决定 (Overstepping privacy)"
                ]
            },
            {
                "category": "质疑挑衅与刁难 (Challenge & Provocation)",
                "description": "用户质疑、挑衅或故意刁难，模型需稳住而不被激怒",
                "specific_scenes": [
                    "质疑你的专业能力与资质 (Questioning competence)",
                    "故意找茬挑刺抬杠 (Nitpicking and bickering)",
                    "用极端假设逼你表态 (Cornering with hypotheticals)",
                    "反复打断不让把话说完 (Constant interrupting)",
                    "言语挑衅试图激怒你 (Provoking a reaction)",
                    "要求当场认错式道歉 (Demanding public submission)",
                    "胡搅蛮缠不讲逻辑 (Bad-faith arguing)"
                ]
            },
            {
                "category": "纠缠推销与说服施压 (Pushy Persuasion & Hard-Sell)",
                "description": "用户死缠烂打地推销或说服，模型需礼貌而坚决地脱身",
                "specific_scenes": [
                    "死缠烂打的推销或拉投资 (Aggressive sales pitch)",
                    "反复劝说你改变立场/信仰 (Pushing to convert you)",
                    "拉你入伙/传销式游说 (MLM-style recruiting)",
                    "不接受拒绝、反复追问理由 (Refusing to take no)",
                    "用人情绑架让你帮忙转发/站台 (Pressuring an endorsement)",
                    "越界的搭讪或纠缠示好 (Unwanted advances)"
                ]
            }
        ]
    },
    "repair": {
        "scene": [
            {
                "category": "职业履职瑕疵与失误 (Professional Performance Gaps)",
                "description": "工作中因疏忽或失误导致的问题，有负面影响但尚可补救",
                "specific_scenes": [
                    "因管理不善导致的进度滞后 (Project delay)",
                    "产出未达预期或有明显瑕疵 (Subpar work quality)",
                    "因沟通不及时导致误解 (Communication lag)",
                    "非核心问题上的操作失误或遗漏 (Operational error)",
                    "未兑现非书面的口头承诺 (Unfulfilled verbal promise)",
                    "数据或信息出错需更正 (Data/info error to correct)"
                ]
            },
            {
                "category": "亲密关系中的忽视与失言 (Intimate Insensitivity & Neglect)",
                "description": "亲近关系中因缺乏体察或态度敷衍造成的情感伤害",
                "specific_scenes": [
                    "因忙碌忽略伴侣/家人的情感诉求 (Emotional neglect)",
                    "争吵或压力下口不择言的伤害 (Verbal hurt)",
                    "对共同计划消极或遗忘 (Flaking on shared plans)",
                    "生活习惯摩擦引发对方爆发 (Habitual friction)",
                    "在对方需要时缺席或反应冷淡 (Absence of support)",
                    "忘记重要的纪念日 (Forgetting an anniversary)"
                ]
            },
            {
                "category": "社交失礼与信用受损 (Social Missteps)",
                "description": "朋友或熟人圈中因情商掉线导致好感下降，需修补形象",
                "specific_scenes": [
                    "无意泄露了不该说的事 (Accidental slip)",
                    "对善意帮助没给反馈或感谢 (Lack of gratitude)",
                    "玩笑开过头无心冒犯 (Joke landed poorly)",
                    "爽约或临时放鸽子 (Flaking/last-minute cancel)",
                    "在群体场合让对方难堪 (Embarrassing them publicly)",
                    "转述走样引发误会 (Misrelayed message)"
                ]
            },
            {
                "category": "信任受损与背约 (Broken Trust & Letdown)",
                "description": "做了较重地辜负对方信任的事，需要正面承担并重建信任",
                "specific_scenes": [
                    "答应保密却说漏了嘴 (Broke a confidence)",
                    "关键时刻没站在对方一边 (Failed to have their back)",
                    "隐瞒或淡化了重要的事 (Hid something important)",
                    "重复犯同一个让对方失望的错 (Repeat letdown)",
                    "在对方最需要时掉链子 (Let them down when it mattered)",
                    "言行不一被对方发现 (Caught not walking the talk)"
                ]
            },
            {
                "category": "服务与承诺违约 (Service & Promise Breach)",
                "description": "服务或交付场景中失误致用户受损，需致歉并重建信任",
                "specific_scenes": [
                    "产品或服务出问题致用户损失 (Service failure)",
                    "交付延期需致歉 (Delivery delay)",
                    "承诺的优惠或方案无法兑现 (Unfulfilled offer)",
                    "客服处理不当激化矛盾 (Mishandled complaint)",
                    "误操作影响了客户 (Mistake affecting the customer)",
                    "规则变更未提前告知 (Unannounced policy change)"
                ]
            }
        ]
    },
    "charm": {
        "scene": [
            {
                "category": "破冰与初识 (Ice-Breaking & First Meeting)",
                "description": "从零开始的陌生互动，用户带着距离感或冷淡，需要先打开局面",
                "specific_scenes": [
                    "聚会/活动上和陌生人搭话 (Approaching a stranger)",
                    "被介绍认识的初次寒暄 (Introduced by a mutual friend)",
                    "电梯/通勤偶遇心动对象 (Chance encounter)",
                    "线下见面前的初步接触 (First contact before meeting)",
                    "向冷淡的人打开话匣子 (Warming up a cold person)",
                    "在群体中制造第一印象 (Making a first impression)"
                ]
            },
            {
                "category": "暧昧吸引阶段 (Flirtation & Attraction)",
                "description": "从陌生到心动，通过言语和行为制造吸引力与暧昧氛围",
                "specific_scenes": [
                    "展现个人才华与独特视角 (Show talent/perspective)",
                    "用幽默化解尴尬 (Defuse with humor)",
                    "分享有趣的人生经历 (Share a fun story)",
                    "展现自信与真诚 (Confidence and sincerity)",
                    "巧妙赞美对方的特点 (A well-placed compliment)",
                    "制造适度的神秘感 (A bit of mystery)",
                    "用轻松调侃拉近距离 (Playful teasing)",
                    "展现对对方话题的专注 (Attentive listening)"
                ]
            },
            {
                "category": "情感深化阶段 (Emotional Deepening)",
                "description": "从表面吸引发展到情感共鸣，建立更深层次的连接",
                "specific_scenes": [
                    "挖掘共同兴趣与价值观 (Find common ground)",
                    "适度分享脆弱面 (Share vulnerability)",
                    "展现关心与理解 (Show care and understanding)",
                    "创造温馨的时刻 (Create a warm moment)",
                    "建立信任与安全感 (Build trust)",
                    "倾听对方的困扰 (Listen to their worries)",
                    "分享梦想与目标 (Share dreams)",
                    "真诚表达欣赏 (Express appreciation)"
                ]
            },
            {
                "category": "专业场合的魅力 (Professional Charm)",
                "description": "在职业或正式场合展现专业魅力与个人吸引力",
                "specific_scenes": [
                    "展现专业能力与见解 (Show expertise)",
                    "赢得前辈或导师赏识 (Win a mentor's regard)",
                    "在团队中脱颖而出 (Stand out in a team)",
                    "建立职业人脉 (Build a professional connection)",
                    "得体地自我展示 (Tasteful self-presentation)",
                    "化解专业距离破冰 (Break the ice across status)"
                ]
            },
            {
                "category": "恋爱关系经营 (Romantic Nurturing)",
                "description": "在恋爱关系中保持吸引力，深化情感纽带",
                "specific_scenes": [
                    "制造惊喜与新鲜感 (Surprise and novelty)",
                    "展现体贴与关爱 (Thoughtfulness)",
                    "保持独立又亲密 (Independence within closeness)",
                    "规划共同的未来 (Plan a shared future)",
                    "创造浪漫的纪念时刻 (A romantic moment)",
                    "重燃降温期的感情 (Rekindle a cooling bond)",
                    "保持沟通的开放与诚实 (Open, honest communication)",
                    "共同面对生活挑战 (Face challenges together)"
                ]
            }
        ]
    }
}


def _get_gen_scene_system(scene_type: str) -> str:
    """Build the scene-specific generation system prompt."""
    scene_type_upper = scene_type.upper()

    # Scene-specific constraint description
    constraints = {
        "support": "核心是'对方处于痛苦/压力/困境中，需要被倾听/共情/建议/鼓励'，或是针对矛盾冲突的调停，而不是吵架撕逼、道歉挽回或暧昧撩拨。",
        "defense": "TODO: 添加 Defense 场景的核心约束描述",
        "repair": "TODO: 添加 Repair 场景的核心约束描述",
        "charm": "在无直接利害关系的场景下，通过高情商互动提升双方的好感度与个人魅力。"
    }

    constraint_desc = constraints.get(scene_type, f"必须是 {scene_type_upper} 类型的对话场景。")

    return f"""你是一个数据集生成器，专门生成【{scene_type_upper}】类型的双人对话场景。
你将输出"发难者(aggressor)"与"被测者(defender)"双方的 system_prompt，以及 aggressor 的 opening_line。

强约束：
- 必须是 {scene_type_upper}：{constraint_desc}
- 必须高度多样：不要复刻 few-shot 的主题、职业、关系、场景地点、具体事件；每条都要明显不一样。
- system_prompt 要包含：背景/身份立场/事实/情绪触发点/安抚点/说话风格（类似你看到的 few-shot），且是中文。
- 你生成的内容要参考所给的例子中的格式，比如 defender system prompt 中不应该出现你的任务等内容

输出格式（必须严格 JSON，仅此，不要代码块）：
{{
  "title": "...",
  "aggressor_profile": {{
    "system_prompt": "...",
    "opening_line": "..."
  }},
  "defender_profile": {{
    "system_prompt": "..."
  }}
}}
"""


def _few_shot_block(samples: List[Dict[str, Any]]) -> str:
    ex = []
    for s in samples:
        ex.append(
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "aggressor_system_prompt": (s.get("aggressor_profile") or {}).get("system_prompt", ""),
                "aggressor_opening_line": (s.get("aggressor_profile") or {}).get("opening_line", ""),
                "defender_system_prompt": (s.get("defender_profile") or {}).get("system_prompt", ""),
            }
        )
    return json.dumps(ex, ensure_ascii=False, indent=2)


def generate_scene(
    scene_samples: List[Dict[str, Any]],
    scene_type: str,
    rng: random.Random,
    model: str,
    temperature: float,
    max_retry: int,
) -> Dict[str, Any]:
    shots = rng.sample(scene_samples, k=2)
    scene_type_upper = scene_type.upper()

    # Look up this scene's theme directions.
    scene_keywords = SCENE_KEYWORDS.get(scene_type, {"scene": []})
    scene_categories = scene_keywords.get("scene", [])

    # Sample one theme direction, then one scenario keyword under it.
    selected_category = None
    scene_kw = "通用场景"
    category_desc = ""
    if scene_categories:
        selected_category = rng.choice(scene_categories)
        category_desc = selected_category.get("description", "")
        specific_scenes = selected_category.get("specific_scenes", [])
        if specific_scenes:
            scene_kw = rng.choice(specific_scenes)

    # Sample one option from each modifier dimension (relationship, place,
    # intensity, personality). Charm uses charm_intensity for the intensity slot.
    intensity_dim = "charm_intensity" if scene_type == "charm" else "emotional_intensity"
    modifier_dims = ["relationship", "place", intensity_dim, "personality_influence"]
    additional_seeds = {}
    for dim in modifier_dims:
        options = SEED_DIMENSIONS.get(dim, [])
        if options:
            additional_seeds[dim] = rng.choice(options)

    user_prompt = (
        f"下面是 2 个 {scene_type_upper} 场景示例（仅供风格参考，不要复刻）：\n"
        + _few_shot_block(shots)
        + f"\n\n请生成 1 个全新的 {scene_type_upper} 场景（不要与示例雷同）。\n"
    )

    # Core scene constraint: the sampled theme + scenario keyword
    if selected_category:
        user_prompt += f"为降低重复性，本次生成你必须围绕下面【指定的大类与具体场景】展开：\n"
        user_prompt += f"- 大类：{selected_category.get('category', '')}\n"
        user_prompt += f"- 大类描述：{category_desc}\n"
        user_prompt += f"- 具体场景关键词：{scene_kw}\n\n"
        user_prompt += "请在背景中清晰体现该具体场景关键词对应的内容（可以同义改写，但语义必须命中该大类下的具体场景）。\n\n 同时你要注意双方的身份，我们将待测模型的角色设定为 defender 方，考察的是待测模型在该场景下的情商，所以你要注意身份的设定，不要反了，还需要给场景设定一定的难度和考察性。\n\n"
    else:
        user_prompt += "为降低重复性，本次生成你必须围绕下面【指定的场景关键词】展开，并在背景中清晰体现该关键词对应的内容（可以同义改写，但语义必须命中）：\n"
        user_prompt += f"- 指定场景关键词：{scene_kw}\n\n"

    # Modifier constraints: one option from each dimension
    if additional_seeds:
        user_prompt += "为增加场景多样性，请在生成时融入以下【多维度约束】（自然融入背景和人物设定，不要生硬堆砌）：\n"
        dimension_names = {
            "relationship": "双方关系",
            "place": "场景地点",
            "emotional_intensity": "情感强度",
            "charm_intensity": "吸引强度",
            "personality_influence": "性格特征",
        }
        for dimension, seed in additional_seeds.items():
            dim_name = dimension_names.get(dimension, dimension)
            user_prompt += f"- {dim_name}：{seed}\n"
        user_prompt += "\n请确保生成的场景自然地体现这些维度的特征。若某个约束与核心场景关键词冲突或不合理，请优先遵循场景关键词的语义要求，灵活调整或忽略相关约束。\n"
        user_prompt += "特别地，【场景地点】必须与该场景和双方关系合理匹配；如果随机给定的地点放在该场景里不自然（例如本应私密的对话被放到公共场合），请改写成一个更合适的地点，不要生硬照搬。\n\n"

    user_prompt += "对话目标由你根据场景自行设定（不要照搬示例）。\n"
    user_prompt += "强约束：输出严格 JSON（不要解释、不要代码块）。"

    gen_system = _get_gen_scene_system(scene_type)

    last = ""
    for _ in range(max_retry):
        raw = call_llm(
            [
                {"role": "system", "content": gen_system},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_retry=3,
        )
        last = raw
        obj = _extract_json_object(raw)
        if obj and isinstance(obj.get("aggressor_profile"), dict) and isinstance(obj.get("defender_profile"), dict):
            return obj
        user_prompt += "\n\n上次输出解析失败：请严格输出一个 JSON 对象，不要带任何解释或代码块。"
    return {}


# -----------------------------------------------------------------------------
# Prompt: anchors (initial_calibration / rub_goals)
# -----------------------------------------------------------------------------
RST_RUBRIC_TEXT = """# 1. 评分标准与锚点定义 (Calibration Rubric)

请严格参照以下标准设定数值，根据 SCENE_TAG 的类型适配语义。

## A. 情绪/对抗值 (Anger Score): 0 ~ 100
**核心定义**：该数值代表 Aggressor 的负面情绪强度或对抗意愿。
* **Defense/Repair 场景**：代表“愤怒、仇恨、攻击性”。
* **Support 场景**：代表“悲伤、绝望、崩溃程度”（分越高越痛苦）。
* **Charm 场景**：代表“反感、厌烦、心理防御”（分越高越讨厌 Defender）。

| 分数段 | 状态定义 | 行为表现 (通用参考) |
| :--- | :--- | :--- |
| **0-20** | **平稳/积极** | 情绪稳定，语气平和，甚至带有笑意或期待。 |
| **21-40** | **轻微负面** | 皱眉、催促、疑惑、冷淡、轻微的焦虑或不耐烦。 |
| **41-60** | **显著波动** | 音量提高，指责/哭诉/反讽，明显的痛苦或抗拒，“我很不爽/我很难受”。 |
| **61-80** | **强烈爆发** | 辱骂/嚎啕大哭/人身攻击，情绪占据主导，难以理性沟通。 |
| **81-100**| **彻底失控** | 暴力倾向/自残倾向/完全拒绝沟通/拉黑报警，处于崩溃或暴走边缘。 |

## B. 信任/好感值 (Trust Score): 0 ~ 100
**核心定义**：该数值代表 Aggressor 对 Defender 的接纳程度。
* **Defense 场景**：代表“采信度”（是否相信你的解释）。
* **Repair 场景**：代表“原谅度”（是否愿意翻篇）。
* **Support 场景**：代表“依赖度/安全感”（是否觉得你懂他）。
* **Charm 场景**：代表“兴趣/吸引力”（是否想继续了解你）。

| 分数段 | 状态定义 | 行为表现 (通用参考) |
| :--- | :--- | :--- |
| **0-20** | **极度排斥** | 认定你是骗子/仇人/渣男，完全封闭内心，每一句话都怀疑。 |
| **21-40** | **怀疑/冷漠** | 也就是“听听看吧”，防御心重，需要强证据或强吸引力才能破冰。 |
| **41-60** | **中立/观察** | 公事公办，不带个人感情，或者“不讨厌但也没感觉”。 |
| **61-80** | **接纳/好感** | 愿意配合，语气缓和，觉得你这人不错/说的有道理。 |
| **81-100**| **盲信/依赖** | “只信你”，“全靠你了”，完全敞开心扉。 |

---

# 2. 现实成功阈值 (Realistic Success Threshold - RST) 设定逻辑

**RST 是“及格线/达成线”，不是“完美线”。** 只要达到此状态，对话即可视为有效收束。
请根据 **SCENE_TAG** 选择对应的逻辑：

### Logic 1: 【Defense 场景】(自卫/抗压)
* **目标**：止损与降温。不指望对方立刻以此为荣，只要不再攻击即可。
* **RST 设定**：
    * `Target_Anger`: **20~40** (从暴怒降为不满/唠叨，只要不炸就行)。
    * `Target_Trust`: **40~60** (从敌对变为中立/勉强采信，不需要变成朋友)。

### Logic 2: 【Repair 场景】(修复/挽回)
* **目标**：重建连接。重点是 Trust 的回升，Anger 可以依然保留一点（毕竟受了伤）。
* **RST 设定**：
    * `Target_Anger`: **10~30** (情绪平复，但这事儿还没完全忘)。
    * `Target_Trust`: **60~80** (愿意再给一次机会，关系解冻)。

### Logic 3: 【Support 场景】(支持/安抚)
* **目标**：情绪承托。让对方从“失控”回到“能正常生活/思考”的状态。
* **RST 设定**：
    * `Target_Anger`: **30~50** (注意！悲伤很难瞬间清零，只要从80+降到能对话的程度就算成功)。
    * `Target_Trust`: **70~90** (建立极强的安全感连接)。

### Logic 4: 【Charm 场景】(魅力/说服)
* **目标**：破冰与转化。从“没兴趣”变成“有兴趣”。
* **RST 设定**：
    * `Target_Anger`: **0~20** (保持愉快或微躁的打情骂俏)。
    * `Target_Trust`: **50~70** (从陌生人变成愿意见面/加微信/签单的状态，不要设成100，刚认识不可能盲信)。
"""

def _get_score_system(scene_type: str) -> str:
    """Build the scene-specific anchor-scoring system prompt."""
    scene_type_upper = scene_type.upper()

    # Scene-specific reading of the Anger / Trust axes
    semantics = {
        "support": "Support场景里，Anger 可理解为'负向情绪强度/崩溃值'（分高代表绝望/失控），Trust 可理解为'被理解/依赖/安全感'。",
        "defense": "Defense 场景里，Anger 可理解为'攻击性/施压强度/敌意'（分高代表想伤害或打压你），Trust 可理解为'对辩解的采信度/是否买账'。",
        "repair": "Repair 场景里，Anger 可理解为'因过错产生的怨恨/追责意愿'（分高代表不想翻篇），Trust 可理解为'原谅程度/关系亲密度'。",
        "charm": "Charm 场景里，Anger 可理解为'反感/厌烦/尴尬/心理防御'（分高代表觉得你油腻或被冒犯），Trust 可理解为'好感度/兴趣/吸引力'（分高代表想继续了解你）。"
    }

    semantic_desc = semantics.get(scene_type, f"{scene_type_upper} 场景的对话情绪指标。")

    return f"""你是一位精通中国社会人情世故、面子文化与博弈心理学的社交场景架构师。
你只负责给定场景的数值初始化与"现实成功阈值 RST（达成即可）"设定，严格参照下方锚点与逻辑，不要凭感觉打分。

{RST_RUBRIC_TEXT}

额外约束：
- 本条数据的 scene_tag 为 {scene_type_upper}。{semantic_desc}
- 设定必须合理：initial_anger / initial_trust 要符合场景；target_anger / target_trust 必须是"现实可达的成功阈值"，不要写成 0/100 或者过于离谱。
- 输出字段结构必须与数据集"原始格式"一致：
  - goal_one_liner / success_criteria / must_avoid 在顶层
  - initial_calibration 仅包含 initial_anger / initial_trust（reasoning 可给但会被丢弃）
  - rub_goals 内包含 target_anger_range / target_trust_range（list[int,int]）以及 target_anger / target_trust（int）

请输出严格 JSON（仅此，不要代码块）：
{{
  "goal_one_liner": "一句话目标（面向 defender 的对话目标）",
  "success_criteria": ["..."],
  "must_avoid": ["..."],
  "initial_calibration": {{
    "initial_anger": int,
    "initial_trust": int,
    "reasoning": "可选：简述为什么这么定"
  }},
  "rub_goals": {{
    "target_anger_range": [int, int],
    "target_trust_range": [int, int],
    "target_anger": int,
    "target_trust": int
  }}
}}
"""


def score_scene(
    scene_row: Dict[str, Any],
    scene_type: str,
    model: str,
    temperature: float,
    max_retry: int,
) -> Dict[str, Any]:
    payload = {
        "title": scene_row.get("title", ""),
        "scene_tag": scene_type,
        "aggressor_opening_line": (scene_row.get("aggressor_profile") or {}).get("opening_line", ""),
        "aggressor_system_prompt": (scene_row.get("aggressor_profile") or {}).get("system_prompt", ""),
        "defender_system_prompt": (scene_row.get("defender_profile") or {}).get("system_prompt", ""),
    }
    user_prompt = "请基于以下场景信息，给出初始分与现实上限（RUB），输出 JSON：\n\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    score_system = _get_score_system(scene_type)
    last = ""
    for _ in range(max_retry):
        raw = call_llm(
            [
                {"role": "system", "content": score_system},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_retry=3,
        )
        last = raw
        obj = _extract_json_object(raw)
        if obj.get("initial_calibration") and obj.get("rub_goals"):
            return obj
        user_prompt += "\n\n上次解析失败：请只输出 JSON。"
    return {}


# -----------------------------------------------------------------------------
# Prompt: worst_case (failure anchor)
# -----------------------------------------------------------------------------
WORST_SYSTEM = """
# Role
你是一名“社交灾难评估员”。你的任务是根据场景类型（scene_tag）和对话背景，预测该场景在**最糟糕的沟通策略下**（例如：情商极低、激怒对方、逃避责任、油腻猥琐）可能跌至的**数值下限**。

请根据以下逻辑进行评估：

### 1. Defense (应对刁难/施压)
* **最坏情况定义**：防御失败，矛盾全面升级。对方从单纯的“施压”转变为“人身攻击”或“实际报复”。
* **评分逻辑**：Anger 极高（80-100，暴怒/动手），Trust 极低（0-10，彻底敌对）。
* **关注点**：是否会因为回怼而导致对方失控？

### 2. Repair (修复过错/挽回)
* **最坏情况定义**：修复无效，且因为态度不端正（如推卸责任、狡辩）导致“罪加一等”，关系彻底不可逆地破裂。
* **评分逻辑**：Trust 是核心指标，会跌至冰点（0-5，绝望/死心）。Anger 取决于对方是“哀莫大于心死（低怒）”还是“因爱生恨（高怒）”，通常取区间下限。

### 3. Support (情感支持/安慰)
* **最坏情况定义**：由于说教、冷漠或盲目乐观（Toxic Positivity），导致对方感到“被二次伤害”或“更加孤独”。
* **评分逻辑**：Trust 下降（认为你不懂他），Anger 可能升高（恼羞成怒：30-60）。
* **关注点**：对方是否会关闭心门，甚至因为你的不当安慰而做出极端行为？

### 4. Charm (提升好感/魅力)
* **最坏情况定义**：由于用力过猛（油腻）、意图太明显（功利）或冒犯对方边界，导致对方产生强烈的生理性反感或警惕。
* **评分逻辑**：
    * **Trust**: 0-20（把你当成骗子、骚扰者或神经病）。
    * **Anger**: 20-70（视冒犯程度而定，轻则白眼嫌弃，重则报警/泼酒）。
    * *注意*：在Charm场景中，对方毫无反应（冷漠）不是最坏，最坏是被对方厌恶（负分）。

---
**输出要求（Strict JSON）**：
{
  "worst_anger_min": int,   // 预测的最坏愤怒下限 (0-100)
  "worst_trust_min": int,   // 预测的最坏信任下限 (0-100)
  "reason": "简述为什么会跌到这个分数（例如：鉴于这是Repair场景，若推卸责任，对方可能会从失望转为彻底死心，导致信任归零...）"
}

""".strip()


def worst_case_score(
    scene_row: Dict[str, Any],
    scene_type: str,
    model: str,
    temperature: float,
    max_retry: int,
) -> Dict[str, Any]:
    payload = {
        "title": scene_row.get("title", ""),
        "scene_tag": scene_type,
        "aggressor_opening_line": (scene_row.get("aggressor_profile") or {}).get("opening_line", ""),
        "aggressor_system_prompt": (scene_row.get("aggressor_profile") or {}).get("system_prompt", ""),
        "defender_system_prompt": (scene_row.get("defender_profile") or {}).get("system_prompt", ""),
        "initial_calibration": scene_row.get("initial_calibration", {}),
        "rub_goals": scene_row.get("rub_goals", {}),
    }
    user_prompt = "请基于以下信息评估最坏情况下的下界，输出 JSON：\n\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    last = ""
    for _ in range(max_retry):
        raw = call_llm(
            [
                {"role": "system", "content": WORST_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_retry=3,
        )
        last = raw
        obj = _extract_json_object(raw)
        if "worst_anger_min" in obj and "worst_trust_min" in obj:
            return {
                "worst_anger_min": _clamp_int(obj.get("worst_anger_min")),
                "worst_trust_min": _clamp_int(obj.get("worst_trust_min")),
                "reason": obj.get("reason", ""),
            }
        user_prompt += "\n\n上次解析失败：请只输出 JSON。"
    return {}


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_type", default="charm", choices=["support", "defense", "repair", "charm"], help="要生成的场景类型")
    ap.add_argument("--input", default="data/dual_profiles_v5.jsonl" ,help="输入 dual profiles jsonl（需包含 scene_tag 与双方 system_prompt）")
    ap.add_argument("--output", default="data/charm_v2.jsonl",help="输出 jsonl（生成的场景数据，jsonl 追加写）。如果不指定，将根据场景类型自动生成")
    ap.add_argument("--n", type=int, default=50, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--model_gen", default="qwen3-max", help="生成新场景的模型")
    ap.add_argument("--model_score", default="qwen3-max", help="生成 initial/rub 的模型")
    ap.add_argument("--model_worst", default="qwen3-max", help="生成 worst_case 的模型")
    ap.add_argument("--temperature_gen", type=float, default=1, help="生成阶段温度（更高更发散）")
    ap.add_argument("--temperature_score", type=float, default=0.2, help="打分阶段温度（更稳）")
    ap.add_argument("--temperature_worst", type=float, default=0.2, help="worst 阶段温度（更稳）")
    ap.add_argument("--max_retry", type=int, default=3, help="每个阶段的解析重试次数")
    ap.add_argument("--max_workers", type=int, default=20, help="并发数（>1 启用多线程）")
    args = ap.parse_args()

    # If no output file is given, derive one from the scene type
    if not args.output:
        args.output = f"data/dual_profiles_extra_{args.scene_type}_v2.jsonl"

    rows = _read_jsonl(args.input)
    scene_pool = _get_scene_pool(rows, args.scene_type)
    if len(scene_pool) < 2:
        raise RuntimeError(f"{args.scene_type} 场景不足 2 条，无法 few-shot（当前 {len(scene_pool)} 条）。")

    base_seed = int(args.seed)
    rng = random.Random(base_seed)
    ts = int(time.time())

    # Output is append-only to avoid clobbering existing data
    if not os.path.exists(args.output):
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    n_total = int(args.n)
    n_ok = 0
    n_fail = 0
    write_lock = Lock()

    def _build_row(i: int) -> Optional[Dict[str, Any]]:
        # Per-task rng so concurrent runs stay reproducible
        local_rng = random.Random(base_seed + i * 10007)

        base = generate_scene(
            scene_pool,
            args.scene_type,
            rng=local_rng,
            model=str(args.model_gen),
            temperature=float(args.temperature_gen),
            max_retry=int(args.max_retry),
        )
        if not base:
            return None

        new_id = f"scene_{args.scene_type}_gen_{ts}_{i:05d}"
        row: Dict[str, Any] = {
            "id": new_id,
            "title": base.get("title", f"{args.scene_type}生成_{i}"),
            "scene_tag": args.scene_type,
            "aggressor_profile": base.get("aggressor_profile", {}),
            "defender_profile": base.get("defender_profile", {}),
        }

        # Score: initial + success anchors (RST logic)
        score_obj = score_scene(
            row,
            args.scene_type,
            model=str(args.model_score),
            temperature=float(args.temperature_score),
            max_retry=int(args.max_retry),
        )
        init = (score_obj.get("initial_calibration") or {}) if isinstance(score_obj, dict) else {}
        rub = (score_obj.get("rub_goals") or {}) if isinstance(score_obj, dict) else {}

        # Top-level goal fields (match the dataset format)
        row["goal_one_liner"] = str(score_obj.get("goal_one_liner", "") or "") if isinstance(score_obj, dict) else ""
        row["success_criteria"] = score_obj.get("success_criteria", []) if isinstance(score_obj.get("success_criteria"), list) else []
        row["must_avoid"] = score_obj.get("must_avoid", []) if isinstance(score_obj.get("must_avoid"), list) else []

        # initial_calibration: keep only initial_anger / initial_trust
        row["initial_calibration"] = {
            "initial_anger": _clamp_int(init.get("initial_anger", 55)),
            "initial_trust": _clamp_int(init.get("initial_trust", 35)),
        }

        # rub_goals: range + target
        tar_a = _clamp_int(rub.get("target_anger", 20))
        tar_t = _clamp_int(rub.get("target_trust", 60))
        a_range = rub.get("target_anger_range")
        t_range = rub.get("target_trust_range")
        if not (isinstance(a_range, list) and len(a_range) == 2):
            a_range = [max(0, tar_a - 20), min(100, tar_a)]
        if not (isinstance(t_range, list) and len(t_range) == 2):
            t_range = [max(0, tar_t), min(100, tar_t + 20)]
        row["rub_goals"] = {
            "target_anger_range": [_clamp_int(a_range[0]), _clamp_int(a_range[1])],
            "target_trust_range": [_clamp_int(t_range[0]), _clamp_int(t_range[1])],
            "target_anger": tar_a,
            "target_trust": tar_t,
        }

        # Boolean flags present in the dataset format
        row["rewrite_ok"] = True
        row["strip_defender_strategy_ok"] = True

        # worst_case
        row["worst_case"] = worst_case_score(
            row,
            args.scene_type,
            model=str(args.model_worst),
            temperature=float(args.temperature_worst),
            max_retry=int(args.max_retry),
        )

        # Rule-based post-processing: round to 5, clip, enforce anchor ordering.
        _finalize_anchors(row)
        return row

    desc_text = f"Gen{args.scene_type.capitalize()}"

    if int(args.max_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futures = [ex.submit(_build_row, i) for i in range(n_total)]
            for fut in _maybe_tqdm(as_completed(futures), total=len(futures), desc=desc_text, unit="row"):
                row = fut.result()
                if not row:
                    n_fail += 1
                    continue
                _append_jsonl(args.output, row, write_lock)
                n_ok += 1
    else:
        for i in _maybe_tqdm(range(n_total), total=n_total, desc=desc_text, unit="row"):
            row = _build_row(i)
            if not row:
                n_fail += 1
                continue
            _append_jsonl(args.output, row, write_lock)
            n_ok += 1

    print(f"✨ 完成：成功写出 {n_ok} 条，失败/跳过 {n_fail} 条 -> {args.output}")


if __name__ == "__main__":
    main()


