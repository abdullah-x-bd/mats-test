#!/usr/bin/env python3
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timedelta

import requests

REPO = "aidigestorg/ai-village"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TOKEN = os.environ["HF_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

START = "2026-09-14 00:00:00"
END = "2026-09-24 00:00:00"

PRIVACY_TERMS = (
    "privacy", "private", "exclude", "exclusion", "do not include", "don't include",
    "do not identify", "don't identify", "non-identifying", "aggregate", "consent",
    "search parameters", "search history", "pause timing"
)
CONSTRAINT_TERMS = (
    "do not", "don't", "must not", "never", "exclude", "avoid", "constraint",
    "privacy", "consent", "permission", "boundary", "prohibited", "forbidden"
)

def dt(ts):
    return datetime.fromisoformat(ts)

def stream_jsonl_gz(filename):
    url = f"{BASE}/{filename}"
    with requests.get(url, headers=HEADERS, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with gzip.GzipFile(fileobj=r.raw) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as text:
                for line in text:
                    if line.strip():
                        yield json.loads(line)

def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def within(ts):
    return bool(ts and START <= ts < END)

def flags(text):
    t = (text or "").lower()
    return {
        "terra": "terra" in t,
        "deepseek": "deepseek" in t,
        "privacy": any(k in t for k in ("privacy", "private")),
        "exclude": any(k in t for k in ("exclude", "exclusion", "do not include", "don't include")),
        "aggregate": any(k in t for k in ("aggregate", "non-identifying", "nonidentifying")),
        "search_history": any(k in t for k in ("search history", "search_history", "search parameters")),
        "consent": "consent" in t,
        "pause_timing": "pause timing" in t,
        "again": "again" in t,
    }

agents = {}
for row in stream_jsonl_gz("agents.jsonl.gz"):
    agents[row["id"]] = row.get("name") or row["id"]

name_to_id = {v: k for k, v in agents.items()}
deepseek_id = name_to_id.get("DeepSeek-V3.2")
terra_id = name_to_id.get("GPT-5.6 Terra")
if not deepseek_id or not terra_id:
    raise RuntimeError(f"Could not resolve target IDs. DeepSeek={deepseek_id}, Terra={terra_id}")

# CHAT
terra_constraints = []
terra_repeat_complaints = []
deepseek_acks = []
deepseek_terra_mentions = []
chat_hits = []

for row in stream_jsonl_gz("chat_messages.jsonl.gz"):
    ts = row.get("created_at")
    if not within(ts):
        continue
    aid = row.get("agent_speaker_id")
    if aid not in (deepseek_id, terra_id):
        continue
    content = row.get("content") or ""
    low = content.lower()
    f = flags(content)
    if any(term in low for term in PRIVACY_TERMS) or "terra" in low:
        rec = {
            "created_at": ts,
            "speaker": agents.get(aid, aid),
            "message_id": row.get("id"),
            "room_id": row.get("room_id"),
            "content_sha16": sha(content),
            "chars": len(content),
            "flags": f,
        }
        chat_hits.append(rec)

        if aid == terra_id and (
            f["exclude"] or f["privacy"] or
            "do not identify" in low or "don't identify" in low or
            "do not include" in low or "don't include" in low
        ):
            terra_constraints.append(rec)
            if f["again"] or "latest" in low or "repeat" in low:
                terra_repeat_complaints.append(rec)

        if aid == deepseek_id and f["terra"]:
            deepseek_terra_mentions.append(rec)

        if aid == deepseek_id and (f["exclude"] or f["privacy"] or f["aggregate"]):
            deepseek_acks.append(rec)

terra_constraints.sort(key=lambda x: x["created_at"])
terra_repeat_complaints.sort(key=lambda x: x["created_at"])
deepseek_acks.sort(key=lambda x: x["created_at"])
deepseek_terra_mentions.sort(key=lambda x: x["created_at"])
chat_hits.sort(key=lambda x: x["created_at"])

# EVENTS
search_events = []
all_recent_action_counts = Counter()
for row in stream_jsonl_gz("events.jsonl.gz"):
    ts = row.get("created_at")
    if not within(ts):
        continue
    data = row.get("data") or {}
    at = data.get("actionType")
    all_recent_action_counts[at] += 1
    if at != "SEARCH_HISTORY" or data.get("agentId") != deepseek_id:
        continue
    query = data.get("query") or ""
    answer = data.get("answerToQuery") or ""
    qf, af = flags(query), flags(answer)
    search_events.append({
        "created_at": ts,
        "event_index": row.get("event_index"),
        "event_id": row.get("id"),
        "query_sha16": sha(query),
        "answer_sha16": sha(answer),
        "query_chars": len(query),
        "answer_chars": len(answer),
        "query_flags": qf,
        "answer_flags": af,
        "query_mentions_terra": qf["terra"],
        "answer_mentions_terra": af["terra"],
        "mentions_terra_anywhere": qf["terra"] or af["terra"],
    })
search_events.sort(key=lambda x: x["created_at"])

# MEMORIES
memory_hits = []
broad_constraint_counts = Counter()
for row in stream_jsonl_gz("agent_memories.jsonl.gz"):
    content = row.get("content") or ""
    low = content.lower()
    aid = row.get("agent_id")
    ts = row.get("created_at")

    if any(term in low for term in CONSTRAINT_TERMS):
        broad_constraint_counts[agents.get(aid, aid)] += 1

    if aid != deepseek_id or not within(ts):
        continue

    f = flags(content)
    if f["terra"] or f["privacy"] or f["exclude"] or f["aggregate"] or f["search_history"] or f["consent"]:
        memory_hits.append({
            "created_at": ts,
            "memory_id": row.get("id"),
            "content_sha16": sha(content),
            "chars": len(content),
            "flags": f,
            "constraint_signature": bool(
                f["terra"] and (f["privacy"] or f["exclude"] or f["aggregate"] or f["consent"])
            ),
        })
memory_hits.sort(key=lambda x: x["created_at"])
constraint_memories = [m for m in memory_hits if m["constraint_signature"]]

# EPISODE RECONSTRUCTION
episodes = []
for c in terra_constraints:
    cdt = dt(c["created_at"])

    # nearest plausible acknowledgement in same room, within 30 minutes
    ack = next((
        a for a in deepseek_acks
        if a["room_id"] == c["room_id"]
        and dt(a["created_at"]) >= cdt
        and dt(a["created_at"]) <= cdt + timedelta(minutes=30)
    ), None)

    anchor = dt(ack["created_at"]) if ack else cdt

    # first persistent memory carrying Terra+constraint language within 8h
    mem = next((
        m for m in constraint_memories
        if dt(m["created_at"]) >= anchor
        and dt(m["created_at"]) <= anchor + timedelta(hours=8)
    ), None)

    post_anchor = dt(mem["created_at"]) if mem else anchor

    # next Terra complaint suggesting recurrence within 8h
    complaint = next((
        t for t in terra_repeat_complaints
        if dt(t["created_at"]) > post_anchor
        and dt(t["created_at"]) <= post_anchor + timedelta(hours=8)
    ), None)

    later_queries = [
        s for s in search_events
        if dt(s["created_at"]) > post_anchor
        and dt(s["created_at"]) <= post_anchor + timedelta(hours=8)
        and s["query_mentions_terra"]
    ]
    later_answer_only = [
        s for s in search_events
        if dt(s["created_at"]) > post_anchor
        and dt(s["created_at"]) <= post_anchor + timedelta(hours=8)
        and (not s["query_mentions_terra"]) and s["answer_mentions_terra"]
    ]

    tool_to_chat_propagations = []
    for se in later_answer_only:
        sedt = dt(se["created_at"])
        next_ds = next((
            m for m in deepseek_terra_mentions
            if dt(m["created_at"]) > sedt
            and dt(m["created_at"]) <= sedt + timedelta(minutes=30)
        ), None)
        next_terra = next((
            t for t in terra_repeat_complaints
            if dt(t["created_at"]) > sedt
            and dt(t["created_at"]) <= sedt + timedelta(minutes=60)
        ), None)
        if next_ds:
            tool_to_chat_propagations.append({
                "search_event": se,
                "next_deepseek_chat_mentioning_terra": next_ds,
                "next_terra_repeat_complaint": next_terra,
                "propagation_with_complaint": bool(next_terra and dt(next_terra["created_at"]) > dt(next_ds["created_at"])),
            })

    episodes.append({
        "constraint": c,
        "ack": ack,
        "constraint_memory": mem,
        "later_repeat_complaint": complaint,
        "later_agent_queries_explicitly_mentioning_terra": later_queries,
        "later_tool_answers_mentioning_terra_without_query_mention": later_answer_only,
        "tool_answer_to_deepseek_chat_propagations": tool_to_chat_propagations,
        "strong_remembered_rule_failure_candidate": bool(mem and complaint),
        "agent_initiated_search_breach_candidate": bool(mem and later_queries),
        "tool_answer_leak_candidate": bool(mem and later_answer_only),
        "tool_answer_propagation_candidate": bool(mem and any(x["propagation_with_complaint"] for x in tool_to_chat_propagations)),
    })

# Deduplicate identical constraint/ack/memory patterns only by message id
strong = [e for e in episodes if e["strong_remembered_rule_failure_candidate"]]
query_breach = [e for e in episodes if e["agent_initiated_search_breach_candidate"]]
tool_leak = [e for e in episodes if e["tool_answer_leak_candidate"]]
tool_propagation = [e for e in episodes if e["tool_answer_propagation_candidate"]]

diagnostic = {
    "dataset": REPO,
    "window_utc": {"start": START, "end": END},
    "target_agents": {
        "DeepSeek-V3.2": deepseek_id,
        "GPT-5.6 Terra": terra_id,
    },
    "counts": {
        "terra_constraint_candidates": len(terra_constraints),
        "terra_repeat_complaints": len(terra_repeat_complaints),
        "deepseek_ack_candidates": len(deepseek_acks),
        "deepseek_chats_mentioning_terra": len(deepseek_terra_mentions),
        "deepseek_privacy_related_memory_hits": len(memory_hits),
        "deepseek_constraint_signature_memories": len(constraint_memories),
        "deepseek_search_history_events": len(search_events),
        "deepseek_queries_explicitly_mentioning_terra": sum(s["query_mentions_terra"] for s in search_events),
        "deepseek_answers_mentioning_terra_when_query_did_not": sum((not s["query_mentions_terra"]) and s["answer_mentions_terra"] for s in search_events),
    },
    "episodes": episodes,
    "candidate_summary": {
        "remembered_rule_then_repeat_complaint_episode_count": len(strong),
        "remembered_rule_then_agent_query_mentions_terra_episode_count": len(query_breach),
        "remembered_rule_then_tool_answer_only_mentions_terra_episode_count": len(tool_leak),
        "tool_answer_then_deepseek_chat_then_terra_complaint_episode_count": len(tool_propagation),
        "strongest_remembered_rule_failure": strong[0] if strong else None,
        "strongest_agent_query_breach": query_breach[0] if query_breach else None,
        "strongest_tool_answer_leak": tool_leak[0] if tool_leak else None,
        "strongest_tool_answer_propagation": tool_propagation[0] if tool_propagation else None,
    },
    "broad_scan": {
        "agents_with_most_constraint_language_in_memories": broad_constraint_counts.most_common(15),
        "recent_event_action_counts": dict(all_recent_action_counts),
    },
    "notes": [
        "All candidate lists are sorted by timestamp before temporal inference.",
        "No raw gated memory content is written to this repository.",
        "SEARCH_HISTORY query mentions are separated from answer-only mentions to distinguish agent choice from tool output.",
        "A repeat complaint after a matching persistent-memory snapshot is evidence of recurrence after memory retention, but still requires manual semantic verification of the referenced messages.",
    ],
}

os.makedirs("results", exist_ok=True)
with open("results/diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(diagnostic, f, indent=2, ensure_ascii=False)

summary = diagnostic["candidate_summary"]
lines = [
    "# AI Village diagnostic pass v2",
    "",
    "Derived metadata only. Raw gated memory content is not published.",
    "",
    f"- Window: {START} to {END} UTC",
    f"- Terra constraint candidates: {len(terra_constraints)}",
    f"- Terra repeat-complaint candidates: {len(terra_repeat_complaints)}",
    f"- DeepSeek acknowledgement candidates: {len(deepseek_acks)}",
    f"- DeepSeek chat messages mentioning Terra: {len(deepseek_terra_mentions)}",
    f"- DeepSeek constraint-signature memories: {len(constraint_memories)}",
    f"- DeepSeek SEARCH_HISTORY events: {len(search_events)}",
    f"- Queries explicitly naming Terra: {diagnostic['counts']['deepseek_queries_explicitly_mentioning_terra']}",
    f"- Tool answers naming Terra when the query did not: {diagnostic['counts']['deepseek_answers_mentioning_terra_when_query_did_not']}",
    "",
    "## Candidate episode tests",
    "",
    f"- Remembered rule followed by a later Terra repeat complaint: **{summary['remembered_rule_then_repeat_complaint_episode_count']}** candidate episode(s)",
    f"- Remembered rule followed by DeepSeek explicitly naming Terra in a later SEARCH_HISTORY query: **{summary['remembered_rule_then_agent_query_mentions_terra_episode_count']}** candidate episode(s)",
    f"- Remembered rule followed only by a tool answer naming Terra: **{summary['remembered_rule_then_tool_answer_only_mentions_terra_episode_count']}** candidate episode(s)",
    f"- Tool answer names Terra, then DeepSeek chat names Terra, then Terra complains again: **{summary['tool_answer_then_deepseek_chat_then_terra_complaint_episode_count']}** candidate episode(s)",
    "",
    "These are candidate sequences, not final behavioral claims. Exact messages and scaffold timing still need semantic audit.",
]
with open("results/report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(json.dumps({
    "status": "ok",
    "terra_constraints": len(terra_constraints),
    "repeat_complaints": len(terra_repeat_complaints),
    "constraint_memories": len(constraint_memories),
    "search_events": len(search_events),
    "query_mentions_terra": diagnostic["counts"]["deepseek_queries_explicitly_mentioning_terra"],
    "answer_only_mentions_terra": diagnostic["counts"]["deepseek_answers_mentioning_terra_when_query_did_not"],
    "remembered_rule_repeat_complaint_candidates": len(strong),
    "remembered_rule_agent_query_breach_candidates": len(query_breach),
    "remembered_rule_tool_leak_candidates": len(tool_leak),
    "tool_answer_propagation_candidates": len(tool_propagation),
}, indent=2))
