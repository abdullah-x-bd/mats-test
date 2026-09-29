#!/usr/bin/env python3
import gzip, io, json, os, re
from collections import Counter, defaultdict
import requests

REPO="aidigestorg/ai-village"
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/main"
TOKEN=os.environ["HF_TOKEN"]
HEADERS={"Authorization":f"Bearer {TOKEN}"}
START="2026-09-14 00:00:00"
END="2026-09-24 00:00:00"

def stream(filename):
    with requests.get(f"{BASE}/{filename}",headers=HEADERS,stream=True,timeout=(30,600)) as r:
        r.raise_for_status(); r.raw.decode_content=False
        with gzip.GzipFile(fileobj=r.raw) as gz, io.TextIOWrapper(gz,encoding="utf-8") as t:
            for line in t:
                if line.strip(): yield json.loads(line)

agents={}
for row in stream("agents.jsonl.gz"):
    agents[row["id"]]=row.get("name") or row["id"]
names=sorted(set(agents.values()), key=len, reverse=True)
_name_pattern = re.compile(
    r"(?<![A-Za-z0-9_.-])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_.-])",
    re.I,
)
_name_map = {n.lower(): n for n in names}

def mentioned_names(text):
    out=set()
    for m in _name_pattern.finditer(text or ""):
        n=_name_map.get(m.group(1).lower())
        if n:
            out.add(n)
    return out

overall={"events":0,"answers_introduce_any_name":0,"queries_name_any_agent":0}
by_requester=defaultdict(lambda:Counter())
introduced_name_counts=Counter()
deepseek_introduced=Counter()
deepseek_total=0
deepseek_answer_introduces=0
deepseek_rows=[]

for row in stream("events.jsonl.gz"):
    ts=row.get("created_at")
    if not ts or not (START <= ts < END): continue
    d=row.get("data") or {}
    if d.get("actionType")!="SEARCH_HISTORY": continue
    requester=agents.get(d.get("agentId"),d.get("agentId"))
    q=d.get("query") or ""; a=d.get("answerToQuery") or ""
    qn=mentioned_names(q); an=mentioned_names(a); intro=an-qn
    overall["events"]+=1
    if qn: overall["queries_name_any_agent"]+=1
    if intro: overall["answers_introduce_any_name"]+=1
    by_requester[requester]["events"]+=1
    if intro: by_requester[requester]["answers_introduce_any_name"]+=1
    for n in intro:
        introduced_name_counts[n]+=1
        by_requester[requester][f"introduced::{n}"]+=1
    if requester=="DeepSeek-V3.2":
        deepseek_total+=1
        if intro: deepseek_answer_introduces+=1
        for n in intro: deepseek_introduced[n]+=1
        if "GPT-5.6 Terra" in intro:
            deepseek_rows.append({
                "created_at":ts,
                "event_index":row.get("event_index"),
                "event_id":row.get("id"),
                "query_named_agents":sorted(qn),
                "answer_named_agents":sorted(an),
                "introduced_agents":sorted(intro)
            })

requester_rates=[]
for requester,c in by_requester.items():
    if c["events"]:
        requester_rates.append({
            "requester":requester,
            "events":c["events"],
            "answer_introduces_any_name":c["answers_introduce_any_name"],
            "rate":round(c["answers_introduce_any_name"]/c["events"],4)
        })
requester_rates.sort(key=lambda x:(-x["events"],x["requester"]))

out={
  "window_utc":{"start":START,"end":END},
  "matching_method":"Boundary-aware longest-name matching; prevents prefix collisions such as GPT-5 inside GPT-5.6 Terra",
  "overall":{
    **overall,
    "answer_introduces_any_name_rate": round(overall["answers_introduce_any_name"]/overall["events"],4) if overall["events"] else None
  },
  "deepseek":{
    "events":deepseek_total,
    "answers_introduce_any_name":deepseek_answer_introduces,
    "answer_introduces_any_name_rate":round(deepseek_answer_introduces/deepseek_total,4) if deepseek_total else None,
    "introduced_agent_counts":deepseek_introduced.most_common(),
    "terra_introduction_events":deepseek_rows
  },
  "all_requesters":requester_rates,
  "introduced_agent_counts_all_searches":introduced_name_counts.most_common()
}
os.makedirs("results",exist_ok=True)
with open("results/search_history_name_leakage.json","w") as f: json.dump(out,f,indent=2)

with open("results/search_history_name_leakage.md","w") as f:
    f.write("# SEARCH_HISTORY name-introduction baseline\n\n")
    f.write(f"- Window: {START} to {END} UTC\n")
    f.write(f"- All SEARCH_HISTORY events: {overall['events']}\n")
    f.write(f"- Answers introducing at least one agent name absent from the query: {overall['answers_introduce_any_name']} ({out['overall']['answer_introduces_any_name_rate']:.1%})\n")
    f.write(f"- DeepSeek SEARCH_HISTORY events: {deepseek_total}\n")
    f.write(f"- DeepSeek answers introducing at least one unqueried agent name: {deepseek_answer_introduces} ({out['deepseek']['answer_introduces_any_name_rate']:.1%})\n")
    f.write(f"- DeepSeek answers introducing GPT-5.6 Terra specifically: {deepseek_introduced.get('GPT-5.6 Terra',0)}\n")
print(json.dumps(out["overall"],indent=2))
print(json.dumps(out["deepseek"],indent=2))
