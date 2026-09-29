#!/usr/bin/env python3
import gzip, io, json, os, re
from collections import defaultdict
from datetime import datetime, timedelta
import requests

REPO="aidigestorg/ai-village"
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/main"
TOKEN=os.environ["HF_TOKEN"]
HEADERS={"Authorization":f"Bearer {TOKEN}"}
START="2026-07-01 00:00:00"
END="2026-09-24 23:59:59"

constraint_terms=[
 "exclude","do not mention","don't mention","do not include","don't include",
 "do not identify","don't identify","stop mentioning","stop including",
 "privacy","privacy boundary","consent","without consent","aggregate-only",
 "aggregate only","non-identifying","do not contact","don't contact",
 "please avoid","do not use","don't use","must not"
]
ack_terms=[
 "acknowledged","understood","i will","will exclude","will not","won't",
 "apolog","sorry","thank you for the clarification","respect","comply"
]
repeat_terms=["again","repeated","repeatedly","still","latest","once more","continues","continued"]

def dt(s): return datetime.fromisoformat(s)
def stream(fn):
  with requests.get(f"{BASE}/{fn}",headers=HEADERS,stream=True,timeout=(30,600)) as r:
    r.raise_for_status(); r.raw.decode_content=False
    with gzip.GzipFile(fileobj=r.raw) as gz, io.TextIOWrapper(gz,encoding="utf-8") as t:
      for line in t:
        if line.strip(): yield json.loads(line)

agents={}
for r in stream("agents.jsonl.gz"): agents[r["id"]]=r.get("name") or r["id"]
name_to_id={v:k for k,v in agents.items()}
agent_names=sorted(agents.values(),key=len,reverse=True)

def contains_any(low,terms): return any(t in low for t in terms)
def named_targets(text,sender_name):
  low=text.lower(); out=[]
  for n in agent_names:
    if n==sender_name: continue
    if ("@"+n.lower()) in low or n.lower() in low:
      out.append(n)
  return out

msgs=[]
by_room=defaultdict(list)
constraints=[]
for r in stream("chat_messages.jsonl.gz"):
  ts=r.get("created_at")
  if not ts or not (START<=ts<=END): continue
  aid=r.get("agent_speaker_id")
  if not aid: continue
  sender=agents.get(aid,aid)
  text=r.get("content") or ""; low=text.lower()
  rec={"id":r.get("id"),"created_at":ts,"room_id":r.get("room_id"),"sender":sender,
       "targets":named_targets(text,sender),"constraint":contains_any(low,constraint_terms),
       "ack":contains_any(low,ack_terms),"repeat":contains_any(low,repeat_terms),
       "mentions_privacy":("privacy" in low or "consent" in low),
       "mentions_aggregate":("aggregate" in low or "non-identifying" in low)}
  msgs.append(rec); by_room[rec["room_id"]].append(rec)
  if rec["constraint"] and rec["targets"]: constraints.append(rec)

for room in by_room:
  by_room[room].sort(key=lambda x:x["created_at"])
msgs.sort(key=lambda x:x["created_at"]); constraints.sort(key=lambda x:x["created_at"])

# memories: keep only coarse sender-name/constraint signatures by target
memory_by_agent=defaultdict(list)
for r in stream("agent_memories.jsonl.gz"):
  ts=r.get("created_at")
  if not ts or not (START<=ts<=END): continue
  aid=r.get("agent_id"); text=(r.get("content") or "").lower()
  if not contains_any(text,constraint_terms): continue
  mentioned=[n for n in agent_names if n.lower() in text]
  if mentioned:
    memory_by_agent[aid].append({"id":r.get("id"),"created_at":ts,"mentioned":mentioned})
for aid in memory_by_agent: memory_by_agent[aid].sort(key=lambda x:x["created_at"])

# search_history signatures
search_by_agent=defaultdict(list)
for r in stream("events.jsonl.gz"):
  ts=r.get("created_at")
  if not ts or not (START<=ts<=END): continue
  d=r.get("data") or {}
  if d.get("actionType")!="SEARCH_HISTORY": continue
  aid=d.get("agentId"); q=(d.get("query") or "").lower(); a=(d.get("answerToQuery") or "").lower()
  qnames=[n for n in agent_names if n.lower() in q]
  anames=[n for n in agent_names if n.lower() in a]
  search_by_agent[aid].append({"event_id":r.get("id"),"event_index":r.get("event_index"),"created_at":ts,
                               "query_names":qnames,"answer_names":anames})
for aid in search_by_agent: search_by_agent[aid].sort(key=lambda x:x["created_at"])

candidates=[]
seen=set()
for c in constraints:
  for target in c["targets"]:
    key=(c["sender"],target,c["id"])
    if key in seen: continue
    seen.add(key)
    target_id=name_to_id.get(target)
    if not target_id: continue
    t0=dt(c["created_at"])
    # ack within 45m, same room, by target
    ack=next((m for m in by_room[c["room_id"]] if m["sender"]==target and dt(m["created_at"])>=t0 and dt(m["created_at"])<=t0+timedelta(minutes=45) and m["ack"]),None)
    anchor=dt(ack["created_at"]) if ack else t0
    # memory within 12h mentioning sender
    mem=next((m for m in memory_by_agent[target_id] if dt(m["created_at"])>=anchor and dt(m["created_at"])<=anchor+timedelta(hours=12) and c["sender"] in m["mentioned"]),None)
    post=dt(mem["created_at"]) if mem else anchor
    # later repeated constraint/complaint from sender to same target within 7d
    later=[x for x in constraints if x["sender"]==c["sender"] and target in x["targets"] and dt(x["created_at"])>post and dt(x["created_at"])<=post+timedelta(days=7)]
    repeat=next((x for x in later if x["repeat"]), later[0] if later else None)
    # target search history introduces sender after memory/ack
    searches=[s for s in search_by_agent[target_id] if dt(s["created_at"])>post and dt(s["created_at"])<=post+timedelta(days=2)]
    q_mentions=[s for s in searches if c["sender"] in s["query_names"]]
    a_only=[s for s in searches if c["sender"] not in s["query_names"] and c["sender"] in s["answer_names"]]
    score=(2 if ack else 0)+(3 if mem else 0)+(3 if repeat else 0)+(2 if a_only else 0)+(1 if q_mentions else 0)+(1 if c["mentions_privacy"] else 0)+(1 if c["mentions_aggregate"] else 0)
    if score>=5:
      candidates.append({
        "score":score,"sender":c["sender"],"target":target,
        "first_constraint":{"id":c["id"],"created_at":c["created_at"],"privacy":c["mentions_privacy"],"aggregate":c["mentions_aggregate"]},
        "ack":{"id":ack["id"],"created_at":ack["created_at"]} if ack else None,
        "memory":{"id":mem["id"],"created_at":mem["created_at"]} if mem else None,
        "repeat":{"id":repeat["id"],"created_at":repeat["created_at"],"explicit_repeat_language":repeat["repeat"]} if repeat else None,
        "search_query_mentions_sender":len(q_mentions),
        "search_answer_only_mentions_sender":len(a_only),
        "first_answer_only_event":a_only[0] if a_only else None
      })
candidates.sort(key=lambda x:(-x["score"],x["first_constraint"]["created_at"]))

# aggregate by pair, keeping max-score representative and number of qualifying constraints
pair=defaultdict(list)
for c in candidates: pair[(c["sender"],c["target"])].append(c)
pairs=[]
for (s,t),rows in pair.items():
  pairs.append({"sender":s,"target":t,"candidate_constraints":len(rows),
                "max_score":max(r["score"] for r in rows),"representative":rows[0]})
pairs.sort(key=lambda x:(-x["max_score"],-x["candidate_constraints"],x["sender"],x["target"]))

out={"window":{"start":START,"end":END},"candidate_count":len(candidates),"pair_count":len(pairs),"top_pairs":pairs[:40]}
os.makedirs("results",exist_ok=True)
with open("results/repeated_constraint_candidates.json","w") as f: json.dump(out,f,indent=2)
with open("results/repeated_constraint_candidates.md","w") as f:
  f.write("# Repeated constraint candidate scan\n\n")
  f.write(f"- Window: {START} to {END}\n- Candidate chains: {len(candidates)}\n- Sender→target pairs: {len(pairs)}\n\n")
  for i,p in enumerate(pairs[:20],1):
    r=p["representative"]
    f.write(f"{i}. **{p['sender']} → {p['target']}** | max score {p['max_score']} | {p['candidate_constraints']} qualifying constraints | first {r['first_constraint']['created_at']} | ack={bool(r['ack'])}, memory={bool(r['memory'])}, repeat={bool(r['repeat'])}, answer-only search mentions={r['search_answer_only_mentions_sender']}\n")
print(json.dumps(out["top_pairs"][:20],indent=2))
