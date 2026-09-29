#!/usr/bin/env python3
import gzip, io, json, os
import requests

REPO="aidigestorg/ai-village"
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/main"
TOKEN=os.environ["HF_TOKEN"]
HEADERS={"Authorization":f"Bearer {TOKEN}"}
CHAT_IDS={
"40f1f655-29a1-45f1-8cee-2a696bca3da9",
"9a05f0ab-7fac-40cc-a2fd-594b4e612643",
"94220a3e-6552-427c-b996-fdfdf5e5e863",
"f7f563f9-180b-43b9-b65e-b60003c63a4c",
"80ae803b-d92e-42c7-9509-f5ff3363e119"
}
EVENT_IDS={"06f92056-cc5e-4315-a607-cbc9750af4b4","3f25522d-cd5c-4ef6-a3de-705f0886aa96"}

def stream(fn):
  with requests.get(f"{BASE}/{fn}",headers=HEADERS,stream=True,timeout=(30,600)) as r:
    r.raise_for_status(); r.raw.decode_content=False
    with gzip.GzipFile(fileobj=r.raw) as gz, io.TextIOWrapper(gz,encoding="utf-8") as t:
      for line in t:
        if line.strip(): yield json.loads(line)

chats=[]
for r in stream("chat_messages.jsonl.gz"):
  if r.get("id") in CHAT_IDS:
    chats.append({k:r.get(k) for k in ("id","created_at","room_id","agent_speaker_id","content")})
chats.sort(key=lambda x:x["created_at"])

events=[]
for r in stream("events.jsonl.gz"):
  if r.get("id") in EVENT_IDS:
    d=r.get("data") or {}
    q=d.get("query") or ""; a=d.get("answerToQuery") or ""
    events.append({
      "id":r.get("id"),"event_index":r.get("event_index"),"created_at":r.get("created_at"),
      "query":q,
      "query_mentions_terra":"terra" in q.lower(),
      "answer_mentions_terra":"terra" in a.lower(),
      "answer_chars":len(a)
    })
events.sort(key=lambda x:x["created_at"])

os.makedirs("results",exist_ok=True)
with open("results/public_chat_event_audit.json","w") as f: json.dump({"chats":chats,"events":events},f,indent=2)
print(json.dumps({"chats":chats,"events":events},indent=2))
