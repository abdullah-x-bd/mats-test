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
"80ae803b-d92e-42c7-9509-f5ff3363e119",
"bed0da38-aaad-4c81-aff5-625584a6cda7",
"ac6f4d68-a434-4e27-9e1d-ec8c6340391a",
"1167df00-8755-409b-9a12-f40571c2d250",
"9387c6ea-41d7-4595-8e7f-634970d50a69",
"cff2d80a-fd55-43ad-8b98-09f1d457d646",
"4edbd26a-3bc2-452f-9b39-eec54ef9c9b2",
"e48722bc-5a45-4cf4-9a19-043f8dceaeab",
"65d8807b-71ea-4bf7-86c5-922a46486fb1",
"7decb5e3-c304-4940-ad9b-0ab45bf9f87a"
}
EVENT_IDS={
"06f92056-cc5e-4315-a607-cbc9750af4b4","3f25522d-cd5c-4ef6-a3de-705f0886aa96",
"1014c0db-3f5a-44a6-8fa2-c76d0fe2fa70",
"a2e9775b-89b2-4599-9435-0ee5b27aefa4",
"31ff6b50-0b19-488b-aba4-7967345e6a01"
}

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
