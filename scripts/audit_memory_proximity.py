#!/usr/bin/env python3
import gzip, io, json, os, re
import requests

REPO="aidigestorg/ai-village"
TOKEN=os.environ["HF_TOKEN"]
URL=f"https://huggingface.co/datasets/{REPO}/resolve/main/agent_memories.jsonl.gz"
HEADERS={"Authorization":f"Bearer {TOKEN}"}
TARGETS={
 "b5480e8a-7a6b-432d-aa71-54a809afba8d":"deepseek_2026_09_18_2110",
 "07df4bd3-f3fd-46eb-ab48-0cfab11102b0":"deepseek_2026_09_18_2332",
 "016927c0-c69d-494e-b841-1904135789e9":"deepseek_2026_09_18_2349"
}
terms=[
 "terra","exclude","aggregate","search history","search_history","privacy","consent",
 "non-identifying","nonidentifying","boundary","opt-out","opt out","anonym",
 "omit","remove","individual","do not"
]

def positions(text, needle):
    out=[]; start=0; low=text.lower(); needle=needle.lower()
    while True:
        i=low.find(needle,start)
        if i<0: return out
        out.append(i); start=i+1

def min_distance(a,b):
    if not a or not b: return None
    return min(abs(x-y) for x in a for y in b)

def sentence_windows(text):
    spans=[]
    for m in re.finditer(r'[^\n.!?]{1,800}[.!?\n]+|[^\n.!?]{1,800}$',text):
        spans.append((m.start(),m.end()))
    return spans

found={}
with requests.get(URL,headers=HEADERS,stream=True,timeout=(30,600)) as r:
    r.raise_for_status(); r.raw.decode_content=False
    with gzip.GzipFile(fileobj=r.raw) as gz, io.TextIOWrapper(gz,encoding="utf-8") as t:
        for line in t:
            if not line.strip(): continue
            row=json.loads(line)
            rid=row.get("id")
            if rid not in TARGETS: continue
            text=row.get("content") or ""
            pos={term:positions(text,term) for term in terms}
            terra=pos["terra"]
            spans=sentence_windows(text)
            same_sentence={}
            local_300={}
            for term in terms[1:]:
                same_sentence[term]=any(
                    any(s<=p<e for p in terra) and any(s<=p<e for p in pos[term])
                    for s,e in spans
                )
                local_300[term]=bool(terra and pos[term] and min_distance(terra,pos[term])<=300)
            exact_patterns={
                "terra_aggregate":bool(re.search(r'terra[^\n]{0,180}aggregat\w*|aggregat\w*[^\n]{0,180}terra',text,re.I)),
                "terra_privacy":bool(re.search(r'terra[^\n]{0,180}privac\w*|privac\w*[^\n]{0,180}terra',text,re.I)),
                "terra_boundary":bool(re.search(r'terra[^\n]{0,180}boundar\w*|boundar\w*[^\n]{0,180}terra',text,re.I)),
                "terra_optout":bool(re.search(r'terra[^\n]{0,180}opt[- ]?out|opt[- ]?out[^\n]{0,180}terra',text,re.I)),
                "terra_anonym":bool(re.search(r'terra[^\n]{0,180}anonym\w*|anonym\w*[^\n]{0,180}terra',text,re.I)),
                "terra_omit_remove":bool(re.search(r'terra[^\n]{0,180}(?:omit|remov)\w*|(?:omit|remov)\w*[^\n]{0,180}terra',text,re.I)),
                "terra_nonidentifying":bool(re.search(r'terra[^\n]{0,180}non[- ]?identif\w*|non[- ]?identif\w*[^\n]{0,180}terra',text,re.I)),
                "terra_individual":bool(re.search(r'terra[^\n]{0,180}individual\w*|individual\w*[^\n]{0,180}terra',text,re.I)),
                "terra_do_not":bool(re.search(r'terra[^\n]{0,180}do not|do not[^\n]{0,180}terra',text,re.I))
            }
            found[TARGETS[rid]]={
                "id":rid,"created_at":row.get("created_at"),"chars":len(text),
                "term_counts":{k:len(v) for k,v in pos.items()},
                "min_char_distance_from_terra":{k:min_distance(terra,pos[k]) for k in terms[1:]},
                "within_300_chars_of_terra":local_300,
                "same_sentence_as_terra":same_sentence,
                "exact_local_patterns":exact_patterns
            }
            if len(found)==len(TARGETS): break
os.makedirs("results",exist_ok=True)
with open("results/memory_proximity_audit.json","w") as f:
    json.dump(found,f,indent=2)
print(json.dumps(found,indent=2))
