# MATS Winter 2027 Stream 2 Test

This repository is the working analysis repository for the **MATS Winter 2027 Stream 2 test**.

## Current task

Analyze the AI Village dataset and identify one concrete, reproducible empirical finding using only:

- `chat_messages`
- `agent_memories`
- `events`

The analysis should explain:

1. the concrete finding and how to reproduce it,
2. why the finding is interesting,
3. how to distinguish a general underlying phenomenon from an artifact of the AI Village setup.

## Data handling

The source AI Village dataset remains on Hugging Face at `aidigestorg/ai-village`.

The large gated `agent_memories.jsonl.gz` file is **not committed to this repository**. GitHub Actions accesses it using the repository secret `HF_TOKEN`. Raw gated records are not intentionally published to the repository.

## Analysis strategy

The first investigation focuses on whether explicit social constraints:

1. are stated and acknowledged in chat,
2. survive into persistent memory,
3. nevertheless fail to govern later behavior.

A leading candidate is the repeated privacy/exclusion interaction involving GPT-5.6 Terra and DeepSeek-V3.2. This candidate is treated as a hypothesis to test, not as a predetermined conclusion. The pipeline is designed so competing findings can be investigated as well.

## Important dataset caveat

AI Village scaffolding changed over time. Any behavioral interpretation should be checked against the project changelog before being treated as model-level evidence.
