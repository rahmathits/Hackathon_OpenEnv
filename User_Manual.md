---
title: EDA OpenEnv Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags:
  - openenv
---

<div align="center">

# 🤖 EDA OpenEnv Agent

**A real-world Reinforcement Learning environment where AI agents learn to perform Exploratory Data Analysis**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Validated-brightgreen)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![Phase 1](https://img.shields.io/badge/Phase%201-PASSED-brightgreen)]()
[![Phase 2](https://img.shields.io/badge/Phase%202-PASSED-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[🚀 Try the Live Demo](https://rahmath1-eda-openenv.hf.space/web/) • [📖 API Docs](https://rahmath1-eda-openenv.hf.space/docs) • [💻 GitHub](https://github.com/rahmathits/Hackathon_OpenEnv)

</div>

---

## 👋 What is this?

Most RL environments are abstract — grids, games, mazes. **EDA OpenEnv** is different.

It models a **real data science workflow** that people actually do every day:

```
Clean your data → Explore it → Engineer features → Train a model
```

An AI agent learns to complete these steps **in the right order**, on a real CSV dataset, and gets rewarded based on the **actual quality** of its work — not just whether it pressed the right button.

> **Think of it as**: a gym environment, but instead of playing Atari, the agent is doing data science.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌍 **Real-world tasks** | Agent works on actual tabular CSV data, not synthetic grids |
| 🏆 **Dynamic grading** | Rewards based on actual data quality — deterministic and reproducible |
| 🔢 **Pipeline enforcement** | Agent must follow the correct order or get penalised |
| 🔁 **Reproducible** | Same task + same seed + same actions = same score every time |
| 🔌 **OpenEnv standard** | Works with any agent via `reset()` / `step()` / `state()` |
| 🌐 **Web UI** | Built-in browser interface to test the environment live |
| 🐳 **Docker ready** | One command to run anywhere |
| ✅ **Fully validated** | Phase 1 + Phase 2 passed — officially in the running |

---

## 🎮 Try it in 30 seconds

1. Open [https://rahmath1-eda-openenv.hf.space/web/](https://rahmath1-eda-openenv.hf.space/web/)
2. Click **Reset** to start a new episode
3. Type `clean_data` in the Action Type field
4. Click **Step** and watch the reward come back
5. Follow the pipeline all the way through

No installation needed!

---

## 🗺️ The Pipeline

The agent must complete actions in this exact order:

```
┌─────────────┐    ┌─────┐    ┌─────────────────────┐    ┌─────────────┐
│  clean_data │ →  │ eda │ →  │ feature_engineering  │ →  │ train_model │
└─────────────┘    └─────┘    └─────────────────────┘    └─────────────┘
                                                                  ↓
                                                        Task-specific action
                                                   (missing / correlation / insight)
```

**Skip a step? You get penalised. Follow the order? You get rewarded.**

---

## 🎯 Tasks

Each episode assigns the agent one of three tasks with increasing difficulty:

### 🟢 Easy — Detect Missing Values
> *"Find all columns in the dataset that have missing values"*

Run the pipeline then execute: `missing`

Graded on: proportion of columns with missing values identified vs. total columns

### 🟡 Medium — Find Correlation
> *"Find the strongest correlation between any two numeric columns"*

Run the pipeline then execute: `correlation`

Graded on fixed thresholds:
`≥ 0.9` → **0.95** · `≥ 0.7` → **0.75** · `≥ 0.5` → **0.55** · `≥ 0.3` → **0.35** · `< 0.3` → **0.15**

### 🔴 Hard — Generate Insight
> *"Generate a meaningful insight about the dataset referencing actual values"*

Run the pipeline then execute: `insight`

Graded on three deterministic sub-scores:
- **Text length** (up to +0.40) — min viable = 30 chars, full credit at 200+
- **Numeric references** (up to +0.30) — how many actual data values are mentioned
- **Column name mentions** (up to +0.30) — how many real column names appear

---

## 🏆 Reward System

All rewards are strictly between **0.02 and 0.98** — never exactly 0 or 1:

```
✅ Correct pipeline order (deterministic):
   clean_data           →  0.25
   eda                  →  0.50
   feature_engineering  →  0.75
   train_model          →  0.95

📊 Task-specific action:
   Scored by dynamic grader (0.02 → 0.98), deterministic

⚠️  Out-of-order action (deterministic penalty):
   Skip 1 step   →  -0.25
   Skip 2 steps  →  -0.50
   Skip 3 steps  →  -0.75

🔁 Repeated action  →  0.05 (small penalty)
🎯 Wrong action     →  0.15 (valid but irrelevant)
```

### Reproducibility Guarantee
- Same DataFrame + same actions = **same score, always**
- `reset(seed=42)` always picks the same task for that seed
- `reset(task_name="detect_missing")` always sets that exact task
- No randomness in graders — pure math on actual data quality

---

## 🔌 Connect Your Own Agent

### Option 1 — HTTP API (simplest)

```bash
# Start a new episode
curl -X POST https://rahmath1-eda-openenv.hf.space/reset \
  -H "Content-Type: application/json" -d "{}"

# Take a step
curl -X POST https://rahmath1-eda-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "clean_data"}}'

# Start with specific task (deterministic)
curl -X POST https://rahmath1-eda-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "detect_missing"}'

# Start with seed (reproducible)
curl -X POST https://rahmath1-eda-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'
```

### Option 2 — Python client

```python
from client import EdaOpenenvEnv
from models import EdaOpenenvAction

with EdaOpenenvEnv(base_url="https://rahmath1-eda-openenv.hf.space") as env:
    result = env.reset()
    print(f"Task: {result.observation.task}")

    for action in ["clean_data", "eda", "feature_engineering", "train_model", "missing"]:
        if result.done:
            break
        result = env.step(EdaOpenenvAction(action_type=action))
        print(f"{action:25} → reward={result.reward:.4f}")
```

### Option 3 — Run locally with Docker

```bash
docker build -t eda-openenv .
docker run -p 8000:8000 eda-openenv
# Open http://localhost:8000/web/
```

---

## 🤖 Run the LLM Baseline Agent

Uses an LLM via HuggingFace router, runs all 3 tasks, outputs structured results:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

python inference.py                          # uses built-in sample dataset
python inference.py --csv your_data.csv      # use your own CSV
```

**Structured stdout output (parsed by validator):**
```
[START] task=detect_missing env=eda_openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=clean_data reward=0.25 done=false error=null
[STEP] step=2 action=eda reward=0.50 done=false error=null
[STEP] step=3 action=feature_engineering reward=0.75 done=false error=null
[STEP] step=4 action=train_model reward=0.95 done=false error=null
[STEP] step=5 action=missing reward=0.83 done=true error=null
[END] success=true steps=5 score=0.656 rewards=0.25,0.50,0.75,0.95,0.83
```

Results saved to `baseline_results.json`.

---

## 🚀 Local Setup

```bash
# Clone
git clone https://github.com/rahmathits/Hackathon_OpenEnv.git
cd Hackathon_OpenEnv

# Install
pip install -r requirements.txt

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or Docker
docker build -t eda-openenv .
docker run -p 8000:8000 eda-openenv

# Open web UI
# http://localhost:8000/web/
```

---

## 🗂️ Project Structure

```
EDA_OpenEnv/
│
├── 📄 inference.py                  # LLM baseline agent — mandatory submission file
├── 📄 pipeline.py                   # Pipeline ordering rules & deterministic rewards
├── 📄 models.py                     # Pydantic models (Action, Observation, Reward)
├── 📄 grader.py                     # Deterministic task graders
├── 📄 client.py                     # Python client SDK
├── 📄 openenv.yaml                  # OpenEnv spec configuration
├── 📄 Dockerfile                    # Container definition
├── 📄 requirements.txt              # Python dependencies
├── 📄 pyproject.toml                # Package configuration
│
└── 📁 server/
    ├── __init__.py
    ├── app.py                       # FastAPI server (openenv-core create_app)
    └── EDA_OpenEnv_environment.py   # Core environment logic
```

---

## 🧪 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Inspect environment state |
| `/health` | GET | Liveness check |
| `/schema` | GET | Action and observation schemas |
| `/web/` | GET | Built-in Gradio web interface |
| `/docs` | GET | Auto-generated Swagger UI |

**Step request:**
```json
{ "action": { "action_type": "clean_data" } }
```

**Step response:**
```json
{
  "observation": {
    "task": "detect_missing",
    "objective": "Identify all columns with missing values",
    "columns": ["age", "salary", "score"],
    "history": ["clean_data"],
    "done": false
  },
  "reward": 0.25,
  "done": false
}
```

---

## 🛠️ Tech Stack

- **[openenv-core](https://pypi.org/project/openenv-core/)** — OpenEnv framework (auto-generates HTTP + WebSocket endpoints)
- **FastAPI + Uvicorn** — API server
- **Pydantic v2** — typed models
- **Pandas / NumPy / Scikit-learn** — EDA tooling
- **OpenAI SDK** — LLM baseline agent (compatible with HF router)
- **Gradio** — built-in web UI (via openenv-core)
- **Docker** — containerised deployment

---

## 📄 License

MIT — free to use, modify and distribute.

---

<div align="center">

Built with ❤️ for the OpenEnv Hackathon • **Phase 1 ✅ Phase 2 ✅ Validated 🏆**

</div>