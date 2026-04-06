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

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[🚀 Try the Live Demo](https://rahmath1-eda-openenv.hf.space/web/) • [📖 API Docs](https://rahmath1-eda-openenv.hf.space/docs) • [💻 GitHub](https://github.com/rahmathits/OpenEn-Hackathon)

</div>

---

## 👋 What is this?

Most RL environments are abstract — grids, games, mazes. **EDA OpenEnv** is different.

It models a **real data science workflow** that people actually do every day:

```
Clean your data → Explore it → Engineer features → Train a model
```

An AI agent learns to complete these steps in the right order, on a real CSV dataset, and gets rewarded based on the actual quality of its work — not just whether it pressed the right button.

> **Think of it as**: a gym environment, but instead of playing Atari, the agent is doing data science.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌍 **Real-world tasks** | Agent works on actual tabular CSV data, not synthetic grids |
| 🏆 **Dynamic grading** | Rewards based on actual data quality — no fixed scores |
| 🔢 **Pipeline enforcement** | Agent must follow the correct order or get penalised |
| 🔌 **OpenEnv standard** | Works with any agent via `reset()` / `step()` / `state()` |
| 🌐 **Web UI** | Built-in browser interface to test the environment live |
| 🐳 **Docker ready** | One command to run anywhere |

---

## 🎮 Try it in 30 seconds

The easiest way to try the environment is through the **live web interface**:

1. Open [https://rahmath1-eda-openenv.hf.space/web/](https://rahmath1-eda-openenv.hf.space/web/)
2. Click **Reset** to start a new episode
3. Type an action like `clean_data` in the Action Type field
4. Click **Step** and watch the reward come back
5. Follow the pipeline all the way to `train_model`

That's it — no installation needed!

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

**Skip a step? You get penalised.** Follow the order? You get rewarded. Simple, but surprisingly challenging for an agent to learn.

---

## 🎯 Tasks

Each episode assigns the agent one of three tasks with increasing difficulty:

### 🟢 Easy — Detect Missing Values
> *"Find all columns in the dataset that have missing values"*

The agent needs to run `clean_data` → `eda` → `feature_engineering` → `train_model` → `missing`

Graded on: how many columns with missing values were identified vs. total columns

### 🟡 Medium — Find Correlation  
> *"Find the strongest correlation between any two numeric columns"*

The agent needs to run the pipeline then execute `correlation`

Graded on: strength of the maximum correlation found (`0.9+` = perfect score)

### 🔴 Hard — Generate Insight
> *"Generate a meaningful insight about the dataset referencing actual values"*

The agent needs to run the pipeline then execute `insight`

Graded on: text length, numeric references, and column name mentions

---

## 🏆 Reward System

All rewards are bounded between **0 and 1**:

```
✅ Correct pipeline order (first time):
   clean_data          →  +0.25
   eda                 →  +0.50
   feature_engineering →  +0.75
   train_model         →  +1.00

📊 Task-specific action:
   Scored dynamically by grader (0.0 → 1.0)

⚠️  Out-of-order action:
   Skip 1 step  →  -0.25
   Skip 2 steps →  -0.50
   Skip 3 steps →  -0.75

🔁 Repeated action:
   →  0.10 (small penalty)
```

---

## 🔌 Connect Your Own Agent

Any Python agent can connect to this environment in just a few lines:

### Option 1 — Connect to the live HF Space

```python
from eda_openenv_client import EDAOpenEnv, EDAAction

EDA_ENV_URL = "https://rahmath1-eda-openenv.hf.space"

with EDAOpenEnv(base_url=EDA_ENV_URL) as env:
    result = env.reset()

    print(f"Task: {result.observation['task']}")
    print(f"Objective: {result.observation['objective']}")

    # Run the pipeline in order
    for action in ["clean_data", "eda", "feature_engineering", "train_model"]:
        if result.done:
            break
        result = env.step(EDAAction(action_type=action))
        print(f"{action:25} → reward={result.reward:.2f} | done={result.done}")
```

### Option 2 — Run locally with Docker

```bash
# Pull and run
docker run -p 8000:8000 rahmath1/eda-openenv:latest

# Then connect
env = EDAOpenEnv(base_url="http://localhost:8000")
```

### Option 3 — Use the HTTP API directly

```bash
# Start a new episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d "{}"

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "clean_data"}}'

# Check state
curl http://localhost:8000/state
```

---

## 🤖 Run the LLM Baseline Agent

The baseline agent uses an LLM (via HuggingFace router) to decide which action to take at each step:

```bash
# Set your credentials
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

# Run across all 3 tasks
python inference.py --csv your_data.csv

# Run with more episodes for stable average
python inference.py --csv your_data.csv --episodes 3 --steps 10
```

**Example output:**
```
══ EDA OpenEnv — Inference Script ══════════════════════════
  Dataset       : titanic.csv (891 rows × 12 cols)
  Model         : Qwen/Qwen2.5-72B-Instruct

══ Task: detect_missing  (difficulty: easy) ══════════════════
  Step 01 | clean_data             | score=0.2500 | ✅
  Step 02 | eda                    | score=0.5000 | ✅
  Step 03 | feature_engineering    | score=0.7500 | ✅
  Step 04 | train_model            | score=1.0000 | ✅
  Step 05 | missing                | score=0.8750 | ✅ DONE

══ BASELINE SCORE SUMMARY ════════════════════════════════════
  Task                      Diff       Steps    Penalties  Avg Reward
  ─────────────────────────────────────────────────────────────
  detect_missing            easy       5        0          0.7750
  find_correlation          medium     6        0          0.6800
  generate_insight          hard       7        1          0.5200
  ─────────────────────────────────────────────────────────────
  BASELINE SCORE                                            0.6583
══════════════════════════════════════════════════════════════
```

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/rahmathits/OpenEn-Hackathon.git
cd OpenEn-Hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with Docker (recommended)
docker build -t eda-openenv .
docker run -p 8000:8000 eda-openenv

# 4. Open the web UI
# Go to http://localhost:8000/web/
```

---

## 🗂️ Project Structure

```
EDA_OpenEnv/
│
├── 📄 inference.py              # LLM baseline agent (mandatory for submission)
├── 📄 pipeline.py               # Pipeline ordering rules & reward shaping
├── 📄 models.py                 # Pydantic models (Action, Observation, Reward)
├── 📄 grader.py                 # Dynamic task graders
├── 📄 client.py                 # Python client SDK
├── 📄 openenv.yaml              # OpenEnv spec configuration
├── 📄 Dockerfile                # Container definition
├── 📄 requirements.txt          # Python dependencies
│
└── 📁 server/
    ├── app.py                   # FastAPI server (auto-wired by openenv-core)
    └── EDA_OpenEnv_environment.py  # Core environment logic
```

---

## 🧪 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode, get initial observation |
| `/step` | POST | Take an action, get reward + next observation |
| `/state` | GET | Inspect full environment state |
| `/health` | GET | Liveness check |
| `/schema` | GET | Action and observation schemas |
| `/web/` | GET | Built-in web interface |
| `/docs` | GET | Auto-generated API documentation |

**Step request format:**
```json
{
  "action": {
    "action_type": "clean_data"
  }
}
```

**Step response format:**
```json
{
  "observation": {
    "task": "detect_missing",
    "objective": "Identify all columns with missing values",
    "columns": ["age", "salary", "score"],
    "history": ["clean_data"],
    "done": false,
    "reward": 0.25
  }
}
```

---

## 📊 Evaluation Criteria

This project was built for the **OpenEnv Hackathon** and addresses all five judging criteria:

| Criterion | Weight | How we address it |
|---|---|---|
| 🌍 Real-world utility | 30% | Models a genuine EDA workflow on real CSV data |
| 🎯 Task & grader quality | 25% | 3 tasks with difficulty progression; dynamic input-sensitive grading |
| 🏗️ Environment design | 20% | Clean state management, shaped rewards in `[0,1]`, proper episode boundaries |
| 💻 Code quality & spec | 15% | OpenEnv-compliant API, typed Pydantic models, Dockerfile, inference script |
| 💡 Creativity & novelty | 10% | Pipeline ordering enforcement with graded penalties is an original mechanic |

---

## 🛠️ Tech Stack

- **[openenv-core](https://pypi.org/project/openenv-core/)** — OpenEnv framework (auto-generates HTTP endpoints)
- **FastAPI + Uvicorn** — API server
- **Pydantic v2** — typed models
- **Pandas / NumPy / Scikit-learn** — EDA tooling
- **OpenAI SDK** — LLM baseline agent
- **Docker** — containerised deployment
- **Gradio** — built-in web UI (via openenv-core)

---

## 🤝 Contributing

Contributions are welcome! If you want to:
- Add new tasks
- Improve the grader logic
- Add new action types
- Fix bugs

Fork the repo, make your changes and open a pull request.

```bash
git fork https://github.com/rahmathits/OpenEn-Hackathon
cd OpenEn-Hackathon
# make your changes
git pull-request
```

---

## 📄 License

MIT — free to use, modify and distribute.

---

<div align="center">

Built with ❤️ for the OpenEnv Hackathon

</div>