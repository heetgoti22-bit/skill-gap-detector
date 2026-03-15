<div align="center">

# 🧠 Skill Gap Detector

**AI-powered labor market intelligence that scrapes real job postings, extracts skills with NLP, detects trends with statistical testing, and recommends learning paths for career growth.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-33%2F33_passing-brightgreen?style=flat)](tests/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

[Live Dashboard](#-live-dashboard) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Docs](#-api-endpoints) · [Methodology](#-methodology--transparency) · [Evaluation](#-extraction-evaluation)

</div>

---

## 🎯 What This Does

Most "skill trend" tools show you static lists. This project builds the **entire pipeline from scratch** — from scraping raw job postings to serving trend forecasts via API.

```
Real job postings → NLP skill extraction → Statistical trend detection → Career recommendations
```

**Built for undergrads and career changers** who want to know: *What should I learn next to maximize my employability?*

### Key Features

- **Real data pipeline** — Scrapes 3,000+ live job postings from Greenhouse, Lever, Remotive, and Hacker News APIs (no API keys needed)
- **NLP skill extraction** — Rule-based taxonomy (120+ skills) + spaCy entity ruler, measured at ~92% precision / ~78% recall
- **Statistical trend detection** — Mann-Kendall non-parametric test + Sen's slope estimator, not just "is the line going up"
- **Forecasting** — Facebook Prophet with linear fallback and 80% prediction intervals
- **Gap analyzer** — Input your skills, get a readiness score and prioritized learning path with course recommendations
- **Career paths** — Role-specific roadmaps (AI/ML Engineer, Data Engineer, Platform Engineer, Security Engineer) with phase-by-phase skill progression
- **Full transparency** — Every component documents what it does AND what it doesn't do. A `/api/v1/methodology` endpoint exposes all limitations publicly

---

## 📸 Live Dashboard

The frontend is a standalone React dashboard with a neural network background, glassmorphism UI, and real-time animations.

**To view locally:**
```bash
open frontend/index.html
```

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Industry overview** | 6 industries with animated bar charts, metric cards, AI-generated insights |
| **Skill tables** | Expandable rows with sparklines, 12-month bar charts, salary data, tech tags |
| **Emerging signals** | Pre-mainstream skills with confidence scores and time-to-mainstream estimates |
| **Global leaderboard** | Top 10 fastest-growing skills across all industries with trend sparklines |
| **Gap analyzer** | Add your skills → get readiness score, priority gaps, course recommendations |
| **Neural network bg** | Live canvas with drifting nodes, pulse signals, and glow effects |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (free, no auth)            │
│  Greenhouse API ─── Lever API ─── Remotive ─── HN Hiring   │
│  (24 companies)     (5 co's)      (7 cats)     (Algolia)   │
└────────────────────────┬────────────────────────────────────┘
                         │  async HTTP + retry/backoff
                         │  token bucket rate limiting
                         │  content-hash deduplication
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SKILL EXTRACTION (rule-based, not custom NER)  │
│                                                             │
│  Layer 1: Taxonomy regex ────── 120+ skills, pre-compiled   │
│  Layer 2: spaCy entity ruler ── pattern matching variants   │
│  Layer 3: BERTopic ──────────── novel cluster discovery     │
│                                                             │
│  Measured: ~92% precision, ~78% recall (50-posting eval)    │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 TREND ANALYSIS + FORECASTING                │
│                                                             │
│  Mann-Kendall test ── non-parametric trend significance     │
│  Sen's slope ──────── robust growth rate estimation         │
│  Classification ───── requires BOTH magnitude AND p < 0.05  │
│  Prophet forecast ─── with linear fallback                  │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LEARNING PATH MAPPER (heuristic)               │
│                                                             │
│  Priority: 0.4×growth + 0.35×demand + 0.25×salary          │
│  Prerequisites: manually curated dependency graph           │
│  Courses: curated from DeepLearning.AI, Coursera, fast.ai  │
│  Career paths: 4 role templates with phased timelines       │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    API + DASHBOARD                          │
│                                                             │
│  FastAPI ──── 8 endpoints + interactive docs at /docs       │
│  React ────── Neural network UI with gap analyzer           │
│  SQLite ───── WAL mode, schema versioning, content dedup    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- macOS / Linux (Windows with WSL works too)
- Python 3.9+
- ~2GB disk space for NLP models

### One-command setup

```bash
chmod +x setup.sh && ./setup.sh
```

This installs everything — dependencies, spaCy model, project structure, all source files.

### Or manual setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/skill-gap-detector.git
cd skill-gap-detector

# Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run full pipeline (scrapes real job postings — takes 5-8 min)
python pipeline.py full

# Start API
python pipeline.py serve
# → http://localhost:8000/docs

# Open dashboard
open frontend/index.html
```

### Available Commands

| Command | What it does | Time |
|---------|-------------|------|
| `python pipeline.py full` | Complete pipeline: scrape → extract → analyze → forecast | 5-8 min |
| `python pipeline.py ingest` | Scrape job postings from 4 sources | 3-5 min |
| `python pipeline.py extract` | NLP skill extraction | 1-2 min |
| `python pipeline.py trends` | Trend analysis + forecasting | 30-60s |
| `python pipeline.py serve` | Start API server on :8000 | instant |
| `python pipeline.py test` | Built-in unit tests (20 assertions) | < 1s |
| `pytest tests/ -v` | Full test suite (33 tests) | ~2s |
| `python eval_extraction.py` | Extraction quality evaluation (50 postings) | ~5s |
| `make run` / `make serve` / `make test` | Makefile shortcuts | — |

---

## 📡 API Endpoints

Start the server with `python pipeline.py serve`, then open `http://localhost:8000/docs` for interactive documentation.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | DB stats and health check |
| GET | `/api/v1/trends?direction=rising` | Skill trends (filter by direction, industry) |
| GET | `/api/v1/skills/{name}` | Detail on a specific skill with courses and prereqs |
| POST | `/api/v1/gap-analysis` | Personalized skill gap analysis |
| GET | `/api/v1/career-paths` | Role-specific career trajectories |
| GET | `/api/v1/novel-topics` | BERTopic-discovered emerging skills |
| GET | `/api/v1/methodology` | Full transparency on methods and limitations |

### Example: Gap Analysis

```bash
curl -X POST http://localhost:8000/api/v1/gap-analysis \
  -H "Content-Type: application/json" \
  -d '{"skills": ["python", "react", "sql"], "target_industry": "AI/ML"}'
```

**Response:**
```json
{
  "readiness_score": 22,
  "methodology": "Heuristic: 0.4*growth + 0.35*demand + 0.25*salary (hand-tuned, not learned)",
  "skills_matched": ["python"],
  "skill_gaps": [
    {"skill": "machine learning", "growth_yoy": 25.3, "current_demand": 5420, "priority_score": 0.84},
    {"skill": "deep learning", "growth_yoy": 21.1, "current_demand": 3180, "priority_score": 0.72}
  ],
  "learning_path": [
    {"skill": "machine learning", "reason": "+25.3% growth, 5420 openings", "hours": 80},
    {"skill": "deep learning", "reason": "Prerequisite for llm fine-tuning", "hours": 90}
  ],
  "estimated_hours": 248,
  "career_paths": [{"role": "AI/ML Engineer", "match_pct": 12, "timeline_months": 24}]
}
```

---

## 🔬 Methodology & Transparency

This project is deliberately honest about what each component does and doesn't do. The `/api/v1/methodology` endpoint returns full details.

| Component | What it IS | What it is NOT |
|-----------|-----------|----------------|
| Skill extraction | Rule-based taxonomy (120+ skills) + spaCy entity ruler | NOT a custom-trained NER model |
| Trend detection | Mann-Kendall non-parametric test (p < 0.05) | NOT just "is the line going up" |
| Growth estimation | Sen's slope (robust to outliers) | NOT simple first/last comparison |
| Forecasting | Facebook Prophet with linear fallback | NOT ARIMA, NOT an ensemble |
| Recommendations | Weighted heuristic (hand-tuned) | NOT a learned/collaborative model |
| Prerequisites | Manually curated dependency graph | NOT inferred from data |

### Known Limitations

| Area | Limitation | Impact |
|------|-----------|--------|
| Data sources | 4 free APIs, ~35 companies | Biased toward tech startups |
| Salary parsing | Regex-based, ~40% hit rate | Sparse salary data |
| HN source | ~30% format deviation | Noisy company/title extraction |
| Industry labels | Taxonomy-based keywords | Company-level nuance lost |
| Seasonality | Not handled in trend detection | Jan/summer dips may be misread |
| Scale | SQLite, single-machine | Designed for < 100k postings |

---

## 📊 Extraction Evaluation

Quality is measured against 50 hand-labeled job postings. Run `python eval_extraction.py` to reproduce.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | ~92% | When we extract a skill, it's correct ~92% of the time |
| **Recall** | ~78% | We catch ~78% of skills a human annotator would find |
| **F1 Score** | ~84% | Solid for a rule-based system |

**Common misses:** Skills not in the taxonomy (scikit-learn, GraphQL, PostgreSQL)
**Common false positives:** Skills mentioned in passing context, not as requirements

---

## 🧪 Testing

33 tests across 8 test classes:

```
tests/
├── TestDatabase        — Insert, dedup, schema versioning, trends
├── TestTaxonomy        — Alias resolution, industry mapping
├── TestExtraction      — Pattern matching, requirements parsing
├── TestTrends          — Mann-Kendall (rising/falling/flat), Sen's slope
├── TestSalary          — Parsing ranges, k-format, bound rejection
├── TestRateLimiter     — Burst allowance, throttling
├── TestAPI             — All endpoints, validation, 404s
└── TestLearningPaths   — Prereq validity, career path consistency
```

```bash
$ pytest tests/ -v
================================ 33 passed in 1.23s ================================
```

---

## 📁 Project Structure

```
skill-gap-detector/
├── pipeline.py              # Full ML pipeline (ingestion → API)
├── eval_extraction.py       # 50-posting quality evaluation
├── setup.sh                 # One-command project setup
├── requirements.txt         # Python dependencies
├── Makefile                 # Convenience commands
├── frontend/
│   └── index.html           # Standalone React dashboard
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # 33 tests across 8 classes
└── data/
    ├── skillgap.db          # SQLite database (auto-created)
    ├── output/
    │   └── dashboard_data.json
    └── eval/
        ├── eval_results.json
        └── eval_report.md
```

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Ingestion | aiohttp, asyncio | Async HTTP with retry/backoff for 4 concurrent sources |
| NLP | spaCy, sentence-transformers, BERTopic | Rule-based extraction + novel topic discovery |
| Statistics | scipy (Mann-Kendall), numpy (Sen's slope) | Non-parametric trend testing |
| Forecasting | Prophet | Time-series forecasting with uncertainty |
| Storage | SQLite (WAL mode) | Simple, zero-config, schema-versioned |
| API | FastAPI, Pydantic, uvicorn | Auto-docs, validation, rate limiting |
| Frontend | React, HTML Canvas | Neural network background, glassmorphism UI |
| Testing | pytest, pytest-asyncio | 33 tests including API integration |

---

## 🗺 Roadmap

- [ ] Add Indeed/LinkedIn APIs for broader coverage
- [ ] Train a real NER model on the 50-posting labeled dataset
- [ ] Context-aware extraction (distinguish "required" vs "mentioned")
- [ ] Seasonal decomposition in trend detection
- [ ] Collaborative filtering for recommendations
- [ ] PostgreSQL + Alembic for production scale
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions

---

## 📄 License

MIT — use it for your portfolio, modify it, learn from it.

---

<div align="center">

**Built as a portfolio project demonstrating end-to-end ML engineering.**

If this helped you, consider giving it a ⭐

</div>
