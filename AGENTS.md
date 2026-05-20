# AGENT.md

## Project Overview

A multilingual AI voice agent enabling Indian users to book restaurant tables on Swiggy Dineout through a phone call, using natural speech in Hindi, English, Telugu, Bengali, or Marathi. The system handles the complete reservation lifecycle: discovery, availability checks, booking, cancellations, and rebooking from history — without requiring app navigation.

## Tech Stack

- **Frontend**: React + Vite
- **Backend**: FastAPI
- **Database**: PostgreSQL (token store), SQLite (local dev fallback)
- **Styling**: TailwindCSS
- **State Management**: Zustand

### Voice Pipeline
- **Telephony**: Twilio Media Streams (WebSocket, mulaw 8kHz)
- **VAD**: Silero VAD (ONNX, per 30ms frame; handles barge-in)
- **Language Classifier**: ECAPA-TDNN or wav2vec2-base + 5-class softmax head (int8 quantized, ~6MB)
- **STT**: 5 × Moonshine Tiny ONNX int8 (one per language: hi, en, te, bn, mr; ~27MB each, ~135MB total)
- **LLM**: Groq (sentence-level streaming)
- **TTS**: Sarvam Bulbul v3 (language-matched voice)
- **Commerce**: Swiggy Builders Club — Dineout MCP (6 tools)

### ML Training
- **Framework**: PyTorch
- **Dataset**: ai4bharat/IndicVoices
- **ONNX Runtime**: int8 quantized inference
- **Augmentation**: librosa (telephony simulation: mu-law 8kHz resample + noise)

## Key Commands

### Backend
```bash
cd backend
uvicorn main:app --reload        # dev server
pytest                           # run all tests
pytest tests/test_api/           # run api tests only
black .                          # format code
```

### Frontend
```bash
cd frontend
npm run dev                      # dev server
npm run build                    # production build
npm run test                     # run tests
npm run lint                     # lint
```

### ML Training
```bash
# Classifier training (Track 2)
cd ml
python -m scripts.train_classifier --epochs 20 --batch-size 64

# Moonshine finetuning per language (Track 3)
python -m scripts.finetune_moonshine --lang hi --epochs 10
python -m scripts.finetune_moonshine --lang en --epochs 10
python -m scripts.finetune_moonshine --lang te --epochs 10
python -m scripts.finetune_moonshine --lang bn --epochs 10
python -m scripts.finetune_moonshine --lang mr --epochs 10

# ONNX export + int8 quantization
python -m scripts.export_onnx --model classifier
python -m scripts.export_onnx --model moonshine-tiny-hi

# Evaluation
python -m scripts.evaluate_classifier --split test
python -m scripts.wer_benchmark --lang hi --test-set b
python -m scripts.wer_benchmark --lang all --test-set a

# Voice pipeline (stub LLM for testing without Groq)
cd backend
python -m voice.pipeline --stage stub
python -m voice.pipeline --stage groq

# Database (PostgreSQL)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=dineout_voice postgres:15
```

## Directory Structure

```
project-name/
├── frontend/
│   ├── src/
│   │   ├── components/          # reusable UI components
│   │   ├── pages/               # route-level components
│   │   ├── hooks/               # custom React hooks
│   │   ├── utils/               # helper functions
│   │   ├── assets/              # images, fonts, static files
│   │   ├── store/               # state management
│   │   ├── services/            # API call functions (fetch/axios wrappers)
│   │   └── main.jsx             # entry point
│   ├── public/
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── backend/
│   ├── core/                    # business logic, domain layer (no HTTP knowledge)
│   ├── api/
│   │   └── v1/                  # versioned route handlers (thin layer)
│   ├── models/                  # SQLAlchemy DB models
│   ├── schemas/                 # Pydantic schemas (request/response)
│   ├── voice/                   # real-time voice pipeline (VAD, classifier routing, TTS)
│   ├── auth/                    # OAuth, token storage/refresh, Twilio SMS
│   ├── utils/
│   │   ├── config.py            # Pydantic BaseSettings class, instantiated as `config`
│   │   ├── logger.py            # custom logger, imported as `logger`
│   │   └── [other helpers]
│   ├── tests/
│   │   ├── test_api/            # mirrors api/v1/ structure
│   │   └── test_core/           # mirrors core/ structure
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # settings and env vars
│   └── requirements.txt
│
├── ml/
│   ├── scripts/                 # training scripts (classifier, moonshine finetuning)
│   ├── evaluate/                # WER benchmarking, classifier evaluation
│   ├── export/                  # ONNX export + int8 quantization
│   ├── notebooks/               # experiment tracking
│   └── models/                  # trained checkpoints (gitignored, downloaded at startup)
│
├── docs/
│   ├── features.json            # canonical feature tracker — always kept up to date
│   └── design.md                # system design specification
├── .env.example                 # committed, no secrets
├── .gitignore
├── README.md
└── AGENT.md
```

## Conventions

### Python (Backend)
- **Package manager: `uv`** — use `uv` for all dependency management (`uv add`, `uv run`, `uv sync`). Never use `pip` directly.
- Every backend file starts with a header comment: `# ----- <4-5 word purpose> @ <file location> -----`
  - Example: `# ----- user authentication logic @ backend/core/auth.py -----`
- Formatter: black (always)
- Naming: snake_case for everything — files, variables, functions, DB columns
- Imports: sorted (isort compatible with black)
- API routes are thin: validate input → call core → return output
- core/ has zero knowledge of HTTP or FastAPI
- Env vars are accessed exclusively via the config object (`from utils.config import config`) — never use `os.environ` directly. 

- Config is a Pydantic BaseSettings class instantiated once in `backend/utils/config.py`. Initialise it as below configuration snippet.

````
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

load_dotenv()

class Settings(BaseSettings):
    """
    Central management for settings and configurations
    Reads .env file
    """

settings = Settings()
````

- All logging uses the custom logger (`from utils.logger import logger`) — never use `print` or the stdlib `logging` module directly. Initialise it as below.

````
import logging
import os
from datetime import datetime

LOGS_DIR="logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE=os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s-%(levelname)s-%(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
````

### JavaScript (Frontend)
- camelCase for variables and functions
- PascalCase for components and types
- snake_case for file names
- All backend API calls go through services/, never directly in components

### General
- Commits: conventional commits format (feat:, fix:, chore:, docs:, test:, refactor:)
- Env vars: never committed, always have a .env.example with keys but no values
- API versioned from day one under /api/v1/
- **README badges**: READMEs should include HTML shield badges (via [shields.io](https://shields.io)) for things like build status, version, license, and tech stack. Use raw HTML `<img>` tags, not Markdown image syntax, so badge layout and alignment can be controlled.

## Development Philosophy
- TDD first: write the test, then the implementation. Never skip.
- Tests mirror the structure of the module they test
- No function ships without a test
- API routes are thin — logic lives in core/
- Explicit over clever — readable code beats smart code

## Agent Roles

This project uses a three-agent workflow. Every task goes through all three stages.

- **Planner**: breaks down the task, identifies edge cases and risks, defines what tests need to exist, produces a written plan. Writes no code. Must check /docs for any relevant design documents before planning.
- **Builder**: implements exactly per the plan — no scope creep, no improvising. Writes tests first, then implementation.
- **Reviewer**: checks correctness, black formatting, snake_case compliance, test coverage, and edge cases. Flags anything that deviates from this AGENT.md. Verifies that docs/features.json has been updated to reflect the work done.

The Planner must finish before the Builder starts.
The Reviewer must approve before any task is considered done.

## Agent Guidelines
- Always run black before considering Python code done
- Always use snake_case — no exceptions for Python files, variables, functions, DB columns
- Never modify files in /docs unless explicitly asked
- Always run tests after making changes — if tests fail, fix before moving on
- Every new backend file must start with the header comment — Reviewer should flag any file missing it
- Never use `os.environ` directly — always use `from utils.config import config`
- Never use `print` or stdlib `logging` — always use `from utils.logger import logger`
- Never put API calls directly in React components — they belong in services/
- Always use `uv` for Python package management — never invoke `pip` directly
- Always check /docs for relevant design documents before starting any task — if a design doc exists for what you're building, it takes precedence
- If a design doc is missing but the task is significant enough to warrant one, flag it to the user before proceeding
- Always update docs/features.json after completing any task — mark features as done, update test status, add new features if they were introduced. follow the schema shape provided

````bash
// docs/features.json
{
  "project": "[project-name]",
  "last_updated": "YYYY-MM-DD",
  "summary": {
    "total": 0,
    "completed": 0,
    "in_progress": 0,
    "planned": 0,
    "tests_passing": 0,
    "tests_failing": 0,
    "tests_missing": 0
  },
  "features": [
    {
      "id": "F001",
      "name": "[Feature Name]",
      "description": "[What it does and why it exists]",
      "status": "planned",
      "priority": "high",
      "module": "backend/core",
      "design_doc": "docs/[relevant-design-doc].md",
      "tests": {
        "status": "missing",
        "files": [],
        "notes": ""
      },
      "subtasks": [
        {
          "id": "F001-1",
          "name": "[Subtask name]",
          "status": "planned"
        }
      ],
      "notes": "",
      "added": "YYYY-MM-DD",
      "completed": null
    }
  ]
}
````

- If something feels out of scope, flag it rather than silently doing it

## Project-Specific Notes

- **External APIs and key storage**: Groq API keys (LLM), Sarvam API keys (Bulbul TTS), Swiggy Builders Club credentials (Dineout MCP), Twilio account SID + auth token — all stored in `.env`, never committed. Swiggy OAuth scopes: Dineout bookings + history only (no payments or food orders).
- **ML model artifacts**: Trained checkpoints live under `ml/models/` and are gitignored — downloaded at startup from a model registry. The `ml/` directory contains training scripts only, not production inference code.
- **Non-standard setup steps**: PostgreSQL must be running (`docker run` command above). ML training requires CUDA-capable GPU. ONNX Runtime must be installed for quantized model inference. All 5 Moonshine models are loaded at backend startup — keep them warm.
- **Language scope (v1)**: Hindi (`hi`), English (`en`), Telugu (`te`), Bengali (`bn`), Marathi (`mr`). Do not add languages without classifier retraining. v2 candidates: Tamil (`ta`), Kannada (`kn`).
- **Classifier routing gotchas**: Low-confidence (`conf < 0.80`) or short utterances (`≤ 5 words`) must NOT update `active_language`. First-utterance default is Hindi. Empty transcripts from STT skip the LLM pipeline entirely.
- **Telephony augmentation**: Always apply `simulate_telephony()` (mu-law 8kHz resample + noise) to 60% of training data. Never apply to validation or test sets — these must remain clean for accurate benchmark reporting.
- **Memory budget**: ~141MB for STT stack in production (5 × Moonshine Tiny int8 at ~27MB each + ~6MB classifier). Stay within this; do not add models without evaluating memory impact.
- **Latency targets**: E2E to first audio byte < 1.5s; VAD 700ms silence gate is unavoidable; classifier < 15ms; Moonshine Tiny < 50ms; Groq TTFT < 200ms; Sarvam Bulbul first sentence < 300ms.
- **Swiggy MCP tools**: 6 tools — `search_restaurants`, `check_table_availability`, `book_table`, `get_booking`, `cancel_booking`, `get_booking_history`. Never fabricate availability or booking confirmations.
- **Files or directories that should never be touched**: `ml/models/` (managed externally), `.env` (real secrets), `docs/design.md` (unless explicitly asked).
- **Deployment target**: FastAPI on a server with static IP for Twilio WebSocket callbacks. Swiggy Builders Club access requires architecture documentation and static IP submission (see `docs/design.md` Appendix A).