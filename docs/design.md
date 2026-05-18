# Dineout Voice Reservation Agent — Design Specification

**Version:** 1.0  
**Status:** Draft  
**Last Updated:** May 2026

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Scope & Constraints](#2-scope--constraints)
3. [System Architecture](#3-system-architecture)
4. [Component Specifications](#4-component-specifications)
5. [Data Architecture](#5-data-architecture)
6. [Conversation Design](#6-conversation-design)
7. [Phased Implementation Plan](#7-phased-implementation-plan)
8. [Benchmark & Evaluation Plan](#8-benchmark--evaluation-plan)
9. [Test Plan](#9-test-plan)
10. [Metrics & Observability](#10-metrics--observability)

---

## 1. Product Overview

A multilingual AI voice agent enabling Indian users to book restaurant tables on Swiggy Dineout through a phone call, using natural speech in their preferred language. The system handles the complete reservation lifecycle: discovery, availability checks, booking, cancellations, and rebooking from history.

### 1.1 Problem Statement

Dineout reservations currently require app navigation. A voice-first interface removes this friction for callers who prefer speech interaction, and creates an accessible channel for users who find app UIs cumbersome for time-sensitive bookings.

### 1.2 Supported Actions

- Search restaurants by location, cuisine, vibe, date, and time
- Check table availability for a given party size and slot
- Place a table reservation
- Retrieve an existing booking by ID
- Cancel a booking
- Look up booking history for rebooking a past restaurant

### 1.3 Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Telephony | Twilio Media Streams | WebSocket, mulaw 8kHz |
| Voice Activity Detection | Silero VAD | Per 30ms frame; handles barge-in |
| Language Classification | Finetuned audio classifier | Tiny encoder + 5-class head on raw audio |
| Speech-to-Text | 5 × Moonshine Tiny (per language) | ONNX, int8 quantized; routed by classifier |
| STT Benchmark Baseline | Sarvam Saaras v1 | Commercial reference to beat |
| LLM Inference | Groq | Streamed output, sentence-level |
| Commerce / Actions | Swiggy Builders Club — Dineout MCP | 6 tools, real transactions |
| Text-to-Speech | Sarvam Bulbul v3 | Language-matched voice |
| Token Store | PostgreSQL | Caller ID → Swiggy OAuth token |

---

## 2. Scope & Constraints

### 2.1 Language Scope

Dineout is a metro behaviour — users own smartphones, plan in advance, and are concentrated in major cities. The v1 language set is deliberately narrow to maximise ML quality per language.

| Language | Primary Coverage | IndicVoices Tag |
|---|---|---|
| Hindi | Delhi, Mumbai, north India | `hi` |
| English | All metros, code-switchers | `en` |
| Telugu | Hyderabad | `te` |
| Bengali | Kolkata | `bn` |
| Marathi | Mumbai, Pune | `mr` |

**v2 candidates:** Tamil (`ta`, Chennai) and Kannada (`kn`, Bengaluru), added based on call volume data.

Narrowing to 5 languages has direct ML benefits: more IndicVoices training data per class, simpler classifier discrimination, and reduced catastrophic forgetting risk during Moonshine finetuning.

### 2.2 Architecture Decisions & Rationale

**Why not Whisper for STT?**  
Whisper pads all audio to a fixed 30-second window regardless of utterance length, paying full compute cost every time. Moonshine uses variable-length input windows and encoder/decoder state caching — scaling inference cost to actual audio duration. Empirically, Moonshine v2 Small is 13.1× faster than Whisper Small on equivalent hardware (148ms vs. ~1.9s). The lack of Whisper's language token is addressed by a separate audio classifier.

**Why 5 separate Moonshine models, not one multilingual model?**  
The Moonshine research ("Flavors of Moonshine") shows accuracy gains come specifically from monolingual training. A multilingual Moonshine finetuned across all 5 languages would dilute this, producing worse accuracy than 5 specialists. At int8 quantization, each Moonshine Tiny is ~27MB; 5 models total is ~135MB — well within budget.

**Why an audio classifier, not text-based LID?**  
Text-based LID (e.g., IndicLID) requires a transcript, but STT requires a language code — a circular dependency. The classifier must run on raw audio before STT.

**Why not a general-purpose audio LID (e.g., facebook/mms-lid)?**  
MMS-LID is a 126-language model applied to a 5-class problem — overparameterised, slower than necessary, and not trained on telephone-quality audio. A classifier trained specifically on IndicVoices for these 5 languages will be smaller, faster, and more accurate on this distribution.

**Why is STT not streaming?**  
A reservation agent cannot act on a partial transcript. "Book a table for" is useless until the utterance completes — party size, location, date, and time are all required. Buffering to end-of-utterance (via VAD silence detection) and running one batch inference step is correct for this use case.

---

## 3. System Architecture

### 3.1 End-to-End Pipeline

```
Inbound Call (Twilio)
  │
  ├─ Caller ID lookup → PostgreSQL token store
  │     ├─ Known + valid token    → proceed
  │     ├─ Known + expired token  → refresh via Swiggy OAuth
  │     └─ Unknown caller         → IN-CALL ONBOARDING FLOW
  │
  └─ Open WebSocket (mulaw 8kHz)
       │
       ↓
Audio Preprocessing
  └─ mulaw 8kHz → PCM 16kHz resample → 30ms frames
       │
       ↓
Silero VAD (per frame)
  ├─ speech_prob > 0.5, during silence → start buffering
  ├─ speech_prob > 0.5, during TTS     → BARGE-IN (kill TTS, flush buffer)
  └─ 700ms silence after speech        → end of utterance → trigger pipeline
       │
       ↓
Language Classifier (full utterance buffer)
  └─ Tiny encoder + 5-class softmax
       ├─ conf > 0.80, > 5 words → update active_language
       ├─ conf < 0.80 or ≤ 5 words → keep previous active_language
       └─ first utterance, low conf → default to Hindi
       │
       ↓ route by active_language
Moonshine STT (language-specific model)
  └─ One of: moonshine-tiny-{hi,en,te,bn,mr}
       └─ Empty transcript → skip, re-open mic
       │
       ↓
Groq LLM + Swiggy Dineout MCP
  └─ System prompt: respond in active_language, ≤ 2 sentences, confirm before booking
       └─ Tools: search_restaurants, check_table_availability, book_table,
                 get_booking, cancel_booking, get_booking_history
       │
       ↓ sentence-level streaming
Sarvam Bulbul v3 TTS
  └─ target_language_code + speaker driven by active_language
       └─ First sentence plays before Groq finishes
       │
       ↓
Twilio Output
  └─ PCM → mulaw 8kHz → WebSocket → caller
```

### 3.2 Latency Budget

| Stage | Target Latency |
|---|---|
| Silero VAD + 700ms silence gate | ~700ms (unavoidable) |
| Language classifier | ~10ms |
| Moonshine Tiny ONNX int8 | ~50ms |
| Groq time to first token | ~200ms |
| Sarvam Bulbul first sentence | ~300ms |
| **Total to first audio byte** | **~1.26s** |

### 3.3 Memory Layout at Runtime

```
Classifier model (ECAPA-TDNN int8):   ~6MB
Moonshine Tiny Hindi (int8):          ~27MB
Moonshine Tiny English (int8):        ~27MB
Moonshine Tiny Telugu (int8):         ~27MB
Moonshine Tiny Bengali (int8):        ~27MB
Moonshine Tiny Marathi (int8):        ~27MB
─────────────────────────────────────────────
Total STT stack in memory:            ~141MB
```

---

## 4. Component Specifications

### 4.1 Voice Activity Detection — Silero VAD

**Input:** 30ms audio frames at 16kHz  
**Output:** speech probability per frame (0.0–1.0)

**Behaviour:**

| Condition | Action |
|---|---|
| `speech_prob > 0.5`, in silence | Begin buffering audio |
| `speech_prob > 0.5`, TTS playing | Barge-in: kill TTS stream, flush buffer, start new buffer |
| 700ms consecutive silence after speech | End of utterance; flush buffer to classifier + STT |

**Metrics to track:** false positive rate (< 5% target), false negative rate (< 3% target), barge-in response time (< 200ms target).

---

### 4.2 Language Classifier

**Architecture:**

```
Input: raw audio buffer (16kHz PCM, full utterance)
  ↓
Feature extraction: mel-spectrogram or MFCC
  ↓
Tiny encoder (one of):
  - wav2vec2-base (95M params) + classification head  ← higher accuracy
  - ECAPA-TDNN (6M params)                            ← faster, smaller
  - MMS-LID encoder finetuned on 5 classes            ← good starting point
  ↓
Linear head: encoder_dim → 5
  ↓
Softmax → [p_hi, p_en, p_te, p_bn, p_mr]
  ↓
Output: predicted language + confidence score
```

**Training dataset:** `ai4bharat/IndicVoices`, filtered and balanced across 5 target languages.

**Class balancing:**

```python
dataset = load_dataset("ai4bharat/IndicVoices")
target_langs = {"hi", "en", "te", "bn", "mr"}
filtered = dataset.filter(lambda x: x["language"] in target_langs)
# Undersample majority, oversample minority to equal class size
```

**Train/val/test split:** 80 / 10 / 10 (same held-out test set used for Moonshine benchmarks).

**Telephony Simulation (acoustic augmentation):**

IndicVoices is recorded at 16–48kHz, high SNR. Twilio delivers 8kHz mu-law (G.711) audio, band-limited to 300Hz–3.4kHz with compression artifacts. Without augmentation, WER and classifier accuracy drops of 30–50% are common in production.

Apply the following augmentation to 60% of training samples:

```python
import librosa
import numpy as np

def simulate_telephony(audio: np.ndarray, sr: int) -> np.ndarray:
    audio_8k = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    audio_mulaw = librosa.mu_compress(audio_8k, mu=255, quantize=True)
    audio_decoded = librosa.mu_expand(audio_mulaw, mu=255, quantize=True)
    noise = np.random.normal(0, 0.002, audio_decoded.shape)
    audio_noisy = audio_decoded + noise
    audio_16k = librosa.resample(audio_noisy, orig_sr=8000, target_sr=16000)
    return audio_16k
```

**Augmentation split:** 60% telephony-simulated, 40% clean.  
**Important:** Do not apply simulation to validation or test sets.

**Routing decision rules:**

| Condition | Action |
|---|---|
| `conf > 0.80` AND utterance > 5 words | Update `active_language`; route to that Moonshine model |
| `conf < 0.80` OR utterance ≤ 5 words | Keep previous `active_language` |
| First utterance, `conf < 0.80` | Default to Hindi |
| Mid-call switch (`conf > 0.80`) | Update `active_language` and Bulbul TTS params |

The short-utterance guard is critical at call start — "haan", "ok", "nahi" are 1-word responses that carry insufficient signal for reliable classification.

**Targets:** 5-class accuracy > 95%, inference latency < 15ms.

---

### 4.3 Moonshine STT — Per-Language Models

Five separate Moonshine Tiny models, each finetuned on a single language from IndicVoices.

**Model sizing rationale:**

| Variant | Params | Latency | Size (int8) |
|---|---|---|---|
| Moonshine Tiny | 27M | ~50ms | ~27MB |
| Moonshine Small | 130M | ~148ms | ~130MB |
| 5 × Tiny total | — | — | ~135MB |
| 5 × Small total | — | — | ~650MB |

Start with Tiny. Upgrade to Small for a specific language only if WER benchmarks show Tiny is insufficient.

**Finetuning:**

```python
from moonshine import MoonshineModel
from datasets import load_dataset

for lang in ["hi", "en", "te", "bn", "mr"]:
    model = MoonshineModel.from_pretrained("UsefulSensors/moonshine-tiny")
    lang_dataset = load_dataset(
        "ai4bharat/IndicVoices", split="train"
    ).filter(lambda x: x["language"] == lang)
    # 80/10/10 split per language
    # Track per-language WER after every eval epoch
    # Early stopping on validation WER
```

**ONNX export and quantization:**

```bash
python -m moonshine.export \
    --model moonshine-tiny-hi \
    --format onnx \
    --quantize int8
# → moonshine-tiny-hi-int8.onnx (~27MB)
```

All 5 models are loaded at startup and kept warm. No cold-start latency per call.

**Empty transcript guard:** if STT returns an empty string, skip the LLM pipeline and re-open the microphone.

**Targets:** per-language WER tracked individually; macro WER < 15%; RTF < 0.3; latency ~50ms.

---

### 4.4 Groq LLM + Swiggy Dineout MCP

**System prompt constraints:**
- Respond in `{active_language}`
- Maximum 2 sentences per response (voice interface)
- Always confirm details before calling `book_table`
- Never fabricate availability or pricing
- On any API error: apologise and offer to retry

**Dineout MCP tools:**

| Tool | Arguments |
|---|---|
| `search_restaurants` | `location`, `cuisine`, `vibe`, `date`, `time` |
| `check_table_availability` | `restaurant_id`, `date`, `time`, `party_size` |
| `book_table` | `restaurant_id`, `date`, `time`, `party_size` |
| `get_booking` | `booking_id` |
| `cancel_booking` | `booking_id` |
| `get_booking_history` | `user_id` |

**Auth:** per-user Swiggy `access_token` injected from PostgreSQL lookup on each call.

**Streaming:** sentence-level — first sentence triggers TTS immediately.

**Failure handling:**

| Condition | Action |
|---|---|
| Response time > 3s | Play filler audio ("ek second..."); retry once |
| 2 retries failed | "Let me connect you to someone" → escalate |
| Swiggy MCP API error | Apologise and retry; never fabricate a booking |

---

### 4.5 Sarvam Bulbul v3 TTS

Language and speaker are set from `active_language` on every response:

```python
LANGUAGE_TO_BULBUL = {
    "hi": {"target_language_code": "hi-IN", "speaker": "anand"},
    "en": {"target_language_code": "en-IN", "speaker": "anand"},
    "te": {"target_language_code": "te-IN", "speaker": "anu"},
    "bn": {"target_language_code": "bn-IN", "speaker": "anu"},
    "mr": {"target_language_code": "mr-IN", "speaker": "anu"},
}

def get_tts_params(lang: str, confidence: float, word_count: int):
    if confidence < 0.80 or word_count <= 5:
        return None  # keep previous voice
    return LANGUAGE_TO_BULBUL.get(lang, LANGUAGE_TO_BULBUL["hi"])
```

Mid-call language switches are detected by the classifier on the next utterance and the TTS voice updates accordingly. Barge-in halts the audio stream immediately.

---

## 5. Data Architecture

### 5.1 PostgreSQL Token Store

```sql
CREATE TABLE user_tokens (
  phone_number     VARCHAR(15) PRIMARY KEY,  -- E.164 format
  swiggy_user_id   TEXT        NOT NULL,
  access_token     TEXT        NOT NULL,     -- encrypted at rest
  refresh_token    TEXT        NOT NULL,     -- encrypted at rest
  token_expiry     TIMESTAMPTZ NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT now(),
  updated_at       TIMESTAMPTZ DEFAULT now()
);
```

**On every inbound call:**

| Lookup result | Action |
|---|---|
| Token valid | Inject `access_token` into Groq MCP auth header |
| Token expired | Refresh via Swiggy OAuth → update DB → proceed |
| Phone unknown | Trigger in-call onboarding flow |

### 5.2 User Onboarding Flows

**Web Onboarding (primary — preferred before first call):**

1. User enters phone number → Twilio OTP verify
2. OTP confirmed → phone ownership proven
3. "Connect Swiggy" → OAuth redirect (scopes: Dineout bookings + history only — no payments or food orders)
4. OAuth callback → exchange `auth_code` for tokens
5. Write to PostgreSQL (phone, tokens, expiry)
6. Success screen: "Call [number] to start booking"

**In-Call Onboarding (fallback — unknown callers):**

```
Caller ID not found in Postgres
  ↓
Agent: "You'll need to link your Swiggy account first.
        Sending a link to your number now."
  ↓
Twilio SMS → onboarding URL?ref={phone_number}
  ↓
Agent: "I'll wait while you do that."
  ↓
Poll Postgres every 2s, max 90s (hard circuit breaker at 90s)
  ├── Token appears → "You're linked! How can I help?"
  └── 90s timeout  → "The link stays valid — call back once done." → graceful hangup
```

---

## 6. Conversation Design

### 6.1 Canonical Reservation Flow

```
Turn 1 — User states intent + details
  "Book a table for 2 at a rooftop place in Koramangala, Saturday 8pm"

Turn 2 — Agent searches and presents options
  search_restaurants(location="Koramangala", vibe="rooftop", ...)
  "Found two options — Skyye Rooftop and The Permit Room. Which one?"

Turn 3 — Agent checks availability
  check_table_availability(restaurant_id, date="Saturday", time="8pm", party_size=2)
  "Skyye has availability at 8pm for 2. Shall I confirm the booking?"

Turn 4 — User confirms; agent books
  book_table(...)
  "Done! Booking confirmed at Skyye Rooftop, Saturday 8pm, 2 guests. Booking ID: SW-4821."
```

### 6.2 Supported Intents

- New reservation (the primary flow above)
- Cancel an existing booking
- Check status of an existing booking by ID
- Rebook a restaurant from past booking history

### 6.3 Edge Case Handling

| Scenario | Behaviour |
|---|---|
| No availability for requested slot | Suggest alternative times or nearby restaurants |
| Multiple restaurant matches | Disambiguation turn ("Which city area do you mean?") |
| Swiggy MCP API failure | Apologise; retry once; escalate on second failure |
| Unsupported language | Classifier low confidence → Hindi default → agent asks to speak one of 5 supported languages |
| Empty transcript | Skip LLM pipeline; re-open microphone |
| User silent for 5s | Agent prompts; no crash |
| Repeated barge-in (3×) | No state corruption; each new utterance processed cleanly |

---

## 7. Phased Implementation Plan

The build is organised into four independent tracks that converge at integration.

---

### Phase 1 — Foundations (Tracks 1, 2, 3 in parallel)

**Duration:** Weeks 1–4

#### Track 1 — Auth & Onboarding

**Goal:** A working OAuth link between a phone number and a Swiggy account, stored securely in PostgreSQL.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 1.1 | Define PostgreSQL schema (`user_tokens` table, encryption at rest) | Schema migration script |
| 1.2 | Build web onboarding UI (phone input → Twilio OTP verify) | Working web page |
| 1.3 | Implement Swiggy OAuth redirect and callback handler | OAuth callback endpoint |
| 1.4 | Token storage and refresh logic | `token_service.py` |
| 1.5 | In-call onboarding: Twilio SMS dispatch + Postgres polling loop | SMS + polling handler |
| 1.6 | Circuit breaker: hard 90s timeout on in-call polling | Tested timeout behaviour |
| 1.7 | Token injection into Groq MCP auth header | Auth middleware |

**Acceptance criteria:**
- A test phone number can complete web onboarding and be found in the DB.
- A simulated expired token is refreshed automatically on lookup.
- An unknown caller receives an SMS and the polling loop resolves within 90s.

---

#### Track 2 — Language Classifier

**Goal:** A trained, quantized, production-ready 5-class audio language identifier.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 2.1 | Filter IndicVoices to 5 target languages | Filtered dataset |
| 2.2 | Balance classes (undersample/oversample) | Balanced dataset |
| 2.3 | Implement telephony augmentation pipeline (`simulate_telephony`) | `augment.py` |
| 2.4 | Define train/val/test split (80/10/10); lock test set | Reproducible splits |
| 2.5 | Select and train encoder backbone (ECAPA-TDNN or wav2vec2-base + head) | Trained model |
| 2.6 | Evaluate: per-language F1, 5-class accuracy, short-utterance slice, confusion matrix | Evaluation report |
| 2.7 | int8 quantize model | Quantized checkpoint |
| 2.8 | Benchmark inference latency (target: < 15ms) | Latency report |

**Acceptance criteria:**
- 5-class accuracy > 95% on held-out test set.
- Inference latency < 15ms on target hardware.
- Confusion matrix reviewed; no systematic cross-language errors.
- Short-utterance accuracy tracked and documented (≤ 5 words).

---

#### Track 3 — Moonshine STT Finetuning

**Goal:** Five language-specific Moonshine Tiny models, finetuned, ONNX-exported, and benchmarked against Sarvam Saaras v1.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 3.1 | Set up per-language finetuning loop on IndicVoices | Training scripts |
| 3.2 | Apply telephony augmentation (same pipeline as Track 2) to training data | Augmented training sets |
| 3.3 | Finetune Moonshine Tiny for each of: hi, en, te, bn, mr | 5 model checkpoints |
| 3.4 | Track per-language WER after every eval epoch; early stopping on val WER | Training logs |
| 3.5 | Export each model to ONNX, int8 quantize (~27MB per model) | 5 ONNX files |
| 3.6 | Benchmark: WER on Test Set A (clean) and Test Set B (telephony-simulated) | Benchmark report |
| 3.7 | Benchmark: WER vs Sarvam Saaras v1 on same test sets | Comparison table |
| 3.8 | Benchmark: RTF and latency p50/p95 on target hardware | Latency report |

**Test Set Design:**

```
Test Set A — Clean IndicVoices (held-out 10%, untouched)
  → Academic baseline; optimistic by design

Test Set B — Telephony-simulated (same 10%, simulation applied)
  → Production proxy; the number that actually matters

Gap (A → B) = acoustic domain shift penalty.
Large gap → increase telephony simulation proportion in training data.
```

**Decision from benchmark:**

| Outcome | Action |
|---|---|
| Moonshine WER competitive + RTF < 0.3 | Ship Moonshine; Sarvam not used in prod |
| Sarvam wins significantly on one language | Use Sarvam for that language only |
| Moonshine RTF > 0.3 on a language | Profile; consider Moonshine Small for that language |

**Acceptance criteria:**
- All 5 models exported and loadable at runtime (~135MB total).
- Macro WER < 15% on Test Set B (telephony-simulated).
- RTF < 0.3 on deployment hardware.
- A vs B gap documented per language.

---

### Phase 2 — Voice Pipeline (Track 4)

**Depends on:** Track 2 (classifier) and Track 3 (Moonshine models) complete.  
**Duration:** Weeks 4–6

**Goal:** A working real-time voice pipeline from Twilio audio in to Sarvam TTS audio out, tested with a stub LLM before Groq is wired.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 4.1 | Twilio WebSocket handler (mulaw 8kHz ingestion) | `twilio_handler.py` |
| 4.2 | Audio preprocessing: mulaw → PCM 16kHz resample, 30ms framing | `audio.py` |
| 4.3 | Silero VAD integration: speech buffering, end-of-utterance detection | `vad.py` |
| 4.4 | Barge-in: kill TTS stream on `speech_prob > 0.5` during playback | Barge-in handler |
| 4.5 | Load all 5 Moonshine models at startup; implement classifier routing | `model_router.py` |
| 4.6 | Language classifier integration; active_language state management | `classifier_service.py` |
| 4.7 | Sarvam Bulbul TTS integration; language-to-voice mapping | `tts_service.py` |
| 4.8 | Sentence-level streaming from LLM to TTS | Streaming pipeline |
| 4.9 | Twilio output: PCM → mulaw 8kHz re-encode → WebSocket | Output handler |
| 4.10 | Stub LLM integration: hardcoded responses for pipeline testing | Stub LLM |
| 4.11 | End-to-end pipeline test with stub LLM | Test pass |

**Acceptance criteria:**
- Full audio round-trip functions with stub LLM.
- Barge-in halts TTS within 200ms.
- Active language updates correctly on mid-call language switch.
- Short utterance guard prevents false language switches.
- Latency to first audio byte < 1.5s under no-load conditions.

---

### Phase 3 — Integration

**Depends on:** Track 1 (Auth) and Track 4 (Voice Pipeline) complete.  
**Duration:** Weeks 6–7

**Goal:** Wire all components together. Replace stub LLM with Groq + Dineout MCP. Validate the complete end-to-end flow.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 5.1 | Replace stub LLM with Groq; configure system prompt | Groq integration |
| 5.2 | Wire Swiggy Builders Club Dineout MCP tools | MCP integration |
| 5.3 | Integrate OAuth token injection from PostgreSQL into Groq auth | Auth integration |
| 5.4 | Implement Groq timeout handling: filler audio, retry, escalation | Timeout handler |
| 5.5 | End-to-end happy path test: Hindi query → booking placed | Test pass |
| 5.6 | End-to-end tests for cancellation, rebooking, error paths | Test suite pass |

**Acceptance criteria:**
- A real table can be booked via voice call in all 5 supported languages.
- Cancellation and rebooking intents work end-to-end.
- Groq timeout and MCP error paths are handled without fabricating bookings.
- Containment rate > 70% in controlled test calls.

---

### Phase 4 — Hardening & Observability

**Duration:** Weeks 7–8

**Goal:** Production-readiness: per-call tracing, business dashboards, load testing, regression suite, alerting.

**Deliverables:**

| # | Task | Output |
|---|---|---|
| 6.1 | Per-call trace: latency per stage, transcript, classifier output, tool calls, outcome | Tracing pipeline |
| 6.2 | Business dashboard: bookings, cancellations, containment, FCR, CSAT, cost | Dashboard |
| 6.3 | Alerts: p95 latency, WER regression, MCP error rate, WebSocket drops, SMS failures | Alert rules |
| 6.4 | Load tests: 10, 50, 100 concurrent calls | Load test report |
| 6.5 | Regression suite: classifier accuracy, per-language WER, RTF, E2E latency p95 | CI pipeline |
| 6.6 | Swiggy Builders Club access request: submit use case, architecture, auth URIs, static IP | Access granted |

**Acceptance criteria:**
- p95 latency < 2s at 50 concurrent calls.
- Regression suite runs on every deploy with automated pass/fail.
- All alert thresholds configured and verified.

---

## 8. Benchmark & Evaluation Plan

### 8.1 Classifier Benchmark

On held-out 10% of IndicVoices (5 target languages):

| Metric | Target |
|---|---|
| 5-class accuracy | > 95% |
| Per-language F1 (hi, en, te, bn, mr) | Tracked separately |
| Short utterance accuracy (≤ 5 words) | Tracked separately |
| Inference latency | < 15ms |
| Confusion matrix | Documented |

### 8.2 Moonshine STT Benchmark

Per-language WER, macro WER, RTF, and latency tracked on two test sets:

| Metric | Moonshine Tiny — Test Set A (clean) | Moonshine Tiny — Test Set B (telephony sim) | Sarvam Saaras v1 |
|---|---|---|---|
| WER — Hindi | | | |
| WER — English | | | |
| WER — Telugu | | | |
| WER — Bengali | | | |
| WER — Marathi | | | |
| Macro avg WER | | | |
| RTF on deployment hardware | | | |
| Latency p50 / p95 | | | |
| WER on short utterances | | | |
| WER on code-switched samples | | | |

The A–B delta per language is the acoustic domain shift penalty. Minimise this during training by increasing the telephony simulation proportion if the gap is large.

---

## 9. Test Plan

### 9.1 Functional Tests

| Test | Description | Pass Condition |
|---|---|---|
| Happy path | Hindi query → classifier → Moonshine → Groq → booking placed | Booking confirmed end-to-end |
| Barge-in | User speaks mid-TTS | TTS halts; new utterance processed correctly |
| Mid-call language switch | Starts Hindi, switches to Telugu | Classifier detects; Bulbul voice updates |
| Short utterances | "haan", "ok", "nahi" | `active_language` does not update |
| Low confidence routing | `conf < 0.80` | Previous language kept; no Moonshine misrouting |
| First utterance default | Unknown caller, low confidence | Defaults to Hindi |
| Silence | User quiet for 5s | Agent prompts; no crash |
| Repeated barge-in | 3 interruptions | No state corruption |
| Rebooking flow | "Book that place I went to last month" | `get_booking_history` called; restaurant identified |
| Cancellation | "Cancel my booking SW-4821" | `cancel_booking` called; confirmation given |

### 9.2 Classifier Tests

| Test | Description |
|---|---|
| 5-class accuracy | Per-language and overall on held-out set |
| Short utterance slice | Accuracy on utterances ≤ 5 words |
| Code-switched samples | Hinglish, Telugu-English, etc. |
| Telephony-simulated audio | Augmented noise applied to test slice |
| Confusion matrix | Which languages are misclassified as which |

### 9.3 Moonshine STT Tests

Per-language WER, macro WER, RTF, latency p50/p95 on both test sets; WER on noisy, short, and code-switched audio slices.

### 9.4 Load Tests

| Test | Description | Target |
|---|---|---|
| Concurrent calls | 10, 50, 100 simultaneous WebSocket connections | No errors |
| Latency under load | p95 at 50 concurrent | < 2s |
| Model inference under load | Classifier + Moonshine RTF | Stays < 0.3 |
| API rate limits | Groq, Swiggy MCP behaviour at limit | Graceful degradation |

### 9.5 Edge Case Tests

| Case | Expected Behaviour |
|---|---|
| Unsupported language | Low confidence → Hindi default → agent asks to speak a supported language |
| Angry caller, fast speech | VAD and classifier function; no crash |
| Background noise | VAD suppresses; classifier degrades gracefully |
| Empty transcript | Skip pipeline; re-open microphone |
| Groq timeout | Filler audio; retry once; escalate on second failure |
| Swiggy MCP error | Apologise; retry; never fabricate a booking |
| In-call onboarding 90s timeout | Graceful hangup with "call back once linked" message |

### 9.6 Regression Suite (every deploy)

- Classifier 5-class accuracy does not drop
- Per-language Moonshine WER does not degrade
- Moonshine RTF does not increase > 10%
- E2E latency p95 does not increase > 10%
- Onboarding SMS delivery success > 95%

---

## 10. Metrics & Observability

### 10.1 System Health Targets

| Metric | Target |
|---|---|
| End-to-end latency | < 1.5s |
| Time to first audio byte | < 800ms |
| VAD false positive rate | < 5% |
| VAD false negative rate | < 3% |
| Barge-in response time | < 200ms |
| Classifier accuracy (5-class) | > 95% |
| Classifier latency | < 15ms |
| Moonshine macro WER | < 15% |
| Moonshine RTF | < 0.3 |
| TTS naturalness (MOS) | > 3.5/5 |
| WebSocket drop rate | < 1% |

### 10.2 Business Targets

| Metric | Target |
|---|---|
| Call containment rate | > 70% |
| Escalation rate | < 30% |
| First Call Resolution (FCR) | > 60% |
| CSAT | > 3.8/5 |
| Call abandonment rate | < 8% |
| Repeat call rate (7 days) | < 20% |
| Onboarding conversion | Track from day 1 |
| Cost per call | Track from day 1 |

### 10.3 Observability

**Per-call trace (every call):**
- Latency at each pipeline stage
- Full transcript + classifier output (language + confidence) per utterance
- Groq tool calls made and results
- TTS language used
- Call outcome (booked / cancelled / escalated / abandoned)

**Business dashboard:**
- Bookings placed, cancellations, repeat callers
- Onboarding conversion rate
- Containment rate, escalation rate, FCR
- Cost per call, CSAT

**Alerts:**
- p95 latency > 2s
- Classifier accuracy drop
- Moonshine WER regression
- Moonshine RTF spike
- Swiggy MCP error rate spike
- WebSocket drop rate > 1%
- SMS delivery failure > 5%

---

## Appendix A — Swiggy Builders Club Access

Access is invite-led. Submit:

- Use case description
- System architecture overview
- OAuth redirect URIs
- Static IP addresses
- Data handling declaration

**OAuth scopes to request:** Dineout bookings + booking history only. Do not request payment or food order scopes.

---

## Appendix B — Build Track Summary

| Track | Dependencies | Duration | Owner |
|---|---|---|---|
| Track 1 — Auth & Onboarding | None | Weeks 1–4 | — |
| Track 2 — Classifier | None | Weeks 1–4 | — |
| Track 3 — Moonshine Finetuning | None | Weeks 1–4 | — |
| Track 4 — Voice Pipeline | Tracks 2, 3 complete | Weeks 4–6 | — |
| Phase 3 — Integration | Tracks 1, 4 complete | Weeks 6–7 | — |
| Phase 4 — Hardening | Integration complete | Weeks 7–8 | — |
