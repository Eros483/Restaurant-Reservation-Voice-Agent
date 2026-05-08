# Dineout Voice Reservation Agent — Design Document

---

## Overview

A multilingual AI voice agent that lets Indian users call a phone number and book restaurant tables on Swiggy Dineout using natural speech — in Hindi, Tamil, Telugu, Kannada, or any other supported Indian language. The system handles the full reservation flow: discovery, availability checks, booking, cancellations, and rebooking history.

Built on a DIY stack with Twilio for telephony, Sarvam for STT/TTS, IndicLID for language detection, Groq for LLM inference, and Swiggy Builders Club MCP for Dineout actions.

---

## Stack at a Glance

| Layer | Tool | Notes |
|---|---|---|
| Telephony | Twilio Media Streams | WebSocket, mulaw 8kHz |
| VAD | Silero VAD | Per 30ms frame, barge-in handling |
| STT | Sarvam Saaras v1 | Indian language optimised |
| Language ID | IndicLID → finetuned if needed | Benchmarked on ai4bharat/IndicVoices |
| LLM + Tool Use | Groq | Fast inference, streamed output |
| Commerce Layer | Swiggy Builders Club — Dineout MCP | 6 tools, real transactions |
| TTS | Sarvam Bulbul | Language-matched voice |
| Token Store | PostgreSQL | Caller ID → Swiggy OAuth token |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TWILIO LAYER                                │
│                                                                      │
│  Inbound call arrives                                                │
│  → Extract caller ID from Twilio webhook                             │
│  → Lookup phone_number in Postgres                                   │
│      Found + token valid   → proceed with access_token              │
│      Found + token expired → refresh via Swiggy OAuth, update DB     │
│      Not found             → IN-CALL ONBOARDING FLOW (see below)     │
│  → Open Media Stream WebSocket                                       │
│  → Receive audio as mulaw 8kHz                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      AUDIO PREPROCESSING                             │
│                                                                      │
│  mulaw 8kHz → PCM 16kHz resample                                     │
│  Frame into 30ms chunks                                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         SILERO VAD                                   │
│                                                                      │
│  Per 30ms frame: speech_prob = model(chunk, 16000)                   │
│                                                                      │
│  speech_prob > 0.5 during silence  → start buffering                │
│  speech_prob > 0.5 during TTS      → BARGE-IN                       │
│    └── kill TTS stream immediately                                   │
│    └── flush old audio buffer                                        │
│    └── start fresh buffer                                            │
│  700ms silence after speech        → end of utterance               │
│    └── flush buffer → trigger STT pipeline                           │
│                                                                      │
│  Metrics: false positive rate, false negative rate,                  │
│           barge-in response time < 200ms                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     SARVAM STT — Saaras v1                           │
│                                                                      │
│  Input:  buffered 16kHz PCM                                          │
│  Param:  language_code = active_language (default: hi-IN)            │
│  Output: transcript text                                             │
│                                                                      │
│  Guard:  empty transcript → skip pipeline, re-open mic               │
│                                                                      │
│  Metrics: WER per language, STT latency                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      LANGUAGE ID LAYER                               │
│                                                                      │
│  Model:  IndicLID (baseline)                                         │
│    → Benchmark on ai4bharat/IndicVoices                              │
│    → If macro F1 < 0.90: finetune on IndicVoices                     │
│         Base: ai4bharat/indic-bert or google/muril                   │
│                                                                      │
│  Rules:                                                              │
│    conf > 0.80 AND utterance > 5 words → update active_language      │
│    conf < 0.80 OR  utterance ≤ 5 words → keep previous language      │
│    mid-call switch detected           → update STT lang next turn    │
│                                                                      │
│  Output: lang_code passed to Groq context + next Sarvam STT call     │
│                                                                      │
│  Metrics: macro F1, per-language accuracy,                           │
│           short-utterance accuracy tracked separately                │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    GROQ LLM + SWIGGY DINEOUT MCP                     │
│                                                                      │
│  System prompt:                                                      │
│    - You are a table reservation assistant for Swiggy Dineout        │
│    - Respond in: {active_language}                                   │
│    - Max 2 sentences per response (phone call)                       │
│    - Always confirm details before calling book_table                │
│    - Never fabricate availability or pricing                         │
│    - On any API error → apologise and offer to try again             │
│                                                                      │
│  Dineout MCP Tools:                                                  │
│    search_restaurants       (location, cuisine, vibe, date, time)    │
│    check_table_availability (restaurant_id, date, time, party_size)  │
│    book_table               (restaurant_id, date, time, party_size)  │
│    get_booking              (booking_id)                             │
│    cancel_booking           (booking_id)                             │
│    get_booking_history      (user_id)                                │
│                                                                      │
│  Auth: per-user Swiggy access_token injected from Postgres lookup    │
│                                                                      │
│  Streaming: sentence-level — first sentence triggers TTS immediately │
│  Timeout:   > 3s → play filler ("ek second...") → retry once        │
│  Fallback:  2 retries fail → "let me connect you to someone"         │
│                                                                      │
│  Metrics: TTFT, containment rate, escalation rate,                   │
│           bookings placed, cancellations, FCR                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     SARVAM TTS — Bulbul                              │
│                                                                      │
│  Language-matched voice from active_language                         │
│  Sentence-level streaming — don't wait for full Groq response        │
│  Barge-in signal → halt audio stream immediately                     │
│                                                                      │
│  Metrics: TTS latency, MOS score (human eval)                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         TWILIO LAYER                                 │
│                                                                      │
│  Re-encode PCM → mulaw 8kHz                                          │
│  Stream audio back to caller over WebSocket                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Swiggy Builders Club

Swiggy Builders Club provides access to 3 MCP servers and 18+ API tools across Swiggy Food, Instamart, and Dineout. This system uses **Dineout only**.

**Access model:** Invite-led. Submit use case, integration architecture, auth redirect URIs, static IP info, and a data-handling declaration. Swiggy reviews and grants access.

**Dineout MCP tools used:**

| Tool | Purpose |
|---|---|
| `search_restaurants` | Location, cuisine, vibe, date/time filters |
| `check_table_availability` | Party size, date, time for a specific restaurant |
| `book_table` | Place the reservation |
| `get_booking` | Read back booking details by ID |
| `cancel_booking` | Cancel an existing reservation |
| `get_booking_history` | Fetch past bookings for rebooking |

**Ground rules (Builders Club ToS):**
- Cannot resell MCP access
- Cannot scrape data beyond provided APIs
- Cannot misrepresent availability or pricing
- Cannot build aggregation layers that obscure Swiggy's brand

**OAuth scopes to request:** Dineout bookings + history only. Do not request food order or payment scopes — overbroad permissions hurt onboarding conversion.

---

## Conversation Flow

```
Turn 1 — Intent + details
  "book a table for 2 at a rooftop place in Koramangala, Saturday 8pm"

Turn 2 — Search + present options
  search_restaurants(location="Koramangala", vibe="rooftop", ...)
  "Found two options — Skyye Rooftop and The Permit Room. Which one?"

Turn 3 — Check availability
  check_table_availability(restaurant_id, date, time, party_size=2)
  "Skyye has availability at 8pm for 2. Shall I confirm the booking?"

Turn 4 — Confirm + book
  book_table(restaurant_id, date="Saturday", time="20:00", party_size=2)
  "Done! Booking confirmed at Skyye Rooftop, Saturday 8pm, 2 guests.
   Your booking ID is SW-4821."
```

**Supported intents:**
- New reservation
- Cancel reservation (`cancel_booking`)
- Check existing booking (`get_booking`)
- Rebook a past restaurant (`get_booking_history` → `book_table`)

**Edge cases to handle:**
- No availability at requested time → suggest alternative slots
- Multiple restaurants match → disambiguation turn
- Restaurant not found → broaden search or ask for clarification
- Booking API failure → apologise, offer retry or escalate

---

## Language ID Layer

### Baseline: IndicLID
Handles 24 Indian languages + English. Free, open source.

```python
from indiclid import IndicLID
lang_detector = IndicLID()
lang, confidence = lang_detector.predict(transcript_text)
# → ("hi", 0.97) or ("ta", 0.91)
```

### Benchmarking Plan
Dataset: `ai4bharat/IndicVoices` — naturalistic speech across 22 Indian languages, multiple speakers.

```
1. Pull IndicVoices samples per language
2. Run each audio clip through Sarvam STT → get transcript
3. Feed transcript into IndicLID → get predicted language
4. Compare predicted vs ground truth label from IndicVoices
5. Compute per-language accuracy + macro F1

If macro F1 > 0.90 → ship IndicLID as-is
If macro F1 < 0.90 → finetune on IndicVoices
  Base model: ai4bharat/indic-bert or google/muril
```

**Watch for during benchmarking:**
- Code-switching samples (Hinglish, Tanglish) — IndicLID may struggle
- Short utterances ("haan", "ok", "nahi") — track accuracy separately
- Noisy audio — IndicVoices has clean audio in parts; real Twilio call accuracy will be lower, so benchmark numbers are optimistic

### Decision rules per utterance

| Condition | Action |
|---|---|
| conf > 0.80 AND > 5 words | Update `active_language` |
| conf < 0.80 | Keep previous `active_language` |
| utterance ≤ 5 words | Keep previous `active_language` regardless of conf |
| Mid-call language switch | Update STT lang code for next utterance |

---

## VAD — Barge-in Handling

```python
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')

# Per 30ms chunk (resampled to 16kHz):
speech_prob = model(chunk, 16000)
is_speaking = speech_prob > 0.5
```

**State transitions:**

```
IDLE
  speech detected           → BUFFERING, start audio buffer

BUFFERING
  speech continues          → append to buffer
  700ms silence             → END_OF_UTTERANCE, flush to STT pipeline
  speech during TTS playing → BARGE_IN

BARGE_IN
  → kill TTS stream immediately
  → discard current Groq response
  → flush audio buffer
  → restart BUFFERING with new audio
```

---

## User Onboarding

### Web Onboarding (primary)

Users link their Swiggy account before ever calling.

```
Page 1: Enter phone number
  [+91] [__________]
  [Send OTP]
         ↓
Page 2: Verify OTP (Twilio Verify)
  Sent to +91 98765 43210
  [_ _ _ _ _ _]
  [Verify]  [Resend in 30s]
         ↓
Page 3: Connect Swiggy account
  [Connect Swiggy Account]  ← OAuth redirect
  
  What we access:
  ✓ Dineout bookings
  ✓ Booking history
  ✗ Payment info (not requested)
  ✗ Food orders (not requested)
         ↓  OAuth callback
Page 4: Done
  ✓ You're all set!
  Call [+91 XXXXX XXXXX] to book tables using your voice.
```

**On OAuth callback — Postgres write:**
```sql
INSERT INTO user_tokens (
  phone_number, swiggy_user_id,
  access_token, refresh_token, token_expiry
) VALUES (...)
ON CONFLICT (phone_number) DO UPDATE
  SET access_token   = EXCLUDED.access_token,
      refresh_token  = EXCLUDED.refresh_token,
      token_expiry   = EXCLUDED.token_expiry,
      updated_at     = now();
```

### In-Call Onboarding (fallback)

Triggered when caller ID is not found in Postgres.

```
UNKNOWN_CALLER detected
  ↓
Agent: "Hi! To book tables via this service, you'll need to link 
        your Swiggy account first. I'm sending a link to your 
        number right now — takes about 30 seconds."
  ↓
Twilio SMS → onboarding URL?ref={phone_number}
  ↓
Agent: "I'll wait while you complete it. Take your time."
  ↓
Poll Postgres every 2s, max 90s

  ├── Record appears within 90s
  │     Agent: "Perfect, you're all linked! How can I help?
  │             Looking to book a table somewhere tonight?"
  │     → Resume normal call flow
  │
  └── 90s timeout — no record
        Agent: "No worries — the link stays valid. Call back 
                once you've connected your account and we'll 
                get you booked right away. Bye!"
        → Graceful hangup
```

**Note:** The polling loop needs a circuit breaker — a hung Postgres connection must not keep the call open past the 90s window.

---

## Database

```sql
CREATE TABLE user_tokens (
  phone_number     VARCHAR(15) PRIMARY KEY,  -- E.164, e.g. "+919876543210"
  swiggy_user_id   TEXT        NOT NULL,
  access_token     TEXT        NOT NULL,     -- encrypted at rest
  refresh_token    TEXT        NOT NULL,     -- encrypted at rest
  token_expiry     TIMESTAMPTZ NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT now(),
  updated_at       TIMESTAMPTZ DEFAULT now()
);
```

**Token resolution on every call:**
```python
async def resolve_user(caller_id: str):
    record = await db.fetchrow(
        "SELECT * FROM user_tokens WHERE phone_number = $1", caller_id
    )
    if not record:
        return None  # trigger onboarding

    if record["token_expiry"] < now():
        new_tokens = await swiggy_oauth.refresh(record["refresh_token"])
        await db.execute(
            "UPDATE user_tokens SET access_token=$1, refresh_token=$2, "
            "token_expiry=$3, updated_at=now() WHERE phone_number=$4",
            new_tokens.access, new_tokens.refresh,
            new_tokens.expiry, caller_id
        )
        return new_tokens.access

    return record["access_token"]
```

**Security:**
- Tokens encrypted at rest (Postgres column-level encryption or storage-level)
- Tokens never logged in call traces
- Phone number is the only identity key — no passwords or PINs needed (Twilio already verified the caller's number)

---

## Latency Budget

| Stage | Target |
|---|---|
| Silero VAD + 700ms silence | ~700ms |
| Sarvam STT | ~300ms |
| IndicLID | ~50ms |
| Groq (time to first token) | ~200ms |
| Sarvam TTS (first sentence) | ~300ms |
| **Total to first audio byte** | **~1.55s** |

To stay under 1.5s: stream Groq output sentence-by-sentence into TTS. Do not wait for the full LLM response before starting audio playback.

---

## Metrics

### System Health

| Metric | Target |
|---|---|
| End-to-end latency | < 1.5s |
| Time to first audio byte | < 800ms |
| VAD false positive rate | < 5% |
| VAD false negative rate | < 3% |
| Barge-in response time | < 200ms |
| STT Word Error Rate (WER) | < 15% per language |
| Lang ID macro F1 | > 0.90 |
| Lang ID on short utterances (< 5 words) | Tracked separately |
| TTS naturalness (MOS) | > 3.5/5 |
| WebSocket drop rate | < 1% |

### Business

| Metric | Target |
|---|---|
| Call containment rate | > 70% |
| Escalation rate | < 30% |
| First Call Resolution (FCR) | > 60% |
| Customer Satisfaction (CSAT) | > 3.8/5 |
| Call abandonment rate | < 8% |
| Repeat call rate (same issue, 7 days) | < 20% |
| Onboarding conversion (SMS sent → linked) | Track from day 1 |
| Cost per call | Track from day 1 |

---

## Test Plan

### Functional Tests

| Test | Description |
|---|---|
| Happy path | Hindi query, full pipeline completes, correct Dineout booking |
| Barge-in | User speaks mid-TTS, TTS stops, new utterance processed correctly |
| Mid-call language switch | Starts Hindi, switches to Tamil, system adapts from utterance 3 |
| Code-switching | Hinglish, Tanglish — STT and lang ID handle gracefully |
| Short utterances | "haan", "ok", "nahi" — lang ID must not flip active language |
| Silence | User quiet for 5s — agent prompts, no crash |
| Long utterance | 30s monologue — buffering and STT handle correctly |
| Repeated barge-in | User interrupts 3 times — no state corruption |
| Reorder flow | "Book that place I went to last month" — history lookup works |
| Cancellation | User cancels existing booking by booking ID |

### Stress / Load Tests

| Test | Description |
|---|---|
| Concurrent calls | 10, 50, 100 simultaneous WebSocket connections |
| Latency under load | p95 latency stays < 2s at 50 concurrent calls |
| API rate limits | Sarvam, Groq, Swiggy MCP — test behaviour when limits hit |
| WebSocket reconnect | Drop mid-call, verify graceful session handling |

### Language Coverage Tests (IndicVoices)

Run for each supported language: Hindi, Tamil, Telugu, Kannada, Bengali, Marathi, Gujarati, Malayalam, Punjabi, Odia, Assamese + others.

- Separate accuracy score per language — do not hide weak languages in macro average
- Test on noisy audio — add synthetic noise to IndicVoices samples to simulate real Twilio call quality
- Benchmark numbers from clean IndicVoices audio will be optimistic; note this clearly

### Edge Cases

| Case | Expected behaviour |
|---|---|
| Unknown/unsupported language | Graceful fallback, ask caller to speak Hindi or English |
| Angry caller, fast speech | VAD and STT still function, no crash |
| Background noise (TV, crowd) | Silero VAD should suppress, STT degrades gracefully |
| Network jitter / packet loss | Simulate in WebSocket stream, verify no hang |
| Empty STT transcript | Skip pipeline, re-open mic — do not send blank prompt to Groq |
| Groq timeout | Play filler audio, retry once, escalate on second failure |
| Swiggy MCP error | Apologise, offer retry — never fabricate a booking confirmation |
| No availability | Suggest alternative times or nearby restaurants |
| Polling circuit breaker | In-call onboarding polling terminates at 90s regardless of DB state |

### Regression Tests (run on every deploy)

- WER does not degrade on benchmark set
- Lang ID macro F1 does not drop
- E2E latency p95 does not increase > 10%
- Onboarding SMS delivery success rate stays > 95%

---

## Observability

**Per-call trace** (stored for every call):
- Latency at each pipeline stage
- Full transcript: STT output + detected language per utterance + Groq tool calls made
- Outcome: booking placed / cancelled / escalated / abandoned

**Business metrics dashboard:**
- Bookings placed, cancellations, repeat callers
- Onboarding conversion rate (SMS sent → account linked)
- Web onboarding vs in-call onboarding split
- Containment rate, escalation rate, FCR
- Cost per call (Twilio + Sarvam + Groq)
- CSAT via post-call SMS ("Rate your experience 1–5")

**Alerts:**
- p95 latency > 2s
- WER regression on benchmark set
- Lang ID F1 drop
- Swiggy MCP error rate spike
- WebSocket drop rate > 1%
- Onboarding SMS delivery failure rate > 5%

---

## What to Build First

Build these two tracks in parallel — they are independent and unblock everything else:

**Track 1 — Auth + Onboarding**
Postgres schema → web onboarding UI → Twilio OTP flow → Swiggy OAuth integration → token storage and refresh logic. This unblocks all Dineout MCP calls.

**Track 2 — Voice Pipeline**
Twilio Media Stream WebSocket → audio preprocessing → Silero VAD → Sarvam STT → IndicLID. Test this end-to-end with a mock LLM response before wiring in Groq and Swiggy MCP.

Once both tracks are working independently, wire them together and run the full happy-path flow.