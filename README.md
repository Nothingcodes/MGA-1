# MGA-1 v1.0: Medical General Architecture for Reliable Medical LLMs

MGA-1 is a model-agnostic, agentic reliability architecture designed to study whether external multi-stage multi-agent orchestration and verification can mitigate medical hallucinations, unsupported clinical claims, and unsafe confidence without modifying underlying model weights.

> **Research Prototype Notice**: This system is a research prototype designed for studying agentic LLM reliability. It does **not** provide clinical diagnosis and does **not** substitute for licensed medical evaluation.

---

## Architecture

The system executes a strictly staged, closed-loop verification pipeline:

```
USER
 ↓
Problem Formulation / Interactive Clinical Intake (Base Model: gemini-3.8-flash)
 ↓
Evidence Grounding (Google Search, URL Context, Code Execution)
 ↓
MGA-1 Orchestrator (gemini-3.5-flash-lite)
 ↓
Dynamic Specialist Swarm (antigravity-preview-09-2026 managed agents, max 5, parallel execution)
 ↓
Disagreement Detection (gemini-3.5-flash-lite)
 ↓
Evidence-Constrained Summarizer (gemini-3.8-flash)
 ↓
Independent Evaluator (gemini-3.8-flash)
 ↓
PASS ──→ Final User Response & Source Provenance
FAIL
 ↓
Revision Plan
 ↓
Orchestrator ──→ Dynamic Expansion / Specialist Reassignment / New Evidence ──→ Re-synthesis ──→ Re-evaluation (Max 3 Loops)
 ↓
(If unresolved after 3 cycles: ABSTAIN)
```

### Core Architecture Stages

1. **Base Model (`gemini-3.8-flash`)**:
   - Conducts interactive intake Q&A to gather patient age, onset, duration, symptom severity, medications, allergies, and red flags.
   - Grounding tools: Google Search, URL Context, Code Execution.
   - Formulates a normalized `TaskSpec` with clinical domains and evidence requirements.

2. **Orchestrator (`gemini-3.5-flash-lite`)**:
   - Determines relevant specialties from a built-in catalog of 12+ specialties.
   - Dynamically provisions between 1 and 5 specialists.
   - Assigns distinct, non-overlapping investigative tasks.
   - Adapts the swarm dynamically on revision cycles based on evaluator feedback.

3. **Specialist Swarm (`antigravity-preview-09-2026`)**:
   - Reusable custom managed agents lazily created via `client.agents.create()`.
   - Independent parallel execution using `Promise.allSettled`.
   - Researches clinical guidelines using Google Search and URL Context.
   - Multi-strategy JSON parsing for Antigravity text outputs.
   - Real-time terminal dashboard with per-specialist activity and latency tracking.

4. **Disagreement Detector (`gemini-3.5-flash-lite`)**:
   - Audits specialist reports for divergent risk stratifications or conflicting clinical guidelines.
   - Highlights disagreements without arbitrarily picking a side.

5. **Evidence-Constrained Summarizer (`gemini-3.8-flash`)**:
   - Synthesizes findings strictly from gathered evidence and specialist reports.
   - Establishes explicit claim provenance: `Claim → Evidence → Specialist`.
   - Prohibits hallucinating new facts or unreferenced citations.

6. **Independent Evaluator (`gemini-3.8-flash`)**:
   - Evaluates candidate answers across 10 structured dimensions:
     - Factuality, Evidence Support, Completeness, Internal Consistency, Medical Safety, Uncertainty Calibration, Unsupported Claims, Critical Errors, Missing Critical Information, Revision Plan.
   - Gates answers: `PASS` proceeds to final output; `FAIL` generates an actionable revision plan for the Orchestrator.

7. **Revision Loop & Abstention**:
   - Maximum 3 revision iterations.
   - If consensus or safety thresholds cannot be established within 3 iterations, the system **ABSTAINS** (`decision = "UNRESOLVED"`), refusing to manufacture false certainty.

---

## Quick Start

### 1. Requirements
- Node.js >= 20.0.0
- `@google/genai` >= 2.3.0 (`2.24.0`)
- `GEMINI_API_KEY` set in environment or `.env`

### 2. Configuration
Copy `.env.example` to `.env` and set your key:
```bash
cp .env.example .env
export GEMINI_API_KEY="your-api-key"
```

Optional model configuration overrides:
```bash
export MGA_BASE_MODEL="gemini-3.8-flash"
export MGA_ORCHESTRATOR_MODEL="gemini-3.5-flash-lite"
export MGA_SUMMARIZER_MODEL="gemini-3.8-flash"
export MGA_EVALUATOR_MODEL="gemini-3.8-flash"
```

### 3. Running MGA-1
```bash
# Syntax verification
npm run check

# Start interactive CLI
npm start
# or
node index.js
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/help` | Display available commands |
| `/new` | Start a new research session |
| `/reset` | Reset current Q&A and session state |
| `/models` | Display active model configuration and recognized registry |
| `/agents` | Display specialist managed agents catalog and registration status |
| `/status` | View current phase, round, revision cycle, and active specialists |
| `/verbose` | Enable verbose debug and interaction trace logging |
| `/quiet` | Reduce internal trace output |
| `/quit` / `/exit` | Cleanly terminate application |

---

## Specialist Catalog

MGA-1 includes 12 built-in specialist roles:
- **Pediatrics** (`mga-pediatrics`)
- **Internal Medicine** (`mga-internal-medicine`)
- **Emergency Medicine** (`mga-emergency-medicine`)
- **Neurology** (`mga-neurology`)
- **Neurosurgery** (`mga-neurosurgery`)
- **Cardiology** (`mga-cardiology`)
- **Oncology** (`mga-oncology`)
- **Pharmacology** (`mga-pharmacology`)
- **Radiology** (`mga-radiology`)
- **Psychiatry** (`mga-psychiatry`)
- **Infectious Disease** (`mga-infectious-disease`)
- **Endocrinology** (`mga-endocrinology`)

---

## Example Demonstration Scenario

```
You:
› My 9-year-old child hit their head yesterday and now has a headache and feels slightly nauseous. What should I do?

[MGA-1] Phase 1/9 — Clinical Intake & Formulation (gemini-3.8-flash)
  ✓ Intake complete
  ✓ Task formulated

[MGA-1] Phase 3/9 — Orchestration (gemini-3.5-flash-lite)
  ✓ Research plan formulated
  → Specialists assigned: Pediatrics, Neurology, Emergency Medicine

[MGA-1] Phase 4/9 — Specialist Swarm (Antigravity Managed Agents)
  ✓ Pediatrics          completed [1.9s]
  ✓ Neurology           completed [1.9s]
  ✓ Emergency Medicine  completed [1.9s]

[MGA-1] Phase 5/9 — Disagreement Analysis (gemini-3.5-flash-lite)
  ⚠ 1 divergence(s) identified between specialist perspectives

[MGA-1] Phase 6/9 — Evidence-Constrained Synthesis (gemini-3.8-flash)
  ✓ Synthesis complete (3 claims traced to evidence)

[MGA-1] Phase 7/9 — Independent Evaluation (gemini-3.8-flash)
  ✓ Evaluation: PASS (Factuality: 95% | Evidence: 93% | Safety: 94%)

╭─ MGA-1 Final Answer ───────────────────────────────────────────────╮
... structured recommendations, PECARN criteria, urgent red flags ...
╰────────────────────────────────────────────────────────────────────╯
```
