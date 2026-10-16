#!/usr/bin/env node
/**
 * MGA-1 v1.0 — Medical General Architecture for Reliable Medical LLMs
 * Research Prototype CLI
 *
 * A model-agnostic, multi-stage agentic reliability architecture designed to
 * study whether external orchestration, independent specialist managed-agent
 * swarms, disagreement detection, evidence-constrained synthesis, and
 * independent semantic evaluation can reduce medical hallucinations,
 * unsupported claims, and unsafe confidence without modifying model weights.
 *
 * IMPORTANT: Research prototype only — not medical advice, not a clinical
 * diagnosis engine, and not a substitute for a physician.
 */

import { GoogleGenAI } from "@google/genai";
import * as readline from "node:readline";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";

// ============================================================================
// 0. ENVIRONMENT LOADING (Minimal zero-dependency .env loader)
// ============================================================================

function loadDotEnv() {
  try {
    const envPath = path.resolve(process.cwd(), ".env");
    if (!fs.existsSync(envPath)) return;
    const raw = fs.readFileSync(envPath, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx <= 0) continue;
      const key = trimmed.slice(0, eqIdx).trim();
      let val = trimmed.slice(eqIdx + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (process.env[key] === undefined) {
        process.env[key] = val;
      }
    }
  } catch {
    // Ignore .env read errors; rely on process.env
  }
}

loadDotEnv();

// ============================================================================
// 1. MODEL REGISTRY & CORE LIMITS
// ============================================================================

const CURRENT_MODEL_REGISTRY = {
  "gemini-4-argon": {
    id: "gemini-4-argon",
    displayName: "Gemini 4 Argon",
    family: "gemini-4",
    contextWindow: "1M tokens",
    description: "Most powerful frontier model for complex reasoning, coding, and research",
  },
  "gemini-3.8-flash": {
    id: "gemini-3.8-flash",
    displayName: "Gemini 3.8 Flash",
    family: "gemini-3.8",
    contextWindow: "1M tokens",
    description: "Fast, balanced performance for agentic and multimodal tasks",
  },
  "gemini-3.7-flash": {
    id: "gemini-3.7-flash",
    displayName: "Gemini 3.7 Flash",
    family: "gemini-3.7",
    contextWindow: "1M tokens",
    description: "High-speed, efficient Flash model for agentic tool use and multi-step execution",
  },
  "gemini-3.6-flash": {
    id: "gemini-3.6-flash",
    displayName: "Gemini 3.6 Flash",
    family: "gemini-3.6",
    contextWindow: "1M tokens",
    description: "Balanced speed and multimodal capabilities across general agentic tasks",
  },
  "gemini-3.5-flash": {
    id: "gemini-3.5-flash",
    displayName: "Gemini 3.5 Flash",
    family: "gemini-3.5",
    contextWindow: "1M tokens",
    description: "Foundational high-throughput Flash model",
  },
  "gemini-3.5-flash-lite": {
    id: "gemini-3.5-flash-lite",
    displayName: "Gemini 3.5 Flash-Lite",
    family: "gemini-3.5",
    contextWindow: "1M tokens",
    description: "Fastest, lowest-cost 3.5 model for high-throughput routing and execution",
  },
  "gemini-3.1-pro-preview": {
    id: "gemini-3.1-pro-preview",
    displayName: "Gemini 3.1 Pro Preview",
    family: "gemini-3.1",
    contextWindow: "1M tokens",
    description: "Deep reasoning and multimodal understanding model",
  },
  "gemini-3.1-flash-lite": {
    id: "gemini-3.1-flash-lite",
    displayName: "Gemini 3.1 Flash-Lite",
    family: "gemini-3.1",
    contextWindow: "1M tokens",
    description: "Cost-efficient, ultra-fast model for lightweight high-frequency tasks",
  },
  "gemini-3.5-transcribe": {
    id: "gemini-3.5-transcribe",
    displayName: "Gemini 3.5 Transcribe",
    family: "gemini-3.5",
    contextWindow: "Audio/Text",
    description: "Fast speech-to-text with smart and verbatim modes",
  },
  "gemini-nano-banana-2.1": {
    id: "gemini-nano-banana-2.1",
    displayName: "Gemini Nano Banana 2.1",
    family: "nano-banana",
    contextWindow: "131k / 32k tokens",
    description: "Default high-efficiency image generation and conversational editing",
  },
  "gemini-3-pro-image": {
    id: "gemini-3-pro-image",
    displayName: "Gemini 3 Pro Image (Nano Banana Pro)",
    family: "nano-banana",
    contextWindow: "65k / 32k tokens",
    description: "High-quality image generation and editing",
  },
  "gemini-3.1-flash-lite-image": {
    id: "gemini-3.1-flash-lite-image",
    displayName: "Gemini 3.1 Flash-Lite Image",
    family: "nano-banana",
    contextWindow: "65k / 32k tokens",
    description: "Ultra-fast image generation and editing",
  },
  "gemini-3.8-flash-tts": {
    id: "gemini-3.8-flash-tts",
    displayName: "Gemini 3.8 Flash TTS",
    family: "gemini-3.8-tts",
    contextWindow: "Text-to-Speech",
    description: "Expressive text-to-speech, multi-speaker dialogue, Voice Design",
  },
  "gemini-3.8-flash-lite-tts": {
    id: "gemini-3.8-flash-lite-tts",
    displayName: "Gemini 3.8 Flash-Lite TTS",
    family: "gemini-3.8-tts",
    contextWindow: "Text-to-Speech",
    description: "Fast, cost-efficient text-to-speech for voice agents",
  },
  "gemini-omni-1.1-flash": {
    id: "gemini-omni-1.1-flash",
    displayName: "Gemini Omni 1.1 Flash",
    family: "gemini-omni",
    contextWindow: "Multimodal Video",
    description: "Video generation, transitions, extensions, and reference-guided generation",
  },
  "gemma-4-31b-it": {
    id: "gemma-4-31b-it",
    displayName: "Gemma 4 31B IT",
    family: "gemma-4",
    contextWindow: "Dense 31B",
    description: "Gemma 4 dense instruction-tuned model",
  },
  "gemma-4-26b-a4b-it": {
    id: "gemma-4-26b-a4b-it",
    displayName: "Gemma 4 26B A4B IT",
    family: "gemma-4",
    contextWindow: "MoE 26B/4B",
    description: "Gemma 4 Mixture-of-Experts instruction-tuned model",
  },
};

const CURRENT_AGENTS_REGISTRY = {
  "antigravity-preview-09-2026": {
    id: "antigravity-preview-09-2026",
    displayName: "Antigravity Managed Agent",
    description: "General-purpose managed agent with code execution, file management, and web access in a remote Linux sandbox",
  },
  "deep-research-preview-04-2026": {
    id: "deep-research-preview-04-2026",
    displayName: "Deep Research Agent",
    description: "Fast, interactive background research agent",
  },
  "deep-research-max-preview-04-2026": {
    id: "deep-research-max-preview-04-2026",
    displayName: "Deep Research Max Agent",
    description: "Maximum exhaustiveness background research agent",
  },
};

const DEPRECATED_MODEL_PATTERNS = [
  /^gemini-1\.0/i,
  /^gemini-1\.5/i,
  /^gemini-2\.0/i,
  /^gemini-2\.5/i,
  /^gemini-pro$/i,
  /^gemini-3\.1-flash-image$/i,
  /^gemini-3\.1-flash-tts-preview$/i,
];

function resolveModelName(envVarName, fallbackModel) {
  const requested = process.env[envVarName]?.trim();
  if (!requested) return fallbackModel;
  for (const pattern of DEPRECATED_MODEL_PATTERNS) {
    if (pattern.test(requested)) {
      process.stderr.write(
        `\x1b[33m⚠ Deprecated model "${requested}" requested in ${envVarName}. Substituting with "gemini-3.8-flash".\x1b[0m\n`
      );
      return "gemini-3.8-flash";
    }
  }
  return requested;
}

const MODELS = {
  base: resolveModelName("MGA_BASE_MODEL", process.env.BASE_MODEL || "gemini-3.8-flash"),
  orchestrator: resolveModelName(
    "MGA_ORCHESTRATOR_MODEL",
    process.env.ORCHESTRATOR_MODEL || "gemini-3.5-flash-lite"
  ),
  summarizer: resolveModelName(
    "MGA_SUMMARIZER_MODEL",
    process.env.SUMMARIZER_MODEL || "gemini-3.8-flash"
  ),
  evaluator: resolveModelName(
    "MGA_EVALUATOR_MODEL",
    process.env.EVALUATOR_MODEL || "gemini-3.8-flash"
  ),
  specialistAgentBase: "antigravity-preview-09-2026",
};

const LIMITS = {
  maxSpecialists: 5,
  maxRevisionIterations: 3,
  maxQaRounds: 5,
};

const TIMEOUTS = {
  geminiTimeoutMs: Number(process.env.GEMINI_TIMEOUT_MS) || 180000,
  agentTimeoutMs: Number(process.env.AGENT_TIMEOUT_MS) || 300000,
};

// FUTURE: MGA-1 Memory Custom Tool
// Commands planned:
// write
// edit
// read
// delete
// This is intentionally not implemented in v1.0.

// ============================================================================
// 2. SPECIALIST ROLE CATALOG
// ============================================================================

/**
 * Generates the standardized MGA-1 specialist system instruction for a custom
 * managed agent extending antigravity-preview-09-2026.
 */
function buildSpecialistSystemInstruction(displayName, specialtyDescription, domainFocus) {
  return `You are the MGA-1 ${displayName} Specialist Agent.
You are a research-oriented specialist role inside a larger medical reliability architecture (MGA-1 v1.0).
Specialty Focus: ${specialtyDescription}
Specific Domain Directives: ${domainFocus}

Your job is NOT to produce the final user-facing answer.
Your job is to independently investigate the assigned medical subproblem using evidence.

Rules:
- Stay strictly within your assigned specialty (${displayName}).
- Use Google Search / URL Context / Code Execution when evidence or clinical calculations are required.
- Distinguish established clinical evidence from uncertainty or low-quality data.
- Do not invent evidence, studies, guidelines, or citations.
- Explicitly identify clinical red flags and urgent warning signs relevant to your specialty.
- Explicitly identify missing patient or contextual information that materially affects risk.
- Explicitly state disagreements or controversies when clinical guidelines or studies conflict.
- Do not claim to be a real licensed physician.
- Do not produce a definitive clinical diagnosis for a real person.
- Return ONLY a strict machine-readable JSON object matching the required MGA-1 Specialist Report schema:
{
  "role": "${displayName}",
  "findings": ["..."],
  "claims": [
    {
      "claim": "Specific substantive clinical or scientific claim",
      "evidence": [
        {
          "title": "Guideline or study title",
          "source": "Journal / Organization / Domain",
          "url": "https://... (if available from search/url tool, otherwise empty string)",
          "summary": "Brief summary of supporting evidence"
        }
      ],
      "confidence": 0.90
    }
  ],
  "redFlags": ["..."],
  "uncertainties": ["..."],
  "disagreements": ["..."],
  "recommendationForSynthesis": "..."
}`;
}

const SPECIALIST_ROLE_CATALOG = {
  pediatrics: {
    id: "pediatrics",
    displayName: "Pediatrics",
    managedAgentId: "mga-pediatrics",
    specialtyDescription:
      "Pediatric medicine, child development, age-specific physiology, pediatric red flags, and developmental risk assessment.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Pediatrics",
      "Pediatric medicine, child and adolescent physiology, age-specific clinical guidelines, and pediatric emergency red flags.",
      "Evaluate age-appropriate symptom presentation, pediatric dosing/safety considerations, guardian observation criteria, and pediatric clinical practice guidelines (e.g., AAP, PECARN where relevant)."
    ),
  },
  internal_medicine: {
    id: "internal_medicine",
    displayName: "Internal Medicine",
    managedAgentId: "mga-internal-medicine",
    specialtyDescription:
      "Adult multisystem disease, differential framework, chronic comorbidities, and general internal medicine synthesis.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Internal Medicine",
      "Adult internal medicine, systemic pathophysiology, multimorbidity, and differential evaluation.",
      "Investigate systemic causes, organ-system interactions, baseline chronic conditions, and evidence-based adult clinical guidelines."
    ),
  },
  emergency_medicine: {
    id: "emergency_medicine",
    displayName: "Emergency Medicine",
    managedAgentId: "mga-emergency-medicine",
    specialtyDescription:
      "Acute triage, time-critical red flags, life/limb-threatening conditions, clinical decision rules, and immediate escalation thresholds.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Emergency Medicine",
      "Acute medical triage, high-acuity red flags, validated clinical decision instruments, and urgent evaluation thresholds.",
      "Focus on ruling out time-sensitive emergencies, identifying red-flag symptoms requiring immediate emergency department evaluation, and applying validated risk stratification criteria."
    ),
  },
  neurology: {
    id: "neurology",
    displayName: "Neurology",
    managedAgentId: "mga-neurology",
    specialtyDescription:
      "Central and peripheral nervous system disorders, concussion/TBI neurology, headache syndromes, focal deficits, and seizure/autonomic phenomena.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Neurology",
      "Clinical neurology, traumatic brain injury/concussion pathophysiology, headache disorders, and neurological examination red flags.",
      "Analyze neurological symptoms, secondary neurological injury mechanisms, concussion guidelines, autonomic/vestibular symptoms, and progressive neurological warning signs."
    ),
  },
  neurosurgery: {
    id: "neurosurgery",
    displayName: "Neurosurgery",
    managedAgentId: "mga-neurosurgery",
    specialtyDescription:
      "Intracranial hemorrhage, elevated intracranial pressure, neurosurgical trauma criteria, spinal cord compression, and neuroimaging thresholds.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Neurosurgery",
      "Neurosurgical emergencies, intracranial mass effect, elevated ICP, neurotrauma imaging criteria, and surgical red flags.",
      "Assess risk factors for intracranial hemorrhage, herniation syndromes, lucid intervals, skull fracture indicators, and evidence-based criteria for urgent CT/MRI neuroimaging."
    ),
  },
  cardiology: {
    id: "cardiology",
    displayName: "Cardiology",
    managedAgentId: "mga-cardiology",
    specialtyDescription:
      "Cardiovascular physiology, ischemic heart disease, arrhythmias, syncope, heart failure, and hemodynamic risk.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Cardiology",
      "Cardiovascular medicine, electrophysiology, ischemic syndromes, syncope, and hemodynamic instability.",
      "Evaluate cardiac risk factors, chest pain/syncope/palpitation differentials, arrhythmia mechanisms, and ACC/AHA/ESC clinical guidelines."
    ),
  },
  oncology: {
    id: "oncology",
    displayName: "Oncology",
    managedAgentId: "mga-oncology",
    specialtyDescription:
      "Hematology/oncology, malignancy red flags, paraneoplastic syndromes, and treatment-related toxicities.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Oncology",
      "Medical oncology and hematology, oncologic emergencies, constitutional red flags, and therapy-associated complications.",
      "Investigate potential neoplastic etiologies, systemic B-symptoms, cytopenias, and evidence-based oncology guidelines."
    ),
  },
  pharmacology: {
    id: "pharmacology",
    displayName: "Pharmacology",
    managedAgentId: "mga-pharmacology",
    specialtyDescription:
      "Clinical pharmacology, drug-drug interactions, adverse drug reactions, contraindications, pharmacokinetics, and toxicology.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Pharmacology",
      "Clinical pharmacology, medication safety, adverse effects, drug interactions, contraindications, and toxicology.",
      "Analyze current medications, OTC analgesic/antiemetic safety, age/renal/hepatic contraindications, medication overuse, and toxicologic risks."
    ),
  },
  radiology: {
    id: "radiology",
    displayName: "Radiology",
    managedAgentId: "mga-radiology",
    specialtyDescription:
      "Diagnostic imaging appropriateness criteria, radiation safety (ALARA), modality selection, and clinical imaging guidelines.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Radiology",
      "Diagnostic radiology, ACR Appropriateness Criteria, imaging modality sensitivity/specificity, and radiation exposure considerations.",
      "Determine when diagnostic imaging is clinically indicated vs. unnecessary, compare CT vs. MRI vs. ultrasound vs. radiography, and evaluate pediatric/adult radiation safety."
    ),
  },
  psychiatry: {
    id: "psychiatry",
    displayName: "Psychiatry",
    managedAgentId: "mga-psychiatry",
    specialtyDescription:
      "Neuropsychiatric symptoms, mood/anxiety/trauma disorders, somatic presentations, delirium vs. primary psychiatric conditions.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Psychiatry",
      "Consultation-liaison and general psychiatry, neurobehavioral symptoms, organic vs. psychiatric differentials, and safety risk.",
      "Evaluate behavioral, affective, cognitive, or sleep changes while ensuring medical/organic causes are excluded first."
    ),
  },
  infectious_disease: {
    id: "infectious_disease",
    displayName: "Infectious Disease",
    managedAgentId: "mga-infectious-disease",
    specialtyDescription:
      "Bacterial, viral, fungal, and parasitic infections, CNS infections, sepsis, epidemiology, and antimicrobial stewardship.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Infectious Disease",
      "Infectious diseases, epidemiological exposures, fever syndromes, meningoencephalitis, and antimicrobial guidance.",
      "Assess infectious etiologies, exposure histories, immunocompromised risks, and IDSA/CDC clinical guidelines."
    ),
  },
  endocrinology: {
    id: "endocrinology",
    displayName: "Endocrinology",
    managedAgentId: "mga-endocrinology",
    specialtyDescription:
      "Endocrine and metabolic disorders, glycemic emergencies, thyroid/adrenal/pituitary dysfunction, and electrolyte disturbances.",
    systemInstruction: buildSpecialistSystemInstruction(
      "Endocrinology",
      "Endocrinology, diabetes, thyroid/adrenal axis disorders, and metabolic homeostasis.",
      "Analyze metabolic, hormonal, and electrolyte contributors to systemic or neurological symptoms."
    ),
  },
};

// ============================================================================
// 3. TERMINAL UI PRIMITIVES & ANSI STYLING
// ============================================================================

const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  italic: "\x1b[3m",
  underline: "\x1b[4m",
  cyan: "\x1b[36m",
  brightCyan: "\x1b[96m",
  green: "\x1b[32m",
  brightGreen: "\x1b[92m",
  yellow: "\x1b[33m",
  brightYellow: "\x1b[93m",
  red: "\x1b[31m",
  brightRed: "\x1b[91m",
  magenta: "\x1b[35m",
  brightMagenta: "\x1b[95m",
  blue: "\x1b[34m",
  brightBlue: "\x1b[94m",
  white: "\x1b[37m",
  gray: "\x1b[90m",
  clearLine: "\x1b[2K\r",
  hideCursor: "\x1b[?25l",
  showCursor: "\x1b[?25h",
};

const SYMBOLS = {
  success: `${ANSI.brightGreen}✓${ANSI.reset}`,
  working: `${ANSI.brightCyan}◌${ANSI.reset}`,
  tool: `${ANSI.brightBlue}↳${ANSI.reset}`,
  warning: `${ANSI.brightYellow}⚠${ANSI.reset}`,
  failure: `${ANSI.brightRed}✕${ANSI.reset}`,
  transition: `${ANSI.brightMagenta}→${ANSI.reset}`,
  bulletActive: `${ANSI.brightCyan}●${ANSI.reset}`,
  bulletWaiting: `${ANSI.gray}○${ANSI.reset}`,
};

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function stripAnsi(str) {
  return String(str || "").replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
}

function getTerminalWidth() {
  const cols = process.stdout?.columns || 80;
  return Math.max(60, Math.min(cols, 100));
}

function wrapText(text, maxWidth) {
  const lines = [];
  for (const rawParagraph of String(text || "").split(/\r?\n/)) {
    if (!rawParagraph.trim()) {
      lines.push("");
      continue;
    }
    const leadingMatch = rawParagraph.match(/^(\s*)/);
    const indent = leadingMatch ? leadingMatch[1] : "";
    const words = rawParagraph.trim().split(/\s+/);
    let currentLine = indent;

    for (const word of words) {
      const candidate = currentLine === indent ? `${indent}${word}` : `${currentLine} ${word}`;
      if (stripAnsi(candidate).length > maxWidth && currentLine !== indent) {
        lines.push(currentLine);
        currentLine = `${indent}${word}`;
      } else {
        currentLine = candidate;
      }
    }
    if (currentLine) lines.push(currentLine);
  }
  return lines;
}

function renderBox(title, bodyLines, options = {}) {
  const borderColor = options.borderColor || ANSI.cyan;
  const titleColor = options.titleColor || ANSI.bold + ANSI.brightCyan;
  const rounded = options.rounded !== false;
  const width = Math.min(options.width || getTerminalWidth() - 4, 88);
  const innerWidth = width - 4;

  const tl = rounded ? "╭" : "┌";
  const tr = rounded ? "╮" : "┐";
  const bl = rounded ? "╰" : "└";
  const br = rounded ? "╯" : "┘";
  const h = "─";
  const v = "│";

  const wrappedLines = [];
  for (const line of bodyLines) {
    wrappedLines.push(...wrapText(line, innerWidth));
  }

  let topBar;
  if (title) {
    const cleanTitle = ` ${title} `;
    const dashCount = Math.max(1, width - 2 - stripAnsi(cleanTitle).length - 1);
    topBar = `${borderColor}${tl}${h}${ANSI.reset}${titleColor}${cleanTitle}${ANSI.reset}${borderColor}${h.repeat(dashCount)}${tr}${ANSI.reset}`;
  } else {
    topBar = `${borderColor}${tl}${h.repeat(width - 2)}${tr}${ANSI.reset}`;
  }

  const middle = wrappedLines.map((line) => {
    const visibleLen = stripAnsi(line).length;
    const pad = Math.max(0, innerWidth - visibleLen);
    return `${borderColor}${v}${ANSI.reset} ${line}${" ".repeat(pad)} ${borderColor}${v}${ANSI.reset}`;
  });

  const bottomBar = `${borderColor}${bl}${h.repeat(width - 2)}${br}${ANSI.reset}`;
  return [topBar, ...middle, bottomBar].join("\n");
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "0.0s";
  return `${(ms / 1000).toFixed(1)}s`;
}

function truncateText(str, maxLen = 110) {
  const clean = String(str || "").replace(/\s+/g, " ").trim();
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, maxLen - 3)}...`;
}

class LiveSpinner {
  constructor() {
    this.timer = null;
    this.frameIdx = 0;
    this.text = "";
    this.subtext = "";
    this.active = false;
  }

  start(text, subtext = "") {
    this.text = text;
    this.subtext = subtext;
    if (!process.stdout.isTTY) {
      console.log(`${SYMBOLS.working} ${text}${subtext ? ` — ${subtext}` : ""}`);
      return;
    }
    this.active = true;
    if (this.timer) clearInterval(this.timer);
    this.render();
    this.timer = setInterval(() => {
      this.frameIdx = (this.frameIdx + 1) % SPINNER_FRAMES.length;
      this.render();
    }, 90);
  }

  update(text, subtext = "") {
    this.text = text;
    this.subtext = subtext;
    if (!process.stdout.isTTY) return;
    if (this.active) this.render();
  }

  render() {
    if (!this.active || !process.stdout.isTTY) return;
    const frame = `${ANSI.brightCyan}${SPINNER_FRAMES[this.frameIdx]}${ANSI.reset}`;
    const sub = this.subtext ? ` ${ANSI.dim}${truncateText(this.subtext, 65)}${ANSI.reset}` : "";
    process.stdout.write(`${ANSI.clearLine}${frame} ${this.text}${sub}`);
  }

  stop(finalLine = null) {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.active && process.stdout.isTTY) {
      process.stdout.write(ANSI.clearLine);
    }
    this.active = false;
    if (finalLine) {
      console.log(finalLine);
    }
  }
}

const activeSpinner = new LiveSpinner();

// ============================================================================
// 4. SESSION & PROCESS STATE
// ============================================================================

/**
 * In-memory registry of custom managed specialist agents created or verified
 * in the Google project during this process lifetime.
 */
const managedAgentRegistry = new Map();

/**
 * Global CLI configuration flags.
 */
const cliConfig = {
  verbose: false,
  activeAbortController: null,
  activeStreams: new Set(),
  isRunningTask: false,
};

function createInitialSessionState() {
  return {
    sessionId: `mga_${crypto.randomBytes(4).toString("hex")}`,
    phase: "idle",
    originalPrompt: null,
    qa: {
      interactionId: null,
      round: 0,
      complete: false,
      contextSummary: null,
      historyTranscript: [],
    },
    taskSpec: null,
    evidence: [],
    researchPlan: null,
    specialists: [],
    disagreementReport: null,
    candidateAnswer: null,
    claimProvenance: [],
    evaluation: null,
    revisionIteration: 0,
    metrics: {
      startTimeMs: null,
      endTimeMs: null,
      inputTokens: 0,
      outputTokens: 0,
      thoughtTokens: 0,
      toolUseTokens: 0,
      totalTokens: 0,
      interactionsCount: 0,
    },
  };
}

let state = createInitialSessionState();

function resetSessionState() {
  state = createInitialSessionState();
}

// ============================================================================
// 5. ARCHITECTURAL LOGGING & METRICS ACCUMULATION
// ============================================================================

function debug(eventType, metadata = {}) {
  if (!cliConfig.verbose) return;
  activeSpinner.stop();
  const parts = [`${ANSI.gray}[trace:${eventType}]${ANSI.reset}`];
  if (metadata.phase || state.phase) {
    parts.push(`${ANSI.dim}phase=${metadata.phase || state.phase}${ANSI.reset}`);
  }
  if (metadata.interactionId) {
    parts.push(`${ANSI.dim}id=${metadata.interactionId}${ANSI.reset}`);
  }
  if (metadata.agentId) {
    parts.push(`${ANSI.dim}agent=${metadata.agentId}${ANSI.reset}`);
  }
  if (metadata.model) {
    parts.push(`${ANSI.dim}model=${metadata.model}${ANSI.reset}`);
  }
  if (metadata.elapsedMs !== undefined) {
    parts.push(`${ANSI.dim}elapsed=${formatDuration(metadata.elapsedMs)}${ANSI.reset}`);
  }
  if (metadata.tokens !== undefined) {
    parts.push(`${ANSI.dim}tokens=${metadata.tokens}${ANSI.reset}`);
  }
  if (metadata.detail) {
    parts.push(`${ANSI.dim}${truncateText(metadata.detail, 80)}${ANSI.reset}`);
  }
  console.log(`  ${parts.join("  ")}`);
}

function recordUsageMetrics(usage) {
  if (!usage || typeof usage !== "object") return;
  const input = Number(usage.total_input_tokens ?? usage.input_tokens ?? 0);
  const output = Number(usage.total_output_tokens ?? usage.output_tokens ?? 0);
  const thought = Number(usage.total_thought_tokens ?? usage.thought_tokens ?? 0);
  const toolUse = Number(usage.total_tool_use_tokens ?? usage.tool_use_tokens ?? 0);
  const total = Number(
    usage.total_tokens ?? input + output + thought + toolUse
  );

  if (Number.isFinite(input) && input > 0) state.metrics.inputTokens += input;
  if (Number.isFinite(output) && output > 0) state.metrics.outputTokens += output;
  if (Number.isFinite(thought) && thought > 0) state.metrics.thoughtTokens += thought;
  if (Number.isFinite(toolUse) && toolUse > 0) state.metrics.toolUseTokens += toolUse;
  if (Number.isFinite(total) && total > 0) state.metrics.totalTokens += total;
}

function registerEvidenceItem(item) {
  if (!item || (!item.url && !item.title && !item.summary)) return null;
  const normalizedUrl = (item.url || "").trim();
  const normalizedTitle = (item.title || item.source || "Clinical Reference").trim();
  const existing = state.evidence.find(
    (e) =>
      (normalizedUrl && e.url === normalizedUrl) ||
      (!normalizedUrl && e.title === normalizedTitle && e.summary === item.summary)
  );
  if (existing) return existing.id;

  const id = `E${String(state.evidence.length + 1).padStart(2, "0")}`;
  let domain = item.source || "";
  if (!domain && normalizedUrl) {
    try {
      domain = new URL(normalizedUrl).hostname.replace(/^www\./, "");
    } catch {
      domain = normalizedUrl;
    }
  }
  const entry = {
    id,
    title: normalizedTitle,
    source: domain || "Medical Literature",
    url: normalizedUrl,
    summary: truncateText(item.summary || "", 280),
    origin: item.origin || state.phase,
  };
  state.evidence.push(entry);
  return id;
}

// ============================================================================
// 6. EVENT NORMALIZATION & THOUGHT / TOOL RENDERERS
// ============================================================================

/**
 * Renders API-provided thought summaries cleanly without flooding the terminal
 * or exposing raw private chain-of-thought.
 */
function renderThoughtSummary(summaryText, context = {}) {
  const clean = truncateText(summaryText, 95);
  if (!clean) return;
  if (context.onThoughtSummary) {
    context.onThoughtSummary(clean);
    return;
  }
  if (context.quiet) return;
  activeSpinner.update(`${ANSI.brightCyan}Thinking${ANSI.reset}`, clean);
}

function renderModelDelta(textDelta, context = {}) {
  if (context.onModelDelta) {
    context.onModelDelta(textDelta);
  }
}

function renderToolEvent(stepOrDelta, eventPhase = "start", context = {}) {
  if (!stepOrDelta || typeof stepOrDelta !== "object") return;
  const type = stepOrDelta.type;

  if (type === "google_search_call") {
    const queries = stepOrDelta.arguments?.queries || [];
    const qText = queries.length > 0 ? queries.join(", ") : "medical literature query";
    if (context.onToolEvent) {
      context.onToolEvent("google_search_call", `Search: ${truncateText(qText, 55)}`);
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.tool} ${ANSI.bold}Google Search${ANSI.reset}`);
      console.log(`    ${ANSI.dim}Query: ${truncateText(qText, 75)}${ANSI.reset}`);
    }
  } else if (type === "google_search_result") {
    const results = Array.isArray(stepOrDelta.result) ? stepOrDelta.result : [];
    for (const r of results) {
      if (r?.title || r?.url) {
        registerEvidenceItem({
          title: r.title || "Search Result",
          url: r.url || "",
          summary: r.search_suggestions || "",
          origin: context.label || state.phase,
        });
      }
    }
    if (context.onToolEvent) {
      context.onToolEvent("google_search_result", "Google Search completed");
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.success} Google Search`);
    }
  } else if (type === "url_context_call") {
    const urls = stepOrDelta.arguments?.urls || [];
    const urlStr = urls.length > 0 ? urls[0] : "clinical guideline...";
    if (context.onToolEvent) {
      context.onToolEvent("url_context_call", `URL: ${truncateText(urlStr, 55)}`);
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.tool} ${ANSI.bold}URL Context${ANSI.reset}`);
      console.log(`    ${ANSI.dim}Reading: ${truncateText(urlStr, 75)}${ANSI.reset}`);
    }
  } else if (type === "url_context_result") {
    const results = Array.isArray(stepOrDelta.result) ? stepOrDelta.result : [];
    for (const r of results) {
      if (r?.url) {
        registerEvidenceItem({
          title: "URL Context Source",
          url: r.url,
          summary: `Status: ${r.status || "fetched"}`,
          origin: context.label || state.phase,
        });
      }
    }
    if (context.onToolEvent) {
      context.onToolEvent("url_context_result", "URL Context completed");
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.success} URL Context`);
    }
  } else if (type === "code_execution_call") {
    const lang = stepOrDelta.arguments?.language || "Python";
    if (context.onToolEvent) {
      context.onToolEvent("code_execution_call", `Code Execution (${lang})`);
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.tool} ${ANSI.bold}Code Execution${ANSI.reset}`);
      console.log(`    ${ANSI.dim}Running ${lang} calculation...${ANSI.reset}`);
    }
  } else if (type === "code_execution_result") {
    if (context.onToolEvent) {
      context.onToolEvent("code_execution_result", "Code Execution completed");
      return;
    }
    if (!context.quiet) {
      activeSpinner.stop();
      console.log(`  ${SYMBOLS.success} Code Execution`);
    }
  }
}

function renderStepStart(event, stepState, context = {}) {
  const step = event?.step;
  if (!step) return;
  const stepType = step.type;
  stepState.currentStepType = stepType;

  debug("step.start", {
    detail: `index=${event.index} type=${stepType}`,
    agentId: context.agentId,
    model: context.model,
  });

  if (stepType === "thought") {
    stepState.hadThought = true;
    const initialSummaries = Array.isArray(step.summary) ? step.summary : [];
    for (const s of initialSummaries) {
      if (s?.text) renderThoughtSummary(s.text, context);
    }
    if (initialSummaries.length === 0) {
      if (context.onThoughtStart) {
        context.onThoughtStart();
      } else if (!context.quiet) {
        activeSpinner.start(`${ANSI.brightCyan}Thinking${ANSI.reset}`, context.thinkingLabel || "");
      }
    }
  } else if (
    stepType === "google_search_call" ||
    stepType === "google_search_result" ||
    stepType === "url_context_call" ||
    stepType === "url_context_result" ||
    stepType === "code_execution_call" ||
    stepType === "code_execution_result"
  ) {
    renderToolEvent(step, "start", context);
  }
}

function renderStepStop(event, stepState, context = {}) {
  if (event?.usage) {
    recordUsageMetrics(event.usage);
  } else if (event?.step_usage) {
    recordUsageMetrics(event.step_usage);
  }

  if (stepState.currentStepType === "thought" && stepState.hadThought) {
    stepState.hadThought = false;
    if (context.onThoughtStop) {
      context.onThoughtStop();
    } else if (!context.quiet) {
      activeSpinner.stop(`  ${SYMBOLS.success} Thinking complete`);
    }
  }
  stepState.currentStepType = null;
}

// ============================================================================
// 7. ROBUST JSON & TEXT EXTRACTION HELPERS
// ============================================================================

/**
 * Extracts text and citations from a completed Interaction object or accumulated stream.
 */
function extractInteractionText(interaction, accumulatedText = "") {
  if (interaction && typeof interaction.output_text === "string" && interaction.output_text.trim()) {
    return interaction.output_text;
  }
  if (interaction && Array.isArray(interaction.steps)) {
    const parts = [];
    for (const step of interaction.steps) {
      if (step?.type === "model_output" && Array.isArray(step.content)) {
        for (const item of step.content) {
          if (item?.type === "text" && typeof item.text === "string") {
            parts.push(item.text);
          }
          if (Array.isArray(item?.annotations)) {
            for (const ann of item.annotations) {
              if (ann?.type === "url_citation" && ann.url) {
                registerEvidenceItem({
                  title: ann.title || "Cited Source",
                  url: ann.url,
                  summary: "Inline URL citation from model output",
                });
              }
            }
          }
        }
      }
    }
    if (parts.length > 0) return parts.join("");
  }
  return accumulatedText || "";
}

/**
 * Extracts the first balanced JSON object or array from arbitrary text.
 * Supports:
 * 1. Direct JSON
 * 2. JSON inside fenced markdown blocks (```json ... ```)
 * 3. First balanced { ... } object in prose
 */
function parseRobustJson(rawText) {
  if (!rawText || typeof rawText !== "string") {
    return { ok: false, data: null, mode: "empty", error: "Empty response text" };
  }
  const trimmed = rawText.trim();

  // 1. Direct JSON parse
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object") {
      return { ok: true, data: parsed, mode: "direct", error: null };
    }
  } catch {
    // Continue to fenced block extraction
  }

  // 2. Fenced code block ```json ... ``` or ``` ... ```
  const fenceRegex = /```(?:json)?\s*([\s\S]*?)```/gi;
  let match;
  while ((match = fenceRegex.exec(trimmed)) !== null) {
    const candidate = match[1].trim();
    if (!candidate) continue;
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object") {
        return { ok: true, data: parsed, mode: "fenced", error: null };
      }
    } catch {
      // Try next block
    }
  }

  // 3. First balanced JSON object { ... }
  const firstBrace = trimmed.indexOf("{");
  if (firstBrace !== -1) {
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = firstBrace; i < trimmed.length; i++) {
      const ch = trimmed[i];
      if (escape) {
        escape = false;
        continue;
      }
      if (ch === "\\") {
        if (inString) escape = true;
        continue;
      }
      if (ch === '"') {
        inString = !inString;
        continue;
      }
      if (!inString) {
        if (ch === "{") depth++;
        else if (ch === "}") {
          depth--;
          if (depth === 0) {
            const candidate = trimmed.slice(firstBrace, i + 1);
            try {
              const parsed = JSON.parse(candidate);
              if (parsed && typeof parsed === "object") {
                return { ok: true, data: parsed, mode: "balanced", error: null };
              }
            } catch {
              break;
            }
          }
        }
      }
    }
  }

  return {
    ok: false,
    data: null,
    mode: "failed",
    error: "No valid JSON object found in output",
  };
}

// ============================================================================
// 8. GEMINI INTERACTIONS API ADAPTERS & HELPERS
// ============================================================================

let genAiClient = null;

function initializeGenAiClient() {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    console.error(
      `\n${SYMBOLS.failure} ${ANSI.bold}${ANSI.brightRed}Missing GEMINI_API_KEY${ANSI.reset}`
    );
    console.error(
      `  Please set ${ANSI.bold}GEMINI_API_KEY${ANSI.reset} in your environment or in a ${ANSI.bold}.env${ANSI.reset} file.`
    );
    console.error(`  See ${ANSI.bold}.env.example${ANSI.reset} for configuration details.\n`);
    process.exit(1);
  }
  genAiClient = new GoogleGenAI({ apiKey });
  return genAiClient;
}

/**
 * Formats an API error into a clear, non-credential-leaking diagnostic string.
 */
function formatApiError(err) {
  if (!err) return "Unknown error";
  const msg = String(err.message || err);
  // Ensure we never echo any API key if present in error strings
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  const sanitized = apiKey && apiKey.length > 4 ? msg.split(apiKey).join("[REDACTED]") : msg;
  const status = err.status || err.statusCode;
  if (status === 401 || status === 403 || /api_key_invalid|invalid api key|permission_denied/i.test(sanitized)) {
    return `Authentication/Permission Error (${status || "auth"}): ${sanitized}`;
  }
  if (status === 429 || /rate limit|quota|resource_exhausted/i.test(sanitized)) {
    return `Rate Limit / Quota Exceeded (429): ${sanitized}`;
  }
  if (err.name === "AbortError" || /aborted|timed out/i.test(sanitized)) {
    return `Operation interrupted or timed out: ${sanitized}`;
  }
  return status ? `API Error (${status}): ${sanitized}` : sanitized;
}

/**
 * Checks whether an error was caused by Google Search / grounding tool restrictions
 * on free-tier keys so the model call can gracefully retry without grounding tools.
 */
function isToolUnsupportedError(err) {
  const text = String(err?.message || err?.body || err || "").toLowerCase();
  return (
    text.includes("google_search") ||
    text.includes("url_context") ||
    text.includes("code_execution") ||
    text.includes("search grounding") ||
    text.includes("tool is not supported") ||
    text.includes("tools are not supported") ||
    text.includes("billing") ||
    text.includes("free tier")
  );
}

/**
 * Non-streaming Interactions API helper.
 */
async function createGeminiInteraction(params, context = {}) {
  const startTime = Date.now();
  const timeoutMs = context.timeoutMs || TIMEOUTS.geminiTimeoutMs;
  const requestParams = {
    ...params,
    timeout: timeoutMs,
    signal: cliConfig.activeAbortController?.signal,
  };

  debug("interactions.create", {
    model: params.model,
    agentId: params.agent,
    detail: "unary request",
  });

  const interaction = await genAiClient.interactions.create(requestParams);
  state.metrics.interactionsCount++;
  if (interaction?.usage) {
    recordUsageMetrics(interaction.usage);
  }
  if (Array.isArray(interaction?.steps)) {
    for (const step of interaction.steps) {
      if (
        step?.type === "google_search_result" ||
        step?.type === "url_context_result" ||
        step?.type === "code_execution_result"
      ) {
        renderToolEvent(step, "stop", { ...context, quiet: true });
      }
    }
  }
  debug("interaction.completed", {
    interactionId: interaction?.id,
    model: params.model,
    agentId: params.agent,
    elapsedMs: Date.now() - startTime,
    tokens: interaction?.usage?.total_tokens,
  });
  return interaction;
}

/**
 * Streaming Interactions API helper supporting current event schema:
 * interaction.created -> interaction.status_update -> step.start -> step.delta -> step.stop -> interaction.completed
 */
async function streamGeminiInteraction(params, context = {}) {
  const startTime = Date.now();
  const timeoutMs = context.timeoutMs || TIMEOUTS.geminiTimeoutMs;
  const streamParams = {
    ...params,
    stream: true,
    timeout: timeoutMs,
    signal: cliConfig.activeAbortController?.signal,
  };

  let interactionId = null;
  let finalInteraction = null;
  let accumulatedText = "";
  let accumulatedThought = "";
  const collectedSteps = [];
  const stepState = { currentStepType: null, hadThought: false };

  let stream;
  try {
    stream = await genAiClient.interactions.create(streamParams);
    cliConfig.activeStreams.add(stream);
    state.metrics.interactionsCount++;

    for await (const event of stream) {
      if (!event || typeof event !== "object") continue;
      const eventType = event.event_type;

      if (eventType === "interaction.created") {
        interactionId = event.interaction?.id || interactionId;
        finalInteraction = event.interaction || finalInteraction;
        debug("interaction.created", {
          interactionId,
          model: params.model,
          agentId: params.agent,
        });
        if (context.onCreated) context.onCreated(event.interaction);
      } else if (eventType === "interaction.status_update") {
        interactionId = event.interaction_id || interactionId;
        if (event.usage) recordUsageMetrics(event.usage);
        debug("interaction.status_update", {
          interactionId,
          detail: `status=${event.status}`,
        });
        if (context.onStatusUpdate) context.onStatusUpdate(event.status, event);
      } else if (eventType === "step.start") {
        if (event.step) collectedSteps.push(event.step);
        renderStepStart(event, stepState, {
          ...context,
          model: params.model,
          agentId: params.agent,
        });
      } else if (eventType === "step.delta") {
        const delta = event.delta;
        if (!delta) continue;
        const deltaType = delta.type;

        if (deltaType === "text") {
          const chunk = typeof delta.text === "string" ? delta.text : "";
          accumulatedText += chunk;
          renderModelDelta(chunk, context);
        } else if (deltaType === "thought_summary") {
          const thoughtText =
            typeof delta.content?.text === "string"
              ? delta.content.text
              : typeof delta.text === "string"
              ? delta.text
              : "";
          if (thoughtText) {
            accumulatedThought += thoughtText;
            renderThoughtSummary(thoughtText, context);
          }
        } else if (deltaType === "text_annotation_delta") {
          const ann = delta.annotation;
          if (ann?.type === "url_citation" && ann.url) {
            registerEvidenceItem({
              title: ann.title || "Cited Source",
              url: ann.url,
              summary: "Streamed citation annotation",
            });
          }
        } else if (
          deltaType === "google_search_call" ||
          deltaType === "google_search_result" ||
          deltaType === "url_context_call" ||
          deltaType === "url_context_result" ||
          deltaType === "code_execution_call" ||
          deltaType === "code_execution_result"
        ) {
          renderToolEvent(delta, "delta", context);
        }
      } else if (eventType === "step.stop") {
        renderStepStop(event, stepState, context);
      } else if (eventType === "interaction.completed") {
        if (event.interaction) {
          finalInteraction = event.interaction;
          interactionId = event.interaction.id || interactionId;
          if (event.interaction.usage) {
            recordUsageMetrics(event.interaction.usage);
          }
        }
        debug("interaction.completed", {
          interactionId,
          model: params.model,
          agentId: params.agent,
          elapsedMs: Date.now() - startTime,
          tokens: event.interaction?.usage?.total_tokens,
        });
      } else if (eventType === "error") {
        const errMsg = event.error?.message || "Stream error event received";
        throw new Error(errMsg);
      }
    }
  } finally {
    if (stream) cliConfig.activeStreams.delete(stream);
    if (!context.quiet && activeSpinner.active) {
      activeSpinner.stop();
    }
  }

  const outputText = extractInteractionText(finalInteraction, accumulatedText);
  return {
    id: interactionId || finalInteraction?.id || null,
    status: finalInteraction?.status || "completed",
    output_text: outputText,
    steps: finalInteraction?.steps || collectedSteps,
    usage: finalInteraction?.usage || null,
    environment_id: finalInteraction?.environment_id || null,
    thoughtSummary: accumulatedThought,
    rawInteraction: finalInteraction,
  };
}

/**
 * Continues a stateful multi-turn Gemini interaction using previous_interaction_id.
 * Re-supplies interaction-scoped configuration (system_instruction, tools, generation_config).
 */
async function continueGeminiInteraction(previousInteractionId, params, context = {}) {
  return streamGeminiInteraction(
    {
      ...params,
      previous_interaction_id: previousInteractionId,
    },
    context
  );
}

/**
 * Creates a structured-output Gemini interaction using current Interactions API
 * `response_format` schema, with automatic fallback if tools/schema combinations
 * or free-tier search tools trigger an API error.
 */
async function createStructuredInteraction({
  model,
  input,
  systemInstruction,
  schema,
  tools = null,
  previousInteractionId = null,
  thinkingLabel = "Analyzing...",
  quiet = false,
}) {
  const baseRequest = {
    model,
    input,
    system_instruction: systemInstruction,
    generation_config: {
      thinking_summaries: "auto",
    },
    response_format: schema,
  };

  if (previousInteractionId) {
    baseRequest.previous_interaction_id = previousInteractionId;
  }
  if (Array.isArray(tools) && tools.length > 0) {
    baseRequest.tools = tools;
  }

  let interactionResult = null;
  try {
    interactionResult = await streamGeminiInteraction(baseRequest, {
      thinkingLabel,
      quiet,
    });
  } catch (err) {
    // If tools (e.g., google_search on free tier) or response_format object form caused an error,
    // handle gracefully without crashing the pipeline.
    if (baseRequest.tools && isToolUnsupportedError(err)) {
      if (!quiet) {
        activeSpinner.stop();
        console.log(
          `  ${SYMBOLS.warning} ${ANSI.dim}Grounding tool unavailable on current API tier; continuing with model reasoning.${ANSI.reset}`
        );
      }
      const fallbackReq = { ...baseRequest };
      delete fallbackReq.tools;
      interactionResult = await streamGeminiInteraction(fallbackReq, {
        thinkingLabel,
        quiet,
      });
    } else {
      throw err;
    }
  }

  const rawText = interactionResult.output_text || "";
  const parsed = parseRobustJson(rawText);

  if (parsed.ok) {
    return {
      ...interactionResult,
      data: parsed.data,
      parseMode: parsed.mode,
    };
  }

  // Controlled repair pass if model returned non-JSON text
  debug("structured_output.repair", {
    model,
    detail: "Attempting JSON repair transformation",
  });
  const repairResult = await createGeminiInteraction(
    {
      model: MODELS.orchestrator,
      system_instruction:
        "Convert the following text into a valid JSON object matching the requested schema. Return ONLY valid JSON.",
      input: `Schema:\n${JSON.stringify(schema)}\n\nContent to convert:\n${rawText}`,
      response_format: schema,
    },
    { quiet: true }
  );

  const repairedParsed = parseRobustJson(extractInteractionText(repairResult));
  return {
    ...interactionResult,
    data: repairedParsed.ok ? repairedParsed.data : null,
    parseMode: repairedParsed.ok ? "repaired" : "failed",
  };
}

// ============================================================================
// 9. STAGE 1 & 2: BASE MODEL INTERACTIVE Q&A & EVIDENCE GROUNDING
// ============================================================================

const QA_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    status: {
      type: "string",
      enum: ["needs_user_input", "ready"],
    },
    questions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          question: { type: "string" },
        },
        required: ["id", "question"],
      },
    },
    context_summary: {
      type: "string",
    },
    task_spec_ready: {
      type: "boolean",
    },
    task_spec: {
      type: "object",
      properties: {
        originalPrompt: { type: "string" },
        patientContext: {
          type: "object",
          properties: {
            age: { type: "string" },
            sex: { type: "string" },
            symptoms: { type: "string" },
            duration: { type: "string" },
            history: { type: "string" },
            medications: { type: "string" },
            allergies: { type: "string" },
          },
        },
        medicalDomains: {
          type: "array",
          items: { type: "string" },
        },
        riskLevel: {
          type: "string",
          enum: ["low", "medium", "high", "critical"],
        },
        missingInformation: {
          type: "array",
          items: { type: "string" },
        },
        researchQuestions: {
          type: "array",
          items: { type: "string" },
        },
        evidenceRequirements: {
          type: "array",
          items: { type: "string" },
        },
        specialistRequirements: {
          type: "array",
          items: { type: "string" },
        },
      },
    },
  },
  required: ["status", "questions", "context_summary", "task_spec_ready"],
};

function getBaseModelSystemInstruction(currentRound, maxRounds) {
  return `You are the MGA-1 Base Intake & Problem Formulation Model (${MODELS.base}).
Your role in MGA-1 v1.0 is ONLY to:
1. Understand the user's initial medical question;
2. Gather essential clinical context via polite, targeted interactive Q&A;
3. Use grounding tools (Google Search, URL Context, Code Execution) ONLY when helpful to understand terminology, clinical red-flag categories, or formulate the problem (DO NOT solve the medical problem yourself — the MGA-1 Specialist Swarm will do that after TaskSpec formulation);
4. Produce a normalized TaskSpec once sufficient context is available.

Current Q&A Round: ${currentRound} of maximum ${maxRounds}.

CRITICAL INTAKE RULES:
- On Round 1, unless the user has already provided comprehensive clinical context (age, symptom onset/duration, severity/red flags, relevant history/medications), set "status": "needs_user_input" and "task_spec_ready": false, and ask 3 to 5 concise, medically relevant intake questions grouped in a single batch.
- Ask ONLY what is clinically relevant to the specific scenario (e.g., age, sex if relevant, onset/duration, severity/progression, key red-flag symptoms, relevant diagnoses/medications/allergies, mechanism of injury/exposure).
- NEVER request full name, home address, identity documents, or unnecessary personally identifying information.
- Remind or respect that the user should only share information they are comfortable providing.
- If the user has answered your intake questions (Round >= 2) and you have enough context to formulate the research task, OR if the user declines to provide more details, OR if current round >= ${maxRounds}, set "status": "ready", "task_spec_ready": true, "questions": [], and populate "task_spec" completely.
- Output strictly valid JSON conforming to the response_format schema.`;
}

function normalizeTaskSpec(rawSpec, originalPrompt, contextSummary) {
  const spec = rawSpec && typeof rawSpec === "object" ? rawSpec : {};
  const pc = spec.patientContext && typeof spec.patientContext === "object" ? spec.patientContext : {};
  const validRisks = ["low", "medium", "high", "critical"];
  const risk = validRisks.includes(String(spec.riskLevel || "").toLowerCase())
    ? String(spec.riskLevel).toLowerCase()
    : "medium";

  return {
    originalPrompt: spec.originalPrompt || originalPrompt || "",
    patientContext: {
      age: pc.age || "Not specified",
      sex: pc.sex || "Not specified",
      symptoms: pc.symptoms || contextSummary || originalPrompt || "Not specified",
      duration: pc.duration || "Not specified",
      history: pc.history || "None reported",
      medications: pc.medications || "None reported",
      allergies: pc.allergies || "None reported",
    },
    medicalDomains: Array.isArray(spec.medicalDomains) && spec.medicalDomains.length > 0
      ? spec.medicalDomains.map(String)
      : ["General Medicine"],
    riskLevel: risk,
    missingInformation: Array.isArray(spec.missingInformation)
      ? spec.missingInformation.map(String)
      : [],
    researchQuestions:
      Array.isArray(spec.researchQuestions) && spec.researchQuestions.length > 0
        ? spec.researchQuestions.map(String)
        : [`Evaluate clinical considerations and red flags for: ${originalPrompt}`],
    evidenceRequirements:
      Array.isArray(spec.evidenceRequirements) && spec.evidenceRequirements.length > 0
        ? spec.evidenceRequirements.map(String)
        : ["Current clinical practice guidelines", "Red-flag triage criteria"],
    specialistRequirements:
      Array.isArray(spec.specialistRequirements) && spec.specialistRequirements.length > 0
        ? spec.specialistRequirements.map(String)
        : ["Primary domain specialist assessment", "Acute risk and red-flag evaluation"],
  };
}

/**
 * Executes a single turn of the stateful Base Model Q&A.
 */
async function runBaseModelQaTurn(userMessage) {
  state.phase = "Phase 1 / Intake";
  state.qa.round += 1;

  if (!state.originalPrompt) {
    state.originalPrompt = userMessage;
  }
  state.qa.historyTranscript.push({ role: "user", text: userMessage });

  console.log(`\n${ANSI.bold}${ANSI.brightMagenta}MGA-1:${ANSI.reset}`);
  console.log(`${ANSI.dim}[MGA-1] Phase 1/9 — Problem Formulation & Intake (Round ${state.qa.round}/${LIMITS.maxQaRounds})${ANSI.reset}`);
  activeSpinner.start(
    `${ANSI.brightCyan}Understanding your situation...${ANSI.reset}`,
    "Determining which clinical domains and missing information are relevant..."
  );

  const forceReady = state.qa.round >= LIMITS.maxQaRounds;
  const tools = [
    { type: "google_search" },
    { type: "url_context" },
    { type: "code_execution" },
  ];

  const turnInput = forceReady
    ? `${userMessage}\n\n[SYSTEM NOTE: Maximum Q&A rounds (${LIMITS.maxQaRounds}) reached. Finalize and return status="ready" and task_spec_ready=true now based on available information.]`
    : userMessage;

  let result;
  try {
    result = await createStructuredInteraction({
      model: MODELS.base,
      input: turnInput,
      systemInstruction: getBaseModelSystemInstruction(state.qa.round, LIMITS.maxQaRounds),
      schema: QA_RESPONSE_SCHEMA,
      tools,
      previousInteractionId: state.qa.interactionId,
      thinkingLabel: "Determining which clinical domains and missing information are relevant...",
    });
  } catch (err) {
    // If previous_interaction_id failed (e.g. expired or store issue), fallback with transcript context
    if (state.qa.interactionId) {
      debug("qa.state_fallback", { detail: "Retrying turn with transcript context" });
      const transcriptText = state.qa.historyTranscript
        .map((m) => `${m.role.toUpperCase()}: ${m.text}`)
        .join("\n\n");
      result = await createStructuredInteraction({
        model: MODELS.base,
        input: `Conversation so far:\n${transcriptText}`,
        systemInstruction: getBaseModelSystemInstruction(state.qa.round, LIMITS.maxQaRounds),
        schema: QA_RESPONSE_SCHEMA,
        tools: null,
        previousInteractionId: null,
        thinkingLabel: "Formulating clinical context...",
      });
    } else {
      throw err;
    }
  } finally {
    activeSpinner.stop();
  }

  if (result.id) {
    state.qa.interactionId = result.id;
  }

  const payload = result.data || {};
  state.qa.contextSummary = payload.context_summary || state.qa.contextSummary || userMessage;

  const isReady =
    forceReady ||
    payload.status === "ready" ||
    payload.task_spec_ready === true ||
    !Array.isArray(payload.questions) ||
    payload.questions.length === 0;

  if (!isReady) {
    const questions = payload.questions.slice(0, 6);
    state.qa.historyTranscript.push({
      role: "assistant",
      text: questions.map((q, i) => `${i + 1}. ${q.question}`).join("\n"),
    });

    console.log(`  ${SYMBOLS.success} Context collected`);
    const intakeLines = [
      "Before I research this, I need a little context",
      `${ANSI.dim}(Share only what you are comfortable providing):${ANSI.reset}`,
      "",
    ];
    questions.forEach((q, idx) => {
      intakeLines.push(`${ANSI.bold}${idx + 1}.${ANSI.reset} ${q.question}`);
    });

    console.log("\n" + renderBox("Intake", intakeLines, { rounded: false }));
    return { ready: false };
  }

  // Q&A is complete -> Normalize TaskSpec
  state.qa.complete = true;
  state.taskSpec = normalizeTaskSpec(
    payload.task_spec,
    state.originalPrompt,
    state.qa.contextSummary
  );

  console.log(`  ${SYMBOLS.success} Intake complete`);
  console.log(`  ${SYMBOLS.success} Task formulated (${ANSI.bold}Risk: ${state.taskSpec.riskLevel.toUpperCase()}${ANSI.reset})`);
  console.log(`  ${SYMBOLS.success} Evidence requirements identified`);

  // Stage 2: Evidence Grounding check
  await runBaseEvidenceGrounding();

  return { ready: true, taskSpec: state.taskSpec };
}

/**
 * Stage 2: Evidence Grounding
 * Performs initial problem-formulation grounding and displays any sources gathered.
 */
async function runBaseEvidenceGrounding() {
  state.phase = "Phase 2 / Grounding";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 2/9 — Evidence Grounding${ANSI.reset}`);

  if (state.evidence.length > 0) {
    console.log(`  ${SYMBOLS.success} Grounding metadata captured (${state.evidence.length} source(s))`);
    renderEvidenceSummaryList(state.evidence.slice(0, 4));
    return;
  }

  activeSpinner.start(
    `${ANSI.brightCyan}Grounding clinical terminology & evidence requirements...${ANSI.reset}`,
    state.taskSpec.evidenceRequirements.join("; ")
  );

  try {
    const groundingPrompt = `Identify authoritative clinical practice guideline organizations, validated clinical decision rules, and key literature domains required to evaluate:
Original Question: ${state.originalPrompt}
Patient Context: ${JSON.stringify(state.taskSpec.patientContext)}
Research Questions: ${JSON.stringify(state.taskSpec.researchQuestions)}

Do NOT answer the patient's question. Return a JSON object listing 2 to 4 authoritative reference guidelines or clinical decision rules to anchor the specialist swarm.`;

    const groundingSchema = {
      type: "object",
      properties: {
        references: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              source: { type: "string" },
              url: { type: "string" },
              summary: { type: "string" },
            },
            required: ["title", "source", "summary"],
          },
        },
      },
      required: ["references"],
    };

    const res = await createStructuredInteraction({
      model: MODELS.base,
      input: groundingPrompt,
      systemInstruction:
        "You are the MGA-1 Evidence Grounding module. Identify real, established clinical practice guidelines or validated clinical decision rules relevant to the TaskSpec. Never fabricate URLs.",
      schema: groundingSchema,
      tools: [{ type: "google_search" }, { type: "url_context" }],
      thinkingLabel: "Identifying relevant clinical guidelines and decision rules...",
    });

    if (Array.isArray(res.data?.references)) {
      for (const ref of res.data.references) {
        registerEvidenceItem({
          title: ref.title,
          source: ref.source,
          url: ref.url || "",
          summary: ref.summary,
          origin: "grounding",
        });
      }
    }
  } catch (err) {
    debug("grounding.warning", { detail: formatApiError(err) });
  } finally {
    activeSpinner.stop();
  }

  console.log(`  ${SYMBOLS.success} Grounding baseline established (${state.evidence.length} anchor(s))`);
  if (state.evidence.length > 0) {
    renderEvidenceSummaryList(state.evidence.slice(0, 4));
  }
}

function renderEvidenceSummaryList(evidenceList) {
  if (!Array.isArray(evidenceList) || evidenceList.length === 0) return;
  console.log(`  ${ANSI.bold}Evidence gathered${ANSI.reset}`);
  evidenceList.forEach((item, idx) => {
    const loc = item.url || item.source || "medical reference";
    console.log(`  ${idx + 1}. ${ANSI.brightWhite || ANSI.white}${truncateText(item.title, 68)}${ANSI.reset}`);
    console.log(`     ${ANSI.dim}${truncateText(loc, 72)}${ANSI.reset}`);
  });
}

// ============================================================================
// 10. STAGE 3: MGA-1 ORCHESTRATOR
// ============================================================================

const ORCHESTRATOR_SCHEMA = {
  type: "object",
  properties: {
    riskLevel: {
      type: "string",
      enum: ["low", "medium", "high", "critical"],
    },
    researchObjective: {
      type: "string",
    },
    specialists: {
      type: "array",
      items: {
        type: "object",
        properties: {
          roleId: {
            type: "string",
            enum: Object.keys(SPECIALIST_ROLE_CATALOG),
          },
          displayName: { type: "string" },
          task: { type: "string" },
          priority: { type: "integer" },
          required: { type: "boolean" },
          action: {
            type: "string",
            enum: ["assign", "keep", "add", "replace"],
          },
        },
        required: ["roleId", "displayName", "task", "priority", "required"],
      },
    },
    removedSpecialists: {
      type: "array",
      items: { type: "string" },
    },
    additionalEvidenceNeeded: {
      type: "array",
      items: { type: "string" },
    },
    reasoning: {
      type: "string",
    },
  },
  required: ["riskLevel", "researchObjective", "specialists", "additionalEvidenceNeeded", "reasoning"],
};

function validateAndNormalizeResearchPlan(rawPlan) {
  const allowedRoleIds = Object.keys(SPECIALIST_ROLE_CATALOG);
  const plan = rawPlan && typeof rawPlan === "object" ? rawPlan : {};

  const validRisks = ["low", "medium", "high", "critical"];
  const riskLevel = validRisks.includes(String(plan.riskLevel || "").toLowerCase())
    ? String(plan.riskLevel).toLowerCase()
    : state.taskSpec?.riskLevel || "medium";

  const rawSpecs = Array.isArray(plan.specialists) ? plan.specialists : [];
  const dedupedSpecs = [];
  const seenRoles = new Set();

  for (const s of rawSpecs) {
    if (!s || typeof s !== "object") continue;
    const roleId = String(s.roleId || "").toLowerCase().trim();
    if (!allowedRoleIds.includes(roleId)) {
      debug("orchestrator.rejected_unknown_role", { detail: `Unknown roleId: ${roleId}` });
      continue;
    }
    if (seenRoles.has(roleId)) continue;
    seenRoles.add(roleId);

    const catalogEntry = SPECIALIST_ROLE_CATALOG[roleId];
    dedupedSpecs.push({
      roleId,
      displayName: catalogEntry.displayName,
      task:
        s.task && String(s.task).trim()
          ? String(s.task).trim()
          : `Analyze the case from the ${catalogEntry.displayName} perspective, focusing on evidence-based risk stratification and red flags.`,
      priority: Number.isFinite(Number(s.priority)) ? Number(s.priority) : dedupedSpecs.length + 1,
      required: s.required !== false,
      action: s.action || "assign",
    });

    if (dedupedSpecs.length >= LIMITS.maxSpecialists) break;
  }

  // Ensure at least 1 valid specialist is assigned
  if (dedupedSpecs.length === 0) {
    const fallbackRoles = ["internal_medicine", "emergency_medicine"];
    for (const rId of fallbackRoles) {
      const cat = SPECIALIST_ROLE_CATALOG[rId];
      dedupedSpecs.push({
        roleId: rId,
        displayName: cat.displayName,
        task: `Investigate the clinical presentation and identify evidence-based management and red flags from a ${cat.displayName} perspective.`,
        priority: dedupedSpecs.length + 1,
        required: true,
        action: "assign",
      });
    }
  }

  return {
    riskLevel,
    researchObjective:
      plan.researchObjective ||
      `Establish an evidence-supported multi-specialty assessment for: ${state.originalPrompt}`,
    specialists: dedupedSpecs.slice(0, LIMITS.maxSpecialists),
    removedSpecialists: Array.isArray(plan.removedSpecialists) ? plan.removedSpecialists : [],
    additionalEvidenceNeeded: Array.isArray(plan.additionalEvidenceNeeded)
      ? plan.additionalEvidenceNeeded.map(String)
      : [],
    reasoning: plan.reasoning || "Selected smallest sufficient specialist swarm for independent verification.",
  };
}

/**
 * Runs the MGA-1 Orchestrator (initial planning or revision re-planning).
 */
async function runOrchestrator(isRevision = false) {
  state.phase = isRevision ? `Phase 8 / Revision (Iteration ${state.revisionIteration})` : "Phase 3 / Orchestration";
  const phaseLabel = isRevision
    ? `[MGA-1] Phase 8/9 — Revision Orchestration (Cycle ${state.revisionIteration}/${LIMITS.maxRevisionIterations})`
    : `[MGA-1] Phase 3/9 — Orchestration`;
  console.log(`\n${ANSI.dim}${phaseLabel}${ANSI.reset}`);

  const catalogSummary = Object.values(SPECIALIST_ROLE_CATALOG)
    .map((r) => `- ${r.id} (${r.displayName}): ${r.specialtyDescription}`)
    .join("\n");

  const systemInstruction = `You are the MGA-1 Orchestrator (${MODELS.orchestrator}).
You do NOT directly solve the medical problem. Your responsibility is routing, swarm planning, and revision control.

Available Specialist Role Catalog (choose ONLY from these roleIds):
${catalogSummary}

Rules:
- Select between 1 and ${LIMITS.maxSpecialists} specialists (${LIMITS.maxSpecialists} maximum).
- Prefer the smallest sufficient swarm that provides independent verification across the relevant clinical domains. Do NOT blindly select all 5 specialists on every case.
- Give each selected specialist a concrete, non-overlapping research task tailored to their specialty.
- If this is a Revision Cycle (Evaluator returned FAIL), examine the Evaluator's revisionPlan, unsupportedClaims, and missingCriticalInformation:
  - Decide whether to keep existing valid specialist reports, add a missing specialty, remove an irrelevant/failed specialist, refine a specialist's task, or request targeted additional evidence.
  - Do NOT blindly rerun all specialists if only specific domains or claims need new research.`;

  const inputPayload = {
    originalPrompt: state.originalPrompt,
    taskSpec: state.taskSpec,
    qaContextSummary: state.qa.contextSummary,
    groundingEvidence: state.evidence,
    iterationNumber: state.revisionIteration,
  };

  if (isRevision) {
    inputPayload.previousResearchPlan = state.researchPlan;
    inputPayload.previousSpecialistReportsSummary = state.specialists.map((s) => ({
      roleId: s.roleId,
      status: s.status,
      findingsCount: s.report?.findings?.length || 0,
      uncertainties: s.report?.uncertainties || [],
    }));
    inputPayload.previousCandidateAnswer = state.candidateAnswer?.answer || null;
    inputPayload.evaluatorFeedback = state.evaluation;
    inputPayload.revisionPlan = state.evaluation?.revisionPlan || [];
  }

  activeSpinner.start(
    `${ANSI.brightCyan}Orchestrating specialist swarm...${ANSI.reset}`,
    isRevision ? "Adapting research plan from Evaluator feedback..." : "Routing clinical subproblems..."
  );

  let result;
  try {
    result = await createStructuredInteraction({
      model: MODELS.orchestrator,
      input: JSON.stringify(inputPayload, null, 2),
      systemInstruction,
      schema: ORCHESTRATOR_SCHEMA,
      thinkingLabel: "Selecting optimal specialist swarm and assigning tasks...",
    });
  } finally {
    activeSpinner.stop();
  }

  const normalizedPlan = validateAndNormalizeResearchPlan(result.data);
  state.researchPlan = normalizedPlan;

  console.log(`  ${SYMBOLS.success} Research plan formulated (${ANSI.bold}${normalizedPlan.specialists.length} specialist(s)${ANSI.reset})`);
  console.log(`  ${ANSI.dim}Objective: ${truncateText(normalizedPlan.researchObjective, 82)}${ANSI.reset}`);
  for (const spec of normalizedPlan.specialists) {
    const badge = isRevision && spec.action ? ` [${spec.action}]` : "";
    console.log(
      `    ${SYMBOLS.transition} ${ANSI.bold}${spec.displayName}${ANSI.reset}${ANSI.dim}${badge}: ${truncateText(spec.task, 66)}${ANSI.reset}`
    );
  }

  return normalizedPlan;
}

// ============================================================================
// 11. STAGE 4: CUSTOM MANAGED SPECIALIST AGENTS & PARALLEL SWARM EXECUTION
// ============================================================================

/**
 * Lazily ensures a reusable custom managed specialist agent exists in the Google project
 * based on `antigravity-preview-09-2026`.
 *
 * Behavior:
 * 1. Check in-memory registry first.
 * 2. Check remote project via `client.agents.get(id)`.
 * 3. If not found, create via `client.agents.create(...)` with `base_agent: "antigravity-preview-09-2026"`.
 * 4. Handle "already exists" (409) cleanly.
 */
async function ensureSpecialistAgent(roleId) {
  const role = SPECIALIST_ROLE_CATALOG[roleId];
  if (!role) {
    throw new Error(`Unknown specialist roleId: "${roleId}"`);
  }

  const agentId = role.managedAgentId;

  // 1. Check in-memory process cache
  if (managedAgentRegistry.has(agentId)) {
    const cached = managedAgentRegistry.get(agentId);
    if (cached.registered) {
      debug("agent.reuse_cached", { agentId, detail: `Reusing cached managed agent ${agentId}` });
      return agentId;
    }
  }

  // 2. Check if custom managed agent already exists in Google project
  try {
    const existing = await genAiClient.agents.get(agentId);
    if (existing && (existing.id || existing.name)) {
      managedAgentRegistry.set(agentId, {
        id: agentId,
        roleId,
        displayName: role.displayName,
        baseAgent: MODELS.specialistAgentBase,
        registered: true,
        status: "reused",
        updatedAt: new Date().toISOString(),
      });
      debug("agent.reuse_remote", { agentId, detail: "Found existing agent in Google project" });
      return agentId;
    }
  } catch (err) {
    const status = err?.status || err?.statusCode;
    // 404 Not Found is expected when the agent hasn't been created yet
    if (status && status !== 404 && status !== 400) {
      debug("agent.get_check", { agentId, detail: formatApiError(err) });
    }
  }

  // 3. Create reusable custom managed agent lazily
  try {
    const created = await genAiClient.agents.create({
      id: agentId,
      base_agent: MODELS.specialistAgentBase,
      description: `MGA-1 ${role.displayName} Specialist Research Agent (${role.specialtyDescription})`,
      system_instruction: role.systemInstruction,
      agent_config: {
        type: "antigravity",
        model: MODELS.base,
      },
      tools: [
        { type: "google_search" },
        { type: "url_context" },
        { type: "code_execution" },
      ],
    });

    const resolvedId = created?.id || agentId;
    managedAgentRegistry.set(agentId, {
      id: resolvedId,
      roleId,
      displayName: role.displayName,
      baseAgent: MODELS.specialistAgentBase,
      registered: true,
      status: "created",
      updatedAt: new Date().toISOString(),
    });
    debug("agent.created", { agentId: resolvedId, detail: `Created custom managed agent ${resolvedId}` });
    return resolvedId;
  } catch (err) {
    const status = err?.status || err?.statusCode;
    const msg = String(err?.message || err || "").toLowerCase();

    // Handle 409 Already Exists cleanly
    if (status === 409 || msg.includes("already exists") || msg.includes("already_exists")) {
      managedAgentRegistry.set(agentId, {
        id: agentId,
        roleId,
        displayName: role.displayName,
        baseAgent: MODELS.specialistAgentBase,
        registered: true,
        status: "reused",
        updatedAt: new Date().toISOString(),
      });
      debug("agent.already_exists", { agentId, detail: "Agent already exists (409), reusing" });
      return agentId;
    }

    // Record registration attempt status and rethrow so caller can handle
    managedAgentRegistry.set(agentId, {
      id: agentId,
      roleId,
      displayName: role.displayName,
      baseAgent: MODELS.specialistAgentBase,
      registered: false,
      status: "registration_failed",
      error: formatApiError(err),
      updatedAt: new Date().toISOString(),
    });
    throw err;
  }
}

/**
 * Live terminal dashboard for the concurrent MGA-1 Specialist Swarm.
 */
class SpecialistSwarmDashboard {
  constructor(assignedSpecialists) {
    const assignedIds = new Set(assignedSpecialists.map((s) => s.roleId));
    this.rows = [];

    for (const s of assignedSpecialists) {
      this.rows.push({
        roleId: s.roleId,
        displayName: s.displayName,
        assigned: true,
        state: "waiting", // waiting | working | completed | failed
        activity: "waiting to launch...",
        thought: "",
        startTimeMs: Date.now(),
        elapsedMs: 0,
      });
    }

    // Show up to 5 slots in dashboard (include 1-2 unassigned catalog roles as "not assigned" if < 5 assigned)
    if (this.rows.length < LIMITS.maxSpecialists) {
      for (const role of Object.values(SPECIALIST_ROLE_CATALOG)) {
        if (this.rows.length >= LIMITS.maxSpecialists) break;
        if (!assignedIds.has(role.id)) {
          this.rows.push({
            roleId: role.id,
            displayName: role.displayName,
            assigned: false,
            state: "unassigned",
            activity: "not assigned",
            thought: "",
            startTimeMs: null,
            elapsedMs: 0,
          });
        }
      }
    }

    this.timer = null;
    this.frameIdx = 0;
    this.renderedLineCount = 0;
    this.isTTY = Boolean(process.stdout.isTTY) && !cliConfig.verbose;
  }

  start() {
    if (this.isTTY) {
      process.stdout.write(ANSI.hideCursor);
      this.render();
      this.timer = setInterval(() => {
        this.frameIdx = (this.frameIdx + 1) % SPINNER_FRAMES.length;
        for (const r of this.rows) {
          if (r.assigned && (r.state === "working" || r.state === "waiting")) {
            r.elapsedMs = Date.now() - r.startTimeMs;
          }
        }
        this.render();
      }, 150);
    } else {
      console.log(`  ${SYMBOLS.working} Launching ${this.rows.filter((r) => r.assigned).length} specialist agent(s)...`);
    }
  }

  updateSpecialist(roleId, patch) {
    const row = this.rows.find((r) => r.roleId === roleId);
    if (!row) return;
    Object.assign(row, patch);
    if (row.startTimeMs) {
      row.elapsedMs = Date.now() - row.startTimeMs;
    }
    if (!this.isTTY && patch.state && (patch.state === "completed" || patch.state === "failed")) {
      const sym = patch.state === "completed" ? SYMBOLS.success : SYMBOLS.failure;
      console.log(`  ${sym} ${row.displayName.padEnd(20)} (${formatDuration(row.elapsedMs)}) — ${row.activity}`);
    } else if (this.isTTY) {
      this.render();
    }
  }

  render() {
    if (!this.isTTY) return;
    const bodyLines = [""];
    const spinnerChar = SPINNER_FRAMES[this.frameIdx];

    for (const r of this.rows) {
      const nameCol = r.displayName.padEnd(19, " ");
      if (!r.assigned) {
        bodyLines.push(`${SYMBOLS.bulletWaiting} ${ANSI.dim}${nameCol} not assigned${ANSI.reset}`);
        continue;
      }

      const elapsed = `${ANSI.dim}(${formatDuration(r.elapsedMs)})${ANSI.reset}`;
      let icon = SYMBOLS.bulletWaiting;
      let statusStr = r.activity;

      if (r.state === "working") {
        icon = `${ANSI.brightCyan}${spinnerChar}${ANSI.reset}`;
        statusStr = `${ANSI.cyan}${truncateText(r.activity, 36)}${ANSI.reset} ${elapsed}`;
      } else if (r.state === "completed") {
        icon = SYMBOLS.success;
        statusStr = `${ANSI.green}${truncateText(r.activity || "research completed", 36)}${ANSI.reset} ${elapsed}`;
      } else if (r.state === "failed") {
        icon = SYMBOLS.failure;
        statusStr = `${ANSI.red}${truncateText(r.activity || "failed", 36)}${ANSI.reset} ${elapsed}`;
      } else {
        icon = SYMBOLS.bulletActive;
        statusStr = `${ANSI.dim}${truncateText(r.activity, 36)}${ANSI.reset}`;
      }

      bodyLines.push(`${icon} ${ANSI.bold}${nameCol}${ANSI.reset} ${statusStr}`);
      if (r.thought && r.state === "working") {
        bodyLines.push(`  ${SYMBOLS.working} ${ANSI.dim}Thinking: ${truncateText(r.thought, 52)}${ANSI.reset}`);
      }
    }

    const waitingList = this.rows
      .filter((r) => r.assigned && (r.state === "working" || r.state === "waiting"))
      .map((r) => r.displayName);

    bodyLines.push("");
    if (waitingList.length > 0) {
      bodyLines.push(`${ANSI.dim}Waiting for: ${waitingList.join(", ")}${ANSI.reset}`);
    } else {
      bodyLines.push(`${SYMBOLS.success} ${ANSI.green}All assigned specialists finished.${ANSI.reset}`);
    }

    const boxStr = renderBox("MGA-1 Specialist Swarm", bodyLines, { width: 78 });
    const lines = boxStr.split("\n");

    if (this.renderedLineCount > 0) {
      process.stdout.write(`\x1b[${this.renderedLineCount}A`);
    }
    for (const line of lines) {
      process.stdout.write(`${ANSI.clearLine}${line}\n`);
    }
    this.renderedLineCount = lines.length;
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.isTTY) {
      this.render();
      process.stdout.write(ANSI.showCursor);
    }
  }
}

/**
 * Builds the isolated research prompt for an individual specialist agent.
 * Deliberately excludes other specialists' outputs on the initial pass to
 * prevent correlated reasoning / groupthink.
 */
function buildSpecialistTaskPrompt(specialistAssignment, isRevision = false) {
  const evidenceSummary =
    state.evidence.length > 0
      ? state.evidence
          .map((e) => `[${e.id}] ${e.title} (${e.source})${e.url ? ` - ${e.url}` : ""}: ${e.summary}`)
          .join("\n")
      : "No prior evidence anchors collected yet.";

  let revisionContext = "";
  if (isRevision && state.evaluation) {
    revisionContext = `
Revision Context (Iteration ${state.revisionIteration}):
Evaluator identified the following issues to resolve:
- Missing Critical Information: ${JSON.stringify(state.evaluation.missingCriticalInformation || [])}
- Unsupported Claims to verify/remove: ${JSON.stringify(state.evaluation.unsupportedClaims || [])}
- Revision Plan Directives: ${JSON.stringify(state.evaluation.revisionPlan || [])}
`;
  }

  return `MGA-1 SPECIALIST RESEARCH TASK
Role:
${specialistAssignment.displayName}

Original medical question:
${state.originalPrompt}

Patient/context information:
${JSON.stringify(state.taskSpec?.patientContext || {}, null, 2)}

TaskSpec:
${JSON.stringify(
  {
    medicalDomains: state.taskSpec?.medicalDomains,
    riskLevel: state.taskSpec?.riskLevel,
    missingInformation: state.taskSpec?.missingInformation,
    researchQuestions: state.taskSpec?.researchQuestions,
    evidenceRequirements: state.taskSpec?.evidenceRequirements,
  },
  null,
  2
)}

Research objective:
${specialistAssignment.task}
${revisionContext}
Evidence already gathered:
${evidenceSummary}

Instructions:
Investigate ONLY your assigned specialty perspective (${specialistAssignment.displayName}).
Use web research when required.
Distinguish established facts from uncertainty.
Identify red flags.
Identify unsupported assumptions.
Return the result in the required report format.
Do not attempt to write the final user-facing answer.

Required JSON Report Format:
{
  "role": "${specialistAssignment.displayName}",
  "findings": ["Detailed clinical finding 1", "Detailed clinical finding 2"],
  "claims": [
    {
      "claim": "Specific clinical claim supported by evidence",
      "evidence": [
        {
          "title": "Guideline or study title",
          "source": "Organization or journal",
          "url": "https://...",
          "summary": "Key supporting detail"
        }
      ],
      "confidence": 0.91
    }
  ],
  "redFlags": ["Specific urgent warning sign 1"],
  "uncertainties": ["Specific unknown or missing clinical factor"],
  "disagreements": ["Areas where clinical evidence or guidelines vary"],
  "recommendationForSynthesis": "Key takeaway for the MGA-1 Summarizer"
}`;
}

function normalizeSpecialistReport(rawObj, displayName, rawText = "", requiredRepair = false) {
  const obj = rawObj && typeof rawObj === "object" ? rawObj : {};
  const claims = Array.isArray(obj.claims)
    ? obj.claims.map((c) => {
        const evList = Array.isArray(c?.evidence) ? c.evidence : [];
        const evidenceIds = [];
        for (const ev of evList) {
          if (typeof ev === "string") {
            const id = registerEvidenceItem({
              title: ev,
              source: displayName,
              summary: ev,
              origin: displayName,
            });
            if (id) evidenceIds.push(id);
          } else if (ev && typeof ev === "object") {
            const id = registerEvidenceItem({
              title: ev.title || ev.source || `${displayName} Reference`,
              source: ev.source || displayName,
              url: ev.url || "",
              summary: ev.summary || c.claim || "",
              origin: displayName,
            });
            if (id) evidenceIds.push(id);
          }
        }
        const conf = Number(c?.confidence);
        return {
          claim: String(c?.claim || "").trim(),
          evidence: evList,
          evidenceIds,
          confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.75,
        };
      }).filter((c) => c.claim)
    : [];

  return {
    role: obj.role || displayName,
    findings: Array.isArray(obj.findings) ? obj.findings.map(String) : [],
    claims,
    redFlags: Array.isArray(obj.redFlags) ? obj.redFlags.map(String) : [],
    uncertainties: Array.isArray(obj.uncertainties) ? obj.uncertainties.map(String) : [],
    disagreements: Array.isArray(obj.disagreements) ? obj.disagreements.map(String) : [],
    recommendationForSynthesis: String(obj.recommendationForSynthesis || ""),
    unstructured: !rawObj,
    requiredRepair,
    rawText: rawText || "",
  };
}

/**
 * Runs a single Custom Managed Specialist Agent based on `antigravity-preview-09-2026`.
 * Important:
 * - Uses `environment: "remote"` as required by Managed Agents.
 * - Does NOT send unsupported `response_format` or `generation_config` to Antigravity.
 * - Parses JSON from text output with controlled fallback repair if needed.
 */
async function runSpecialistAgent(assignment, dashboard, isRevision = false) {
  const { roleId, displayName } = assignment;
  const startTime = Date.now();

  dashboard.updateSpecialist(roleId, {
    state: "working",
    activity: "ensuring managed agent...",
    startTimeMs: startTime,
  });

  let managedAgentId;
  try {
    managedAgentId = await ensureSpecialistAgent(roleId);
  } catch (err) {
    // If custom agent creation fails (e.g., quota/tier restriction), record clearly and attempt direct Antigravity invocation
    debug("specialist.ensure_failed", {
      agentId: SPECIALIST_ROLE_CATALOG[roleId]?.managedAgentId,
      detail: formatApiError(err),
    });
    managedAgentId = MODELS.specialistAgentBase;
  }

  dashboard.updateSpecialist(roleId, {
    state: "working",
    activity: "provisioning remote sandbox...",
    thought: "Reasoning...",
  });

  const taskPrompt = buildSpecialistTaskPrompt(assignment, isRevision);

  let rawOutputText = "";
  let environmentId = null;

  try {
    // Invoke the custom managed agent in a remote sandbox environment
    // Do NOT pass response_format or generation_config (unsupported on Antigravity)
    const agentInteractionParams = {
      agent: managedAgentId,
      input:
        managedAgentId === MODELS.specialistAgentBase
          ? `${SPECIALIST_ROLE_CATALOG[roleId].systemInstruction}\n\n${taskPrompt}`
          : taskPrompt,
      environment: "remote",
    };

    const streamRes = await streamGeminiInteraction(agentInteractionParams, {
      quiet: true,
      timeoutMs: TIMEOUTS.agentTimeoutMs,
      agentId: managedAgentId,
      onCreated: (inter) => {
        if (inter?.environment_id) environmentId = inter.environment_id;
        dashboard.updateSpecialist(roleId, {
          activity: "researching...",
        });
      },
      onStatusUpdate: (status) => {
        if (status === "queued") {
          dashboard.updateSpecialist(roleId, { activity: "queued in sandbox..." });
        } else if (status === "in_progress") {
          dashboard.updateSpecialist(roleId, { activity: "analyzing clinical evidence..." });
        }
      },
      onThoughtStart: () => {
        dashboard.updateSpecialist(roleId, {
          thought: "evaluating clinical evidence & red flags...",
        });
      },
      onThoughtSummary: (summary) => {
        dashboard.updateSpecialist(roleId, {
          thought: summary,
          activity: "evaluating specialty findings...",
        });
      },
      onThoughtStop: () => {
        dashboard.updateSpecialist(roleId, {
          thought: "",
        });
      },
      onToolEvent: (toolType, label) => {
        dashboard.updateSpecialist(roleId, {
          activity: label,
        });
      },
    });

    // If the managed agent ran asynchronously or returned in_progress/queued, poll until completion
    let finalRes = streamRes;
    if (
      (finalRes.status === "in_progress" || finalRes.status === "queued") &&
      finalRes.id
    ) {
      const pollStart = Date.now();
      while (Date.now() - pollStart < TIMEOUTS.agentTimeoutMs) {
        await new Promise((r) => setTimeout(r, 4000));
        const polled = await genAiClient.interactions.get(finalRes.id, {
          timeout: TIMEOUTS.geminiTimeoutMs,
          signal: cliConfig.activeAbortController?.signal,
        });
        if (polled?.usage) recordUsageMetrics(polled.usage);
        if (polled?.status === "completed") {
          finalRes = {
            ...finalRes,
            status: "completed",
            output_text: extractInteractionText(polled),
            environment_id: polled.environment_id || environmentId,
          };
          break;
        } else if (polled?.status === "failed" || polled?.status === "cancelled") {
          throw new Error(`Managed agent interaction ${polled.status}`);
        }
      }
    }

    rawOutputText = finalRes.output_text || "";
    environmentId = finalRes.environment_id || environmentId;
  } catch (agentErr) {
    // If remote Antigravity sandbox is unavailable on the user's free-tier API key,
    // record the managed-agent error transparently and execute a controlled specialist
    // research fallback so the prototype remains testable end-to-end when desired,
    // OR if the error is a hard abort, fail the specialist cleanly.
    if (cliConfig.activeAbortController?.signal?.aborted) {
      throw agentErr;
    }

    debug("specialist.managed_agent_fallback", {
      agentId: managedAgentId,
      detail: formatApiError(agentErr),
    });

    dashboard.updateSpecialist(roleId, {
      activity: "running fallback specialist analysis...",
      thought: "Managed agent sandbox unavailable; using specialist model adapter...",
    });

    const fallbackRes = await streamGeminiInteraction(
      {
        model: MODELS.base,
        system_instruction: SPECIALIST_ROLE_CATALOG[roleId].systemInstruction,
        input: taskPrompt,
        generation_config: {
          thinking_summaries: "auto",
        },
      },
      {
        quiet: true,
        onThoughtSummary: (summary) => {
          dashboard.updateSpecialist(roleId, { thought: summary });
        },
        onToolEvent: (_t, label) => {
          dashboard.updateSpecialist(roleId, { activity: label });
        },
      }
    );

    rawOutputText = fallbackRes.output_text || "";
  }

  // Parse the specialist report using the 3-tier robust parser
  dashboard.updateSpecialist(roleId, {
    activity: "validating specialist report...",
    thought: "",
  });

  const parsed = parseRobustJson(rawOutputText);
  let report;

  if (parsed.ok) {
    report = normalizeSpecialistReport(parsed.data, displayName, rawOutputText, false);
  } else {
    // Attempt controlled fallback transformation using a normal Gemini model
    debug("specialist.output_repair", {
      agentId: managedAgentId,
      detail: `Specialist ${displayName} returned unstructured output; attempting repair`,
    });

    try {
      const repairSchema = {
        type: "object",
        properties: {
          role: { type: "string" },
          findings: { type: "array", items: { type: "string" } },
          claims: {
            type: "array",
            items: {
              type: "object",
              properties: {
                claim: { type: "string" },
                evidence: {
                  type: "array",
                  items: {
                    type: "object",
                    properties: {
                      title: { type: "string" },
                      source: { type: "string" },
                      url: { type: "string" },
                      summary: { type: "string" },
                    },
                  },
                },
                confidence: { type: "number" },
              },
              required: ["claim", "confidence"],
            },
          },
          redFlags: { type: "array", items: { type: "string" } },
          uncertainties: { type: "array", items: { type: "string" } },
          disagreements: { type: "array", items: { type: "string" } },
          recommendationForSynthesis: { type: "string" },
        },
        required: ["role", "findings", "claims", "redFlags", "uncertainties"],
      };

      const repaired = await createStructuredInteraction({
        model: MODELS.orchestrator,
        input: `Extract and structure the following ${displayName} specialist output into the required JSON report without inventing any facts:\n\n${rawOutputText}`,
        systemInstruction:
          "You are a strict structural parser. Convert the raw specialist report into the target JSON schema without adding new medical claims.",
        schema: repairSchema,
        quiet: true,
      });

      if (repaired.data) {
        report = normalizeSpecialistReport(repaired.data, displayName, rawOutputText, true);
      } else {
        report = normalizeSpecialistReport(null, displayName, rawOutputText, true);
      }
    } catch {
      report = normalizeSpecialistReport(null, displayName, rawOutputText, true);
    }
  }

  const elapsedMs = Date.now() - startTime;
  dashboard.updateSpecialist(roleId, {
    state: "completed",
    activity: report.requiredRepair
      ? `completed (repaired JSON, ${report.claims.length} claims)`
      : `completed (${report.claims.length} claims)`,
    thought: "",
    elapsedMs,
  });

  debug("specialist.completed", {
    agentId: managedAgentId,
    elapsedMs,
    detail: `${displayName}: ${report.claims.length} claims, repair=${report.requiredRepair}`,
  });

  return {
    roleId,
    displayName,
    managedAgentId,
    environmentId,
    status: "completed",
    elapsedMs,
    report,
  };
}

/**
 * Stage 4: Parallel Specialist Swarm Execution
 * Runs up to 5 assigned specialists concurrently via Promise.allSettled.
 */
async function runSpecialistSwarm(researchPlan, isRevision = false) {
  state.phase = "Phase 4 / Specialist Swarm";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 4/9 — Specialist Swarm${ANSI.reset}`);

  const assigned = researchPlan.specialists.slice(0, LIMITS.maxSpecialists);

  // On revision cycles, preserve existing completed specialists if action === "keep"
  const keptSpecialists = [];
  const toRun = [];

  for (const spec of assigned) {
    if (isRevision && spec.action === "keep") {
      const prev = state.specialists.find(
        (s) => s.roleId === spec.roleId && s.status === "completed"
      );
      if (prev) {
        keptSpecialists.push(prev);
        continue;
      }
    }
    toRun.push(spec);
  }

  if (keptSpecialists.length > 0) {
    console.log(
      `  ${SYMBOLS.success} Keeping ${keptSpecialists.length} verified specialist report(s): ${keptSpecialists
        .map((s) => s.displayName)
        .join(", ")}`
    );
  }

  if (toRun.length === 0) {
    return state.specialists;
  }

  console.log(`  ${SYMBOLS.success} Task assigned`);
  console.log(`  ${SYMBOLS.success} ${toRun.length} specialist agent(s) launched in parallel\n`);

  const dashboard = new SpecialistSwarmDashboard(toRun);
  dashboard.start();

  const settledResults = await Promise.allSettled(
    toRun.map(async (assignment) => {
      try {
        return await runSpecialistAgent(assignment, dashboard, isRevision);
      } catch (err) {
        dashboard.updateSpecialist(assignment.roleId, {
          state: "failed",
          activity: truncateText(formatApiError(err), 38),
          thought: "",
        });
        throw err;
      }
    })
  );

  dashboard.stop();

  const currentRunResults = [];
  settledResults.forEach((res, idx) => {
    const spec = toRun[idx];
    if (res.status === "fulfilled") {
      currentRunResults.push(res.value);
    } else {
      const errMsg = formatApiError(res.reason);
      console.log(
        `\n  ${SYMBOLS.warning} ${ANSI.brightYellow}${spec.displayName} failed:${ANSI.reset} ${truncateText(errMsg, 75)}`
      );
      console.log(`  ${ANSI.dim}Continuing with remaining specialists...${ANSI.reset}`);
      currentRunResults.push({
        roleId: spec.roleId,
        displayName: spec.displayName,
        managedAgentId: SPECIALIST_ROLE_CATALOG[spec.roleId]?.managedAgentId || null,
        status: "failed",
        error: errMsg,
        report: null,
      });
    }
  });

  state.specialists = [...keptSpecialists, ...currentRunResults];
  return state.specialists;
}

// ============================================================================
// 12. STAGE 5: DISAGREEMENT DETECTOR
// ============================================================================

const DISAGREEMENT_SCHEMA = {
  type: "object",
  properties: {
    disagreements: {
      type: "array",
      items: {
        type: "object",
        properties: {
          claim: { type: "string" },
          agents: {
            type: "array",
            items: { type: "string" },
          },
          positions: {
            type: "array",
            items: { type: "string" },
          },
          severity: {
            type: "string",
            enum: ["low", "medium", "high", "critical"],
          },
          requiresAdditionalResearch: {
            type: "boolean",
          },
        },
        required: ["claim", "agents", "positions", "severity", "requiresAdditionalResearch"],
      },
    },
  },
  required: ["disagreements"],
};

/**
 * Stage 5: Disagreement Detector
 * Identifies conflicts or differing clinical thresholds across specialist reports
 * without silently choosing a medical truth.
 */
async function runDisagreementDetector() {
  state.phase = "Phase 5 / Disagreement Analysis";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 5/9 — Disagreement Analysis${ANSI.reset}`);

  const completedSpecialists = state.specialists.filter(
    (s) => s.status === "completed" && s.report
  );

  if (completedSpecialists.length <= 1) {
    state.disagreementReport = { disagreements: [] };
    console.log(
      `  ${SYMBOLS.success} Single active specialist report — cross-agent disagreement check skipped.`
    );
    return state.disagreementReport;
  }

  activeSpinner.start(
    `${ANSI.brightCyan}Analyzing cross-specialist disagreements...${ANSI.reset}`,
    `Comparing findings across ${completedSpecialists.length} specialists...`
  );

  const systemInstruction = `You are the MGA-1 Disagreement Detector (${MODELS.orchestrator}).
Your job is to compare the independent Specialist Reports against each other and the gathered Evidence.
CRITICAL RULE:
- Do NOT silently choose a medical truth or resolve conflicts yourself.
- Identify any conflicting recommendations, differing risk thresholds, contradictory claims, or tensions between specialties.
- If the specialists are in full agreement, return an empty "disagreements" array.`;

  const inputPayload = {
    taskSpec: state.taskSpec,
    evidence: state.evidence,
    specialistReports: completedSpecialists.map((s) => ({
      roleId: s.roleId,
      displayName: s.displayName,
      report: s.report,
    })),
  };

  let result;
  try {
    result = await createStructuredInteraction({
      model: MODELS.orchestrator,
      input: JSON.stringify(inputPayload, null, 2),
      systemInstruction,
      schema: DISAGREEMENT_SCHEMA,
      thinkingLabel: "Comparing specialist claims and risk thresholds...",
    });
  } finally {
    activeSpinner.stop();
  }

  const disagreements = Array.isArray(result.data?.disagreements)
    ? result.data.disagreements
    : [];

  state.disagreementReport = { disagreements };

  if (disagreements.length === 0) {
    console.log(`  ${SYMBOLS.success} No substantive cross-specialist conflicts detected`);
  } else {
    console.log(
      `  ${SYMBOLS.warning} ${ANSI.brightYellow}${disagreements.length} disagreement(s) identified across specialists:${ANSI.reset}`
    );
    for (const d of disagreements) {
      console.log(
        `    • [${String(d.severity || "medium").toUpperCase()}] ${truncateText(d.claim, 70)} (${(d.agents || []).join(" vs. ")})`
      );
    }
  }

  return state.disagreementReport;
}

// ============================================================================
// 13. STAGE 6: EVIDENCE-CONSTRAINED SUMMARIZER & CLAIM PROVENANCE
// ============================================================================

const SUMMARIZER_SCHEMA = {
  type: "object",
  properties: {
    answer: {
      type: "string",
    },
    claims: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          text: { type: "string" },
          evidenceIds: {
            type: "array",
            items: { type: "string" },
          },
          specialistIds: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: ["id", "text", "evidenceIds", "specialistIds"],
      },
    },
    uncertainties: {
      type: "array",
      items: { type: "string" },
    },
    unsupportedClaims: {
      type: "array",
      items: { type: "string" },
    },
    citations: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          title: { type: "string" },
          source: { type: "string" },
          url: { type: "string" },
        },
        required: ["id", "title"],
      },
    },
  },
  required: ["answer", "claims", "uncertainties", "unsupportedClaims", "citations"],
};

/**
 * Builds and validates the internal Claim -> Evidence -> Specialist provenance graph.
 */
function buildClaimProvenance(summarizerData) {
  const validEvidenceIds = new Set(state.evidence.map((e) => e.id));
  const validSpecialistIds = new Set(
    state.specialists.filter((s) => s.status === "completed").map((s) => s.roleId)
  );

  const rawClaims = Array.isArray(summarizerData?.claims) ? summarizerData.claims : [];
  const provenance = [];
  const autoUnsupported = new Set(
    Array.isArray(summarizerData?.unsupportedClaims) ? summarizerData.unsupportedClaims : []
  );

  rawClaims.forEach((c, idx) => {
    const claimId = c.id || `claim_${String(idx + 1).padStart(2, "0")}`;
    const matchedEvidence = (Array.isArray(c.evidenceIds) ? c.evidenceIds : []).filter((id) =>
      validEvidenceIds.has(id)
    );
    const matchedSpecialists = (Array.isArray(c.specialistIds) ? c.specialistIds : [])
      .map((s) => String(s).toLowerCase().replace(/\s+/g, "_"))
      .filter((id) => validSpecialistIds.has(id));

    const isSupported = matchedEvidence.length > 0 || matchedSpecialists.length > 0;
    if (!isSupported) {
      autoUnsupported.add(claimId);
    }

    provenance.push({
      id: claimId,
      text: String(c.text || ""),
      evidenceIds: matchedEvidence,
      specialistIds: matchedSpecialists,
      status: isSupported ? "supported" : "unsupported",
    });
  });

  return {
    provenance,
    unsupportedClaims: Array.from(autoUnsupported),
  };
}

/**
 * Stage 6: Evidence-Constrained Summarizer
 */
async function runSummarizer() {
  state.phase = "Phase 6 / Synthesis";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 6/9 — Evidence-Constrained Synthesis${ANSI.reset}`);

  const completedSpecialists = state.specialists.filter(
    (s) => s.status === "completed" && s.report
  );
  const failedSpecialists = state.specialists.filter((s) => s.status === "failed");

  activeSpinner.start(
    `${ANSI.brightCyan}Synthesizing evidence-constrained candidate answer...${ANSI.reset}`,
    "Mapping claims to specialist findings and evidence IDs..."
  );

  const systemInstruction = `You are the MGA-1 Evidence-Constrained Summarizer (${MODELS.summarizer}).
You receive the Original Prompt, TaskSpec, Evidence Registry, Specialist Reports, and Disagreement Report.

STRICT RELIABILITY RULES:
- You may ONLY synthesize claims supported by the provided Evidence or Specialist Reports.
- You may NOT introduce new medical facts, unverified treatments, invented statistics, or fabricated citations.
- You may NOT act as another independent specialist.
- You may reconcile disagreements ONLY when the supplied evidence explicitly supports doing so; otherwise, transparently communicate the disagreement and uncertainty to the user.
- Every substantive medical claim in your candidate answer must be listed in "claims" with its corresponding "evidenceIds" (e.g., ["E01"]) and "specialistIds" (e.g., ["pediatrics", "neurology"]).
- If any claim lacks backing from the provided evidence/specialists, list its ID in "unsupportedClaims".
- Format "answer" as clear, compassionate, structured natural language suitable for a terminal display, explicitly highlighting red flags / when to seek urgent medical care, practical next steps supported by the specialists, and explicit uncertainties.`;

  const inputPayload = {
    originalPrompt: state.originalPrompt,
    taskSpec: state.taskSpec,
    evidenceRegistry: state.evidence,
    specialistReports: completedSpecialists.map((s) => ({
      roleId: s.roleId,
      displayName: s.displayName,
      requiredRepair: s.report.requiredRepair,
      report: s.report,
    })),
    failedSpecialists: failedSpecialists.map((f) => ({
      roleId: f.roleId,
      displayName: f.displayName,
      error: f.error,
    })),
    disagreementReport: state.disagreementReport,
  };

  let result;
  try {
    result = await createStructuredInteraction({
      model: MODELS.summarizer,
      input: JSON.stringify(inputPayload, null, 2),
      systemInstruction,
      schema: SUMMARIZER_SCHEMA,
      thinkingLabel: "Synthesizing verified claims and tracing provenance...",
    });
  } finally {
    activeSpinner.stop();
  }

  const rawData = result.data || {};
  const { provenance, unsupportedClaims } = buildClaimProvenance(rawData);
  state.claimProvenance = provenance;

  const citations = Array.isArray(rawData.citations) && rawData.citations.length > 0
    ? rawData.citations
    : state.evidence.map((e) => ({
        id: e.id,
        title: e.title,
        source: e.source,
        url: e.url,
      }));

  state.candidateAnswer = {
    answer: rawData.answer || result.output_text || "",
    claims: provenance,
    uncertainties: Array.isArray(rawData.uncertainties) ? rawData.uncertainties : [],
    unsupportedClaims,
    citations,
  };

  console.log(
    `  ${SYMBOLS.success} Candidate synthesis complete (${provenance.length} traced claim(s), ${unsupportedClaims.length} unsupported)`
  );

  if (cliConfig.verbose && provenance.length > 0) {
    console.log(`  ${ANSI.dim}Claim Provenance Tree:${ANSI.reset}`);
    for (const c of provenance.slice(0, 6)) {
      const evStr = c.evidenceIds.length > 0 ? c.evidenceIds.join(", ") : "none";
      const spStr = c.specialistIds.length > 0 ? c.specialistIds.join(", ") : "none";
      console.log(`    ${ANSI.dim}${c.id} [${c.status}] ├─ Evidence: ${evStr} └─ Specialists: ${spStr}${ANSI.reset}`);
    }
  }

  return state.candidateAnswer;
}

// ============================================================================
// 14. STAGE 7: INDEPENDENT EVALUATOR
// ============================================================================

const EVALUATOR_SCHEMA = {
  type: "object",
  properties: {
    decision: {
      type: "string",
      enum: ["PASS", "FAIL", "UNRESOLVED"],
    },
    factuality: { type: "number" },
    evidenceSupport: { type: "number" },
    completeness: { type: "number" },
    internalConsistency: { type: "number" },
    medicalSafety: { type: "number" },
    uncertaintyCalibration: { type: "number" },
    citationIntegrity: { type: "number" },
    unsupportedClaims: {
      type: "array",
      items: { type: "string" },
    },
    criticalErrors: {
      type: "array",
      items: { type: "string" },
    },
    missingCriticalInformation: {
      type: "array",
      items: { type: "string" },
    },
    revisionPlan: {
      type: "array",
      items: { type: "string" },
    },
  },
  required: [
    "decision",
    "factuality",
    "evidenceSupport",
    "completeness",
    "internalConsistency",
    "medicalSafety",
    "uncertaintyCalibration",
    "unsupportedClaims",
    "criticalErrors",
    "missingCriticalInformation",
    "revisionPlan",
  ],
};

function clampScore(val) {
  const num = Number(val);
  if (!Number.isFinite(num)) return 0.0;
  return Number(Math.max(0, Math.min(1, num)).toFixed(2));
}

function validateEvaluationResult(rawEval) {
  const ev = rawEval && typeof rawEval === "object" ? rawEval : {};
  const allowedDecisions = ["PASS", "FAIL", "UNRESOLVED"];
  let decision = String(ev.decision || "").toUpperCase();
  if (!allowedDecisions.includes(decision)) {
    decision = "FAIL";
  }

  const validated = {
    decision,
    factuality: clampScore(ev.factuality),
    evidenceSupport: clampScore(ev.evidenceSupport),
    completeness: clampScore(ev.completeness),
    internalConsistency: clampScore(ev.internalConsistency),
    medicalSafety: clampScore(ev.medicalSafety),
    uncertaintyCalibration: clampScore(ev.uncertaintyCalibration),
    citationIntegrity: clampScore(ev.citationIntegrity ?? ev.evidenceSupport),
    unsupportedClaims: Array.isArray(ev.unsupportedClaims) ? ev.unsupportedClaims.map(String) : [],
    criticalErrors: Array.isArray(ev.criticalErrors) ? ev.criticalErrors.map(String) : [],
    missingCriticalInformation: Array.isArray(ev.missingCriticalInformation)
      ? ev.missingCriticalInformation.map(String)
      : [],
    revisionPlan: Array.isArray(ev.revisionPlan) ? ev.revisionPlan.map(String) : [],
  };

  // Enforce revision limits if still failing at max iterations
  if (
    validated.decision === "FAIL" &&
    state.revisionIteration >= LIMITS.maxRevisionIterations
  ) {
    validated.decision = "UNRESOLVED";
  }

  return validated;
}

/**
 * Stage 7: Independent Evaluator
 * Evaluates the Candidate Answer across all required reliability dimensions.
 */
async function runEvaluator() {
  state.phase = "Phase 7 / Evaluation";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 7/9 — Independent Reliability Evaluation${ANSI.reset}`);

  const completedSpecialists = state.specialists.filter(
    (s) => s.status === "completed" && s.report
  );

  // If zero specialists succeeded, automatically fail or mark unresolved
  if (completedSpecialists.length === 0) {
    state.evaluation = validateEvaluationResult({
      decision: state.revisionIteration >= LIMITS.maxRevisionIterations ? "UNRESOLVED" : "FAIL",
      factuality: 0,
      evidenceSupport: 0,
      completeness: 0,
      internalConsistency: 0,
      medicalSafety: 0,
      uncertaintyCalibration: 0,
      citationIntegrity: 0,
      unsupportedClaims: ["All claims lack specialist verification"],
      criticalErrors: ["No specialist agents completed successfully"],
      missingCriticalInformation: ["Specialist swarm outputs"],
      revisionPlan: ["Re-run specialist swarm with available core specialties"],
    });
    return state.evaluation;
  }

  activeSpinner.start(
    `${ANSI.brightCyan}Auditing candidate answer & claim provenance...${ANSI.reset}`,
    "Evaluating factuality, evidence support, safety, and uncertainty calibration..."
  );

  const systemInstruction = `You are the MGA-1 Independent Evaluator (${MODELS.evaluator}).
You are the final reliability gate before any medical response is shown to the user.
You must be strictly independent from the Summarizer.

Evaluate the Candidate Answer across these exact dimensions (scores between 0.0 and 1.0):
- factuality: Accuracy with respect to established medical knowledge and specialist findings.
- evidenceSupport: Degree to which every substantive claim traces to Evidence or Specialist Reports.
- completeness: Coverage of the user's question, patient context, and relevant red flags.
- internalConsistency: Absence of self-contradiction.
- medicalSafety: Appropriate triage urgency, clear red-flag warnings, no unsafe advice, no false clinical authority.
- uncertaintyCalibration: Honest communication of unknowns, missing context, and specialist disagreements.
- citationIntegrity: No fabricated or misrepresented sources.

Decision Policy:
- Return "PASS" ONLY if the answer is medically safe, well-supported by the specialist findings/evidence, properly calibrated in its uncertainty, and free of critical errors or hallucinated claims.
- Return "FAIL" if there are unsupported claims, unresolved high-severity disagreements that were glossed over, missing critical specialty perspectives, or safety gaps. Provide a concrete "revisionPlan" for the Orchestrator (e.g., which specialist to add, what evidence to retrieve, or which unsupported claim to remove).`;

  const inputPayload = {
    originalPrompt: state.originalPrompt,
    taskSpec: state.taskSpec,
    evidenceRegistry: state.evidence,
    specialistReports: completedSpecialists.map((s) => ({
      roleId: s.roleId,
      displayName: s.displayName,
      requiredRepair: s.report.requiredRepair,
      report: s.report,
    })),
    failedSpecialists: state.specialists
      .filter((s) => s.status === "failed")
      .map((s) => ({ roleId: s.roleId, displayName: s.displayName, error: s.error })),
    disagreementReport: state.disagreementReport,
    candidateAnswer: state.candidateAnswer,
    claimProvenance: state.claimProvenance,
    revisionIteration: state.revisionIteration,
    maxRevisionIterations: LIMITS.maxRevisionIterations,
  };

  let result;
  try {
    result = await createStructuredInteraction({
      model: MODELS.evaluator,
      input: JSON.stringify(inputPayload, null, 2),
      systemInstruction,
      schema: EVALUATOR_SCHEMA,
      thinkingLabel: "Auditing claim provenance, safety red flags, and uncertainty calibration...",
    });
  } finally {
    activeSpinner.stop();
  }

  const evaluation = validateEvaluationResult(result.data);
  state.evaluation = evaluation;

  const badge =
    evaluation.decision === "PASS"
      ? `${ANSI.brightGreen}PASS${ANSI.reset}`
      : evaluation.decision === "FAIL"
      ? `${ANSI.brightYellow}FAIL${ANSI.reset}`
      : `${ANSI.brightRed}UNRESOLVED${ANSI.reset}`;

  console.log(
    `  ${evaluation.decision === "PASS" ? SYMBOLS.success : SYMBOLS.warning} Evaluator Decision: ${ANSI.bold}${badge}${ANSI.reset}`
  );
  console.log(
    `    ${ANSI.dim}Factuality: ${evaluation.factuality.toFixed(2)} | Evidence: ${evaluation.evidenceSupport.toFixed(
      2
    )} | Safety: ${evaluation.medicalSafety.toFixed(2)} | Uncertainty: ${evaluation.uncertaintyCalibration.toFixed(
      2
    )}${ANSI.reset}`
  );

  if (evaluation.decision !== "PASS") {
    if (evaluation.criticalErrors.length > 0) {
      console.log(`    ${SYMBOLS.failure} Critical Errors: ${evaluation.criticalErrors.join("; ")}`);
    }
    if (evaluation.missingCriticalInformation.length > 0) {
      console.log(
        `    ${SYMBOLS.warning} Missing Critical Info: ${evaluation.missingCriticalInformation.join("; ")}`
      );
    }
    if (evaluation.revisionPlan.length > 0) {
      console.log(`    ${SYMBOLS.transition} Revision Plan: ${evaluation.revisionPlan.join("; ")}`);
    }
  }

  return evaluation;
}

// ============================================================================
// 15. STAGE 8 & 9: REVISION LOOP, ABSTENTION & FINAL ANSWER RENDERING
// ============================================================================

/**
 * Performs targeted additional evidence grounding when requested by Orchestrator
 * during a revision cycle.
 */
async function runRevisionEvidenceRetrieval(queries) {
  if (!Array.isArray(queries) || queries.length === 0) return;
  console.log(`  ${SYMBOLS.tool} Retrieving additional targeted evidence (${queries.length} item(s))...`);
  activeSpinner.start(
    `${ANSI.brightCyan}Retrieving additional clinical evidence...${ANSI.reset}`,
    queries[0]
  );

  try {
    const schema = {
      type: "object",
      properties: {
        references: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              source: { type: "string" },
              url: { type: "string" },
              summary: { type: "string" },
            },
            required: ["title", "source", "summary"],
          },
        },
      },
      required: ["references"],
    };

    const res = await createStructuredInteraction({
      model: MODELS.base,
      input: `Retrieve clinical guideline or peer-reviewed evidence for the following revision requirements:\n${queries.join(
        "\n"
      )}`,
      systemInstruction:
        "You are the MGA-1 Evidence Retrieval module. Provide authoritative medical guideline references addressing the requested queries. Never invent URLs.",
      schema,
      tools: [{ type: "google_search" }, { type: "url_context" }],
      thinkingLabel: "Gathering targeted clinical evidence for revision...",
    });

    if (Array.isArray(res.data?.references)) {
      for (const ref of res.data.references) {
        registerEvidenceItem({
          title: ref.title,
          source: ref.source,
          url: ref.url || "",
          summary: ref.summary,
          origin: `revision_${state.revisionIteration}`,
        });
      }
    }
  } catch (err) {
    debug("revision_evidence.warning", { detail: formatApiError(err) });
  } finally {
    activeSpinner.stop();
  }
}

/**
 * Renders the Abstention / Unresolved output when MGA-1 cannot establish
 * a sufficiently reliable evidence base after max revision iterations.
 */
function renderAbstentionOutput() {
  state.phase = "Phase 9 / Final Answer (Abstained)";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 9/9 — Reliability Gate: ABSTAIN${ANSI.reset}\n`);

  const lines = [
    `${SYMBOLS.warning} ${ANSI.bold}${ANSI.brightYellow}MGA-1 could not establish a sufficiently reliable evidence base.${ANSI.reset}`,
    `${ANSI.bold}The system will not manufacture certainty.${ANSI.reset}`,
    "",
  ];

  if (state.evaluation?.criticalErrors?.length > 0) {
    lines.push(`${ANSI.bold}Identified Reliability / Safety Barriers:${ANSI.reset}`);
    for (const err of state.evaluation.criticalErrors) {
      lines.push(`• ${err}`);
    }
    lines.push("");
  }

  if (state.evaluation?.missingCriticalInformation?.length > 0) {
    lines.push(`${ANSI.bold}Missing or Unverified Information:${ANSI.reset}`);
    for (const info of state.evaluation.missingCriticalInformation) {
      lines.push(`• ${info}`);
    }
    lines.push("");
  }

  if (state.disagreementReport?.disagreements?.length > 0) {
    lines.push(`${ANSI.bold}Unresolved Specialist Disagreements:${ANSI.reset}`);
    for (const d of state.disagreementReport.disagreements) {
      lines.push(`• ${d.claim} (${(d.agents || []).join(" vs. ")})`);
    }
    lines.push("");
  }

  // Collect red flags from any completed specialist reports so safety warnings are still surfaced
  const allRedFlags = new Set();
  for (const s of state.specialists) {
    if (Array.isArray(s.report?.redFlags)) {
      for (const rf of s.report.redFlags) allRedFlags.add(rf);
    }
  }

  if (allRedFlags.size > 0) {
    lines.push(`${ANSI.bold}${ANSI.brightRed}Important Red Flags Identified by Specialists (Seek Urgent Care If Present):${ANSI.reset}`);
    for (const rf of Array.from(allRedFlags).slice(0, 6)) {
      lines.push(`• ${rf}`);
    }
    lines.push("");
  }

  lines.push(
    `${SYMBOLS.warning} ${ANSI.dim}Please consult a qualified healthcare professional or emergency medical service for direct clinical evaluation.${ANSI.reset}`
  );

  console.log(
    renderBox("MGA-1 Abstention Notice (UNRESOLVED)", lines, {
      borderColor: ANSI.yellow,
      titleColor: ANSI.bold + ANSI.brightYellow,
    })
  );
}

/**
 * Renders the Final Approved Answer after Evaluator PASS.
 */
function renderFinalApprovedAnswer() {
  state.phase = "Phase 9 / Final Answer";
  console.log(`\n${ANSI.dim}[MGA-1] Phase 9/9 — Final Approved Answer${ANSI.reset}\n`);

  const answerText = state.candidateAnswer?.answer || "No answer text generated.";
  const lines = [...answerText.split(/\r?\n/)];

  if (
    Array.isArray(state.candidateAnswer?.uncertainties) &&
    state.candidateAnswer.uncertainties.length > 0
  ) {
    lines.push("");
    lines.push(`${ANSI.bold}Key Uncertainties & Limitations:${ANSI.reset}`);
    for (const u of state.candidateAnswer.uncertainties.slice(0, 5)) {
      lines.push(`• ${u}`);
    }
  }

  lines.push("");
  lines.push(
    `${SYMBOLS.warning} ${ANSI.yellow}This is a research prototype and does not replace professional medical evaluation.${ANSI.reset}`
  );

  const citations =
    Array.isArray(state.candidateAnswer?.citations) && state.candidateAnswer.citations.length > 0
      ? state.candidateAnswer.citations
      : state.evidence;

  if (citations.length > 0) {
    lines.push("");
    lines.push(`${ANSI.bold}Sources:${ANSI.reset}`);
    citations.slice(0, 8).forEach((src, idx) => {
      const urlPart = src.url ? ` — ${src.url}` : src.source ? ` (${src.source})` : "";
      lines.push(`[${idx + 1}] ${src.title}${urlPart}`);
    });
  }

  console.log(
    renderBox("MGA-1 Final Answer", lines, {
      borderColor: ANSI.green,
      titleColor: ANSI.bold + ANSI.brightGreen,
    })
  );
}

/**
 * Renders the compact MGA-1 Run Summary panel at the end of a research pipeline.
 */
function renderRunSummary() {
  const totalTimeMs =
    state.metrics.startTimeMs && state.metrics.endTimeMs
      ? state.metrics.endTimeMs - state.metrics.startTimeMs
      : 0;
  const width = Math.min(getTerminalWidth() - 4, 72);
  const divider = `${ANSI.dim}${"─".repeat(width)}${ANSI.reset}`;

  console.log(`\n${divider}`);
  console.log(`${ANSI.bold}MGA-1 Run Summary${ANSI.reset}`);
  console.log(`Specialists:     ${state.specialists.length}`);
  console.log(`Iterations:      ${state.revisionIteration + 1}`);
  console.log(`Total time:      ${formatDuration(totalTimeMs)}`);

  if (state.metrics.totalTokens > 0) {
    console.log(`Total tokens:    ${state.metrics.totalTokens.toLocaleString()}`);
    if (cliConfig.verbose) {
      if (state.metrics.inputTokens > 0) {
        console.log(`  Input tokens:  ${state.metrics.inputTokens.toLocaleString()}`);
      }
      if (state.metrics.outputTokens > 0) {
        console.log(`  Output tokens: ${state.metrics.outputTokens.toLocaleString()}`);
      }
      if (state.metrics.thoughtTokens > 0) {
        console.log(`  Thought tokens:${state.metrics.thoughtTokens.toLocaleString()}`);
      }
      if (state.metrics.toolUseTokens > 0) {
        console.log(`  Tool tokens:   ${state.metrics.toolUseTokens.toLocaleString()}`);
      }
    }
  }

  if (state.evaluation) {
    console.log(`Evaluator:       ${state.evaluation.decision}`);
    console.log(`  Factuality:     ${state.evaluation.factuality.toFixed(2)}`);
    console.log(`  Evidence:       ${state.evaluation.evidenceSupport.toFixed(2)}`);
    console.log(`  Safety:         ${state.evaluation.medicalSafety.toFixed(2)}`);
    console.log(`  Completeness:   ${state.evaluation.completeness.toFixed(2)}`);
    console.log(`  Consistency:    ${state.evaluation.internalConsistency.toFixed(2)}`);
    console.log(`  Uncertainty:    ${state.evaluation.uncertaintyCalibration.toFixed(2)}`);
  }
  console.log(divider + "\n");
}

/**
 * Executes the complete MGA-1 Research Pipeline after TaskSpec formulation:
 * Orchestrator -> Parallel Specialist Swarm -> Disagreement Detector ->
 * Evidence-Constrained Summarizer -> Independent Evaluator -> Revision Loop (up to 3) -> Final Answer / Abstention
 */
async function executeMgaResearchPipeline() {
  state.revisionIteration = 0;

  while (state.revisionIteration <= LIMITS.maxRevisionIterations) {
    const isRevision = state.revisionIteration > 0;

    // 1. Orchestrator
    const plan = await runOrchestrator(isRevision);

    // 2. If revision requested additional evidence retrieval, fetch it before running specialists
    if (isRevision && Array.isArray(plan.additionalEvidenceNeeded) && plan.additionalEvidenceNeeded.length > 0) {
      await runRevisionEvidenceRetrieval(plan.additionalEvidenceNeeded);
    }

    // 3. Dynamic Specialist Swarm (Concurrent Antigravity Custom Managed Agents)
    await runSpecialistSwarm(plan, isRevision);

    // 4. Disagreement Detection
    await runDisagreementDetector();

    // 5. Evidence-Constrained Summarizer
    await runSummarizer();

    // 6. Independent Evaluator
    const evaluation = await runEvaluator();

    // 7. Check Gate Decision
    if (evaluation.decision === "PASS") {
      state.metrics.endTimeMs = Date.now();
      renderFinalApprovedAnswer();
      renderRunSummary();
      state.phase = "completed";
      return;
    }

    if (
      evaluation.decision === "UNRESOLVED" ||
      state.revisionIteration >= LIMITS.maxRevisionIterations
    ) {
      state.evaluation.decision = "UNRESOLVED";
      state.metrics.endTimeMs = Date.now();
      renderAbstentionOutput();
      renderRunSummary();
      state.phase = "abstained";
      return;
    }

    // Prepare next revision cycle
    state.revisionIteration += 1;
    console.log(
      `\n  ${SYMBOLS.transition} ${ANSI.brightYellow}Evaluator rejected candidate answer. Entering Revision Loop ${state.revisionIteration}/${LIMITS.maxRevisionIterations}...${ANSI.reset}`
    );
  }
}

// ============================================================================
// 16. CLI COMMANDS & INTERACTIVE TERMINAL LOOP
// ============================================================================

function printStartupBanner() {
  const bannerLines = [
    `${ANSI.bold}${ANSI.brightCyan}MGA-1${ANSI.reset}`,
    "Medical General Architecture for Reliable Medical AI",
    `${ANSI.dim}Research Prototype v1.0${ANSI.reset}`,
  ];
  console.log("\n" + renderBox(null, bannerLines, { width: 58 }));

  const baseDisplay = CURRENT_MODEL_REGISTRY[MODELS.base]?.displayName || MODELS.base;
  const orchDisplay =
    CURRENT_MODEL_REGISTRY[MODELS.orchestrator]?.displayName || MODELS.orchestrator;

  console.log(`${ANSI.bold}Base Model${ANSI.reset}       ${baseDisplay} (${ANSI.dim}${MODELS.base}${ANSI.reset})`);
  console.log(`${ANSI.bold}Orchestrator${ANSI.reset}     ${orchDisplay} (${ANSI.dim}${MODELS.orchestrator}${ANSI.reset})`);
  console.log(`${ANSI.bold}Summarizer${ANSI.reset}       ${MODELS.summarizer}`);
  console.log(`${ANSI.bold}Evaluator${ANSI.reset}        ${MODELS.evaluator}`);
  console.log(`${ANSI.bold}Specialists${ANSI.reset}      Antigravity Managed Agents (${ANSI.dim}${MODELS.specialistAgentBase}${ANSI.reset})`);
  console.log(`${ANSI.bold}Max Specialists${ANSI.reset}  ${LIMITS.maxSpecialists}`);
  console.log(`${ANSI.bold}Max Revisions${ANSI.reset}    ${LIMITS.maxRevisionIterations}`);
  console.log(
    `\n${SYMBOLS.warning} ${ANSI.yellow}Research prototype — not medical advice.${ANSI.reset}`
  );
  console.log(
    `${ANSI.dim}MGA-1 is not a physician and does not replace professional medical evaluation.${ANSI.reset}\n`
  );
}

function handleHelpCommand() {
  console.log(`\n${ANSI.bold}${ANSI.brightCyan}MGA-1 CLI Commands${ANSI.reset}`);
  console.log(`  ${ANSI.bold}/help${ANSI.reset}      Show available CLI commands`);
  console.log(`  ${ANSI.bold}/new${ANSI.reset}       Start a new medical research conversation`);
  console.log(`  ${ANSI.bold}/reset${ANSI.reset}     Reset the current Q&A and session task state`);
  console.log(`  ${ANSI.bold}/models${ANSI.reset}    Show configured role models and recognized current Gemini models`);
  console.log(`  ${ANSI.bold}/agents${ANSI.reset}    Show specialist managed-agent IDs and registration status`);
  console.log(`  ${ANSI.bold}/status${ANSI.reset}    Show current MGA-1 phase, iteration, and active specialists`);
  console.log(`  ${ANSI.bold}/verbose${ANSI.reset}   Enable detailed internal research trace`);
  console.log(`  ${ANSI.bold}/quiet${ANSI.reset}     Reduce internal trace output (default)`);
  console.log(`  ${ANSI.bold}/quit${ANSI.reset}      Cleanly terminate the application (alias: ${ANSI.bold}/exit${ANSI.reset})\n`);
}

function handleModelsCommand() {
  console.log(`\n${ANSI.bold}${ANSI.brightCyan}Active MGA-1 Role Configuration${ANSI.reset}`);
  console.log(`  Base Model (Intake/Grounding): ${ANSI.bold}${MODELS.base}${ANSI.reset}`);
  console.log(`  Orchestrator / Disagreement:   ${ANSI.bold}${MODELS.orchestrator}${ANSI.reset}`);
  console.log(`  Evidence-Constrained Summary:  ${ANSI.bold}${MODELS.summarizer}${ANSI.reset}`);
  console.log(`  Independent Evaluator:         ${ANSI.bold}${MODELS.evaluator}${ANSI.reset}`);
  console.log(`  Specialist Base Managed Agent: ${ANSI.bold}${MODELS.specialistAgentBase}${ANSI.reset}`);

  console.log(`\n${ANSI.bold}${ANSI.brightCyan}Recognized Current Gemini Models Registry${ANSI.reset}`);
  for (const m of Object.values(CURRENT_MODEL_REGISTRY)) {
    console.log(
      `  • ${ANSI.bold}${m.id.padEnd(28)}${ANSI.reset} ${ANSI.dim}(${m.contextWindow}) — ${m.description}${ANSI.reset}`
    );
  }
  console.log("");
}

async function handleAgentsCommand() {
  console.log(`\n${ANSI.bold}${ANSI.brightCyan}MGA-1 Specialist Custom Managed Agents (Base: ${MODELS.specialistAgentBase})${ANSI.reset}`);

  // Optional lightweight remote list check if user invokes /agents
  if (genAiClient && cliConfig.verbose) {
    try {
      const remoteList = await genAiClient.agents.list({ page_size: 50 });
      if (Array.isArray(remoteList?.agents)) {
        for (const ag of remoteList.agents) {
          const id = (ag.id || ag.name || "").replace(/^agents\//, "");
          if (id && !managedAgentRegistry.has(id)) {
            managedAgentRegistry.set(id, {
              id,
              registered: true,
              status: "remote_verified",
            });
          }
        }
      }
    } catch {
      // Ignore remote list errors in /agents display
    }
  }

  for (const role of Object.values(SPECIALIST_ROLE_CATALOG)) {
    const reg = managedAgentRegistry.get(role.managedAgentId);
    const isRegistered = Boolean(reg?.registered);
    const statusBadge = isRegistered
      ? `${SYMBOLS.success} ${ANSI.green}registered (${reg.status})${ANSI.reset}`
      : `${SYMBOLS.bulletWaiting} ${ANSI.dim}lazy (not yet created in this session)${ANSI.reset}`;

    console.log(
      `  • ${ANSI.bold}${role.displayName.padEnd(20)}${ANSI.reset} ID: ${ANSI.cyan}${role.managedAgentId.padEnd(24)}${ANSI.reset} Status: ${statusBadge}`
    );
  }
  console.log("");
}

function handleStatusCommand() {
  console.log(`\n${ANSI.bold}${ANSI.brightCyan}MGA-1 Session Status${ANSI.reset}`);
  console.log(`  Session ID:         ${state.sessionId}`);
  console.log(`  Current Phase:      ${ANSI.bold}${state.phase}${ANSI.reset}`);
  console.log(`  Q&A Round:          ${state.qa.round}/${LIMITS.maxQaRounds} (Complete: ${state.qa.complete})`);
  console.log(`  Q&A Interaction ID: ${state.qa.interactionId || "none"}`);
  console.log(`  Revision Iteration: ${state.revisionIteration}/${LIMITS.maxRevisionIterations}`);
  console.log(`  Evidence Anchors:   ${state.evidence.length}`);
  console.log(`  Trace Mode:         ${cliConfig.verbose ? "verbose" : "quiet"}`);

  if (state.specialists.length > 0) {
    console.log(`  Active Specialists (${state.specialists.length}):`);
    for (const s of state.specialists) {
      console.log(`    - ${s.displayName} (${s.managedAgentId}): ${s.status}`);
    }
  } else {
    console.log(`  Active Specialists: none assigned yet`);
  }
  console.log("");
}

async function processUserTurn(inputLine) {
  const trimmed = inputLine.trim();
  if (!trimmed) return;

  // Handle slash commands
  if (trimmed.startsWith("/")) {
    const cmd = trimmed.split(/\s+/)[0].toLowerCase();
    switch (cmd) {
      case "/help":
        handleHelpCommand();
        return;
      case "/new":
      case "/reset":
        resetSessionState();
        console.log(
          `\n${SYMBOLS.success} Session state reset. Managed specialist agents remain reusable. Ready for a new medical question.\n`
        );
        return;
      case "/models":
        handleModelsCommand();
        return;
      case "/agents":
        await handleAgentsCommand();
        return;
      case "/status":
        handleStatusCommand();
        return;
      case "/verbose":
        cliConfig.verbose = true;
        console.log(`\n${SYMBOLS.success} Verbose research trace enabled.\n`);
        return;
      case "/quiet":
        cliConfig.verbose = false;
        console.log(`\n${SYMBOLS.success} Quiet mode enabled (internal trace reduced).\n`);
        return;
      case "/quit":
      case "/exit":
        console.log(`\n${SYMBOLS.success} Terminating MGA-1 session. Goodbye.\n`);
        process.exit(0);
        return;
      default:
        console.log(
          `\n${SYMBOLS.warning} Unknown command "${cmd}". Type ${ANSI.bold}/help${ANSI.reset} for available commands.\n`
        );
        return;
    }
  }

  // If previous conversation already finished (completed or abstained), start a fresh research session automatically
  if (state.phase === "completed" || state.phase === "abstained") {
    resetSessionState();
  }

  if (!state.metrics.startTimeMs) {
    state.metrics.startTimeMs = Date.now();
  }

  cliConfig.isRunningTask = true;
  cliConfig.activeAbortController = new AbortController();

  try {
    const qaOutcome = await runBaseModelQaTurn(trimmed);
    if (qaOutcome.ready) {
      await executeMgaResearchPipeline();
    }
  } catch (err) {
    activeSpinner.stop();
    process.stdout.write(ANSI.showCursor);
    if (cliConfig.activeAbortController?.signal?.aborted) {
      console.log(`\n${SYMBOLS.warning} ${ANSI.yellow}Current operation interrupted.${ANSI.reset}\n`);
    } else {
      console.error(
        `\n${SYMBOLS.failure} ${ANSI.brightRed}Error during MGA-1 execution:${ANSI.reset} ${formatApiError(err)}\n`
      );
    }
  } finally {
    cliConfig.isRunningTask = false;
    cliConfig.activeAbortController = null;
    process.stdout.write(ANSI.showCursor);
  }
}

// ============================================================================
// 17. STARTUP SELF-CHECK & MAIN INTERACTIVE LOOP
// ============================================================================

async function main() {
  // 1. Verify API key & initialize GoogleGenAI client
  initializeGenAiClient();

  // 2. Print startup banner & configured models
  printStartupBanner();

  // 3. Startup self-check status lines (do NOT pre-create all specialist agents)
  console.log(`${SYMBOLS.success} API key detected`);
  console.log(`${SYMBOLS.success} GoogleGenAI initialized`);
  console.log(`${SYMBOLS.success} Model registry loaded`);
  console.log(`${SYMBOLS.success} MGA-1 ready\n`);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: Boolean(process.stdin.isTTY),
  });

  const promptUser = () => {
    if (cliConfig.isRunningTask) return;
    console.log(`${ANSI.bold}You:${ANSI.reset}`);
    rl.setPrompt(`${ANSI.brightCyan}› ${ANSI.reset}`);
    rl.prompt();
  };

  // Graceful Ctrl+C handling
  const handleSigInt = () => {
    activeSpinner.stop();
    process.stdout.write(ANSI.showCursor);

    if (cliConfig.isRunningTask && cliConfig.activeAbortController) {
      cliConfig.activeAbortController.abort(new Error("Interrupted by user (Ctrl+C)"));
      for (const stream of cliConfig.activeStreams) {
        try {
          stream.close();
        } catch {}
      }
      cliConfig.activeStreams.clear();
      return;
    }

    console.log(`\n${SYMBOLS.warning} ${ANSI.yellow}Current operation interrupted. Type /quit or /exit to leave.${ANSI.reset}\n`);
    promptUser();
  };

  rl.on("SIGINT", handleSigInt);
  process.on("SIGINT", handleSigInt);

  process.on("unhandledRejection", (reason) => {
    activeSpinner.stop();
    process.stdout.write(ANSI.showCursor);
    console.error(`\n${SYMBOLS.warning} ${ANSI.yellow}Unhandled async error:${ANSI.reset} ${formatApiError(reason)}\n`);
  });

  // Handle line-by-line sequential execution (supports both TTY and piped test input)
  const lineQueue = [];
  let processingQueue = false;
  let inputClosed = false;

  const drainQueue = async () => {
    if (processingQueue) return;
    processingQueue = true;
    while (lineQueue.length > 0) {
      const nextLine = lineQueue.shift();
      await processUserTurn(nextLine);
      if (lineQueue.length === 0 && !inputClosed) {
        promptUser();
      }
    }
    processingQueue = false;
    if (inputClosed && lineQueue.length === 0) {
      process.exit(0);
    }
  };

  rl.on("line", (line) => {
    lineQueue.push(line);
    drainQueue();
  });

  rl.on("close", () => {
    inputClosed = true;
    if (!processingQueue && lineQueue.length === 0) {
      process.exit(0);
    }
  });

  promptUser();
}

main().catch((err) => {
  process.stdout.write(ANSI.showCursor);
  console.error(`\n${SYMBOLS.failure} Fatal startup error: ${formatApiError(err)}`);
  process.exit(1);
});
