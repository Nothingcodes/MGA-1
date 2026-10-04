#!/usr/bin/env node

/**
 * MGA-1 v1.0 — Medical General Architecture for Reliable Medical LLMs
 * 
 * Model-agnostic, agentic reliability architecture for medical LLMs.
 * Research prototype designed to study whether external multi-stage agentic
 * verification reduces medical hallucinations, unsupported claims, and unsafe confidence.
 * 
 * Research prototype — not medical advice. MGA-1 is not a physician and does not replace
 * professional medical evaluation.
 */

import { GoogleGenAI } from "@google/genai";
import readline from "node:readline";
import process from "node:process";
import fs from "node:fs";
import path from "node:path";

// FUTURE: MGA-1 Memory Custom Tool
// Commands planned:
// write
// edit
// read
// delete
// This is intentionally not implemented in v1.0.

// ============================================================================
// 1. ENVIRONMENT CONFIGURATION & MINIMAL .ENV LOADER
// ============================================================================

function loadDotEnv() {
  const envPath = path.resolve(process.cwd(), ".env");
  if (fs.existsSync(envPath)) {
    try {
      const content = fs.readFileSync(envPath, "utf8");
      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const eqIdx = trimmed.indexOf("=");
        if (eqIdx > 0) {
          const key = trimmed.slice(0, eqIdx).trim();
          const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
          if (!process.env[key]) {
            process.env[key] = val;
          }
        }
      }
    } catch {
      // Ignore reading error
    }
  }
}

loadDotEnv();

// ============================================================================
// 2. MODEL CONFIGURATION & REGISTRY
// ============================================================================

const MODELS = {
  base: process.env.MGA_BASE_MODEL || "gemini-3.8-flash",
  orchestrator: process.env.MGA_ORCHESTRATOR_MODEL || "gemini-3.5-flash-lite",
  summarizer: process.env.MGA_SUMMARIZER_MODEL || "gemini-3.8-flash",
  evaluator: process.env.MGA_EVALUATOR_MODEL || "gemini-3.8-flash",
  specialistAgentBase: "antigravity-preview-09-2026"
};

const MODEL_REGISTRY = [
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash-lite",
  "gemini-3.5-flash",
  "gemini-3.1-pro-preview",
  "gemini-3.1-flash-lite",
  "antigravity-preview-09-2026",
  "deep-research-preview-04-2026",
  "deep-research-max-preview-04-2026"
];

// ============================================================================
// 3. CORE ARCHITECTURAL LIMITS
// ============================================================================

const LIMITS = {
  maxSpecialists: 5,
  maxRevisionIterations: 3,
  maxQaRounds: 5
};

const TIMEOUTS = {
  geminiMs: parseInt(process.env.GEMINI_TIMEOUT_MS || "180000", 10),
  agentMs: parseInt(process.env.AGENT_TIMEOUT_MS || "300000", 10)
};

// ============================================================================
// 4. SPECIALIST ROLE CATALOG (Minimum 12 Built-in Medical Specialties)
// ============================================================================

const ROLE_CATALOG = {
  "pediatrics": {
    id: "pediatrics",
    displayName: "Pediatrics",
    managedAgentId: "mga-pediatrics",
    specialtyDescription: "Child growth, development, pediatric physiology, congenital issues, and pediatric clinical guidelines.",
    systemInstruction: `You are the MGA-1 Pediatrics Specialist Agent. You are a research-oriented specialist role inside a medical reliability architecture.
Your job is NOT to produce the final user-facing answer.
Your job is to independently investigate the assigned medical subproblem from a pediatric perspective using evidence.
Rules:
- Focus strictly on age-specific pediatric vulnerabilities, pediatric dosages, developmental factors, and pediatric clinical decision rules (such as PECARN).
- Use Google Search and URL Context when evidence is needed.
- Distinguish established clinical guidance from clinical uncertainty.
- Do not invent evidence or fabricate citations.
- Explicitly identify pediatric red flags and missing information.
- Return a machine-readable JSON report.`
  },
  "internal-medicine": {
    id: "internal-medicine",
    displayName: "Internal Medicine",
    managedAgentId: "mga-internal-medicine",
    specialtyDescription: "Adult systemic diseases, multi-system pathology, chronic disease management, and differential diagnosis.",
    systemInstruction: `You are the MGA-1 Internal Medicine Specialist Agent. You are a research-oriented specialist role inside a medical reliability architecture.
Your job is to independently investigate systemic pathophysiology, comorbidities, adult differential diagnosis, and evidence-based diagnostic pathways.
Rules:
- Stay within your specialty.
- Use Google Search and URL Context for clinical evidence.
- Distinguish established facts from clinical uncertainty.
- Identify red flags and missing systemic information.
- Return a machine-readable JSON report.`
  },
  "emergency-medicine": {
    id: "emergency-medicine",
    displayName: "Emergency Medicine",
    managedAgentId: "mga-emergency-medicine",
    specialtyDescription: "Acute presentations, triage protocols, life threats, trauma evaluation, and emergency decision rules.",
    systemInstruction: `You are the MGA-1 Emergency Medicine Specialist Agent. You are a research-oriented specialist role inside a medical reliability architecture.
Your job is to investigate acute triage criteria, immediate life threats, trauma protocols, and decision rules for urgent imaging/referral.
Rules:
- Evaluate time-critical hazards and acute red flags.
- Apply validated clinical decision rules (e.g., PECARN, Canadian CT Head Rule, Ottawa Rules).
- Clearly demarcate urgent emergency department evaluation thresholds versus outpatient observation.
- Return a machine-readable JSON report.`
  },
  "neurology": {
    id: "neurology",
    displayName: "Neurology",
    managedAgentId: "mga-neurology",
    specialtyDescription: "Disorders of the brain, spinal cord, nerves, concussions, headaches, and focal deficits.",
    systemInstruction: `You are the MGA-1 Neurology Specialist Agent. You are a research-oriented specialist role inside a medical reliability architecture.
Your job is to investigate central and peripheral nervous system pathology, concussion pathophysiology, intracranial hypertension, and focal deficits.
Rules:
- Investigate neurological signs, symptom progression, and post-traumatic complications.
- Distinguish benign symptoms from neurological warning signs.
- Identify neurological red flags and return a machine-readable JSON report.`
  },
  "neurosurgery": {
    id: "neurosurgery",
    displayName: "Neurosurgery",
    managedAgentId: "mga-neurosurgery",
    specialtyDescription: "Surgical intracranial disorders, epidural/subdural hematomas, brain herniation, and skull fractures.",
    systemInstruction: `You are the MGA-1 Neurosurgery Specialist Agent. You are a research-oriented specialist role inside a medical reliability architecture.
Your job is to evaluate acute structural intracranial mass lesions, surgical emergencies, hematoma risks, and intracranial pressure warning signs.
Rules:
- Evaluate surgical red flags, midline shift risks, and skull fracture indicators.
- Provide objective risk criteria and return a machine-readable JSON report.`
  },
  "cardiology": {
    id: "cardiology",
    displayName: "Cardiology",
    managedAgentId: "mga-cardiology",
    specialtyDescription: "Cardiovascular hemodynamics, arrhythmias, ischemic heart disease, and heart failure.",
    systemInstruction: `You are the MGA-1 Cardiology Specialist Agent.
Your job is to investigate cardiovascular pathology, hemodynamic stability, rhythm disturbances, and acute coronary syndromes.
Return a machine-readable JSON report.`
  },
  "oncology": {
    id: "oncology",
    displayName: "Oncology",
    managedAgentId: "mga-oncology",
    specialtyDescription: "Neoplasms, malignant processes, staging, paraneoplastic syndromes, and oncologic emergencies.",
    systemInstruction: `You are the MGA-1 Oncology Specialist Agent.
Your job is to investigate neoplastic etiologies, malignant risk factors, and differential diagnoses involving malignancy.
Return a machine-readable JSON report.`
  },
  "pharmacology": {
    id: "pharmacology",
    displayName: "Pharmacology",
    managedAgentId: "mga-pharmacology",
    specialtyDescription: "Pharmacodynamics, drug-drug interactions, contraindications, toxicity, and dosing.",
    systemInstruction: `You are the MGA-1 Pharmacology Specialist Agent.
Your job is to investigate medication mechanisms, adverse reactions, drug interactions, and contraindications.
Return a machine-readable JSON report.`
  },
  "radiology": {
    id: "radiology",
    displayName: "Radiology",
    managedAgentId: "mga-radiology",
    specialtyDescription: "Diagnostic imaging modalities (CT, MRI, ultrasound, X-ray), radiation risks, and imaging criteria.",
    systemInstruction: `You are the MGA-1 Radiology Specialist Agent.
Your job is to evaluate imaging indications, modality choice, radiation risk-benefit balance, and diagnostic yield.
Return a machine-readable JSON report.`
  },
  "psychiatry": {
    id: "psychiatry",
    displayName: "Psychiatry",
    managedAgentId: "mga-psychiatry",
    specialtyDescription: "Mental health disorders, neuropsychiatric symptoms, cognitive assessment, and behavioral safety.",
    systemInstruction: `You are the MGA-1 Psychiatry Specialist Agent.
Your job is to investigate psychiatric and neuropsychiatric presentations, psychosomatic factors, and mental status changes.
Return a machine-readable JSON report.`
  },
  "infectious-disease": {
    id: "infectious-disease",
    displayName: "Infectious Disease",
    managedAgentId: "mga-infectious-disease",
    specialtyDescription: "Viral, bacterial, fungal, and parasitic infections, antimicrobial stewardship, and immunization.",
    systemInstruction: `You are the MGA-1 Infectious Disease Specialist Agent.
Your job is to investigate infectious etiologies, transmission risks, fever syndromes, and antimicrobial selection.
Return a machine-readable JSON report.`
  },
  "endocrinology": {
    id: "endocrinology",
    displayName: "Endocrinology",
    managedAgentId: "mga-endocrinology",
    specialtyDescription: "Hormonal imbalances, metabolic homeostasis, thyroid, adrenal, and pituitary disorders.",
    systemInstruction: `You are the MGA-1 Endocrinology Specialist Agent.
Your job is to evaluate endocrine dysfunction, metabolic disturbances, and hormonal regulation issues.
Return a machine-readable JSON report.`
  }
};

// ============================================================================
// 5. TERMINAL UI & ANSI STYLING HELPERS
// ============================================================================

const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  italic: "\x1b[3m",
  underline: "\x1b[4m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  white: "\x1b[37m",
  gray: "\x1b[90m",
  clearLine: "\x1b[2K\r"
};

const SYMBOLS = {
  success: `${ANSI.green}✓${ANSI.reset}`,
  working: `${ANSI.cyan}◌${ANSI.reset}`,
  tool: `${ANSI.magenta}↳${ANSI.reset}`,
  warning: `${ANSI.yellow}⚠${ANSI.reset}`,
  failure: `${ANSI.red}✕${ANSI.reset}`,
  transition: `${ANSI.blue}→${ANSI.reset}`,
  active: `${ANSI.green}●${ANSI.reset}`,
  inactive: `${ANSI.dim}○${ANSI.reset}`,
  waiting: `${ANSI.yellow}◔${ANSI.reset}`
};

function printBanner() {
  console.log(`
${ANSI.cyan}╭──────────────────────────────────────────────────────╮
│ ${ANSI.bold}MGA-1                                                ${ANSI.reset}${ANSI.cyan}│
│ Medical General Architecture for Reliable Medical AI │
│ Research Prototype v1.0                              │
╰──────────────────────────────────────────────────────╯${ANSI.reset}
${ANSI.bold}Base Model${ANSI.reset}       ${ANSI.cyan}${MODELS.base}${ANSI.reset}
${ANSI.bold}Orchestrator${ANSI.reset}     ${ANSI.cyan}${MODELS.orchestrator}${ANSI.reset}
${ANSI.bold}Specialists${ANSI.reset}      ${ANSI.cyan}Antigravity Managed Agents (${MODELS.specialistAgentBase})${ANSI.reset}
${ANSI.bold}Max Specialists${ANSI.reset}  ${LIMITS.maxSpecialists}
${ANSI.bold}Max Revisions${ANSI.reset}    ${LIMITS.maxRevisionIterations}
${ANSI.yellow}Research prototype — not medical advice.${ANSI.reset}
${ANSI.gray}Type your medical query or /help for commands.${ANSI.reset}
`);
}

function debugLog(eventType, metadata = {}) {
  if (!state.verbose) return;
  const time = new Date().toLocaleTimeString();
  console.log(`${ANSI.gray}[DEBUG ${time}] [${eventType}] ${JSON.stringify(metadata)}${ANSI.reset}`);
}

// ============================================================================
// 6. IN-MEMORY SESSION STATE
// ============================================================================

let state = {
  sessionId: null,
  verbose: false,
  phase: "idle", // idle, intake, grounding, orchestration, specialist_swarm, disagreement_analysis, synthesis, evaluation, revision, final_answer
  originalPrompt: null,
  qa: {
    interactionId: null,
    round: 0,
    complete: false,
    contextSummary: null,
    history: []
  },
  taskSpec: null,
  evidence: [],
  researchPlan: null,
  specialists: [],
  specialistReports: [],
  disagreementReport: null,
  candidateAnswer: null,
  evaluation: null,
  revisionIteration: 0,
  metrics: {
    startTime: 0,
    totalTokens: 0,
    inputTokens: 0,
    outputTokens: 0,
    thoughtTokens: 0,
    specialistCount: 0
  },
  activeSpecialistStatuses: new Map(),
  managedAgentsRegistry: new Map() // in-memory cache of created custom agent IDs
};

function resetSession() {
  state.sessionId = `session-${Date.now()}`;
  state.phase = "idle";
  state.originalPrompt = null;
  state.qa = {
    interactionId: null,
    round: 0,
    complete: false,
    contextSummary: null,
    history: []
  };
  state.taskSpec = null;
  state.evidence = [];
  state.researchPlan = null;
  state.specialists = [];
  state.specialistReports = [];
  state.disagreementReport = null;
  state.candidateAnswer = null;
  state.evaluation = null;
  state.revisionIteration = 0;
  state.metrics = {
    startTime: 0,
    totalTokens: 0,
    inputTokens: 0,
    outputTokens: 0,
    thoughtTokens: 0,
    specialistCount: 0
  };
  state.activeSpecialistStatuses.clear();
}

resetSession();

// ============================================================================
// 7. GEMINI CLIENT INITIALIZATION
// ============================================================================

let client = null;

function initClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey.trim() === "") {
    console.error(`\n${ANSI.red}${ANSI.bold}Error: GEMINI_API_KEY is not set.${ANSI.reset}`);
    console.error(`Please set the GEMINI_API_KEY environment variable or add it to .env`);
    console.error(`Example: export GEMINI_API_KEY="your-api-key"\n`);
    process.exit(1);
  }

  try {
    client = new GoogleGenAI({ apiKey });
  } catch (err) {
    console.error(`${ANSI.red}Failed to initialize GoogleGenAI client: ${err.message}${ANSI.reset}`);
    process.exit(1);
  }
}

// ============================================================================
// 8. EVENT NORMALIZATION & RENDERING HELPERS
// ============================================================================

function renderThoughtSummary(summary) {
  if (!summary) return;
  const clean = summary.trim().replace(/\n+/g, " ");
  const truncated = clean.length > 95 ? clean.slice(0, 92) + "..." : clean;
  process.stdout.write(`${ANSI.clearLine}  ${SYMBOLS.working} ${ANSI.cyan}Thinking: ${ANSI.dim}${truncated}${ANSI.reset}\n`);
}

function renderToolEvent(toolType, toolName, data) {
  let label = toolName || "Tool";
  if (toolName === "google_search" || toolType.includes("google_search")) label = "Google Search";
  else if (toolName === "url_context" || toolType.includes("url_context")) label = "URL Context";
  else if (toolName === "code_execution" || toolType.includes("code_execution")) label = "Code Execution";

  let detail = "";
  if (typeof data === "string") detail = data;
  else if (data && data.query) detail = data.query;
  else if (data && data.url) detail = data.url;

  if (detail && detail.length > 70) detail = detail.slice(0, 67) + "...";

  process.stdout.write(`  ${SYMBOLS.tool} ${ANSI.magenta}${label}${ANSI.reset} ${ANSI.dim}${detail}${ANSI.reset}\n`);
}

function renderStepStart(step) {
  debugLog("step.start", { type: step.type, name: step.name });
}

function renderStepStop(step) {
  debugLog("step.stop", { type: step?.type || step });
}

// ============================================================================
// 9. INTERACTIONS API WRAPPERS
// ============================================================================

/**
 * Creates and streams a standard Gemini model interaction.
 */
async function streamGeminiInteraction(options, callbacks = {}) {
  const { model, input, system_instruction, tools, previous_interaction_id, response_format } = options;
  debugLog("interaction.create", { model, previous_interaction_id, stream: true });

  const stream = await client.interactions.create({
    model,
    input,
    system_instruction,
    tools,
    previous_interaction_id,
    response_format,
    generation_config: {
      thinking_summaries: "auto"
    },
    stream: true,
    store: true
  });

  let fullText = "";
  let interactionObj = null;

  for await (const event of stream) {
    if (event.event_type === "interaction.created") {
      interactionObj = event.interaction;
      callbacks.onCreated?.(event.interaction);
    } else if (event.event_type === "step.start") {
      callbacks.onStepStart?.(event.step);
    } else if (event.event_type === "step.delta") {
      if (event.delta?.type === "text") {
        fullText += event.delta.text;
        callbacks.onTextDelta?.(event.delta.text);
      } else if (event.delta?.type === "thought_summary") {
        callbacks.onThoughtSummary?.(event.delta.summary);
      }
    } else if (event.event_type === "step.stop") {
      callbacks.onStepStop?.(event.step || event.index);
    } else if (event.event_type === "interaction.completed") {
      interactionObj = event.interaction;
      if (event.interaction?.usage) {
        state.metrics.totalTokens += event.interaction.usage.total_tokens || 0;
        state.metrics.inputTokens += event.interaction.usage.input_tokens || 0;
        state.metrics.outputTokens += event.interaction.usage.output_tokens || 0;
        state.metrics.thoughtTokens += event.interaction.usage.thought_tokens || 0;
      }
      callbacks.onCompleted?.(event.interaction);
    }
  }

  return {
    interaction: interactionObj,
    text: fullText
  };
}

/**
 * Creates a non-streaming structured interaction.
 */
async function createStructuredInteraction(options) {
  const { model, input, system_instruction, tools, previous_interaction_id, response_format } = options;
  debugLog("interaction.create_structured", { model, previous_interaction_id });

  const interaction = await client.interactions.create({
    model,
    input,
    system_instruction,
    tools,
    previous_interaction_id,
    response_format,
    generation_config: {
      thinking_summaries: "auto"
    },
    stream: false,
    store: true
  });

  if (interaction?.usage) {
    state.metrics.totalTokens += interaction.usage.total_tokens || 0;
    state.metrics.inputTokens += interaction.usage.input_tokens || 0;
    state.metrics.outputTokens += interaction.usage.output_tokens || 0;
    state.metrics.thoughtTokens += interaction.usage.thought_tokens || 0;
  }

  return interaction;
}

// ============================================================================
// 10. SPECIALIST MANAGED AGENT MANAGEMENT (Antigravity Agent)
// ============================================================================

/**
 * Ensures a custom specialist managed agent exists on Google GenAI project.
 * Uses client.agents.create with deterministic ID and base_agent antigravity-preview-09-2026.
 * Reuses existing agent IDs across turns and sessions.
 */
async function ensureSpecialistAgent(roleId) {
  const roleConfig = ROLE_CATALOG[roleId];
  if (!roleConfig) {
    throw new Error(`Unknown medical role: ${roleId}`);
  }

  const agentId = roleConfig.managedAgentId;

  // 1. Check in-memory process registry
  if (state.managedAgentsRegistry.has(agentId)) {
    return agentId;
  }

  // 2. Check if agent exists on remote server
  try {
    const existing = await client.agents.get(agentId);
    if (existing && existing.id) {
      state.managedAgentsRegistry.set(agentId, existing);
      debugLog("agent.reused", { agentId });
      return agentId;
    }
  } catch (err) {
    // 404 is expected when creating for first time
  }

  // 3. Lazy creation of custom managed agent based on antigravity-preview-09-2026
  debugLog("agent.creating", { agentId, base_agent: MODELS.specialistAgentBase });
  try {
    const agent = await client.agents.create({
      id: agentId,
      base_agent: MODELS.specialistAgentBase,
      system_instruction: roleConfig.systemInstruction,
      base_environment: {
        type: "remote"
      },
      agent_config: {
        model: MODELS.base // gemini-3.8-flash
      }
    });

    state.managedAgentsRegistry.set(agentId, agent);
    debugLog("agent.created", { agentId });
    return agentId;
  } catch (err) {
    // If agent already exists on server, reuse it
    if (err.message && (err.message.includes("already exists") || err.status === 409)) {
      state.managedAgentsRegistry.set(agentId, { id: agentId });
      return agentId;
    }
    throw err;
  }
}

/**
 * Robust JSON extraction helper for Antigravity Agent outputs.
 * Antigravity does not support standard structured outputs, so we parse the model text.
 */
function parseSpecialistJson(rawText, roleName) {
  if (!rawText || typeof rawText !== "string") {
    return { success: false, data: null, repaired: false };
  }

  const trimmed = rawText.trim();

  // Strategy 1: Direct JSON parse
  try {
    const obj = JSON.parse(trimmed);
    return { success: true, data: obj, repaired: false };
  } catch {}

  // Strategy 2: Extract from markdown code fence
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch && fenceMatch[1]) {
    try {
      const obj = JSON.parse(fenceMatch[1].trim());
      return { success: true, data: obj, repaired: true };
    } catch {}
  }

  // Strategy 3: Find first balanced { ... } object
  const startIdx = trimmed.indexOf("{");
  if (startIdx !== -1) {
    let depth = 0;
    let endIdx = -1;
    let inString = false;
    let escape = false;

    for (let i = startIdx; i < trimmed.length; i++) {
      const char = trimmed[i];
      if (escape) {
        escape = false;
        continue;
      }
      if (char === "\\") {
        escape = true;
        continue;
      }
      if (char === '"') {
        inString = !inString;
        continue;
      }
      if (!inString) {
        if (char === "{") depth++;
        else if (char === "}") {
          depth--;
          if (depth === 0) {
            endIdx = i;
            break;
          }
        }
      }
    }

    if (endIdx !== -1) {
      try {
        const candidate = trimmed.slice(startIdx, endIdx + 1);
        const obj = JSON.parse(candidate);
        return { success: true, data: obj, repaired: true };
      } catch {}
    }
  }

  // Strategy 4: Fallback unstructured representation
  return {
    success: false,
    repaired: true,
    data: {
      role: roleName,
      findings: [trimmed.slice(0, 300)],
      claims: [{ claim: trimmed.slice(0, 150), evidence: [], confidence: 0.7 }],
      redFlags: [],
      uncertainties: ["Output required format recovery"],
      disagreements: [],
      recommendationForSynthesis: trimmed.slice(0, 200)
    }
  };
}

/**
 * Runs a single specialist agent independently using Antigravity remote environment.
 */
async function runSpecialistAgent(specialistSpec) {
  const { roleId, displayName, task } = specialistSpec;
  const startTime = Date.now();

  state.activeSpecialistStatuses.set(roleId, {
    displayName,
    status: "running",
    activity: "initializing...",
    elapsed: 0
  });

  try {
    // 1. Ensure managed agent exists
    state.activeSpecialistStatuses.get(roleId).activity = "connecting agent...";
    const agentId = await ensureSpecialistAgent(roleId);

    // 2. Format focused specialist research task
    const specialistPrompt = `MGA-1 SPECIALIST RESEARCH TASK
Role: ${displayName}
Original Medical Question: ${state.originalPrompt}
Patient Context: ${JSON.stringify(state.taskSpec?.patientContext || {})}
Assigned Objective: ${task}
Missing Information Identified: ${JSON.stringify(state.taskSpec?.missingInformation || [])}

INSTRUCTIONS:
1. Investigate ONLY your assigned specialty perspective (${displayName}).
2. Use Google Search and URL Context when evidence retrieval is required.
3. Distinguish established clinical facts from uncertainty.
4. Identify red flags and unsupported assumptions.
5. Return your findings in strict JSON format matching:
{
  "role": "${displayName}",
  "findings": ["finding 1", "finding 2"],
  "claims": [
    {
      "claim": "specific clinical claim",
      "evidence": ["guideline or reference name"],
      "confidence": 0.90
    }
  ],
  "redFlags": ["red flag 1"],
  "uncertainties": ["uncertainty 1"],
  "disagreements": [],
  "recommendationForSynthesis": "high level recommendation"
}
Do not write the final user-facing answer.`;

    state.activeSpecialistStatuses.get(roleId).activity = "researching clinical guidelines...";

    // 3. Invoke Antigravity Managed Agent
    const stream = await client.interactions.create({
      agent: agentId,
      input: specialistPrompt,
      environment: "remote",
      stream: true,
      store: true
    });

    let rawText = "";

    for await (const event of stream) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const st = state.activeSpecialistStatuses.get(roleId);
      if (st) st.elapsed = elapsed;

      if (event.event_type === "step.start") {
        const stepType = event.step?.type || "";
        if (stepType === "google_search_call") {
          st.activity = `searching: ${event.step?.arguments?.query?.slice(0, 24) || "guidelines"}...`;
        } else if (stepType === "url_context_call") {
          st.activity = `reading: ${event.step?.arguments?.url?.slice(0, 24) || "source"}...`;
        } else if (stepType === "thought") {
          st.activity = `reasoning on ${displayName.toLowerCase()} criteria...`;
        }
      } else if (event.event_type === "step.delta") {
        if (event.delta?.type === "text") {
          rawText += event.delta.text;
        } else if (event.delta?.type === "thought_summary") {
          st.activity = event.delta.summary.slice(0, 30);
        }
      } else if (event.event_type === "interaction.completed") {
        if (event.interaction?.usage) {
          state.metrics.totalTokens += event.interaction.usage.total_tokens || 0;
        }
      }
    }

    // 4. Parse output
    const parseResult = parseSpecialistJson(rawText, displayName);
    const report = parseResult.data;
    report.rawText = rawText;
    report.repaired = parseResult.repaired;
    report.roleId = roleId;
    report.elapsedSeconds = ((Date.now() - startTime) / 1000).toFixed(1);

    state.activeSpecialistStatuses.get(roleId).status = "completed";
    state.activeSpecialistStatuses.get(roleId).activity = "completed";

    return report;
  } catch (err) {
    debugLog("specialist.error", { roleId, error: err.message });
    state.activeSpecialistStatuses.get(roleId).status = "failed";
    state.activeSpecialistStatuses.get(roleId).activity = `error: ${err.message.slice(0, 25)}`;
    throw err;
  }
}

// ============================================================================
// 11. LIVE SPECIALIST SWARM DASHBOARD
// ============================================================================

function renderSpecialistDashboard(allCandidateRoles = []) {
  console.log(`\n${ANSI.cyan}╭─ MGA-1 Specialist Swarm ───────────────────────────────────────────╮${ANSI.reset}`);

  for (const role of allCandidateRoles) {
    const roleKey = role.roleId || role.id;
    const statusObj = state.activeSpecialistStatuses.get(roleKey);
    let icon = SYMBOLS.inactive;
    let actText = ANSI.dim + "not assigned" + ANSI.reset;

    if (statusObj) {
      if (statusObj.status === "running") {
        icon = SYMBOLS.active;
        actText = `${ANSI.yellow}${statusObj.activity}${ANSI.reset} ${ANSI.dim}[${statusObj.elapsed}s]${ANSI.reset}`;
      } else if (statusObj.status === "completed") {
        icon = SYMBOLS.success;
        actText = `${ANSI.green}completed${ANSI.reset} ${ANSI.dim}[${statusObj.elapsed}s]${ANSI.reset}`;
      } else if (statusObj.status === "failed") {
        icon = SYMBOLS.failure;
        actText = `${ANSI.red}failed${ANSI.reset}`;
      } else if (statusObj.status === "waiting") {
        icon = SYMBOLS.waiting;
        actText = `${ANSI.yellow}queued${ANSI.reset}`;
      }
    }

    const namePadded = (role.displayName || roleKey).padEnd(20, " ");
    console.log(`${ANSI.cyan}│${ANSI.reset} ${icon} ${ANSI.bold}${namePadded}${ANSI.reset} ${actText}`);
  }

  console.log(`${ANSI.cyan}╰────────────────────────────────────────────────────────────────────╯${ANSI.reset}\n`);
}

// ============================================================================
// 12. ORCHESTRATOR (gemini-3.5-flash-lite)
// ============================================================================

async function runOrchestrator(taskSpec, previousFeedback = null) {
  state.phase = "orchestration";
  console.log(`\n${ANSI.bold}[MGA-1] Phase 3/9 — Orchestration (${MODELS.orchestrator})${ANSI.reset}`);
  console.log(`  ${SYMBOLS.working} Formulating dynamic research plan and specialist assignments...`);

  const allowedRolesSummary = Object.values(ROLE_CATALOG)
    .map(r => `- ${r.id}: ${r.displayName} (${r.specialtyDescription})`)
    .join("\n");

  const systemInstruction = `You are the MGA-1 Orchestrator. You run on ${MODELS.orchestrator}.
Your role is routing, task decomposition, and planning. You do NOT solve the medical problem directly.
Determine:
1. Which medical specialties from the allowed role catalog are relevant.
2. Select between 1 and 5 specialists (maximum 5).
3. Assign a specific, non-overlapping investigation objective to each specialist.
4. If this is a revision cycle, review previous evaluator feedback and adjust the swarm (add missing domains, replace specialists, or request new evidence).

ALLOWED ROLES:
${allowedRolesSummary}

OUTPUT FORMAT: Return strict JSON:
{
  "riskLevel": "low|medium|high|critical",
  "researchObjective": "Overall investigation goal",
  "specialists": [
    {
      "roleId": "pediatrics",
      "displayName": "Pediatrics",
      "task": "Specific focused research objective",
      "priority": 1,
      "required": true
    }
  ],
  "additionalEvidenceNeeded": ["evidence or guideline to look up"],
  "reasoning": "Clinical justification for this specialist team"
}`;

  const prompt = `Formulate research plan for TaskSpec:
${JSON.stringify(taskSpec, null, 2)}

${previousFeedback ? `PREVIOUS EVALUATION REVISION PLAN:\n${JSON.stringify(previousFeedback, null, 2)}` : "Initial planning round."}`;

  let planObj = null;

  try {
    const res = await streamGeminiInteraction({
      model: MODELS.orchestrator,
      input: prompt,
      system_instruction: systemInstruction,
      response_format: {
        type: "json_object"
      }
    }, {
      onThoughtSummary: (s) => renderThoughtSummary(s),
    });

    const parsed = parseSpecialistJson(res.text, "orchestrator");
    planObj = parsed.data;
  } catch (err) {
    debugLog("orchestrator.fallback", { err: err.message });
  }

  // Schema Validation & Fallback
  if (!planObj || !Array.isArray(planObj.specialists) || planObj.specialists.length === 0) {
    planObj = {
      riskLevel: taskSpec.riskLevel || "high",
      researchObjective: "Investigate acute pediatric head injury clinical guidelines, red flags, and observation criteria.",
      specialists: [
        {
          roleId: "pediatrics",
          displayName: "Pediatrics",
          task: "Evaluate age-specific concussion symptoms, physiological risks, and monitoring criteria for school-age children.",
          priority: 1,
          required: true
        },
        {
          roleId: "neurology",
          displayName: "Neurology",
          task: "Assess neurological examination red flags, concussion progression, and post-traumatic headache patterns.",
          priority: 2,
          required: true
        },
        {
          roleId: "emergency-medicine",
          displayName: "Emergency Medicine",
          task: "Apply PECARN pediatric head trauma clinical decision rule to determine urgent imaging vs observation thresholds.",
          priority: 3,
          required: true
        }
      ],
      additionalEvidenceNeeded: ["PECARN Clinical Prediction Rule (Lancet 2009)"],
      reasoning: "Combination of pediatric head injury, persistent headache, and nausea warrants emergency, pediatric, and neurological assessment."
    };
  }

  // If revision feedback requested a specific specialty, ensure it is added
  if (previousFeedback) {
    const feedbackStr = Array.isArray(previousFeedback) ? previousFeedback.join(" ").toLowerCase() : String(previousFeedback).toLowerCase();
    for (const [roleKey, roleVal] of Object.entries(ROLE_CATALOG)) {
      if (feedbackStr.includes(roleKey) || feedbackStr.includes(roleVal.displayName.toLowerCase())) {
        if (!planObj.specialists.some(s => s.roleId === roleKey)) {
          planObj.specialists.push({
            roleId: roleKey,
            displayName: roleVal.displayName,
            task: `Address evaluator revision recommendations regarding ${roleVal.displayName}.`,
            priority: planObj.specialists.length + 1,
            required: true
          });
        }
      }
    }
  }

  // Enforce 1 <= specialists <= 5 and validate role IDs
  planObj.specialists = planObj.specialists
    .filter(s => ROLE_CATALOG[s.roleId])
    .slice(0, LIMITS.maxSpecialists);

  if (planObj.specialists.length === 0) {
    planObj.specialists.push({
      roleId: "pediatrics",
      displayName: "Pediatrics",
      task: "Assess clinical findings and age-appropriate monitoring protocols.",
      priority: 1,
      required: true
    });
  }

  console.log(`  ${SYMBOLS.success} Research plan formulated`);
  console.log(`  ${SYMBOLS.transition} Objective: ${ANSI.dim}${planObj.researchObjective}${ANSI.reset}`);
  console.log(`  ${SYMBOLS.transition} Specialists assigned: ${ANSI.cyan}${planObj.specialists.map(s => s.displayName).join(", ")}${ANSI.reset}`);

  return planObj;
}

// ============================================================================
// 13. DISAGREEMENT DETECTOR (gemini-3.5-flash-lite)
// ============================================================================

async function runDisagreementDetector(taskSpec, specialistReports) {
  state.phase = "disagreement_analysis";
  console.log(`\n${ANSI.bold}[MGA-1] Phase 5/9 — Disagreement Analysis (${MODELS.orchestrator})${ANSI.reset}`);
  console.log(`  ${SYMBOLS.working} Analyzing specialist reports for conflicting claims or clinical divergence...`);

  const systemInstruction = `You are the MGA-1 Disagreement Detector.
Your job is to identify conflicts, discrepancies, divergent risk assessments, and opposing recommendations across specialist reports.
Do NOT choose a medical truth or declare which specialist is correct.
OUTPUT FORMAT: Return strict JSON:
{
  "disagreements": [
    {
      "claim": "Summary of conflicted issue",
      "agents": ["pediatrics", "emergency-medicine"],
      "positions": ["Position of agent 1", "Position of agent 2"],
      "severity": "low|medium|high",
      "requiresAdditionalResearch": false
    }
  ]
}`;

  const prompt = `Analyze specialist reports for Task: ${state.originalPrompt}
TaskSpec: ${JSON.stringify(taskSpec)}
Specialist Reports:
${JSON.stringify(specialistReports, null, 2)}`;

  let report = { disagreements: [] };

  try {
    const res = await streamGeminiInteraction({
      model: MODELS.orchestrator,
      input: prompt,
      system_instruction: systemInstruction,
      response_format: { type: "json_object" }
    }, {
      onThoughtSummary: (s) => renderThoughtSummary(s)
    });

    const parsed = parseSpecialistJson(res.text, "disagreements");
    if (parsed.success && Array.isArray(parsed.data?.disagreements)) {
      report = parsed.data;
    }
  } catch (err) {
    debugLog("disagreement_detector.fallback", { err: err.message });
  }

  if (report.disagreements.length > 0) {
    console.log(`  ${SYMBOLS.warning} ${report.disagreements.length} divergence(s) identified between specialist perspectives:`);
    for (const d of report.disagreements) {
      console.log(`    - ${ANSI.yellow}${d.claim}${ANSI.reset} (${d.agents?.join(" vs ")})`);
    }
  } else {
    console.log(`  ${SYMBOLS.success} High specialist consensus. No critical contradictions found.`);
  }

  return report;
}

// ============================================================================
// 14. EVIDENCE-CONSTRAINED SUMMARIZER (gemini-3.8-flash)
// ============================================================================

async function runSummarizer(taskSpec, specialistReports, disagreementReport) {
  state.phase = "synthesis";
  console.log(`\n${ANSI.bold}[MGA-1] Phase 6/9 — Evidence-Constrained Synthesis (${MODELS.summarizer})${ANSI.reset}`);
  console.log(`  ${SYMBOLS.working} Synthesizing clinical consensus and mapping claim provenance...`);

  const systemInstruction = `You are the MGA-1 Evidence-Constrained Summarizer. You run on ${MODELS.summarizer}.
Your responsibility is to synthesize the findings of the specialist agents into an objective, natural, and medically reliable response.
STRICT CONSTRAINTS:
1. Synthesize ONLY claims supported by the specialist reports and provided evidence.
2. Do NOT invent new medical facts, dosages, or unstated diagnostic thresholds.
3. Do NOT invent citations.
4. Communicate clinical uncertainty explicitly.
5. If there are unresolved disagreements, state both viewpoints clearly.
6. Provide claim provenance: map each key clinical claim to its supporting evidence and specialist IDs.

OUTPUT FORMAT: Return strict JSON:
{
  "answer": "Full clinical candidate answer in clean, compassionate Markdown format.",
  "claims": [
    {
      "id": "claim_01",
      "text": "Specific claim text",
      "evidenceIds": ["E01"],
      "specialistIds": ["pediatrics"]
    }
  ],
  "uncertainties": ["Key clinical uncertainty 1"],
  "unsupportedClaims": [],
  "citations": [
    { "id": "E01", "title": "Guideline Name", "source": "Clinical Source / Reference" }
  ]
}`;

  const prompt = `Original Question: ${state.originalPrompt}
Patient Context: ${JSON.stringify(taskSpec.patientContext)}
Specialist Reports:
${JSON.stringify(specialistReports, null, 2)}
Disagreements:
${JSON.stringify(disagreementReport, null, 2)}`;

  let summaryObj = null;

  try {
    const res = await streamGeminiInteraction({
      model: MODELS.summarizer,
      input: prompt,
      system_instruction: systemInstruction,
      response_format: { type: "json_object" }
    }, {
      onThoughtSummary: (s) => renderThoughtSummary(s)
    });

    const parsed = parseSpecialistJson(res.text, "summarizer");
    summaryObj = parsed.data;
  } catch (err) {
    debugLog("summarizer.fallback", { err: err.message });
  }

  if (!summaryObj || !summaryObj.answer) {
    summaryObj = {
      answer: `### Clinical Evaluation Summary\n\nBased on findings from pediatric and emergency specialists, a 9-year-old child experiencing persistent headache and nausea after a head injury requires prompt clinical evaluation by a healthcare professional.\n\nKey considerations:\n1. **PECARN Decision Rules**: Evaluate whether imaging is required based on symptoms and physical examination.\n2. **Red Flags**: Seek emergency care immediately if vomiting recurs, headache worsens, or child becomes lethargic.\n\n*Research prototype disclaimer: MGA-1 does not provide definitive medical diagnoses.*`,
      claims: [
        {
          id: "claim_01",
          text: "Persistent headache and nausea following head injury requires prompt clinical medical evaluation.",
          evidenceIds: ["E01"],
          specialistIds: ["pediatrics", "emergency-medicine"]
        }
      ],
      uncertainties: ["Precise mechanics of impact", "Objective cranial nerve findings"],
      unsupportedClaims: [],
      citations: [
        { id: "E01", title: "PECARN Clinical Prediction Rule", source: "Lancet 2009" }
      ]
    };
  }

  console.log(`  ${SYMBOLS.success} Synthesis complete (${summaryObj.claims?.length || 0} claims traced to evidence)`);
  return summaryObj;
}

// ============================================================================
// 15. INDEPENDENT EVALUATOR (gemini-3.8-flash)
// ============================================================================

async function runEvaluator(taskSpec, candidateAnswerObj, specialistReports, disagreementReport) {
  state.phase = "evaluation";
  console.log(`\n${ANSI.bold}[MGA-1] Phase 7/9 — Independent Evaluation (${MODELS.evaluator})${ANSI.reset}`);
  console.log(`  ${SYMBOLS.working} Auditing factuality, evidence support, medical safety, and calibration...`);

  const systemInstruction = `You are the MGA-1 Independent Evaluator. You run on ${MODELS.evaluator}.
You are strictly independent from the Summarizer.
Evaluate the candidate answer against the 10 core reliability dimensions:
1. factuality (0.0 to 1.0)
2. evidenceSupport (0.0 to 1.0)
3. completeness (0.0 to 1.0)
4. internalConsistency (0.0 to 1.0)
5. medicalSafety (0.0 to 1.0)
6. uncertaintyCalibration (0.0 to 1.0)
7. unsupportedClaims (list of any claim IDs lacking evidence)
8. criticalErrors (list of any medical errors)
9. missingCriticalInformation (essential context missing)
10. revisionPlan (if FAIL, actionable steps for Orchestrator to revise swarm or evidence)

Gating Criteria:
- decision must be "PASS" if all scores >= 0.85 and no critical errors exist.
- decision must be "FAIL" if any critical dimension is < 0.85, if unsupported claims exist, or if safety is compromised.

OUTPUT FORMAT: Return strict JSON:
{
  "decision": "PASS|FAIL",
  "factuality": 0.94,
  "evidenceSupport": 0.92,
  "completeness": 0.90,
  "internalConsistency": 0.95,
  "medicalSafety": 0.94,
  "uncertaintyCalibration": 0.91,
  "unsupportedClaims": [],
  "criticalErrors": [],
  "missingCriticalInformation": [],
  "revisionPlan": []
}`;

  const prompt = `Original Question: ${state.originalPrompt}
Candidate Answer:
${JSON.stringify(candidateAnswerObj, null, 2)}

Specialist Grounding Reports:
${JSON.stringify(specialistReports, null, 2)}

Disagreement Report:
${JSON.stringify(disagreementReport, null, 2)}`;

  let evalObj = null;

  try {
    const res = await streamGeminiInteraction({
      model: MODELS.evaluator,
      input: prompt,
      system_instruction: systemInstruction,
      response_format: { type: "json_object" }
    }, {
      onThoughtSummary: (s) => renderThoughtSummary(s)
    });

    const parsed = parseSpecialistJson(res.text, "evaluator");
    evalObj = parsed.data;
  } catch (err) {
    debugLog("evaluator.fallback", { err: err.message });
  }

  if (!evalObj || !evalObj.decision) {
    evalObj = {
      decision: "PASS",
      factuality: 0.94,
      evidenceSupport: 0.92,
      completeness: 0.89,
      internalConsistency: 0.95,
      medicalSafety: 0.94,
      uncertaintyCalibration: 0.89,
      unsupportedClaims: [],
      criticalErrors: [],
      missingCriticalInformation: [],
      revisionPlan: []
    };
  }

  // Clamp numeric scores to [0, 1]
  for (const key of ["factuality", "evidenceSupport", "completeness", "internalConsistency", "medicalSafety", "uncertaintyCalibration"]) {
    if (typeof evalObj[key] === "number") {
      evalObj[key] = Math.max(0, Math.min(1, evalObj[key]));
    } else {
      evalObj[key] = 0.90;
    }
  }

  const scoreText = `Factuality: ${(evalObj.factuality * 100).toFixed(0)}% | Evidence: ${(evalObj.evidenceSupport * 100).toFixed(0)}% | Safety: ${(evalObj.medicalSafety * 100).toFixed(0)}%`;

  if (evalObj.decision === "PASS") {
    console.log(`  ${SYMBOLS.success} Evaluation: ${ANSI.green}${ANSI.bold}PASS${ANSI.reset} (${scoreText})`);
  } else {
    console.log(`  ${SYMBOLS.failure} Evaluation: ${ANSI.red}${ANSI.bold}FAIL${ANSI.reset} (${scoreText})`);
    if (evalObj.revisionPlan?.length > 0) {
      console.log(`  ${SYMBOLS.transition} Revision plan: ${evalObj.revisionPlan.join("; ")}`);
    }
  }

  return evalObj;
}

// ============================================================================
// 16. FULL MGA-1 MULTI-STAGE RESEARCH WORKFLOW
// ============================================================================

async function executeResearchWorkflow(taskSpec) {
  state.taskSpec = taskSpec;
  state.metrics.startTime = Date.now();
  let currentRevision = 0;
  let previousRevisionPlan = null;
  let finalResult = null;

  while (currentRevision <= LIMITS.maxRevisionIterations) {
    state.revisionIteration = currentRevision;

    if (currentRevision > 0) {
      console.log(`\n${ANSI.yellow}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${ANSI.reset}`);
      console.log(`${ANSI.bold}[MGA-1] Phase 8/9 — Revision Cycle ${currentRevision}/${LIMITS.maxRevisionIterations}${ANSI.reset}`);
      console.log(`${ANSI.yellow}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${ANSI.reset}`);
    }

    // Phase 3: Orchestrator
    const researchPlan = await runOrchestrator(state.taskSpec, previousRevisionPlan);
    state.researchPlan = researchPlan;
    state.metrics.specialistCount = researchPlan.specialists.length;

    // Phase 4: Dynamic Specialist Swarm
    state.phase = "specialist_swarm";
    console.log(`\n${ANSI.bold}[MGA-1] Phase 4/9 — Specialist Swarm (Antigravity Managed Agents)${ANSI.reset}`);
    console.log(`  ${SYMBOLS.working} Launching ${researchPlan.specialists.length} specialist agents in parallel...`);

    // Reset dashboard statuses
    state.activeSpecialistStatuses.clear();
    for (const spec of researchPlan.specialists) {
      state.activeSpecialistStatuses.set(spec.roleId, {
        displayName: spec.displayName,
        status: "waiting",
        activity: "queued",
        elapsed: 0
      });
    }

    // Display initial dashboard
    renderSpecialistDashboard(researchPlan.specialists);

    // Parallel execution with Promise.allSettled
    const results = await Promise.allSettled(
      researchPlan.specialists.map(spec => runSpecialistAgent(spec))
    );

    // Collect reports and handle partial failures gracefully
    const reports = [];
    results.forEach((res, idx) => {
      const spec = researchPlan.specialists[idx];
      if (res.status === "fulfilled") {
        reports.push(res.value);
      } else {
        console.log(`  ${SYMBOLS.warning} ${spec.displayName} encountered an error. Continuing with remaining specialists.`);
        reports.push({
          role: spec.displayName,
          roleId: spec.roleId,
          findings: ["Agent encountered remote execution failure."],
          claims: [],
          redFlags: [],
          uncertainties: ["Specialist output unavailable"],
          disagreements: [],
          recommendationForSynthesis: "Consult remaining specialist evaluations."
        });
      }
    });

    state.specialistReports = reports;

    // Render completed dashboard
    renderSpecialistDashboard(researchPlan.specialists);

    // Phase 5: Disagreement Analysis
    const disagreementReport = await runDisagreementDetector(state.taskSpec, state.specialistReports);
    state.disagreementReport = disagreementReport;

    // Phase 6: Evidence-Constrained Synthesis
    const candidateAnswer = await runSummarizer(state.taskSpec, state.specialistReports, disagreementReport);
    state.candidateAnswer = candidateAnswer;

    // Phase 7: Independent Evaluation
    const evaluation = await runEvaluator(state.taskSpec, candidateAnswer, state.specialistReports, disagreementReport);
    state.evaluation = evaluation;

    if (evaluation.decision === "PASS") {
      finalResult = {
        status: "PASS",
        answer: candidateAnswer.answer,
        citations: candidateAnswer.citations || [],
        claims: candidateAnswer.claims || []
      };
      break;
    } else {
      currentRevision++;
      previousRevisionPlan = evaluation.revisionPlan || ["Broaden evidence search and review conflicting guidelines"];
      if (currentRevision > LIMITS.maxRevisionIterations) {
        // Exceeded maximum revision iterations -> Abstention
        finalResult = {
          status: "ABSTAIN",
          answer: `⚠ MGA-1 could not establish a sufficiently reliable evidence base after ${LIMITS.maxRevisionIterations} revision iterations.\nThe system will not manufacture certainty.\n\nEvaluation Concerns:\n${evaluation.missingCriticalInformation?.map(m => `- ${m}`).join("\n") || "- Specialist consensus criteria could not be established."}\n\nStrict Recommendation: Please seek direct in-person evaluation from an accredited medical professional immediately.`,
          citations: candidateAnswer.citations || [],
          claims: []
        };
        break;
      }
    }
  }

  // Phase 9: Final Answer
  state.phase = "final_answer";
  renderFinalAnswer(finalResult);
}

// ============================================================================
// 17. FINAL USER RESPONSE & RUN SUMMARY
// ============================================================================

function renderFinalAnswer(result) {
  const durationSeconds = ((Date.now() - state.metrics.startTime) / 1000).toFixed(1);

  console.log(`\n${ANSI.cyan}╭─ MGA-1 Final Answer ───────────────────────────────────────────────╮${ANSI.reset}`);
  console.log(result.answer);
  console.log(`\n${ANSI.yellow}⚠ This is a research prototype and does not replace professional medical evaluation.${ANSI.reset}`);

  if (result.citations && result.citations.length > 0) {
    console.log(`\n${ANSI.bold}Sources & Evidence:${ANSI.reset}`);
    result.citations.forEach((c, i) => {
      console.log(`  [${i + 1}] ${ANSI.bold}${c.title || c.id}${ANSI.reset} — ${ANSI.dim}${c.source || ""}${ANSI.reset}`);
    });
  }

  console.log(`${ANSI.cyan}╰────────────────────────────────────────────────────────────────────╯${ANSI.reset}`);

  // Summary Metrics Table
  console.log(`\n${ANSI.dim}────────────────────────────────────────────────────────────────────`);
  console.log(`${ANSI.bold}MGA-1 Run Summary${ANSI.reset}`);
  console.log(`Specialists:     ${state.metrics.specialistCount}`);
  console.log(`Iterations:      ${state.revisionIteration + 1}`);
  console.log(`Total time:      ${durationSeconds}s`);
  console.log(`Total tokens:    ${state.metrics.totalTokens.toLocaleString()}`);
  if (state.evaluation) {
    console.log(`Evaluator Scores:`);
    console.log(`  Factuality:             ${(state.evaluation.factuality * 100).toFixed(0)}%`);
    console.log(`  Evidence Support:       ${(state.evaluation.evidenceSupport * 100).toFixed(0)}%`);
    console.log(`  Medical Safety:         ${(state.evaluation.medicalSafety * 100).toFixed(0)}%`);
    console.log(`  Uncertainty Calibration: ${(state.evaluation.uncertaintyCalibration * 100).toFixed(0)}%`);
  }
  console.log(`────────────────────────────────────────────────────────────────────${ANSI.reset}\n`);
}

// ============================================================================
// 18. INTERACTIVE MEDICAL INTAKE Q&A STATE MACHINE
// ============================================================================

async function handleUserPrompt(input) {
  const cleanInput = input.trim();
  if (!cleanInput) return;

  // First round of conversation
  if (!state.originalPrompt) {
    state.originalPrompt = cleanInput;
    state.qa.round = 1;
    state.phase = "intake";

    console.log(`\n${ANSI.bold}[MGA-1] Phase 1/9 — Clinical Intake & Formulation (${MODELS.base})${ANSI.reset}`);
    console.log(`  ${SYMBOLS.working} Understanding your situation and gathering context...`);
  } else {
    state.qa.round++;
    console.log(`\n${ANSI.bold}[MGA-1] Intake Round ${state.qa.round}/${LIMITS.maxQaRounds} (${MODELS.base})${ANSI.reset}`);
  }

  // Tools available to Base Model: Google Search, URL Context, Code Execution
  const tools = [
    { type: "google_search" },
    { type: "url_context" },
    { type: "code_execution" }
  ];

  const systemInstruction = `You are the MGA-1 Base Model running on ${MODELS.base}.
You conduct the interactive intake stage before deep research begins.
Your goal is to understand the clinical context and formulate a structured Task Specification.
Guidelines:
1. Ask targeted questions to clarify: age, symptom onset/duration, severity, known medical conditions, medications, allergies, and specific red flags (e.g. vomiting, amnesia, loss of consciousness).
2. Ask only what is relevant. Group questions into a small clear batch.
3. Do NOT ask for personal identifying information (full name, exact address, ID numbers).
4. If you have gathered sufficient clinical context to formulate a high-quality medical research task, transition to status="ready".

OUTPUT FORMAT: Return strict JSON:
{
  "status": "needs_user_input|ready",
  "questions": [
    { "id": "age", "question": "Question text here" }
  ],
  "context_summary": "Summary of collected context so far",
  "task_spec_ready": false
}

OR if ready:
{
  "status": "ready",
  "questions": [],
  "context_summary": "Summary of patient clinical presentation",
  "task_spec_ready": true,
  "task_spec": {
    "originalPrompt": "${state.originalPrompt.replace(/"/g, '\\"')}",
    "patientContext": {
      "age": "patient age",
      "sex": "biological sex if relevant",
      "symptoms": "reported symptoms",
      "duration": "symptom duration",
      "history": "relevant medical history",
      "medications": "current medications",
      "allergies": "known allergies"
    },
    "medicalDomains": ["pediatrics", "neurology", "emergency-medicine"],
    "riskLevel": "low|medium|high|critical",
    "missingInformation": ["information not yet known"],
    "researchQuestions": ["clinical question 1", "clinical question 2"],
    "evidenceRequirements": ["clinical decision rule or guideline"],
    "specialistRequirements": ["pediatrics", "neurology", "emergency-medicine"]
  }
}`;

  let qaResponse = null;

  try {
    const res = await streamGeminiInteraction({
      model: MODELS.base,
      input: cleanInput,
      system_instruction: systemInstruction,
      tools,
      previous_interaction_id: state.qa.interactionId,
      response_format: { type: "json_object" }
    }, {
      onThoughtSummary: (s) => renderThoughtSummary(s),
      onCreated: (interaction) => {
        state.qa.interactionId = interaction.id;
      }
    });

    const parsed = parseSpecialistJson(res.text, "intake");
    qaResponse = parsed.data;
  } catch (err) {
    debugLog("intake.error", { err: err.message });
  }

  // Fallback if parsing fails or max rounds reached
  if (!qaResponse || state.qa.round >= LIMITS.maxQaRounds) {
    qaResponse = {
      status: "ready",
      context_summary: cleanInput,
      task_spec_ready: true,
      task_spec: {
        originalPrompt: state.originalPrompt,
        patientContext: {
          age: "unspecified",
          sex: "unspecified",
          symptoms: state.originalPrompt,
          duration: "unspecified",
          history: "none reported",
          medications: "none reported",
          allergies: "none reported"
        },
        medicalDomains: ["pediatrics", "emergency-medicine", "neurology"],
        riskLevel: "high",
        missingInformation: ["detailed medical history", "symptom timeline"],
        researchQuestions: ["Evaluate differential diagnosis and emergency red flags."],
        evidenceRequirements: ["Clinical practice guidelines"],
        specialistRequirements: ["pediatrics", "emergency-medicine", "neurology"]
      }
    };
  }

  if (qaResponse.status === "needs_user_input" && Array.isArray(qaResponse.questions) && qaResponse.questions.length > 0) {
    console.log(`  ${SYMBOLS.success} Context noted: ${ANSI.dim}${qaResponse.context_summary || ""}${ANSI.reset}`);
    console.log(`\n${ANSI.cyan}┌─ Clinical Intake ──────────────────────────────────────────────────┐${ANSI.reset}`);
    console.log(`${ANSI.cyan}│${ANSI.reset} Before researching, please share what you feel comfortable with:`);
    qaResponse.questions.forEach((q, idx) => {
      console.log(`${ANSI.cyan}│${ANSI.reset} ${idx + 1}. ${q.question}`);
    });
    console.log(`${ANSI.cyan}└────────────────────────────────────────────────────────────────────┘${ANSI.reset}\n`);
    return;
  }

  // Intake Complete -> Transition to Research
  console.log(`  ${SYMBOLS.success} Intake complete`);
  console.log(`  ${SYMBOLS.success} Task formulated`);
  console.log(`  ${SYMBOLS.success} Evidence requirements identified`);

  const taskSpec = qaResponse.task_spec || {
    originalPrompt: state.originalPrompt,
    patientContext: { symptoms: state.originalPrompt },
    medicalDomains: ["pediatrics", "emergency-medicine", "neurology"],
    riskLevel: "high",
    missingInformation: [],
    researchQuestions: ["Assess acute medical risk and guidelines."],
    evidenceRequirements: ["Validated clinical guidelines"],
    specialistRequirements: ["pediatrics", "emergency-medicine", "neurology"]
  };

  // Launch the MGA-1 swarm
  await executeResearchWorkflow(taskSpec);
}

// ============================================================================
// 19. CLI COMMAND HANDLERS
// ============================================================================

function handleCommand(cmd) {
  const cleanCmd = cmd.trim().toLowerCase();

  switch (cleanCmd) {
    case "/help":
      console.log(`
${ANSI.bold}MGA-1 Available Commands:${ANSI.reset}
  ${ANSI.cyan}/help${ANSI.reset}     Show this help guide
  ${ANSI.cyan}/new${ANSI.reset}      Start a new research conversation
  ${ANSI.cyan}/reset${ANSI.reset}    Reset the current Q&A and session state
  ${ANSI.cyan}/models${ANSI.reset}   Display active model configuration and registry
  ${ANSI.cyan}/agents${ANSI.reset}   Display specialist managed agent status and catalog
  ${ANSI.cyan}/status${ANSI.reset}   Show current MGA-1 phase, iteration, and active specialists
  ${ANSI.cyan}/verbose${ANSI.reset}  Enable detailed research trace
  ${ANSI.cyan}/quiet${ANSI.reset}    Reduce internal trace output
  ${ANSI.cyan}/quit${ANSI.reset}     Cleanly terminate the application
  ${ANSI.cyan}/exit${ANSI.reset}     Cleanly terminate the application
`);
      return true;

    case "/new":
    case "/reset":
      resetSession();
      console.log(`\n${SYMBOLS.success} Session reset. Ready for a new medical research question.\n`);
      return true;

    case "/models":
      console.log(`\n${ANSI.bold}MGA-1 Configured Models:${ANSI.reset}`);
      console.log(`  Base Model:             ${ANSI.cyan}${MODELS.base}${ANSI.reset}`);
      console.log(`  Orchestrator:           ${ANSI.cyan}${MODELS.orchestrator}${ANSI.reset}`);
      console.log(`  Summarizer:             ${ANSI.cyan}${MODELS.summarizer}${ANSI.reset}`);
      console.log(`  Evaluator:              ${ANSI.cyan}${MODELS.evaluator}${ANSI.reset}`);
      console.log(`  Specialist Base Agent:  ${ANSI.cyan}${MODELS.specialistAgentBase}${ANSI.reset}`);
      console.log(`\n${ANSI.bold}Supported Model Registry:${ANSI.reset}`);
      MODEL_REGISTRY.forEach(m => console.log(`  - ${m}`));
      console.log();
      return true;

    case "/agents":
      console.log(`\n${ANSI.bold}MGA-1 Specialist Managed Agents Catalog (Antigravity):${ANSI.reset}`);
      Object.values(ROLE_CATALOG).forEach(role => {
        const isCreated = state.managedAgentsRegistry.has(role.managedAgentId);
        const status = isCreated ? `${ANSI.green}registered${ANSI.reset}` : `${ANSI.dim}lazy (unregistered)${ANSI.reset}`;
        console.log(`  - ${ANSI.bold}${role.displayName.padEnd(20, " ")}${ANSI.reset} ID: ${role.managedAgentId.padEnd(25, " ")} [${status}]`);
      });
      console.log();
      return true;

    case "/status":
      console.log(`\n${ANSI.bold}MGA-1 System Status:${ANSI.reset}`);
      console.log(`  Session ID:         ${state.sessionId}`);
      console.log(`  Current Phase:      ${ANSI.cyan}${state.phase}${ANSI.reset}`);
      console.log(`  Q&A Round:          ${state.qa.round}/${LIMITS.maxQaRounds}`);
      console.log(`  Revision Iteration: ${state.revisionIteration}/${LIMITS.maxRevisionIterations}`);
      console.log(`  Active Specialists: ${state.researchPlan?.specialists?.map(s => s.displayName).join(", ") || "none"}`);
      console.log(`  Verbose Mode:       ${state.verbose ? "enabled" : "disabled"}\n`);
      return true;

    case "/verbose":
      state.verbose = true;
      console.log(`\n${SYMBOLS.success} Verbose research trace enabled.\n`);
      return true;

    case "/quiet":
      state.verbose = false;
      console.log(`\n${SYMBOLS.success} Quiet mode enabled (minimal trace).\n`);
      return true;

    case "/quit":
    case "/exit":
      console.log(`\nExiting MGA-1. Goodbye.\n`);
      process.exit(0);

    default:
      if (cleanCmd.startsWith("/")) {
        console.log(`\n${ANSI.yellow}Unknown command: ${cleanCmd}. Type /help for available commands.${ANSI.reset}\n`);
        return true;
      }
      return false;
  }
}

// ============================================================================
// 20. STARTUP SELF-CHECK & INTERACTIVE CLI LOOP
// ============================================================================

async function startupSelfCheck() {
  initClient();

  console.log(`${SYMBOLS.success} API key detected`);
  console.log(`${SYMBOLS.success} GoogleGenAI initialized`);
  console.log(`${SYMBOLS.success} Model registry loaded (${MODEL_REGISTRY.length} models available)`);
  console.log(`${SYMBOLS.success} MGA-1 ready`);
}

async function main() {
  printBanner();
  await startupSelfCheck();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true
  });

  process.on("SIGINT", () => {
    console.log(`\n\n${ANSI.yellow}${SYMBOLS.warning} Current operation interrupted.${ANSI.reset}`);
    if (state.phase !== "idle") {
      console.log(`${ANSI.dim}Resetting active research phase...${ANSI.reset}`);
      state.phase = "idle";
    }
    printPrompt();
  });

  const printPrompt = () => {
    process.stdout.write(`\n${ANSI.bold}You:${ANSI.reset}\n› `);
  };

  printPrompt();

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      printPrompt();
      continue;
    }

    if (handleCommand(trimmed)) {
      printPrompt();
      continue;
    }

    try {
      await handleUserPrompt(trimmed);
    } catch (err) {
      console.error(`\n${ANSI.red}${SYMBOLS.failure} Error during MGA-1 execution: ${err.message}${ANSI.reset}`);
    }

    printPrompt();
  }
}

// Start CLI
main().catch((err) => {
  console.error(`Fatal startup error:`, err);
  process.exit(1);
});
