const form = document.querySelector("#negotiation-form");
const topicInput = document.querySelector("#topic");
const profileInput = document.querySelector("#profile");
const providerInput = document.querySelector("#llm-provider");
const providerHelpNode = document.querySelector("#provider-help");
const modelLevelInput = document.querySelector("#model-level");
const argumentInput = document.querySelector("#argument");
const agreementInput = document.querySelector("#agreement");
const projectDocsInput = document.querySelector("#project-docs");
const projectDocDirectoryInput = document.querySelector("#project-doc-directory");
const messagesNode = document.querySelector("#messages");
const validatedDecisionsNode = document.querySelector("#validated-decisions");
const decisionsEmptyNode = document.querySelector("#decisions-empty");
const sendButton = document.querySelector("#send");
const startButton = document.querySelector("#start");
const resetButton = document.querySelector("#reset");
const saveSessionButton = document.querySelector("#save-session");
const loadSessionButton = document.querySelector("#load-session");
const savedSessionSelect = document.querySelector("#saved-session-select");
const microphoneButton = document.querySelector("#microphone");
const helpAnswerButton = document.querySelector("#help-answer");
const framingReportButton = document.querySelector("#framing-report");
const microphoneStatus = document.querySelector("#microphone-status");
const autoReadInput = document.querySelector("#auto-read");
const ttsProviderInput = document.querySelector("#tts-provider");
const voicePanel = document.querySelector(".voice-panel");
const elevenLabsKeyField = document.querySelector("#elevenlabs-key-field");
const elevenLabsApiKeyInput = document.querySelector("#elevenlabs-api-key");
const voiceSelect = document.querySelector("#voice");
const previewVoiceButton = document.querySelector("#preview-voice");
const stopVoiceButton = document.querySelector("#stop-voice");
const blockingLoader = document.querySelector("#blocking-loader");
const blockingLoaderTitle = document.querySelector("#blocking-loader-title");
const blockingLoaderText = document.querySelector("#blocking-loader-text");
const sessionCostValue = document.querySelector("#session-cost-value");
const sessionCostMeta = document.querySelector("#session-cost-meta");

let history = [];
let validatedDecisions = [];
let hasStarted = false;
let documentSessionId = "";
let documentSessionSignature = "";
let sessionSourceFiles = { agreement: "", project_documents: [] };
let recognition = null;
let isListening = false;
let isRecognitionActive = false;
let transcriptBase = "";
let finalTranscript = "";
let latestInterimTranscript = "";
let sessionCostUsd = 0;
let sessionOpenAICalls = 0;
let sessionOpenAITokens = 0;
let hasUnpricedOpenAIUsage = false;
let lastReport = null;
let lastReportMarkdown = "";
let sessionDisplayName = "";
let isControlRecording = false;
let recognitionMode = "manual";
let currentAudio = null;
let currentAudioUrl = "";
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSynthesizer = window.speechSynthesis;

function appendMessage(role, content) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const author = document.createElement("strong");
  author.textContent =
    role === "user"
      ? "Developpeur"
      : role === "error"
          ? "Erreur"
          : role === "helper"
            ? "Aide"
            : role === "report"
              ? "Rapport de cadrage"
              : role === "improvement"
                ? "Amelioration du rapport"
                : "Client";

  const body = document.createElement("p");
  body.textContent = content;

  message.append(author, body);
  messagesNode.append(message);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function normalizeDecision(decision, defaultProfile = profileInput.value) {
  if (decision && typeof decision === "object") {
    const source = ["document", "client", "codev_user"].includes(decision.source)
      ? decision.source
      : "client";
    const profile = ["sales", "technical", "boss"].includes(decision.profile)
      ? decision.profile
      : defaultProfile;
    return {
      text: String(decision.text || "").replace(/\s+/g, " ").trim(),
      evidence: String(decision.evidence || "").replace(/\s+/g, " ").trim(),
      source,
      profile,
    };
  }
  return {
    text: String(decision || "").replace(/\s+/g, " ").trim(),
    evidence: "",
    source: "client",
    profile: defaultProfile,
  };
}

function getDecisionPresentation(decision) {
  if (decision.source === "document") {
    return { icon: "DOC", label: "Documentation projet", className: "document" };
  }
  if (decision.source === "codev_user") {
    return { icon: "C", label: "Utilisateur CODEV", className: "codev-user" };
  }
  const profiles = {
    sales: { icon: "€", label: "Client commercial", className: "client-sales" },
    technical: { icon: "</>", label: "Client développeur", className: "client-technical" },
    boss: { icon: "◆", label: "Client responsable", className: "client-boss" },
  };
  return profiles[decision.profile] || profiles.sales;
}

function normalizeDecisionComparisonText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function decisionTextsAreSimilar(left, right) {
  const leftText = normalizeDecisionComparisonText(left);
  const rightText = normalizeDecisionComparisonText(right);
  if (!leftText || !rightText) {
    return false;
  }
  if (leftText === rightText) {
    return true;
  }
  if (Math.min(leftText.length, rightText.length) >= 24 && (
    leftText.includes(rightText) || rightText.includes(leftText)
  )) {
    return true;
  }
  const leftTokens = new Set(leftText.split(" "));
  const rightTokens = new Set(rightText.split(" "));
  const overlap = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  return overlap / Math.max(Math.min(leftTokens.size, rightTokens.size), 1) >= 0.72;
}

function deduplicateValidatedDecisions() {
  const unique = [];
  for (const decision of validatedDecisions) {
    const duplicate = unique.find(
      (known) =>
        known.source === decision.source &&
        (decisionTextsAreSimilar(known.text, decision.text) ||
          (known.evidence &&
            decision.evidence &&
            decisionTextsAreSimilar(known.evidence, decision.evidence))),
    );
    if (!duplicate) {
      unique.push(decision);
    } else if (!duplicate.evidence && decision.evidence) {
      duplicate.evidence = decision.evidence;
    }
  }
  validatedDecisions = unique;
}

function renderValidatedDecisions() {
  deduplicateValidatedDecisions();
  validatedDecisionsNode.innerHTML = "";
  decisionsEmptyNode.hidden = validatedDecisions.length > 0;

  for (const decision of validatedDecisions) {
    const presentation = getDecisionPresentation(decision);
    const item = document.createElement("li");
    item.classList.add(`decision-${presentation.className}`);
    item.tabIndex = 0;
    const icon = document.createElement("span");
    icon.className = "decision-icon";
    icon.textContent = presentation.icon;
    icon.setAttribute("aria-hidden", "true");
    const content = document.createElement("span");
    content.className = "decision-content";
    const text = document.createElement("span");
    text.textContent = decision.text;
    const origin = document.createElement("small");
    origin.textContent = presentation.label;
    content.append(text, origin);
    item.append(icon, content);
    if (decision.evidence) {
      const evidence = document.createElement("span");
      evidence.className = "decision-evidence";
      evidence.id = `decision-evidence-${validatedDecisionsNode.children.length}`;
      evidence.setAttribute("role", "tooltip");
      evidence.textContent = `Extrait : « ${decision.evidence} »`;
      item.setAttribute("aria-describedby", evidence.id);
      item.append(evidence);
    } else {
      item.title = "Aucun extrait disponible pour cette ancienne décision.";
    }
    validatedDecisionsNode.append(item);
  }
}

function addValidatedDecisions(decisions) {
  if (!Array.isArray(decisions)) {
    return;
  }

  for (const rawDecision of decisions) {
    const decision = normalizeDecision(rawDecision);
    const duplicate = validatedDecisions.find(
      (known) =>
        known.source === decision.source &&
        (decisionTextsAreSimilar(known.text, decision.text) ||
          (known.evidence &&
            decision.evidence &&
            decisionTextsAreSimilar(known.evidence, decision.evidence))),
    );
    if (!decision.text || duplicate) {
      if (duplicate && !duplicate.evidence && decision.evidence) {
        duplicate.evidence = decision.evidence;
      }
      continue;
    }
    validatedDecisions.push(decision);
  }

  renderValidatedDecisions();
}

function formatSessionCost(value) {
  return new Intl.NumberFormat("fr-FR", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 6,
    maximumFractionDigits: 6,
  }).format(value);
}

function formatCount(value, singular, plural) {
  return `${new Intl.NumberFormat("fr-FR").format(value)} ${value === 1 ? singular : plural}`;
}

function renderSessionCost() {
  sessionCostValue.textContent = formatSessionCost(sessionCostUsd);
  const parts = [
    formatCount(sessionOpenAICalls, "appel", "appels"),
    formatCount(sessionOpenAITokens, "token", "tokens"),
  ];
  if (hasUnpricedOpenAIUsage) {
    parts.push("tarif partiel");
  }
  sessionCostMeta.textContent = parts.join(" · ");
}

function addOpenAIUsage(usage) {
  if (!usage || typeof usage !== "object") {
    return;
  }
  sessionCostUsd += Number(usage.estimated_cost_usd || 0);
  sessionOpenAICalls += Number(usage.api_calls || 0);
  sessionOpenAITokens += Number(usage.total_tokens || 0);
  hasUnpricedOpenAIUsage ||= usage.fully_priced === false;
  renderSessionCost();
}

function appendReport(report, markdown) {
  const message = document.createElement("article");
  message.className = "message report";

  const title = document.createElement("strong");
  const globalScore = Number(report.global_score || 0);
  title.textContent = `Rapport de cadrage - maturite ${globalScore} %`;

  const summary = document.createElement("p");
  summary.textContent = report.executive_summary || "Synthese non disponible.";

  const scoreGrid = document.createElement("div");
  scoreGrid.className = "score-grid";
  const scores = Array.isArray(report.scores) ? report.scores : [];
  for (const item of scores) {
    const scoreItem = document.createElement("div");
    const scoreName = document.createElement("span");
    const scoreValue = document.createElement("strong");
    const scoreReason = document.createElement("small");
    scoreName.textContent = item.name || "Critere";
    scoreValue.textContent = `${Number(item.score || 0)} %`;
    scoreReason.textContent = item.reason || "";
    scoreItem.append(scoreName, scoreValue, scoreReason);
    scoreGrid.append(scoreItem);
  }

  const criticalTitle = document.createElement("h2");
  criticalTitle.textContent = "Points critiques";
  const criticalList = document.createElement("ul");
  const criticalPoints = Array.isArray(report.critical_points) ? report.critical_points : [];
  for (const point of criticalPoints.slice(0, 5)) {
    const item = document.createElement("li");
    item.textContent = point;
    criticalList.append(item);
  }
  if (criticalList.children.length === 0) {
    const item = document.createElement("li");
    item.textContent = "Aucun point critique identifie.";
    criticalList.append(item);
  }

  const downloadButton = document.createElement("button");
  downloadButton.className = "secondary-action report-download";
  downloadButton.type = "button";
  downloadButton.textContent = "Telecharger le rapport";
  downloadButton.addEventListener("click", () => downloadReport(markdown || "", report));

  const improveButton = document.createElement("button");
  improveButton.className = "secondary-action report-improve";
  improveButton.type = "button";
  improveButton.textContent = "Ameliorer le rapport";
  improveButton.addEventListener("click", () => improveReport(report, improveButton));

  const actions = document.createElement("div");
  actions.className = "report-actions";
  actions.append(downloadButton, improveButton);

  message.append(title, summary, scoreGrid, criticalTitle, criticalList, actions);
  messagesNode.append(message);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function appendImprovement(improvement) {
  const message = document.createElement("article");
  message.className = "message improvement";

  const title = document.createElement("strong");
  title.textContent = `Plan d'amelioration - cible ${Number(improvement.target_score || 0)} %`;

  const summary = document.createElement("p");
  summary.textContent = improvement.summary || "Analyse non disponible.";

  const actionList = document.createElement("ul");
  const priorityActions = Array.isArray(improvement.priority_actions)
    ? improvement.priority_actions
    : [];
  for (const action of priorityActions) {
    const item = document.createElement("li");
    const axis = action.axis || "Axe";
    const currentScore = Number(action.current_score || 0);
    const targetScore = Number(action.target_score || 0);
    const expectedImpact = action.expected_impact || "";
    item.textContent = `${axis} (${currentScore} % -> ${targetScore} %) : ${action.action || ""} ${expectedImpact}`;
    actionList.append(item);
  }
  if (actionList.children.length === 0) {
    const item = document.createElement("li");
    item.textContent = "Aucune action prioritaire identifiee.";
    actionList.append(item);
  }

  const questionTitle = document.createElement("h2");
  questionTitle.textContent = "Questions a trancher";
  const questionList = document.createElement("ul");
  const questions = Array.isArray(improvement.questions_to_answer)
    ? improvement.questions_to_answer
    : [];
  for (const question of questions) {
    const item = document.createElement("li");
    item.textContent = question;
    questionList.append(item);
  }
  if (questionList.children.length === 0) {
    const item = document.createElement("li");
    item.textContent = "Aucune question complementaire identifiee.";
    questionList.append(item);
  }

  message.append(title, summary, actionList, questionTitle, questionList);
  messagesNode.append(message);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function showBlockingLoader(title, text) {
  blockingLoaderTitle.textContent = title;
  blockingLoaderText.textContent = text;
  blockingLoader.hidden = false;
}

function hideBlockingLoader() {
  blockingLoader.hidden = true;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading || !hasStarted;
  helpAnswerButton.disabled = isLoading || !hasStarted;
  framingReportButton.disabled = isLoading || !hasStarted;
  startButton.disabled = isLoading || hasStarted;
  saveSessionButton.disabled = isLoading || !hasStarted;
  loadSessionButton.disabled = isLoading || !savedSessionSelect.value;
  sendButton.textContent = isLoading ? "Question en cours..." : "Envoyer la reponse";
  startButton.textContent = isLoading ? "Client en cours..." : "Demarrer la discussion";
}

function setComposerEnabled(isEnabled) {
  argumentInput.disabled = !isEnabled;
  microphoneButton.disabled = !isEnabled || !recognition;
  helpAnswerButton.disabled = !isEnabled;
  framingReportButton.disabled = !isEnabled;
  sendButton.disabled = !isEnabled;
}

function setSetupEnabled(isEnabled) {
  topicInput.disabled = !isEnabled;
  providerInput.disabled = !isEnabled;
  modelLevelInput.disabled = !isEnabled;
  agreementInput.disabled = !isEnabled;
  projectDocsInput.disabled = !isEnabled;
  projectDocDirectoryInput.disabled = !isEnabled;
}

function hasSourceContent() {
  return Boolean(topicInput.value.trim() || agreementInput.files[0]);
}

function getProjectDocumentFiles() {
  const selectedFiles = [...projectDocsInput.files];
  const directoryFiles = [...projectDocDirectoryInput.files].filter((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith(".md") || name.endsWith(".markdown");
  });
  return [...selectedFiles, ...directoryFiles];
}

function getProjectDocumentSignature(files) {
  const filesSignature = files
    .map((file) => `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`)
    .join("|");
  return `${providerInput.value}|${filesSignature}`;
}

function renderProviderHelp() {
  if (providerInput.value === "ollama") {
    providerHelpNode.textContent =
      "Ollama local — installez Ollama puis lancez-le sur http://127.0.0.1:11434 :\n" +
      '$env:OLLAMA_HOST="127.0.0.1:11434"; ollama serve\n' +
      "Modeles requis : ollama pull qwen3:4b, ollama pull qwen3:8b, " +
      "ollama pull qwen3:14b et ollama pull nomic-embed-text. " +
      "Les noms et le port peuvent etre modifies avec les variables OLLAMA_* documentees dans le README.";
  } else if (providerInput.value === "anthropic") {
    providerHelpNode.textContent =
      "Anthropic — definissez la cle avant de lancer CODEV :\n" +
      '$env:ANTHROPIC_API_KEY="votre_cle_anthropic"\n' +
      "Les documents utilisent un index vectoriel local sans seconde cle API. " +
      "Les modeles peuvent etre modifies avec ANTHROPIC_LIGHT_MODEL, ANTHROPIC_MEDIUM_MODEL et ANTHROPIC_STRONG_MODEL.";
  } else if (providerInput.value === "google") {
    providerHelpNode.textContent =
      "Google Gemini — definissez la cle Google AI avant de lancer CODEV :\n" +
      '$env:GOOGLE_API_KEY="votre_cle_google"\n' +
      "Le modele d'embedding par defaut est models/gemini-embedding-001.";
  } else if (providerInput.value === "azure") {
    providerHelpNode.textContent =
      "Azure OpenAI — configurez la cle, l'endpoint et les noms de deployments :\n" +
      '$env:AZURE_OPENAI_API_KEY="votre_cle_azure"\n' +
      '$env:AZURE_OPENAI_ENDPOINT="https://votre-ressource.openai.azure.com"\n' +
      '$env:AZURE_OPENAI_API_VERSION="2025-03-01-preview"\n' +
      "Variables de deployments : AZURE_OPENAI_LIGHT_DEPLOYMENT, AZURE_OPENAI_MEDIUM_DEPLOYMENT, " +
      "AZURE_OPENAI_STRONG_DEPLOYMENT et AZURE_OPENAI_EMBEDDING_DEPLOYMENT.";
  } else {
    providerHelpNode.textContent =
      "OpenAI — definissez la cle dans les variables d'environnement du systeme avant de lancer CODEV :\n" +
      '$env:OPENAI_API_KEY="votre_cle_openai"\n' +
      "Aucune cle n'est saisie ni stockee dans l'interface.";
  }
}

async function ensureDocumentSession() {
  const files = getProjectDocumentFiles();
  if (files.length === 0) {
    documentSessionSignature = "";
    return;
  }

  const signature = getProjectDocumentSignature(files);
  if (documentSessionId && documentSessionSignature === signature) {
    return;
  }

  const payload = new FormData();
  payload.append("provider", providerInput.value);
  for (const file of files) {
    payload.append("project_docs", file, file.webkitRelativePath || file.name);
  }

  const response = await fetch("/api/document-session", {
    method: "POST",
    body: payload,
  });
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Documentation projet invalide.");
  }

  addOpenAIUsage(data.llm_usage || data.openai_usage);
  documentSessionId = data.document_session_id;
  documentSessionSignature = signature;
  sessionSourceFiles.project_documents = files.map(
    (file) => file.webkitRelativePath || file.name,
  );
  setMicrophoneStatus(
    `Documentation indexee: ${data.sources} source(s), ${data.chunks} extrait(s), mode ${data.retrieval_mode}.`,
  );
}

function buildPayload(argument = "") {
  const payload = new FormData();
  payload.append("topic", topicInput.value.trim());
  payload.append("profile", profileInput.value);
  payload.append("provider", providerInput.value);
  payload.append("model_level", modelLevelInput.value);
  payload.append("argument", argument);
  payload.append("history", JSON.stringify(history));
  payload.append("validated_decisions", JSON.stringify(validatedDecisions));
  payload.append("document_session_id", documentSessionId);
  if (agreementInput.files[0]) {
    payload.append("agreement", agreementInput.files[0]);
  }
  return payload;
}

async function fetchWithDocumentSession(endpoint, argument = "") {
  await ensureDocumentSession();
  let response = await fetch(endpoint, {
    method: "POST",
    body: buildPayload(argument),
  });
  let data = await readJsonResponse(response);

  if (response.status === 409 && documentSessionId && getProjectDocumentFiles().length > 0) {
    documentSessionId = "";
    documentSessionSignature = "";
    await ensureDocumentSession();
    response = await fetch(endpoint, {
      method: "POST",
      body: buildPayload(argument),
    });
    data = await readJsonResponse(response);
  }

  return { response, data };
}

function renderTtsProvider() {
  const usesElevenLabs = ttsProviderInput.value === "elevenlabs";
  voicePanel.classList.toggle("is-elevenlabs", usesElevenLabs);
  elevenLabsKeyField.hidden = !usesElevenLabs;
  populateVoices();
}

function setMicrophoneStatus(message, isError = false) {
  microphoneStatus.textContent = message;
  microphoneStatus.classList.toggle("error-text", isError);
}

async function readJsonResponse(response) {
  const body = await response.text();
  try {
    return body ? JSON.parse(body) : {};
  } catch {
    return {
      detail: body || "Reponse serveur invalide.",
    };
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markdownToPrintableHtml(markdown) {
  const lines = markdown.split("\n");
  return lines
    .map((line) => {
      if (line.startsWith("# ")) {
        return `<h1>${escapeHtml(line.slice(2))}</h1>`;
      }
      if (line.startsWith("## ")) {
        return `<h2>${escapeHtml(line.slice(3))}</h2>`;
      }
      if (line.startsWith("- ")) {
        return `<li>${escapeHtml(line.slice(2))}</li>`;
      }
      if (!line.trim()) {
        return "";
      }
      return `<p>${escapeHtml(line)}</p>`;
    })
    .join("\n")
    .replaceAll("</li>\n<li>", "</li><li>");
}

function downloadReport(markdown, report) {
  const html = `<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Rapport de cadrage CODEV</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.5; color: #17211b; }
    h1, h2 { color: #0d766e; }
    li { margin: 6px 0; }
    @media print { body { margin: 20mm; } }
  </style>
</head>
<body>
${markdownToPrintableHtml(markdown)}
<script>window.reportData = ${JSON.stringify(report).replaceAll("<", "\\u003c")};</script>
</body>
</html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "rapport-cadrage-codev.html";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function refreshSavedSessions(selectedId = "") {
  const response = await fetch("/api/saved-sessions");
  const data = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(data.detail || "Liste des sauvegardes indisponible.");
  }
  savedSessionSelect.innerHTML = "";
  const sessions = Array.isArray(data.sessions) ? data.sessions : [];
  if (sessions.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Aucune sauvegarde";
    savedSessionSelect.append(option);
  }
  for (const session of sessions) {
    const option = document.createElement("option");
    option.value = session.id;
    const savedAt = session.saved_at ? new Date(session.saved_at).toLocaleString("fr-FR") : "";
    option.textContent = `${session.name || "Discussion CODEV"}${savedAt ? ` - ${savedAt}` : ""}${session.has_report ? " - rapport" : ""}`;
    option.selected = session.id === selectedId;
    savedSessionSelect.append(option);
  }
  loadSessionButton.disabled = !savedSessionSelect.value;
}

async function saveSession() {
  if (!hasStarted) {
    return;
  }

  const session = {
    format: "codev-session",
    version: 1,
    saved_at: new Date().toISOString(),
    topic: topicInput.value.trim(),
    profile: profileInput.value,
    provider: providerInput.value,
    model_level: modelLevelInput.value,
    display_name: sessionDisplayName,
    history,
    validated_decisions: validatedDecisions,
    report: lastReport,
    report_markdown: lastReportMarkdown,
    document_session_id: documentSessionId,
    source_files: {
      agreement: agreementInput.files[0]?.name || sessionSourceFiles.agreement,
      project_documents: getProjectDocumentFiles().length
        ? getProjectDocumentFiles().map((file) => file.webkitRelativePath || file.name)
        : sessionSourceFiles.project_documents,
    },
    llm_usage: {
      estimated_cost_usd: sessionCostUsd,
      api_calls: sessionOpenAICalls,
      total_tokens: sessionOpenAITokens,
      fully_priced: !hasUnpricedOpenAIUsage,
    },
  };
  const response = await fetch("/api/saved-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(session),
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(data.detail || "Sauvegarde impossible.");
  }
  addOpenAIUsage(data.llm_usage || data.openai_usage);
  sessionDisplayName = data.display_name || sessionDisplayName;
  await refreshSavedSessions(data.id);
  setMicrophoneStatus("Discussion et documentation sauvegardees localement.");
}

function parseLoadedHistory(value) {
  if (!Array.isArray(value)) {
    throw new Error("Historique de discussion invalide.");
  }
  return value
    .filter(
      (item) =>
        item &&
        ["user", "assistant"].includes(item.role) &&
        typeof item.content === "string" &&
        item.content.trim(),
    )
    .map((item) => ({ role: item.role, content: item.content.trim() }));
}

async function loadSession(rawSession) {
  if (!rawSession || typeof rawSession !== "object" || Array.isArray(rawSession)) {
    throw new Error("Ce fichier n'est pas une sauvegarde CODEV compatible.");
  }
  const isCurrentSession =
    rawSession.format === "codev-session" && Number(rawSession.version) === 1;
  const isLegacySession =
    !rawSession.format &&
    Array.isArray(rawSession.history) &&
    (typeof rawSession.topic === "string" || Array.isArray(rawSession.validated_decisions));
  if (!isCurrentSession && !isLegacySession) {
    if (Array.isArray(rawSession.scores) && "global_score" in rawSession) {
      throw new Error(
        "Ce fichier contient uniquement un rapport, pas la discussion permettant de la reprendre.",
      );
    }
    if (rawSession.format === "codev-session") {
      throw new Error(`Version de sauvegarde CODEV non prise en charge: (${rawSession.version}).`);
    }
    throw new Error("Ce JSON n'est pas une sauvegarde de discussion CODEV.");
  }

  let restoredDocumentSessionId = "";
  let restorationUsage = null;
  const savedDocumentSessionId = typeof rawSession.document_session_id === "string"
    ? rawSession.document_session_id
    : "";
  if (savedDocumentSessionId) {
    const restorePayload = new FormData();
    restorePayload.append("document_session_id", savedDocumentSessionId);
    restorePayload.append("provider", rawSession.provider || "openai");
    const restoreResponse = await fetch("/api/document-session/restore", {
      method: "POST",
      body: restorePayload,
    });
    const restoreData = await readJsonResponse(restoreResponse);
    if (restoreResponse.ok) {
      restoredDocumentSessionId = restoreData.document_session_id;
      restorationUsage = restoreData.llm_usage || restoreData.openai_usage;
    }
  }

  stopSpeech();
  history = parseLoadedHistory(rawSession.history);
  validatedDecisions = Array.isArray(rawSession.validated_decisions)
    ? rawSession.validated_decisions
        .map((decision) => normalizeDecision(decision, rawSession.profile || "sales"))
        .filter((decision) => decision.text)
        .slice(0, 100)
    : [];
  lastReport = rawSession.report && typeof rawSession.report === "object"
    ? rawSession.report
    : null;
  lastReportMarkdown = typeof rawSession.report_markdown === "string"
    ? rawSession.report_markdown
    : "";
  sessionDisplayName = typeof rawSession.display_name === "string"
    ? rawSession.display_name
    : "";
  topicInput.value = typeof rawSession.topic === "string" ? rawSession.topic : "";
  if ([...profileInput.options].some((option) => option.value === rawSession.profile)) {
    profileInput.value = rawSession.profile;
  }
  if ([...providerInput.options].some((option) => option.value === rawSession.provider)) {
    providerInput.value = rawSession.provider;
  } else {
    providerInput.value = "openai";
  }
  renderProviderHelp();
  if ([...modelLevelInput.options].some((option) => option.value === rawSession.model_level)) {
    modelLevelInput.value = rawSession.model_level;
  }

  const usage = rawSession.llm_usage || rawSession.openai_usage || {};
  sessionCostUsd = Math.max(0, Number(usage.estimated_cost_usd) || 0);
  sessionOpenAICalls = Math.max(0, Number(usage.api_calls) || 0);
  sessionOpenAITokens = Math.max(0, Number(usage.total_tokens) || 0);
  hasUnpricedOpenAIUsage = usage.fully_priced === false;
  addOpenAIUsage(restorationUsage);
  const sourceFiles = rawSession.source_files || {};
  sessionSourceFiles = {
    agreement: typeof sourceFiles.agreement === "string" ? sourceFiles.agreement : "",
    project_documents: Array.isArray(sourceFiles.project_documents)
      ? sourceFiles.project_documents.filter((name) => typeof name === "string")
      : [],
  };
  documentSessionId = restoredDocumentSessionId;
  documentSessionSignature = "";
  agreementInput.value = "";
  projectDocsInput.value = "";
  projectDocDirectoryInput.value = "";
  argumentInput.value = "";

  messagesNode.innerHTML = "";
  for (const item of history) {
    appendMessage(item.role, item.content);
  }
  if (lastReport) {
    appendReport(lastReport, lastReportMarkdown);
  }
  if (history.length === 0) {
    appendMessage("assistant", "La sauvegarde ne contient aucun echange.");
  }

  hasStarted = history.length > 0;
  renderValidatedDecisions();
  renderSessionCost();
  setSetupEnabled(!hasStarted);
  setComposerEnabled(hasStarted);
  setLoading(false);

  const hadSources = Boolean(
    sessionSourceFiles.agreement || sessionSourceFiles.project_documents.length,
  );
  if (hasStarted && hadSources && !documentSessionId) {
    agreementInput.disabled = false;
    projectDocsInput.disabled = false;
    projectDocDirectoryInput.disabled = false;
    setMicrophoneStatus(
      "Discussion chargee. Reselectionnez les documents source avant de poursuivre.",
    );
  } else if (documentSessionId) {
    setMicrophoneStatus(
      `Discussion et documentation restaurees (${sessionSourceFiles.project_documents.length} fichier(s)).`,
    );
  } else {
    setMicrophoneStatus("Discussion chargee, vous pouvez la reprendre.");
  }
}

async function improveReport(report, button) {
  button.disabled = true;
  button.textContent = "Analyse en cours...";
  showBlockingLoader(
    "Amelioration du rapport...",
    "Analyse des axes faibles et preparation des actions d'amelioration.",
  );
  setLoading(true);

  const payload = new FormData();
  payload.append("provider", providerInput.value);
  payload.append("model_level", modelLevelInput.value);
  payload.append("report", JSON.stringify(report));

  try {
    const response = await fetch("/api/improve-report", {
      method: "POST",
      body: payload,
    });
    const data = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

    addOpenAIUsage(data.llm_usage || data.openai_usage);
    appendImprovement(data.improvement || {});
    button.textContent = "Analyse generee";
  } catch (error) {
    appendMessage("error", error.message);
    button.disabled = false;
    button.textContent = "Ameliorer le rapport";
  } finally {
    hideBlockingLoader();
    setLoading(false);
  }
}

function getFrenchVoices() {
  if (!speechSynthesizer) {
    return [];
  }

  const voices = speechSynthesizer.getVoices();
  const frenchVoices = voices.filter((voice) => voice.lang.toLowerCase().startsWith("fr"));
  return frenchVoices.length ? frenchVoices : voices;
}

function populateVoices() {
  if (ttsProviderInput.value === "elevenlabs") {
    populateElevenLabsVoices();
    return;
  }

  const voices = getFrenchVoices();
  voiceSelect.innerHTML = "";

  if (!speechSynthesizer || voices.length === 0) {
    voiceSelect.disabled = true;
    stopVoiceButton.disabled = true;
    previewVoiceButton.disabled = true;
    const option = document.createElement("option");
    option.textContent = "TTS non disponible";
    voiceSelect.append(option);
    return;
  }

  voiceSelect.disabled = false;
  stopVoiceButton.disabled = false;
  previewVoiceButton.disabled = false;
  voices.forEach((voice, index) => {
    const option = document.createElement("option");
    option.value = voice.voiceURI;
    option.textContent = `${voice.name} (${voice.lang})`;
    option.dataset.provider = "local";
    if (index === 0) {
      option.selected = true;
    }
    voiceSelect.append(option);
  });
}

async function populateElevenLabsVoices() {
  voiceSelect.innerHTML = "";
  const loadingOption = document.createElement("option");
  loadingOption.textContent = "Chargement des voix...";
  voiceSelect.append(loadingOption);
  voiceSelect.disabled = true;
  previewVoiceButton.disabled = true;

  const apiKey = elevenLabsApiKeyInput.value.trim();
  if (!apiKey) {
    loadingOption.textContent = "Cle ElevenLabs requise";
    stopVoiceButton.disabled = true;
    previewVoiceButton.disabled = true;
    return;
  }

  const payload = new FormData();
  payload.append("elevenlabs_api_key", apiKey);

  try {
    const response = await fetch("/api/elevenlabs/voices", {
      method: "POST",
      body: payload,
    });
    const data = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(data.detail || "Voix ElevenLabs indisponibles.");
    }

    voiceSelect.innerHTML = "";
    const voices = Array.isArray(data.voices) ? data.voices : [];
    for (const voice of voices) {
      const option = document.createElement("option");
      option.value = voice.voice_id;
      option.textContent = voice.category ? `${voice.name} (${voice.category})` : voice.name;
      option.dataset.provider = "elevenlabs";
      option.dataset.previewUrl = voice.preview_url || "";
      voiceSelect.append(option);
    }

    if (voices.length === 0) {
      const option = document.createElement("option");
      option.textContent = "Aucune voix disponible";
      voiceSelect.append(option);
      voiceSelect.disabled = true;
      stopVoiceButton.disabled = true;
      previewVoiceButton.disabled = true;
      return;
    }

    voiceSelect.disabled = false;
    stopVoiceButton.disabled = false;
    previewVoiceButton.disabled = false;
    setMicrophoneStatus(`Voix ElevenLabs chargees: ${voices.length}.`);
  } catch (error) {
    voiceSelect.innerHTML = "";
    const option = document.createElement("option");
    option.textContent = "Erreur ElevenLabs";
    voiceSelect.append(option);
    voiceSelect.disabled = true;
    stopVoiceButton.disabled = true;
    previewVoiceButton.disabled = true;
    setMicrophoneStatus(error.message, true);
  }
}

function getSelectedVoice() {
  return getFrenchVoices().find((voice) => voice.voiceURI === voiceSelect.value) || null;
}

function stopSpeech() {
  if (speechSynthesizer) {
    speechSynthesizer.cancel();
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = "";
  }
}

async function speakText(text) {
  if (!autoReadInput.checked || !text.trim()) {
    return;
  }

  stopSpeech();
  if (ttsProviderInput.value === "elevenlabs") {
    await speakWithElevenLabs(text);
    return;
  }

  if (!speechSynthesizer) {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "fr-FR";
  utterance.rate = 1;
  utterance.pitch = 1;

  const selectedVoice = getSelectedVoice();
  if (selectedVoice) {
    utterance.voice = selectedVoice;
    utterance.lang = selectedVoice.lang;
  }

  speechSynthesizer.speak(utterance);
}

async function speakWithElevenLabs(text) {
  const apiKey = elevenLabsApiKeyInput.value.trim();
  const voiceId = voiceSelect.value;
  if (!apiKey || !voiceId || voiceSelect.disabled) {
    setMicrophoneStatus("Selectionnez une voix ElevenLabs valide.", true);
    return;
  }

  const payload = new FormData();
  payload.append("text", text);
  payload.append("voice_id", voiceId);
  payload.append("elevenlabs_api_key", apiKey);

  try {
    const response = await fetch("/api/elevenlabs/speech", {
      method: "POST",
      body: payload,
    });
    if (!response.ok) {
      const data = await readJsonResponse(response);
      throw new Error(data.detail || "Lecture ElevenLabs impossible.");
    }

    const audioBlob = await response.blob();
    currentAudioUrl = URL.createObjectURL(audioBlob);
    currentAudio = new Audio(currentAudioUrl);
    currentAudio.addEventListener("ended", () => {
      if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = "";
      }
      currentAudio = null;
    });
    await currentAudio.play();
  } catch (error) {
    setMicrophoneStatus(error.message, true);
  }
}

async function previewSelectedVoice() {
  stopSpeech();
  if (ttsProviderInput.value === "elevenlabs") {
    const selectedOption = voiceSelect.selectedOptions[0];
    const previewUrl = selectedOption?.dataset.previewUrl || "";
    if (!previewUrl) {
      setMicrophoneStatus("Aucun extrait audio disponible pour cette voix.", true);
      return;
    }
    currentAudio = new Audio(previewUrl);
    currentAudio.addEventListener("ended", () => {
      currentAudio = null;
    });
    try {
      await currentAudio.play();
    } catch {
      setMicrophoneStatus("Lecture de l'extrait ElevenLabs impossible.", true);
    }
    return;
  }

  if (!speechSynthesizer || voiceSelect.disabled) {
    setMicrophoneStatus("Aucun extrait local disponible.", true);
    return;
  }

  const utterance = new SpeechSynthesisUtterance("Bonjour, ceci est un extrait de cette voix.");
  utterance.lang = "fr-FR";
  const selectedVoice = getSelectedVoice();
  if (selectedVoice) {
    utterance.voice = selectedVoice;
    utterance.lang = selectedVoice.lang;
  }
  speechSynthesizer.speak(utterance);
}

function setListening(nextIsListening) {
  isListening = nextIsListening;
  microphoneButton.classList.toggle("is-listening", isListening);
  microphoneButton.setAttribute("aria-pressed", String(isListening));
  if (!isListening) {
    microphoneButton.textContent = "Micro Ctrl";
    return;
  }
  microphoneButton.textContent = recognitionMode === "control" ? "Relacher Ctrl" : "Arreter le micro";
}

function renderTranscript(interimTranscript = "") {
  const spokenText = `${finalTranscript} ${interimTranscript}`.trim();
  argumentInput.value = `${transcriptBase} ${spokenText}`.trim();
  argumentInput.focus();
}

function resetTranscriptState() {
  transcriptBase = "";
  finalTranscript = "";
  latestInterimTranscript = "";
}

function setupSpeechRecognition() {
  if (!SpeechRecognition) {
    setComposerEnabled(false);
    setMicrophoneStatus("Reconnaissance vocale non supportee par ce navigateur.", true);
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "fr-FR";
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.addEventListener("start", () => {
    isRecognitionActive = true;
    transcriptBase = argumentInput.value.trim();
    finalTranscript = "";
    latestInterimTranscript = "";
    setListening(true);
    setMicrophoneStatus("Ecoute en cours...");
  });

  recognition.addEventListener("result", (event) => {
    let interimTranscript = "";

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript.trim();
      if (event.results[index].isFinal) {
        finalTranscript = `${finalTranscript} ${transcript}`.trim();
        latestInterimTranscript = "";
      } else {
        interimTranscript = `${interimTranscript} ${transcript}`.trim();
      }
    }

    latestInterimTranscript = interimTranscript;
    renderTranscript(interimTranscript);
    setMicrophoneStatus(interimTranscript ? `Brouillon: ${interimTranscript}` : "Ecoute en cours...");
  });

  recognition.addEventListener("end", () => {
    isRecognitionActive = false;
    setListening(false);
    isControlRecording = false;
    if (!finalTranscript && latestInterimTranscript) {
      finalTranscript = latestInterimTranscript;
      latestInterimTranscript = "";
    }
    if (finalTranscript) {
      renderTranscript();
      setMicrophoneStatus("Texte ajoute a votre reponse.");
    } else {
      setMicrophoneStatus("");
    }
  });

  recognition.addEventListener("error", (event) => {
    const messages = {
      "not-allowed": "Acces au micro refuse par le navigateur.",
      "no-speech": "Aucune parole detectee.",
      "audio-capture": "Aucun micro disponible.",
    };
    setMicrophoneStatus(messages[event.error] || "Erreur de reconnaissance vocale.", true);
    isRecognitionActive = false;
    setListening(false);
    isControlRecording = false;
  });
}

function startRecognition(mode = "manual") {
  if (!recognition || isRecognitionActive || isListening || !hasStarted || argumentInput.disabled) {
    return;
  }

  try {
    recognitionMode = mode;
    isRecognitionActive = true;
    recognition.start();
  } catch {
    isRecognitionActive = false;
    setMicrophoneStatus("Le micro est deja en cours d'utilisation.", true);
  }
}

function stopRecognition() {
  if (!recognition || (!isRecognitionActive && !isListening)) {
    return;
  }

  isRecognitionActive = false;
  isControlRecording = false;
  setMicrophoneStatus("Arret du micro...");
  try {
    recognition.stop();
  } catch {
    setListening(false);
  }
}

microphoneButton.addEventListener("click", () => {
  if (!recognition) {
    return;
  }

  if (isRecognitionActive || isListening) {
    stopRecognition();
    return;
  }

  isControlRecording = false;
  startRecognition("manual");
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Control" || event.repeat) {
    return;
  }

  if (!recognition || isRecognitionActive || isListening || !hasStarted || argumentInput.disabled) {
    return;
  }

  event.preventDefault();
  isControlRecording = true;
  startRecognition("control");
});

document.addEventListener("keyup", (event) => {
  if (event.key !== "Control" || !isControlRecording) {
    return;
  }

  event.preventDefault();
  isControlRecording = false;
  stopRecognition();
});

startButton.addEventListener("click", async () => {
  if (!hasSourceContent()) {
    appendMessage("error", "La description ou le PDF de l'evolution/correction est obligatoire.");
    return;
  }

  if (speechSynthesizer) {
    stopSpeech();
  }
  if (recognition && isListening) {
    stopRecognition();
  }

  history = [];
  validatedDecisions = [];
  lastReport = null;
  lastReportMarkdown = "";
  sessionDisplayName = "";
  hasStarted = false;
  messagesNode.innerHTML = "";
  renderValidatedDecisions();
  argumentInput.value = "";
  resetTranscriptState();
  setComposerEnabled(false);
  setLoading(true);

  try {
    sessionSourceFiles.agreement = agreementInput.files[0]?.name || "";
    const { response, data } = await fetchWithDocumentSession("/api/negotiate");

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

    addOpenAIUsage(data.llm_usage || data.openai_usage);
    addValidatedDecisions(data.validated_decisions);
    history.push({ role: "assistant", content: data.reply });
    hasStarted = true;
    appendMessage("assistant", data.reply);
    speakText(data.reply);
    setSetupEnabled(false);
    setComposerEnabled(true);
  } catch (error) {
    appendMessage("error", error.message);
    setComposerEnabled(false);
  } finally {
    setLoading(false);
    if (hasStarted) {
      argumentInput.focus();
    }
  }
});

helpAnswerButton.addEventListener("click", async () => {
  if (!hasStarted) {
    appendMessage("error", "Demarrez la discussion avant de demander de l'aide.");
    return;
  }

  setLoading(true);

  try {
    const { response, data } = await fetchWithDocumentSession(
      "/api/help-answer",
      argumentInput.value.trim(),
    );

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

    addOpenAIUsage(data.llm_usage || data.openai_usage);
    appendMessage("helper", data.reply);
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    setLoading(false);
    argumentInput.focus();
  }
});

framingReportButton.addEventListener("click", async () => {
  if (!hasStarted) {
    appendMessage("error", "Demarrez la discussion avant de generer le rapport.");
    return;
  }

  showBlockingLoader(
    "Generation du rapport...",
    "Analyse de la discussion, calcul du score et preparation des points critiques.",
  );
  setLoading(true);

  try {
    const { response, data } = await fetchWithDocumentSession("/api/framing-report");

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

    addOpenAIUsage(data.llm_usage || data.openai_usage);
    lastReport = data.report || null;
    lastReportMarkdown = data.markdown || "";
    appendReport(data.report || {}, data.markdown || "");
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    hideBlockingLoader();
    setLoading(false);
    argumentInput.focus();
  }
});

argumentInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }

  event.preventDefault();
  if (hasStarted && argumentInput.value.trim() && !sendButton.disabled) {
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const argument = argumentInput.value.trim();

  if (!hasStarted) {
    appendMessage("error", "Demarrez la discussion pour laisser le client poser la premiere question.");
    return;
  }

  if (!argument) {
    appendMessage("error", "La reponse du developpeur est obligatoire.");
    return;
  }

  appendMessage("user", argument);
  if (recognition && isListening) {
    resetTranscriptState();
    stopRecognition();
  }
  setLoading(true);

  try {
    const { response, data } = await fetchWithDocumentSession("/api/negotiate", argument);

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

    addOpenAIUsage(data.llm_usage || data.openai_usage);
    addValidatedDecisions(data.validated_decisions);
    history.push({ role: "user", content: argument }, { role: "assistant", content: data.reply });
    appendMessage("assistant", data.reply);
    speakText(data.reply);
    argumentInput.value = "";
    resetTranscriptState();
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    setLoading(false);
    argumentInput.focus();
  }
});

resetButton.addEventListener("click", () => {
  if (speechSynthesizer) {
    stopSpeech();
  }
  if (recognition && isListening) {
    stopRecognition();
  }
  history = [];
  validatedDecisions = [];
  sessionCostUsd = 0;
  sessionOpenAICalls = 0;
  sessionOpenAITokens = 0;
  hasUnpricedOpenAIUsage = false;
  lastReport = null;
  lastReportMarkdown = "";
  sessionDisplayName = "";
  hasStarted = false;
  documentSessionId = "";
  documentSessionSignature = "";
  sessionSourceFiles = { agreement: "", project_documents: [] };
  renderValidatedDecisions();
  renderSessionCost();
  topicInput.value = "";
  modelLevelInput.value = "medium";
  argumentInput.value = "";
  agreementInput.value = "";
  projectDocsInput.value = "";
  projectDocDirectoryInput.value = "";
  messagesNode.innerHTML = "";
  appendMessage(
    "assistant",
    "Decrivez l'evolution ou la correction, ajoutez un PDF si utile, puis demarrez la discussion.",
  );
  setSetupEnabled(true);
  setComposerEnabled(false);
  setLoading(false);
});

saveSessionButton.addEventListener("click", async () => {
  saveSessionButton.disabled = true;
  try {
    await saveSession();
  } catch (error) {
    appendMessage("error", error.message || "Sauvegarde impossible.");
  } finally {
    saveSessionButton.disabled = !hasStarted;
  }
});

savedSessionSelect.addEventListener("change", () => {
  loadSessionButton.disabled = !savedSessionSelect.value;
});

loadSessionButton.addEventListener("click", async () => {
  const savedSessionId = savedSessionSelect.value;
  if (!savedSessionId) {
    return;
  }
  try {
    const response = await fetch(`/api/saved-sessions/${encodeURIComponent(savedSessionId)}`);
    const data = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Sauvegarde introuvable.");
    }
    await loadSession(data);
  } catch (error) {
    appendMessage("error", error.message || "Sauvegarde CODEV invalide.");
  }
});

stopVoiceButton.addEventListener("click", () => {
  stopSpeech();
});

previewVoiceButton.addEventListener("click", previewSelectedVoice);

providerInput.addEventListener("change", () => {
  documentSessionId = "";
  documentSessionSignature = "";
  renderProviderHelp();
});
ttsProviderInput.addEventListener("change", renderTtsProvider);
elevenLabsApiKeyInput.addEventListener("change", () => {
  if (ttsProviderInput.value === "elevenlabs") {
    populateElevenLabsVoices();
  }
});

if (speechSynthesizer) {
  speechSynthesizer.addEventListener("voiceschanged", populateVoices);
}

setupSpeechRecognition();
renderProviderHelp();
renderTtsProvider();
renderValidatedDecisions();
renderSessionCost();
setComposerEnabled(false);
refreshSavedSessions().catch((error) => {
  setMicrophoneStatus(error.message || "Liste des sauvegardes indisponible.", true);
});
