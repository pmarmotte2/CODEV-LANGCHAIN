const form = document.querySelector("#negotiation-form");
const topicInput = document.querySelector("#topic");
const profileInput = document.querySelector("#profile");
const modelLevelInput = document.querySelector("#model-level");
const apiTokenInput = document.querySelector("#api-token");
const tokenHintNode = document.querySelector("#token-hint");
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

let history = [];
let validatedDecisions = [];
let hasStarted = false;
let documentSessionId = "";
let documentSessionSignature = "";
let recognition = null;
let isListening = false;
let isRecognitionActive = false;
let transcriptBase = "";
let finalTranscript = "";
let latestInterimTranscript = "";
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

function normalizeDecision(decision) {
  return String(decision || "").replace(/\s+/g, " ").trim();
}

function renderValidatedDecisions() {
  validatedDecisionsNode.innerHTML = "";
  decisionsEmptyNode.hidden = validatedDecisions.length > 0;

  for (const decision of validatedDecisions) {
    const item = document.createElement("li");
    item.textContent = decision;
    validatedDecisionsNode.append(item);
  }
}

function addValidatedDecisions(decisions) {
  if (!Array.isArray(decisions)) {
    return;
  }

  const knownDecisions = new Set(validatedDecisions.map((decision) => decision.toLowerCase()));
  for (const rawDecision of decisions) {
    const decision = normalizeDecision(rawDecision);
    if (!decision || knownDecisions.has(decision.toLowerCase())) {
      continue;
    }
    validatedDecisions.push(decision);
    knownDecisions.add(decision.toLowerCase());
  }

  renderValidatedDecisions();
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
  addValidatedDecisions(report.clarified_points);
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
  modelLevelInput.disabled = !isEnabled;
  apiTokenInput.disabled = !isEnabled;
  agreementInput.disabled = !isEnabled;
  projectDocsInput.disabled = !isEnabled;
  projectDocDirectoryInput.disabled = !isEnabled;
}

function hasSourceContent() {
  return Boolean(topicInput.value.trim() || agreementInput.files[0]);
}

function hasApiToken() {
  return Boolean(apiTokenInput.value.trim());
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
  return files
    .map((file) => `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`)
    .join("|");
}

async function ensureDocumentSession() {
  const files = getProjectDocumentFiles();
  if (files.length === 0) {
    documentSessionId = "";
    documentSessionSignature = "";
    return;
  }

  const signature = getProjectDocumentSignature(files);
  if (documentSessionId && documentSessionSignature === signature) {
    return;
  }

  const payload = new FormData();
  payload.append("api_token", apiTokenInput.value.trim());
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

  documentSessionId = data.document_session_id;
  documentSessionSignature = signature;
  setMicrophoneStatus(
    `Documentation indexee: ${data.sources} source(s), ${data.chunks} extrait(s), mode ${data.retrieval_mode}.`,
  );
}

function buildPayload(argument = "") {
  const payload = new FormData();
  payload.append("topic", topicInput.value.trim());
  payload.append("profile", profileInput.value);
  payload.append("model_level", modelLevelInput.value);
  payload.append("api_token", apiTokenInput.value.trim());
  payload.append("argument", argument);
  payload.append("history", JSON.stringify(history));
  payload.append("validated_decisions", JSON.stringify(validatedDecisions));
  payload.append("document_session_id", documentSessionId);
  if (agreementInput.files[0]) {
    payload.append("agreement", agreementInput.files[0]);
  }
  return payload;
}

function renderProviderHelp() {
  apiTokenInput.placeholder = "github_pat_...";
  tokenHintNode.textContent = "";
  tokenHintNode.append(
    document.createTextNode("Utilisez un fine-grained token GitHub avec Models en lecture. "),
  );
  const link = document.createElement("a");
  link.href =
    "https://github.com/settings/personal-access-tokens/new?name=CODEV%20GitHub%20Models&user_models=read";
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "Creer le token";
  tokenHintNode.append(link);
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

async function improveReport(report, button) {
  if (!hasApiToken()) {
    appendMessage("error", "Le token API est obligatoire.");
    apiTokenInput.focus();
    return;
  }

  button.disabled = true;
  button.textContent = "Analyse en cours...";
  showBlockingLoader(
    "Amelioration du rapport...",
    "Analyse des axes faibles et preparation des actions d'amelioration.",
  );
  setLoading(true);

  const payload = new FormData();
  payload.append("api_token", apiTokenInput.value.trim());
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

  if (!hasApiToken()) {
    appendMessage("error", "Le token API est obligatoire pour demarrer la discussion.");
    apiTokenInput.focus();
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
  hasStarted = false;
  messagesNode.innerHTML = "";
  renderValidatedDecisions();
  argumentInput.value = "";
  resetTranscriptState();
  setComposerEnabled(false);
  setLoading(true);

  try {
    await ensureDocumentSession();
    const response = await fetch("/api/negotiate", {
      method: "POST",
      body: buildPayload(),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

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

  if (!hasApiToken()) {
    appendMessage("error", "Le token API est obligatoire.");
    apiTokenInput.focus();
    return;
  }

  setLoading(true);

  try {
    await ensureDocumentSession();
    const response = await fetch("/api/help-answer", {
      method: "POST",
      body: buildPayload(argumentInput.value.trim()),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

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

  if (!hasApiToken()) {
    appendMessage("error", "Le token API est obligatoire.");
    apiTokenInput.focus();
    return;
  }

  showBlockingLoader(
    "Generation du rapport...",
    "Analyse de la discussion, calcul du score et preparation des points critiques.",
  );
  setLoading(true);

  try {
    await ensureDocumentSession();
    const response = await fetch("/api/framing-report", {
      method: "POST",
      body: buildPayload(),
    });
    const data = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

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

  if (!hasApiToken()) {
    appendMessage("error", "Le token API est obligatoire.");
    apiTokenInput.focus();
    return;
  }

  appendMessage("user", argument);
  if (recognition && isListening) {
    resetTranscriptState();
    stopRecognition();
  }
  setLoading(true);

  try {
    await ensureDocumentSession();
    const response = await fetch("/api/negotiate", {
      method: "POST",
      body: buildPayload(argument),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Erreur inconnue.");
    }

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
  hasStarted = false;
  documentSessionId = "";
  documentSessionSignature = "";
  renderValidatedDecisions();
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
  renderProviderHelp();
});

stopVoiceButton.addEventListener("click", () => {
  stopSpeech();
});

previewVoiceButton.addEventListener("click", previewSelectedVoice);

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
setComposerEnabled(false);
