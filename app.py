import os
import re
import uuid
import json
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader


GITHUB_MODELS_LIGHT_MODEL = os.getenv("GITHUB_MODELS_LIGHT_MODEL", "openai/gpt-4.1-nano")
GITHUB_MODELS_MEDIUM_MODEL = os.getenv("GITHUB_MODELS_MEDIUM_MODEL", "openai/gpt-4.1-mini")
GITHUB_MODELS_STRONG_MODEL = os.getenv("GITHUB_MODELS_STRONG_MODEL", "openai/gpt-4.1")
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL",
    "https://models.github.ai/inference",
)
ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
GITHUB_MODELS_EMBEDDING_MODEL = os.getenv(
    "GITHUB_MODELS_EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)
MAX_PDF_CHARS = 20_000
MAX_DOCUMENT_CHARS = 60_000
MAX_DOCUMENT_SESSION_CHARS = 500_000
DOCUMENT_CHUNK_CHARS = 1_200
DOCUMENT_CHUNK_OVERLAP = 180
MAX_RETRIEVED_DOCUMENT_CHUNKS = 6
MAX_HISTORY_MESSAGES = 12
MODEL_LEVELS = {
    "light": GITHUB_MODELS_LIGHT_MODEL,
    "medium": GITHUB_MODELS_MEDIUM_MODEL,
    "strong": GITHUB_MODELS_STRONG_MODEL,
}
CLIENT_PROFILES = {
    "sales": {
        "label": "Commercial",
        "prompt_file": "role_commercial.txt",
    },
    "technical": {
        "label": "Developpeur",
        "prompt_file": "role_developpeur.txt",
    },
    "boss": {
        "label": "Responsable produit",
        "prompt_file": "role_responsable_produit.txt",
    },
}
TOKEN_PATTERN = re.compile(r"\w{3,}", re.UNICODE)
DOCUMENT_SESSIONS: dict[str, "DocumentSession"] = {}

app = FastAPI(title="Assistant de CODEV")
app.mount("/static", StaticFiles(directory="static"), name="static")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@dataclass
class DocumentChunk:
    source: str
    text: str
    terms: Counter[str]
    vector: list[float]


@dataclass
class DocumentSession:
    session_id: str
    chunks: list[DocumentChunk]
    source_count: int
    retrieval_mode: str
    vector_index: object | None = None


class GitHubModelsError(Exception):
    pass


class GitHubModelsChatCompletions:
    def __init__(self, client: "GitHubModelsClient") -> None:
        self.client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> SimpleNamespace:
        data = self.client.post_json(
            "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
        )
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=choice.get("message", {}).get("content", ""))
            )
            for choice in data.get("choices", [])
        ]
        return SimpleNamespace(choices=choices)


class GitHubModelsChat:
    def __init__(self, client: "GitHubModelsClient") -> None:
        self.completions = GitHubModelsChatCompletions(client)


class GitHubModelsEmbeddings:
    def __init__(self, client: "GitHubModelsClient") -> None:
        self.client = client

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        data = self.client.post_json(
            "/embeddings",
            {
                "model": model,
                "input": input,
            },
        )
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=item.get("embedding", []))
                for item in data.get("data", [])
            ]
        )


class GitHubModelsClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat = GitHubModelsChat(self)
        self.embeddings = GitHubModelsEmbeddings(self)

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise GitHubModelsError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise GitHubModelsError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise GitHubModelsError("Reponse GitHub Models invalide.") from exc


class ElevenLabsError(Exception):
    pass


def call_elevenlabs_json(path: str, api_key: str) -> dict[str, object]:
    token = api_key.strip()
    if not token:
        raise HTTPException(status_code=400, detail="La cle API ElevenLabs est obligatoire.")

    request = urllib.request.Request(
        f"{ELEVENLABS_BASE_URL}{path}",
        headers={
            "xi-api-key": token,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ElevenLabsError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ElevenLabsError("Reponse ElevenLabs invalide.") from exc


def call_elevenlabs_audio(path: str, api_key: str, payload: dict[str, object]) -> bytes:
    token = api_key.strip()
    if not token:
        raise HTTPException(status_code=400, detail="La cle API ElevenLabs est obligatoire.")

    request = urllib.request.Request(
        f"{ELEVENLABS_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": token,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ElevenLabsError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsError(str(exc)) from exc


def get_github_models_client(api_token: str) -> GitHubModelsClient:
    token = api_token.strip() or os.getenv("GITHUB_MODELS_TOKEN", "")
    if not token:
        raise HTTPException(
            status_code=400,
            detail="Le token GitHub Models est obligatoire.",
        )
    return GitHubModelsClient(api_key=token, base_url=GITHUB_MODELS_BASE_URL)


def get_llm_config(api_token: str, model_level: str) -> tuple[GitHubModelsClient, str]:
    model = MODEL_LEVELS.get(model_level, GITHUB_MODELS_MEDIUM_MODEL)
    return get_github_models_client(api_token), model


def get_embedding_config(api_token: str) -> tuple[GitHubModelsClient, str]:
    return get_github_models_client(api_token), GITHUB_MODELS_EMBEDDING_MODEL


def get_embedding_vectors(client: GitHubModelsClient, model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def build_optional_turbovec_index(vectors: list[list[float]]) -> tuple[object | None, str]:
    if not vectors:
        return None, "vector"

    try:
        import numpy as np
        from turbovec import IdMapIndex
    except Exception:
        return None, "vector"

    try:
        vector_array = np.asarray(vectors, dtype=np.float32)
        ids = np.arange(len(vectors), dtype=np.uint64)
        index = IdMapIndex(dim=len(vectors[0]), bit_width=4)
        index.add_with_ids(vector_array, ids)
        return index, "turbovec"
    except Exception:
        return None, "vector"


def extract_pdf_text(file: UploadFile | None, max_chars: int = MAX_PDF_CHARS) -> str:
    if file is None:
        return ""

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Le fichier doit etre un PDF.")

    try:
        reader = PdfReader(file.file)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="PDF illisible ou invalide.") from exc

    text = "\n\n".join(page for page in pages if page)
    return text[:max_chars]


async def extract_project_document_text(files: list[UploadFile] | None) -> str:
    if not files:
        return ""

    sections: list[str] = []
    remaining_chars = MAX_DOCUMENT_CHARS
    for file in files:
        if remaining_chars <= 0:
            break

        filename = file.filename or "document"
        lowered = filename.lower()
        text = ""

        if lowered.endswith(".pdf"):
            file.file.seek(0)
            text = extract_pdf_text(file, remaining_chars)
        elif lowered.endswith((".md", ".markdown")):
            raw_content = await file.read()
            text = raw_content.decode("utf-8", errors="replace")
        else:
            continue

        text = text.strip()
        if not text:
            continue

        section = f"Document: {filename}\n{text[:remaining_chars]}"
        sections.append(section)
        remaining_chars -= len(section)

    return "\n\n---\n\n".join(sections)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def chunk_text(text: str) -> list[str]:
    clean_text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + DOCUMENT_CHUNK_CHARS, len(clean_text))
        if end < len(clean_text):
            paragraph_break = clean_text.rfind("\n\n", start, end)
            if paragraph_break > start + 300:
                end = paragraph_break
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = max(end - DOCUMENT_CHUNK_OVERLAP, start + 1)
    return chunks


async def read_project_document(file: UploadFile, max_chars: int) -> tuple[str, str]:
    filename = file.filename or "document"
    lowered = filename.lower()

    if lowered.endswith(".pdf"):
        file.file.seek(0)
        return filename, extract_pdf_text(file, max_chars)
    if lowered.endswith((".md", ".markdown")):
        raw_content = await file.read()
        return filename, raw_content.decode("utf-8", errors="replace")[:max_chars]
    return filename, ""


async def build_document_session(
    files: list[UploadFile] | None,
    api_token: str,
) -> DocumentSession:
    if not files:
        raise HTTPException(status_code=400, detail="Aucune documentation projet fournie.")

    chunk_sources: list[str] = []
    chunk_texts: list[str] = []
    chunk_terms: list[Counter[str]] = []
    source_names: set[str] = set()
    remaining_chars = MAX_DOCUMENT_SESSION_CHARS

    for file in files:
        if remaining_chars <= 0:
            break

        source, text = await read_project_document(file, remaining_chars)
        text = text.strip()
        if not text:
            continue

        source_names.add(source)
        remaining_chars -= len(text)
        for chunk in chunk_text(text):
            terms = Counter(tokenize(f"{source}\n{chunk}"))
            if terms:
                chunk_sources.append(source)
                chunk_texts.append(chunk)
                chunk_terms.append(terms)

    if not chunk_texts:
        raise HTTPException(
            status_code=400,
            detail="Aucun contenu exploitable trouve dans la documentation projet.",
        )

    client, embedding_model = get_embedding_config(api_token)
    try:
        raw_vectors = get_embedding_vectors(
            client,
            embedding_model,
            [f"{source}\n{text}" for source, text in zip(chunk_sources, chunk_texts)],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de generer les embeddings de la documentation: {exc}",
        ) from exc

    vectors = [normalize_vector(vector) for vector in raw_vectors]
    if not vectors:
        raise HTTPException(
            status_code=400,
            detail="Aucun embedding genere pour la documentation projet.",
        )

    chunks = [
        DocumentChunk(source=source, text=text, terms=terms, vector=vector)
        for source, text, terms, vector in zip(chunk_sources, chunk_texts, chunk_terms, vectors)
    ]
    vector_index, retrieval_mode = build_optional_turbovec_index(vectors)

    session_id = str(uuid.uuid4())
    session = DocumentSession(
        session_id=session_id,
        chunks=chunks,
        source_count=len(source_names),
        retrieval_mode=retrieval_mode,
        vector_index=vector_index,
    )
    DOCUMENT_SESSIONS[session_id] = session
    return session


def score_lexical_chunks(session: DocumentSession, query_terms: Counter[str]) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not query_terms:
        return scores

    for index, chunk in enumerate(session.chunks):
        overlap = query_terms.keys() & chunk.terms.keys()
        if not overlap:
            continue
        score = sum(query_terms[term] * chunk.terms[term] for term in overlap)
        scores[index] = score / max(sum(chunk.terms.values()), 1)
    return scores


def score_vector_chunks(
    session: DocumentSession,
    query_vector: list[float],
) -> dict[int, float]:
    if not query_vector:
        return {}

    if session.vector_index is not None:
        try:
            import numpy as np

            query_array = np.asarray([query_vector], dtype=np.float32)
            _, ids = session.vector_index.search(
                query_array,
                k=min(MAX_RETRIEVED_DOCUMENT_CHUNKS * 3, len(session.chunks)),
            )
            raw_ids = ids[0] if getattr(ids, "ndim", 1) > 1 else ids
            return {
                int(chunk_id): dot_product(query_vector, session.chunks[int(chunk_id)].vector)
                for chunk_id in raw_ids
                if 0 <= int(chunk_id) < len(session.chunks)
            }
        except Exception:
            pass

    return {
        index: dot_product(query_vector, chunk.vector)
        for index, chunk in enumerate(session.chunks)
    }


def retrieve_project_context(
    document_session_id: str,
    query: str,
    api_token: str,
) -> str:
    if not document_session_id:
        return ""

    session = DOCUMENT_SESSIONS.get(document_session_id)
    if session is None:
        raise HTTPException(
            status_code=400,
            detail="Session documentaire inconnue. Rechargez la documentation projet.",
        )

    query_terms = Counter(tokenize(query))
    if not query_terms:
        selected_chunks = session.chunks[: min(MAX_RETRIEVED_DOCUMENT_CHUNKS, len(session.chunks))]
    else:
        client, embedding_model = get_embedding_config(api_token)
        try:
            query_vector = normalize_vector(get_embedding_vectors(client, embedding_model, [query])[0])
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Impossible de generer l'embedding de recherche documentaire: {exc}",
            ) from exc

        lexical_scores = score_lexical_chunks(session, query_terms)
        vector_scores = score_vector_chunks(session, query_vector)
        combined_scores: dict[int, float] = {}
        for index in set(lexical_scores) | set(vector_scores):
            combined_scores[index] = (
                (0.35 * lexical_scores.get(index, 0.0))
                + (0.65 * vector_scores.get(index, 0.0))
            )
        scored_chunks = sorted(
            ((score, session.chunks[index]) for index, score in combined_scores.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        selected_chunks = [
            chunk for _, chunk in scored_chunks[:MAX_RETRIEVED_DOCUMENT_CHUNKS]
        ] or session.chunks[: min(3, len(session.chunks))]

    sections = [
        f"Source: {chunk.source}\n{chunk.text}"
        for chunk in selected_chunks
    ]
    return "\n\n---\n\n".join(sections)


def get_client_profile(profile: str) -> dict[str, str]:
    profile_config = CLIENT_PROFILES.get(profile, CLIENT_PROFILES["sales"])
    prompt_filename = profile_config["prompt_file"]
    path = os.path.join(PROMPTS_DIR, prompt_filename)
    with open(path, encoding="utf-8") as prompt_file:
        profile_prompt = prompt_file.read().strip()
    return {
        "label": profile_config["label"],
        "prompt": profile_prompt,
    }


def load_prompt_template(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as prompt_file:
        return prompt_file.read().strip()


def build_change_context(change_document_text: str) -> str:
    if change_document_text:
        return f"\n\nDocument de l'evolution/correction extrait du PDF:\n{change_document_text}"
    return "\n\nAucun PDF d'evolution/correction n'a ete fourni."


def build_project_context(project_document_text: str) -> str:
    if project_document_text:
        return f"\n\nDocumentation projet disponible:\n{project_document_text}"
    return "\n\nAucune documentation projet n'a ete fournie."


def build_system_prompt(
    change_description: str,
    change_document_text: str,
    project_document_text: str,
    profile: str,
) -> str:
    description = change_description or "Aucune description texte n'a ete fournie."
    client_profile = get_client_profile(profile)
    return load_prompt_template("client_questions.txt").format(
        description=description,
        change_context=build_change_context(change_document_text),
        project_context=build_project_context(project_document_text),
        client_profile_prompt=client_profile["prompt"],
    )


def build_answer_help_prompt(
    change_description: str,
    change_document_text: str,
    project_document_text: str,
) -> str:
    description = change_description or "Aucune description texte n'a ete fournie."
    return load_prompt_template("answer_help.txt").format(
        description=description,
        change_context=build_change_context(change_document_text),
        project_context=build_project_context(project_document_text),
    )


def build_framing_report_prompt(
    change_description: str,
    change_document_text: str,
    project_document_text: str,
) -> str:
    description = change_description or "Aucune description texte n'a ete fournie."
    return load_prompt_template("framing_report.txt").format(
        description=description,
        change_context=build_change_context(change_document_text),
        project_context=build_project_context(project_document_text),
    )


def build_report_markdown(report: dict[str, object]) -> str:
    scores = report.get("scores") if isinstance(report.get("scores"), list) else []
    sections = [
        "# Rapport de cadrage CODEV",
        "",
        f"Score de maturite global: {int(report.get('global_score') or 0)} %",
        "",
        "## Synthese",
        str(report.get("executive_summary") or "Non renseigne."),
        "",
        "## Scores",
    ]

    for item in scores:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "Critere"
        score = int(item.get("score") or 0)
        reason = item.get("reason") or "Non renseigne."
        sections.append(f"- {name}: {score} % - {reason}")

    for key, title in [
        ("critical_points", "Points critiques a clarifier"),
        ("clarified_points", "Points deja clarifies"),
        ("residual_risks", "Risques residuels"),
        ("acceptance_criteria", "Criteres d'acceptation proposes"),
        ("next_actions", "Prochaines actions"),
    ]:
        values = report.get(key) if isinstance(report.get(key), list) else []
        sections.extend(["", f"## {title}"])
        if values:
            sections.extend(f"- {value}" for value in values)
        else:
            sections.append("- Aucun element identifie.")

    return "\n".join(sections)


def build_report_improvement_prompt(report: dict[str, object]) -> str:
    return load_prompt_template("report_improvement.txt").format(
        report_json=json.dumps(report, ensure_ascii=False, indent=2)
    )


def parse_report_response(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise HTTPException(status_code=502, detail="Rapport IA invalide.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="Rapport IA invalide.") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Rapport IA invalide.")
    return parsed


def build_fallback_improvement(report: dict[str, object]) -> dict[str, object]:
    scores = report.get("scores") if isinstance(report.get("scores"), list) else []
    weak_scores = sorted(
        [item for item in scores if isinstance(item, dict)],
        key=lambda item: int(item.get("score") or 0),
    )[:3]
    current_global_score = int(report.get("global_score") or 0)
    target_score = min(max(current_global_score + 20, 70), 90)

    priority_actions = []
    questions_to_answer = []
    for item in weak_scores:
        axis = str(item.get("name") or "Axe a clarifier")
        current_score = int(item.get("score") or 0)
        priority_actions.append(
            {
                "axis": axis,
                "current_score": current_score,
                "target_score": min(max(current_score + 25, 70), 90),
                "action": f"Clarifier les attendus sur l'axe {axis.lower()} et les transformer en criteres d'acceptation verifiables.",
                "expected_impact": "Reduction des zones floues avant developpement.",
            }
        )
        questions_to_answer.append(
            f"Quelles decisions explicites manquent encore pour securiser l'axe {axis.lower()} ?"
        )

    critical_points = report.get("critical_points")
    if isinstance(critical_points, list):
        questions_to_answer.extend(str(point) for point in critical_points[:5])

    return {
        "target_score": target_score,
        "summary": "Le score peut progresser en transformant les points faibles du rapport en decisions explicites, criteres d'acceptation et contraintes mesurables.",
        "priority_actions": priority_actions,
        "questions_to_answer": questions_to_answer[:8],
        "quick_wins": [
            "Nommer les droits d'acces attendus.",
            "Definir les donnees incluses et exclues.",
            "Fixer les volumes ou limites de performance.",
            "Ajouter des criteres d'acceptation testables.",
        ],
        "definition_of_ready": [
            "Les utilisateurs concernes sont identifies.",
            "Les donnees manipulees sont listees.",
            "Les cas limites principaux sont couverts.",
            "Les criteres d'acceptation sont mesurables.",
        ],
    }


def build_opening_prompt() -> str:
    return load_prompt_template("opening_question.txt")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/document-session")
async def create_document_session(
    api_token: Annotated[str, Form()] = "",
    project_docs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, object]:
    session = await build_document_session(project_docs, api_token)
    return {
        "document_session_id": session.session_id,
        "chunks": len(session.chunks),
        "sources": session.source_count,
        "retrieval_mode": session.retrieval_mode,
    }


@app.post("/api/elevenlabs/voices")
async def elevenlabs_voices(
    elevenlabs_api_key: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        data = call_elevenlabs_json(
            "/v2/voices?page_size=100&include_total_count=false",
            elevenlabs_api_key,
        )
    except ElevenLabsError as exc:
        raise HTTPException(status_code=502, detail=f"Erreur ElevenLabs: {exc}") from exc

    voices = data.get("voices", [])
    if not isinstance(voices, list):
        raise HTTPException(status_code=502, detail="Reponse ElevenLabs invalide.")

    return {
        "voices": [
            {
                "voice_id": voice.get("voice_id"),
                "name": voice.get("name") or "Voix sans nom",
                "category": voice.get("category"),
                "preview_url": voice.get("preview_url"),
            }
            for voice in voices
            if isinstance(voice, dict) and voice.get("voice_id")
        ]
    }


@app.post("/api/elevenlabs/speech")
async def elevenlabs_speech(
    text: Annotated[str, Form()],
    voice_id: Annotated[str, Form()],
    elevenlabs_api_key: Annotated[str, Form()],
) -> Response:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte a lire est obligatoire.")
    if not voice_id.strip():
        raise HTTPException(status_code=400, detail="La voix ElevenLabs est obligatoire.")

    try:
        audio = call_elevenlabs_audio(
            f"/v1/text-to-speech/{voice_id.strip()}?output_format=mp3_44100_128",
            elevenlabs_api_key,
            {
                "text": text.strip(),
                "model_id": ELEVENLABS_TTS_MODEL,
            },
        )
    except ElevenLabsError as exc:
        raise HTTPException(status_code=502, detail=f"Erreur ElevenLabs: {exc}") from exc

    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/negotiate")
async def negotiate(
    topic: Annotated[str, Form()],
    argument: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "sales",
    model_level: Annotated[str, Form()] = "medium",
    api_token: Annotated[str, Form()] = "",
    history: Annotated[str, Form()] = "[]",
    document_session_id: Annotated[str, Form()] = "",
    agreement: Annotated[UploadFile | None, File()] = None,
    project_docs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, object]:
    change_document_text = extract_pdf_text(agreement)
    if not topic.strip() and not change_document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="La description ou le PDF de l'evolution/correction est obligatoire.",
        )

    try:
        import json

        raw_history = json.loads(history)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Historique invalide.") from exc

    retrieval_query = "\n".join(
        [
            topic.strip(),
            argument.strip() or build_opening_prompt(),
            *[
                item.get("content", "")
                for item in raw_history[-4:]
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ],
        ]
    )
    project_document_text = retrieve_project_context(
        document_session_id,
        retrieval_query,
        api_token,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                topic.strip(),
                change_document_text,
                project_document_text,
                profile,
            ),
        }
    ]
    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    if argument.strip():
        messages.append({"role": "user", "content": argument.strip()})
    elif raw_history:
        raise HTTPException(status_code=400, detail="La reponse du developpeur est obligatoire.")
    else:
        messages.append({"role": "user", "content": build_opening_prompt()})

    client, model = get_llm_config(api_token, model_level)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""
    return {
        "reply": content.strip(),
    }


@app.post("/api/help-answer")
async def help_answer(
    topic: Annotated[str, Form()],
    argument: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "sales",
    model_level: Annotated[str, Form()] = "medium",
    api_token: Annotated[str, Form()] = "",
    history: Annotated[str, Form()] = "[]",
    document_session_id: Annotated[str, Form()] = "",
    agreement: Annotated[UploadFile | None, File()] = None,
    project_docs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, object]:
    change_document_text = extract_pdf_text(agreement)
    if not topic.strip() and not change_document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="La description ou le PDF de l'evolution/correction est obligatoire.",
        )

    try:
        import json

        raw_history = json.loads(history)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Historique invalide.") from exc

    client_profile = get_client_profile(profile)
    retrieval_query = "\n".join(
        [
            topic.strip(),
            argument.strip(),
            client_profile["label"],
            *[
                item.get("content", "")
                for item in raw_history[-4:]
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ],
        ]
    )
    project_document_text = retrieve_project_context(
        document_session_id,
        retrieval_query,
        api_token,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)

    messages = [
        {
            "role": "system",
            "content": build_answer_help_prompt(
                topic.strip(),
                change_document_text,
                project_document_text,
            ),
        },
        {
            "role": "user",
            "content": f"Profil actuel du client: {client_profile['label']}.",
        },
    ]
    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    draft = argument.strip()
    messages.append(
        {
            "role": "user",
            "content": (
                f"Voici mon brouillon de reponse: {draft}\nAide-moi a l'ameliorer sans repondre a ma place."
                if draft
                else "Aide-moi a preparer une reponse au dernier message du client sans repondre a ma place."
            ),
        }
    )

    client, model = get_llm_config(api_token, model_level)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )

    content = response.choices[0].message.content or ""
    return {
        "reply": content.strip(),
    }


@app.post("/api/framing-report")
async def framing_report(
    topic: Annotated[str, Form()],
    profile: Annotated[str, Form()] = "sales",
    model_level: Annotated[str, Form()] = "medium",
    api_token: Annotated[str, Form()] = "",
    history: Annotated[str, Form()] = "[]",
    document_session_id: Annotated[str, Form()] = "",
    agreement: Annotated[UploadFile | None, File()] = None,
    project_docs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, object]:
    change_document_text = extract_pdf_text(agreement)
    if not topic.strip() and not change_document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="La description ou le PDF de l'evolution/correction est obligatoire.",
        )

    try:
        raw_history = json.loads(history)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Historique invalide.") from exc

    client_profile = get_client_profile(profile)
    retrieval_query = "\n".join(
        [
            topic.strip(),
            client_profile["label"],
            *[
                item.get("content", "")
                for item in raw_history[-8:]
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ],
        ]
    )
    project_document_text = retrieve_project_context(
        document_session_id,
        retrieval_query,
        api_token,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)

    messages = [
        {
            "role": "system",
            "content": build_framing_report_prompt(
                topic.strip(),
                change_document_text,
                project_document_text,
            ),
        },
        {
            "role": "user",
            "content": f"Profil client utilise pendant la discussion: {client_profile['label']}.",
        },
    ]
    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": """
Genere le rapport de cadrage et le score de maturite au format JSON demande.
Important:
- Analyse la discussion complete dans l'ordre chronologique.
- Les reponses du developpeur et les validations du client doivent etre prises en compte comme des clarifications.
- Si un point etait manquant dans la demande initiale mais a ete precise dans la discussion, ne le compte plus comme manquant.
- Liste ce point dans `clarified_points` et ajuste le score de l'axe concerne a la hausse.
""".strip(),
        }
    )

    client, model = get_llm_config(api_token, model_level)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    report = parse_report_response(content.strip())
    return {
        "report": report,
        "markdown": build_report_markdown(report),
    }


@app.post("/api/improve-report")
async def improve_report(
    report: Annotated[str, Form()],
    model_level: Annotated[str, Form()] = "medium",
    api_token: Annotated[str, Form()] = "",
) -> dict[str, object]:
    try:
        parsed_report = json.loads(report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Rapport invalide.") from exc

    if not isinstance(parsed_report, dict):
        raise HTTPException(status_code=400, detail="Rapport invalide.")

    try:
        messages = [
            {
                "role": "system",
                "content": build_report_improvement_prompt(parsed_report),
            },
            {
                "role": "user",
                "content": "Explique les actions a mener pour ameliorer le score de maturite du rapport.",
            },
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prompt d'amelioration du rapport invalide.",
        ) from exc

    client, model = get_llm_config(api_token, model_level)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except GitHubModelsError as exc:
        raise HTTPException(status_code=502, detail=f"Erreur du fournisseur LLM: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Erreur inattendue pendant l'analyse du rapport.",
        ) from exc

    content = response.choices[0].message.content or ""
    try:
        improvement = parse_report_response(content.strip())
    except HTTPException:
        improvement = build_fallback_improvement(parsed_report)

    return {
        "improvement": improvement,
    }
