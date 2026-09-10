import os
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

import re
import uuid
import json
import urllib.error
import urllib.request
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LangChainOpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader
from typing_extensions import TypedDict


OPENAI_LIGHT_MODEL = os.getenv("OPENAI_LIGHT_MODEL", "gpt-5-nano")
OPENAI_MEDIUM_MODEL = os.getenv("OPENAI_MEDIUM_MODEL", "gpt-5.6-luna")
OPENAI_STRONG_MODEL = os.getenv("OPENAI_STRONG_MODEL", "gpt-4.1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LIGHT_MODEL = os.getenv("OLLAMA_LIGHT_MODEL", "qwen3:4b")
OLLAMA_MEDIUM_MODEL = os.getenv("OLLAMA_MEDIUM_MODEL", "qwen3:8b")
OLLAMA_STRONG_MODEL = os.getenv("OLLAMA_STRONG_MODEL", "qwen3:14b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
MAX_PDF_CHARS = 20_000
MAX_DOCUMENT_CHARS = 60_000
MAX_DOCUMENT_SESSION_CHARS = 500_000
DOCUMENT_CHUNK_CHARS = 1_200
DOCUMENT_CHUNK_OVERLAP = 180
MAX_RETRIEVED_DOCUMENT_CHUNKS = 6
MAX_HISTORY_MESSAGES = 12
DOCUMENT_INDEX_VERSION = 1
SESSION_DATA_DIR = Path(__file__).resolve().parent / "data" / "sessions"
PROVIDER_MODELS = {
    "openai": {
        "light": OPENAI_LIGHT_MODEL,
        "medium": OPENAI_MEDIUM_MODEL,
        "strong": OPENAI_STRONG_MODEL,
    },
    "ollama": {
        "light": OLLAMA_LIGHT_MODEL,
        "medium": OLLAMA_MEDIUM_MODEL,
        "strong": OLLAMA_STRONG_MODEL,
    },
}
OPENAI_PRICING_USD_PER_MILLION = {
    "openai:gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "openai:gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "openai:gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "openai:text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
}


class DecisionOutput(BaseModel):
    text: str = Field(description="Decision validee, formulee comme une contrainte projet.")
    evidence: str = Field(description="Extrait exact qui confirme cette decision.")


class NegotiationOutput(BaseModel):
    reply: str = Field(description="Reponse du persona au developpeur.")
    project_decisions: list[DecisionOutput] = Field(
        description="Decisions explicitement confirmees par la documentation projet."
    )
    client_decisions: list[DecisionOutput] = Field(
        description="Decisions explicitement confirmees dans la reponse du persona."
    )
    codev_user_decisions: list[DecisionOutput] = Field(
        description="Decisions explicitement confirmees par l'utilisateur CODEV."
    )


class SavedSessionTitle(BaseModel):
    title: str = Field(description="Titre metier distinctif de 3 a 5 mots.")


class MaturityScore(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    reason: str


class FramingReport(BaseModel):
    global_score: int = Field(ge=0, le=100)
    scores: list[MaturityScore]
    executive_summary: str
    critical_points: list[str]
    clarified_points: list[str]
    residual_risks: list[str]
    acceptance_criteria: list[str]
    next_actions: list[str]


class PriorityAction(BaseModel):
    axis: str
    current_score: int = Field(ge=0, le=100)
    target_score: int = Field(ge=0, le=100)
    action: str
    expected_impact: str


class ReportImprovement(BaseModel):
    target_score: int = Field(ge=0, le=100)
    summary: str
    priority_actions: list[PriorityAction]
    questions_to_answer: list[str]
    quick_wins: list[str]
    definition_of_ready: list[str]
CLIENT_PROFILES = {
    "sales": {
        "label": "Commercial",
        "skill_file": "personas/commercial/SKILL.md",
    },
    "technical": {
        "label": "Developpeur",
        "skill_file": "personas/developpeur/SKILL.md",
    },
    "boss": {
        "label": "Responsable produit",
        "skill_file": "personas/responsable_produit/SKILL.md",
    },
}
TOKEN_PATTERN = re.compile(r"\w{3,}", re.UNICODE)
DOCUMENT_SESSIONS: dict[str, "DocumentSession"] = {}

app = FastAPI(title="Assistant de CODEV")
app.mount("/static", StaticFiles(directory="static"), name="static")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


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
    embedding_model: str
    source_files: list[tuple[str, bytes]]
    vector_index: object | None = None


class LLMProviderError(Exception):
    pass


class LangChainChatCompletions:
    def __init__(self, client: "LLMClient") -> None:
        self.client = client

    def _build_model(self, model: str, reasoning_effort: str | None) -> Any:
        if self.client.provider == "ollama":
            return ChatOllama(
                model=model,
                base_url=self.client.base_url,
                validate_model_on_init=True,
            )
        model_options: dict[str, object] = {
            "model": model,
            "api_key": self.client.api_key,
            "base_url": self.client.base_url,
            "timeout": 120,
            "max_retries": 2,
        }
        if reasoning_effort is not None:
            model_options["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(**model_options)

    @staticmethod
    def _legacy_response(message: Any, model: str) -> SimpleNamespace:
        usage_metadata = message.usage_metadata or {}
        input_details = usage_metadata.get("input_token_details") or {}
        usage = {
            "prompt_tokens": int(usage_metadata.get("input_tokens") or 0),
            "completion_tokens": int(usage_metadata.get("output_tokens") or 0),
            "total_tokens": int(usage_metadata.get("total_tokens") or 0),
            "prompt_tokens_details": {
                "cached_tokens": int(input_details.get("cache_read") or 0),
            },
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=str(message.content or "")))],
            model=model,
            usage=usage,
        )

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
        reasoning_effort: str | None = None,
    ) -> SimpleNamespace:
        chat_model = self._build_model(model, reasoning_effort)
        if response_format is not None:
            chat_model = chat_model.bind(response_format=response_format)
        try:
            message = chat_model.invoke(messages)
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc
        return self._legacy_response(message, f"{self.client.provider}:{model}")

    def create_structured(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        reasoning_effort: str | None = None,
    ) -> SimpleNamespace:
        structured_options: dict[str, object] = {
            "method": "json_schema",
            "include_raw": True,
        }
        if self.client.provider == "openai":
            structured_options["strict"] = True
        structured_model = self._build_model(model, reasoning_effort).with_structured_output(
            schema,
            **structured_options,
        )
        try:
            result = structured_model.invoke(messages)
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc
        parsing_error = result.get("parsing_error")
        parsed = result.get("parsed")
        raw = result.get("raw")
        if parsing_error is not None or parsed is None or raw is None:
            detail = str(parsing_error or "reponse structuree absente")
            raise LLMProviderError(
                f"Reponse structuree {self.client.provider} invalide: {detail}"
            )
        response = self._legacy_response(raw, f"{self.client.provider}:{model}")
        response.parsed = parsed
        return response


class LLMChat:
    def __init__(self, client: "LLMClient") -> None:
        self.completions = LangChainChatCompletions(client)


class LangChainEmbeddings:
    def __init__(self, client: "LLMClient") -> None:
        self.client = client

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        try:
            if self.client.provider == "ollama":
                embedding_model = OllamaEmbeddings(
                    model=model,
                    base_url=self.client.base_url,
                )
            else:
                embedding_model = LangChainOpenAIEmbeddings(
                    model=model,
                    api_key=self.client.api_key,
                    base_url=self.client.base_url,
                )
            embeddings = embedding_model.embed_documents(input)
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = sum(len(encoding.encode(text)) for text in input)
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=embedding) for embedding in embeddings],
            model=f"{self.client.provider}:{model}",
            usage={"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        )


class LLMClient:
    def __init__(self, provider: str, api_key: str, base_url: str) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat = LLMChat(self)
        self.embeddings = LangChainEmbeddings(self)


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


def normalize_provider(provider: str) -> str:
    selected = provider.strip().lower()
    if selected not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail="Fournisseur LLM non pris en charge.")
    return selected


def get_provider_client(provider: str) -> LLMClient:
    selected = normalize_provider(provider)
    if selected == "ollama":
        return LLMClient(provider=selected, api_key="", base_url=OLLAMA_BASE_URL)
    token = os.getenv("OPENAI_API_KEY", "").strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail="La variable d'environnement OPENAI_API_KEY est obligatoire.",
        )
    return LLMClient(provider=selected, api_key=token, base_url=OPENAI_BASE_URL)


def get_llm_config(provider: str, model_level: str) -> tuple[LLMClient, str, str | None]:
    selected = normalize_provider(provider)
    models = PROVIDER_MODELS[selected]
    model = models.get(model_level, models["medium"])
    reasoning_effort = None
    if selected == "openai" and model_level == "medium" and (
        model == "gpt-5.6" or model.startswith("gpt-5.6-")
    ):
        reasoning_effort = "none"
    return get_provider_client(selected), model, reasoning_effort


def get_embedding_config(provider: str) -> tuple[LLMClient, str]:
    selected = normalize_provider(provider)
    model = OLLAMA_EMBEDDING_MODEL if selected == "ollama" else OPENAI_EMBEDDING_MODEL
    return get_provider_client(selected), model


def summarize_llm_usage(
    model: str,
    usage: dict[str, object],
    *,
    embedding: bool = False,
) -> dict[str, object]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = (
        int(prompt_details.get("cached_tokens") or 0)
        if isinstance(prompt_details, dict)
        else 0
    )
    pricing = OPENAI_PRICING_USD_PER_MILLION.get(model)
    estimated_cost_usd = None
    if model.startswith("ollama:"):
        estimated_cost_usd = 0.0
    if pricing is not None:
        uncached_tokens = max(prompt_tokens - cached_tokens, 0)
        input_cost = (
            (uncached_tokens * pricing["input"])
            + (cached_tokens * pricing["cached_input"])
        ) / 1_000_000
        output_cost = 0.0 if embedding else (completion_tokens * pricing["output"]) / 1_000_000
        estimated_cost_usd = input_cost + output_cost
    return {
        "estimated_cost_usd": estimated_cost_usd,
        "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
        "api_calls": 1,
        "fully_priced": estimated_cost_usd is not None,
    }


def merge_llm_usage(*summaries: dict[str, object]) -> dict[str, object]:
    priced_costs = [
        float(summary["estimated_cost_usd"])
        for summary in summaries
        if summary.get("estimated_cost_usd") is not None
    ]
    return {
        "estimated_cost_usd": sum(priced_costs),
        "total_tokens": sum(int(summary.get("total_tokens") or 0) for summary in summaries),
        "api_calls": sum(int(summary.get("api_calls") or 0) for summary in summaries),
        "fully_priced": all(bool(summary.get("fully_priced", True)) for summary in summaries),
    }


def get_embedding_vectors(
    client: LLMClient,
    model: str,
    texts: list[str],
) -> tuple[list[list[float]], dict[str, object]]:
    if not texts:
        return [], merge_llm_usage()

    response = client.embeddings.create(model=model, input=texts)
    return (
        [item.embedding for item in response.data],
        summarize_llm_usage(response.model, response.usage, embedding=True),
    )


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


async def read_project_document(file: UploadFile, max_chars: int) -> tuple[str, str, bytes]:
    filename = file.filename or "document"
    lowered = filename.lower()
    file.file.seek(0)
    raw_content = await file.read()

    if lowered.endswith(".pdf"):
        try:
            from io import BytesIO

            reader = PdfReader(BytesIO(raw_content))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(page for page in pages if page)[:max_chars]
        except Exception as exc:
            raise HTTPException(status_code=400, detail="PDF illisible ou invalide.") from exc
        return filename, text, raw_content
    if lowered.endswith((".md", ".markdown")):
        return filename, raw_content.decode("utf-8", errors="replace")[:max_chars], raw_content
    return filename, "", raw_content


def get_persisted_session_dir(session_id: str) -> Path:
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,119}", session_id):
        raise HTTPException(status_code=400, detail="Identifiant de sauvegarde invalide.")
    return SESSION_DATA_DIR / session_id


def build_saved_session_label(topic: str) -> str:
    normalized = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode("ascii")
    ignored_words = {
        "nous", "souhaitons", "souhaite", "ajouter", "ajout", "dans", "depuis", "avec",
        "pour", "afin", "que", "les", "des", "une", "un", "notre", "leur", "leurs",
        "permettant", "permettre", "vers", "fonctionnalite", "disposer", "correction",
        "evolution", "rapidement", "idealement", "avoir", "etre", "devra", "devrait",
        "application", "ecran", "interface", "page", "systeme",
    }
    raw_words = re.findall(r"[a-z0-9]+", normalized.lower())
    words = []
    quoted_words = [
        word
        for quoted_text in re.findall(r'["“”]([^"“”]+)["“”]', topic)
        for word in re.findall(
            r"[a-z0-9]+",
            unicodedata.normalize("NFKD", quoted_text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower(),
        )
    ]
    for word in quoted_words + raw_words:
        if word in ignored_words or word in words or len(word) < 3:
            continue
        words.append(word)
    return " ".join(words[:4]).capitalize() or "Discussion CODEV"


def sanitize_saved_session_label(label: str, fallback_topic: str) -> str:
    words = re.findall(r"[\w'-]+", re.sub(r"\s+", " ", label).strip(), re.UNICODE)[:5]
    return " ".join(words).strip(" -'") or build_saved_session_label(fallback_topic)


def generate_saved_session_label(topic: str, provider: str) -> tuple[str, dict[str, object]]:
    client, model, reasoning_effort = get_llm_config(provider, "light")
    response = client.chat.completions.create_structured(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu nommes des sessions de cadrage logiciel. Produis un titre metier distinctif "
                    "de 3 a 5 mots, directement deduit de la description. Garde les noms de modules, "
                    "objets metier et actions importantes. Evite les termes generiques comme projet, "
                    "besoin, evolution ou fonctionnalite. N'invente aucune information."
                ),
            },
            {"role": "user", "content": topic},
        ],
        schema=SavedSessionTitle,
        reasoning_effort=reasoning_effort,
    )
    raw_label = response.parsed.title
    return (
        sanitize_saved_session_label(raw_label, topic),
        summarize_llm_usage(response.model, response.usage),
    )


def build_saved_session_id(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")[:50]
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S-%f")
    return f"{slug or 'discussion-codev'}-{timestamp}"


def persist_document_session(
    session: DocumentSession,
    source_files: list[tuple[str, bytes]],
) -> None:
    session_dir = get_persisted_session_dir(session.session_id)
    documents_dir = session_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=False)

    documents = []
    for source_name, raw_content in source_files:
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".pdf", ".md", ".markdown"}:
            suffix = ".bin"
        stored_name = f"{uuid.uuid4()}{suffix}"
        (documents_dir / stored_name).write_bytes(raw_content)
        documents.append({"original_name": source_name, "stored_name": stored_name})

    index_data = {
        "version": DOCUMENT_INDEX_VERSION,
        "session_id": session.session_id,
        "embedding_model": session.embedding_model,
        "vector_dimension": len(session.chunks[0].vector) if session.chunks else 0,
        "source_count": session.source_count,
        "documents": documents,
        "chunks": [
            {
                "source": chunk.source,
                "text": chunk.text,
                "vector": chunk.vector,
            }
            for chunk in session.chunks
        ],
    }
    (session_dir / "index.json").write_text(
        json.dumps(index_data, ensure_ascii=False),
        encoding="utf-8",
    )


def load_persisted_document_session(
    session_id: str,
    provider: str = "openai",
) -> tuple[DocumentSession, dict[str, object]]:
    session_dir = get_persisted_session_dir(session_id)
    index_path = session_dir / "index.json"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Sauvegarde documentaire introuvable.")

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        if index_data.get("version") != DOCUMENT_INDEX_VERSION:
            raise ValueError("version incompatible")
        raw_chunks = index_data.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError("index vide")
        sources = [str(item["source"]) for item in raw_chunks]
        texts = [str(item["text"]) for item in raw_chunks]
        vectors = [[float(value) for value in item["vector"]] for item in raw_chunks]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="Index documentaire sauvegarde invalide.") from exc

    embedding_usage = merge_llm_usage()
    client, embedding_model = get_embedding_config(provider)
    expected_embedding_model = f"{normalize_provider(provider)}:{embedding_model}"
    saved_model = str(index_data.get("embedding_model") or "")
    rebuilt_embeddings = False
    if saved_model != expected_embedding_model:
        try:
            raw_vectors, embedding_usage = get_embedding_vectors(
                client,
                embedding_model,
                [f"{source}\n{text}" for source, text in zip(sources, texts)],
            )
            vectors = [normalize_vector(vector) for vector in raw_vectors]
            saved_model = expected_embedding_model
            rebuilt_embeddings = True
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Impossible de reconstruire l'index documentaire: {exc}",
            ) from exc

    if len(vectors) != len(texts) or not vectors or any(not vector for vector in vectors):
        raise HTTPException(status_code=409, detail="Embeddings documentaires sauvegardes invalides.")
    vector_dimension = len(vectors[0])
    if any(len(vector) != vector_dimension for vector in vectors):
        raise HTTPException(status_code=409, detail="Dimensions d'embeddings incompatibles.")

    if rebuilt_embeddings:
        index_data["embedding_model"] = saved_model
        index_data["vector_dimension"] = vector_dimension
        for raw_chunk, vector in zip(raw_chunks, vectors):
            raw_chunk["vector"] = vector
        try:
            index_path.write_text(
                json.dumps(index_data, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail="Impossible de mettre a jour l'index documentaire sauvegarde.",
            ) from exc

    chunks = [
        DocumentChunk(
            source=source,
            text=text,
            terms=Counter(tokenize(f"{source}\n{text}")),
            vector=vector,
        )
        for source, text, vector in zip(sources, texts, vectors)
    ]
    vector_index, retrieval_mode = build_optional_turbovec_index(vectors)
    stored_source_files = []
    for document in index_data.get("documents") or []:
        try:
            stored_name = str(document["stored_name"])
            if not re.fullmatch(r"[a-f0-9-]+\.(?:pdf|md|markdown|bin)", stored_name):
                continue
            raw_content = (session_dir / "documents" / stored_name).read_bytes()
            stored_source_files.append((str(document["original_name"]), raw_content))
        except (KeyError, OSError, TypeError):
            continue
    session = DocumentSession(
        session_id=session_id,
        chunks=chunks,
        source_count=int(index_data.get("source_count") or len(set(sources))),
        retrieval_mode=retrieval_mode,
        embedding_model=saved_model,
        source_files=stored_source_files,
        vector_index=vector_index,
    )
    DOCUMENT_SESSIONS[session_id] = session
    return session, embedding_usage


async def build_document_session(
    files: list[UploadFile] | None,
    provider: str = "openai",
) -> tuple[DocumentSession, dict[str, object]]:
    if not files:
        raise HTTPException(status_code=400, detail="Aucune documentation projet fournie.")

    chunk_sources: list[str] = []
    chunk_texts: list[str] = []
    chunk_terms: list[Counter[str]] = []
    source_names: set[str] = set()
    source_files: list[tuple[str, bytes]] = []
    remaining_chars = MAX_DOCUMENT_SESSION_CHARS

    for file in files:
        if remaining_chars <= 0:
            break

        source, text, raw_content = await read_project_document(file, remaining_chars)
        text = text.strip()
        if not text:
            continue

        source_names.add(source)
        source_files.append((source, raw_content))
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

    selected_provider = normalize_provider(provider)
    client, embedding_model = get_embedding_config(selected_provider)
    try:
        raw_vectors, embedding_usage = get_embedding_vectors(
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
        embedding_model=f"{selected_provider}:{embedding_model}",
        source_files=source_files,
        vector_index=vector_index,
    )
    DOCUMENT_SESSIONS[session_id] = session
    return session, embedding_usage


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
    provider: str = "openai",
) -> tuple[str, dict[str, object]]:
    if not document_session_id:
        return "", merge_llm_usage()

    session = DOCUMENT_SESSIONS.get(document_session_id)
    restoration_usage = merge_llm_usage()
    if session is None:
        try:
            session, restoration_usage = load_persisted_document_session(
                document_session_id,
                provider,
            )
        except HTTPException as exc:
            raise HTTPException(
                status_code=409,
                detail="Session documentaire inconnue. Rechargez la documentation projet.",
            ) from exc

    query_terms = Counter(tokenize(query))
    if not query_terms:
        selected_chunks = session.chunks[: min(MAX_RETRIEVED_DOCUMENT_CHUNKS, len(session.chunks))]
        embedding_usage = restoration_usage
    else:
        selected_provider = normalize_provider(provider)
        client, embedding_model = get_embedding_config(selected_provider)
        expected_embedding_model = f"{selected_provider}:{embedding_model}"
        if session.embedding_model != expected_embedding_model:
            raise HTTPException(
                status_code=409,
                detail="Le fournisseur a change. Rechargez la documentation projet.",
            )
        try:
            query_vectors, query_usage = get_embedding_vectors(client, embedding_model, [query])
            embedding_usage = merge_llm_usage(restoration_usage, query_usage)
            query_vector = normalize_vector(query_vectors[0])
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
    return "\n\n---\n\n".join(sections), embedding_usage


def get_client_profile(profile: str) -> dict[str, str]:
    profile_config = CLIENT_PROFILES.get(profile, CLIENT_PROFILES["sales"])
    path = SKILLS_DIR / profile_config["skill_file"]
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


def parse_validated_decisions(raw_decisions: str) -> list[dict[str, str]]:
    if not raw_decisions:
        return []
    try:
        parsed_decisions = json.loads(raw_decisions)
    except Exception:
        return []
    if not isinstance(parsed_decisions, list):
        return []
    decisions = []
    for decision in parsed_decisions:
        if isinstance(decision, dict):
            text = re.sub(r"\s+", " ", str(decision.get("text") or "")).strip()
            evidence = re.sub(r"\s+", " ", str(decision.get("evidence") or "")).strip()
            source = str(decision.get("source") or "client")
            profile = str(decision.get("profile") or "sales")
        else:
            text = re.sub(r"\s+", " ", str(decision)).strip()
            evidence = ""
            source = "client"
            profile = "sales"
        if text:
            decisions.append({"text": text, "evidence": evidence, "source": source, "profile": profile})
    return decisions[:100]


def normalize_decision_comparison_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


def decision_texts_are_similar(left: str, right: str) -> bool:
    left_normalized = normalize_decision_comparison_text(left)
    right_normalized = normalize_decision_comparison_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    if min(len(left_normalized), len(right_normalized)) >= 24 and (
        left_normalized in right_normalized or right_normalized in left_normalized
    ):
        return True
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    overlap = len(left_tokens & right_tokens)
    containment = overlap / max(min(len(left_tokens), len(right_tokens)), 1)
    return containment >= 0.72 or SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.82


def filter_new_validated_decisions(
    existing: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    accepted: list[dict[str, str]] = []
    for candidate in candidates:
        comparisons = existing + accepted
        duplicate = any(
            candidate["source"] == known["source"]
            and (
                decision_texts_are_similar(candidate["text"], known["text"])
                or (
                    candidate.get("evidence")
                    and known.get("evidence")
                    and decision_texts_are_similar(candidate["evidence"], known["evidence"])
                )
            )
            for known in comparisons
        )
        if not duplicate:
            accepted.append(candidate)
    return accepted


def evidence_is_in_text(evidence: str, source_text: str) -> bool:
    evidence_normalized = normalize_decision_comparison_text(evidence)
    source_normalized = normalize_decision_comparison_text(source_text)
    return bool(evidence_normalized and source_normalized and evidence_normalized in source_normalized)


def evidence_supports_decision(decision: str, evidence: str) -> bool:
    ignored = {
        "afin", "avec", "dans", "des", "doit", "du", "elle", "export", "contrat",
        "contrats", "les", "pour", "que", "qui", "sera", "sont", "une",
    }
    decision_tokens = set(normalize_decision_comparison_text(decision).split()) - ignored
    evidence_tokens = set(normalize_decision_comparison_text(evidence).split()) - ignored
    overlap = len(decision_tokens & evidence_tokens)
    containment = overlap / max(min(len(decision_tokens), len(evidence_tokens)), 1)
    return (overlap >= 2 and containment >= 0.40) or (overlap >= 3 and containment >= 0.25)


def find_supporting_excerpt(decision: str, source_text: str) -> str:
    excerpts = [
        excerpt.strip()
        for excerpt in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if excerpt.strip()
    ]
    supported = [excerpt for excerpt in excerpts if evidence_supports_decision(decision, excerpt)]
    if not supported:
        return ""
    return max(
        supported,
        key=lambda excerpt: SequenceMatcher(
            None,
            normalize_decision_comparison_text(decision),
            normalize_decision_comparison_text(excerpt),
        ).ratio(),
    )[:500]


def validate_and_classify_decision_sources(
    candidates: list[dict[str, str]],
    document_text: str,
    codev_user_text: str,
    client_reply: str,
) -> list[dict[str, str]]:
    validated = []
    source_texts = {
        "document": document_text,
        "codev_user": codev_user_text,
        "client": client_reply,
    }
    for candidate in candidates:
        evidence = candidate.get("evidence") or ""
        declared_source_text = source_texts.get(candidate["source"], "")
        evidence_is_valid = (
            evidence_is_in_text(evidence, declared_source_text)
            and evidence_supports_decision(candidate["text"], evidence)
        )
        if not evidence_is_valid:
            if candidate["source"] == "document":
                continue
            repaired_evidence = find_supporting_excerpt(candidate["text"], declared_source_text)
            if not repaired_evidence:
                continue
            candidate = dict(candidate)
            candidate["evidence"] = repaired_evidence
        validated.append(candidate)
    return validated


class NegotiationGraphState(TypedDict, total=False):
    client: LLMClient
    model: str
    reasoning_effort: str | None
    messages: list[dict[str, str]]
    profile: str
    document_text: str
    codev_user_text: str
    existing_decisions: list[dict[str, str]]
    response: Any
    reply: str
    decisions: list[dict[str, str]]


def generate_negotiation_response(state: NegotiationGraphState) -> dict[str, object]:
    response = state["client"].chat.completions.create_structured(
        model=state["model"],
        messages=state["messages"],
        schema=NegotiationOutput,
        reasoning_effort=state.get("reasoning_effort"),
    )
    parsed: NegotiationOutput = response.parsed
    decisions = []
    for raw_decisions, source in [
        (parsed.project_decisions, "document"),
        (parsed.client_decisions, "client"),
        (parsed.codev_user_decisions, "codev_user"),
    ]:
        for decision in raw_decisions:
            text = re.sub(r"\s+", " ", decision.text).strip()
            evidence = re.sub(r"\s+", " ", decision.evidence).strip()
            if text:
                decisions.append({
                    "text": text,
                    "evidence": evidence[:500],
                    "source": source,
                    "profile": state["profile"],
                })
    reply = parsed.reply.strip()
    return {"response": response, "reply": reply, "decisions": decisions}


def validate_negotiation_decisions(state: NegotiationGraphState) -> dict[str, object]:
    decisions = validate_and_classify_decision_sources(
        state.get("decisions", []),
        state.get("document_text", ""),
        state.get("codev_user_text", ""),
        state.get("reply", ""),
    )
    decisions = filter_new_validated_decisions(state.get("existing_decisions", []), decisions)
    return {"decisions": decisions}


_negotiation_graph_builder = StateGraph(NegotiationGraphState)
_negotiation_graph_builder.add_node("generate", generate_negotiation_response)
_negotiation_graph_builder.add_node("validate", validate_negotiation_decisions)
_negotiation_graph_builder.add_edge(START, "generate")
_negotiation_graph_builder.add_edge("generate", "validate")
_negotiation_graph_builder.add_edge("validate", END)
NEGOTIATION_GRAPH = _negotiation_graph_builder.compile()


def append_validated_decisions_context(
    project_document_text: str,
    validated_decisions: list[dict[str, str]],
) -> str:
    if not validated_decisions:
        return project_document_text
    decisions_text = "\n".join(
        f"- [{decision['source']}/{decision['profile']}] {decision['text']}"
        + (f" | preuve: {decision['evidence']}" if decision.get("evidence") else "")
        for decision in validated_decisions
    )
    return "\n\n".join(
        section
        for section in [
            project_document_text.strip(),
            f"Decisions deja validees pendant le cadrage:\n{decisions_text}",
        ]
        if section
    )


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


def normalize_report(parsed: dict[str, object]) -> dict[str, object]:
    scores = parsed.get("scores")
    if isinstance(scores, list):
        normalized_scores = []
        for item in scores:
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            try:
                score = int(normalized_item.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            normalized_item["score"] = max(0, min(score, 100))
            normalized_scores.append(normalized_item)
        parsed["scores"] = normalized_scores
        weakest_scores = sorted(item["score"] for item in normalized_scores)[:3]
        parsed["global_score"] = (
            round(sum(weakest_scores) / len(weakest_scores)) if weakest_scores else 0
        )
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


@app.post("/api/document-session")
async def create_document_session(
    provider: Annotated[str, Form()] = "openai",
    project_docs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, object]:
    session, openai_usage = await build_document_session(project_docs, provider)
    return {
        "document_session_id": session.session_id,
        "chunks": len(session.chunks),
        "sources": session.source_count,
        "retrieval_mode": session.retrieval_mode,
        "llm_usage": openai_usage,
    }


@app.post("/api/document-session/restore")
async def restore_document_session(
    document_session_id: Annotated[str, Form()],
    provider: Annotated[str, Form()] = "openai",
) -> dict[str, object]:
    session = DOCUMENT_SESSIONS.get(document_session_id)
    openai_usage = merge_llm_usage()
    _, embedding_model = get_embedding_config(provider)
    expected_embedding_model = f"{normalize_provider(provider)}:{embedding_model}"
    if session is None or session.embedding_model != expected_embedding_model:
        session, openai_usage = load_persisted_document_session(document_session_id, provider)
    return {
        "document_session_id": session.session_id,
        "chunks": len(session.chunks),
        "sources": session.source_count,
        "retrieval_mode": session.retrieval_mode,
        "llm_usage": openai_usage,
    }


@app.get("/api/saved-sessions")
def list_saved_sessions() -> dict[str, object]:
    sessions = []
    if SESSION_DATA_DIR.is_dir():
        for session_path in SESSION_DATA_DIR.iterdir():
            metadata_path = session_path / "session.json"
            if not session_path.is_dir() or not metadata_path.is_file():
                continue
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                topic = str(data.get("topic") or "Discussion CODEV")
                sessions.append({
                    "id": session_path.name,
                    "name": str(data.get("display_name") or build_saved_session_label(topic)),
                    "saved_at": str(data.get("saved_at") or ""),
                    "has_report": isinstance(data.get("report"), dict),
                })
            except (OSError, json.JSONDecodeError):
                continue
    sessions.sort(key=lambda item: item["saved_at"], reverse=True)
    return {"sessions": sessions}


@app.post("/api/saved-sessions")
def save_discussion_session(payload: Annotated[dict[str, object], Body()]) -> dict[str, object]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise HTTPException(status_code=400, detail="La discussion est vide.")

    topic = str(payload.get("topic") or "Discussion CODEV").strip()
    provider = normalize_provider(str(payload.get("provider") or "openai"))
    existing_display_name = str(payload.get("display_name") or "").strip()
    if existing_display_name:
        display_name = sanitize_saved_session_label(existing_display_name, topic)
        naming_usage = merge_llm_usage()
    else:
        try:
            display_name, naming_usage = generate_saved_session_label(topic, provider)
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=f"Nom de sauvegarde impossible: {exc}") from exc
    saved_session_id = build_saved_session_id(display_name)
    saved_path = get_persisted_session_dir(saved_session_id)
    document_session_id = str(payload.get("document_session_id") or "")
    document_session = DOCUMENT_SESSIONS.get(document_session_id)
    if document_session is None and document_session_id:
        try:
            document_session, _ = load_persisted_document_session(document_session_id, provider)
        except HTTPException:
            document_session = None

    try:
        if document_session is not None:
            persisted_session = DocumentSession(
                session_id=saved_session_id,
                chunks=document_session.chunks,
                source_count=document_session.source_count,
                retrieval_mode=document_session.retrieval_mode,
                embedding_model=document_session.embedding_model,
                source_files=document_session.source_files,
                vector_index=document_session.vector_index,
            )
            persist_document_session(persisted_session, persisted_session.source_files)
            DOCUMENT_SESSIONS[saved_session_id] = persisted_session
            payload["document_session_id"] = saved_session_id
        else:
            saved_path.mkdir(parents=True, exist_ok=False)
            payload["document_session_id"] = ""

        payload["format"] = "codev-session"
        payload["version"] = 1
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        payload["storage_id"] = saved_session_id
        payload["display_name"] = display_name
        previous_usage = payload.get("llm_usage") or payload.get("openai_usage")
        payload["llm_usage"] = merge_llm_usage(
            previous_usage if isinstance(previous_usage, dict) else merge_llm_usage(),
            naming_usage,
        )
        payload.pop("openai_usage", None)
        (saved_path / "session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Sauvegarde impossible: {exc}") from exc

    return {
        "id": saved_session_id,
        "display_name": display_name,
        "saved_at": payload["saved_at"],
        "llm_usage": naming_usage,
    }


@app.get("/api/saved-sessions/{saved_session_id}")
def load_discussion_session(saved_session_id: str) -> dict[str, object]:
    saved_path = get_persisted_session_dir(saved_session_id)
    session_path = saved_path / "session.json"
    if not session_path.is_file():
        raise HTTPException(status_code=404, detail="Sauvegarde introuvable.")
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="Sauvegarde illisible.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=409, detail="Sauvegarde invalide.")

    document_text = ""
    index_path = saved_path / "index.json"
    if index_path.is_file():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            document_text = "\n".join(
                str(chunk.get("text") or "")
                for chunk in index_data.get("chunks") or []
                if isinstance(chunk, dict)
            )
        except (OSError, json.JSONDecodeError):
            pass
    history = data.get("history") if isinstance(data.get("history"), list) else []
    codev_text = "\n".join(
        [str(data.get("topic") or "")]
        + [
            str(item.get("content") or "")
            for item in history
            if isinstance(item, dict) and item.get("role") == "user"
        ]
    )
    client_text = "\n".join(
        str(item.get("content") or "")
        for item in history
        if isinstance(item, dict) and item.get("role") == "assistant"
    )
    decisions = parse_validated_decisions(
        json.dumps(data.get("validated_decisions") or [], ensure_ascii=False)
    )
    decisions = validate_and_classify_decision_sources(
        decisions,
        document_text,
        codev_text,
        client_text,
    )
    data["validated_decisions"] = filter_new_validated_decisions([], decisions)
    return data


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
    provider: Annotated[str, Form()] = "openai",
    model_level: Annotated[str, Form()] = "medium",
    history: Annotated[str, Form()] = "[]",
    validated_decisions: Annotated[str, Form()] = "[]",
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
    project_document_text, embedding_usage = retrieve_project_context(
        document_session_id,
        retrieval_query,
        provider,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)
    document_evidence_text = project_document_text
    existing_decisions = parse_validated_decisions(validated_decisions)
    project_document_text = append_validated_decisions_context(
        project_document_text,
        existing_decisions,
    )

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

    client, model, reasoning_effort = get_llm_config(provider, model_level)
    graph_result = NEGOTIATION_GRAPH.invoke({
        "client": client,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": messages,
        "profile": profile,
        "document_text": document_evidence_text,
        "codev_user_text": "\n".join(
            part for part in [topic.strip(), argument.strip()] if part
        ),
        "existing_decisions": existing_decisions,
    })
    response = graph_result["response"]
    reply = graph_result.get("reply", "")
    new_decisions = graph_result.get("decisions", [])
    if not reply:
        raise HTTPException(status_code=502, detail="Reponse de negociation du fournisseur invalide.")
    return {
        "reply": reply,
        "validated_decisions": new_decisions,
        "llm_usage": merge_llm_usage(
            embedding_usage,
            summarize_llm_usage(response.model, response.usage),
        ),
    }


@app.post("/api/help-answer")
async def help_answer(
    topic: Annotated[str, Form()],
    argument: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "sales",
    provider: Annotated[str, Form()] = "openai",
    model_level: Annotated[str, Form()] = "medium",
    history: Annotated[str, Form()] = "[]",
    validated_decisions: Annotated[str, Form()] = "[]",
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
    project_document_text, embedding_usage = retrieve_project_context(
        document_session_id,
        retrieval_query,
        provider,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)
    project_document_text = append_validated_decisions_context(
        project_document_text,
        parse_validated_decisions(validated_decisions),
    )

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

    client, model, reasoning_effort = get_llm_config(provider, model_level)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
    )

    content = response.choices[0].message.content or ""
    return {
        "reply": content.strip(),
        "llm_usage": merge_llm_usage(
            embedding_usage,
            summarize_llm_usage(response.model, response.usage),
        ),
    }


@app.post("/api/framing-report")
async def framing_report(
    topic: Annotated[str, Form()],
    profile: Annotated[str, Form()] = "sales",
    provider: Annotated[str, Form()] = "openai",
    model_level: Annotated[str, Form()] = "medium",
    history: Annotated[str, Form()] = "[]",
    validated_decisions: Annotated[str, Form()] = "[]",
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
    project_document_text, embedding_usage = retrieve_project_context(
        document_session_id,
        retrieval_query,
        provider,
    )
    if not project_document_text and project_docs:
        project_document_text = await extract_project_document_text(project_docs)
    project_document_text = append_validated_decisions_context(
        project_document_text,
        parse_validated_decisions(validated_decisions),
    )

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
Genere le rapport de cadrage et le score de maturite selon la structure imposee par l'application.
Important:
- Analyse la discussion complete dans l'ordre chronologique.
- Les reponses du developpeur et les validations du client doivent etre prises en compte comme des clarifications.
- Si un point etait manquant dans la demande initiale mais a ete precise dans la discussion, ne le compte plus comme manquant.
- Liste ce point dans `clarified_points` et ajuste le score de l'axe concerne a la hausse.
- Applique strictement le bareme conservateur du prompt et justifie chaque score uniquement par des preuves explicites.
- Une question ou une suggestion non validee n'est jamais un point clarifie.
- Ne transforme pas une exigence fonctionnelle en preuve UX, securite, performance ou exploitabilite si cet aspect n'a pas ete discute.
""".strip(),
        }
    )

    client, model, reasoning_effort = get_llm_config(provider, model_level)
    response = client.chat.completions.create_structured(
        model=model,
        messages=messages,
        schema=FramingReport,
        reasoning_effort=reasoning_effort,
    )

    report = normalize_report(response.parsed.model_dump())
    return {
        "report": report,
        "markdown": build_report_markdown(report),
        "llm_usage": merge_llm_usage(
            embedding_usage,
            summarize_llm_usage(response.model, response.usage),
        ),
    }


@app.post("/api/improve-report")
async def improve_report(
    report: Annotated[str, Form()],
    provider: Annotated[str, Form()] = "openai",
    model_level: Annotated[str, Form()] = "medium",
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

    client, model, reasoning_effort = get_llm_config(provider, model_level)
    try:
        response = client.chat.completions.create_structured(
            model=model,
            messages=messages,
            schema=ReportImprovement,
            reasoning_effort=reasoning_effort,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=f"Erreur du fournisseur LLM: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Erreur inattendue pendant l'analyse du rapport.",
        ) from exc

    improvement = response.parsed.model_dump()

    return {
        "improvement": improvement,
        "llm_usage": summarize_llm_usage(response.model, response.usage),
    }
