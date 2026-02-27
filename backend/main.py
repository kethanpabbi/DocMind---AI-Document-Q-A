from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb

app = FastAPI(title="DocMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = "./chroma_db"
COLLECTION  = "documents"
_ef         = embedding_functions.DefaultEmbeddingFunction()
_client     = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection():
    return _client.get_or_create_collection(name=COLLECTION, embedding_function=_ef)

PROVIDERS = {
    "anthropic": {
        "label": "Anthropic", "icon": "🤖",
        "key_prefix": "sk-ant-",
        "key_url": "https://console.anthropic.com",
        "models": {
            "claude-haiku-4-5-20251001": {"label": "Claude Haiku 3.5",  "tag": "cheapest"},
            "claude-sonnet-4-6":         {"label": "Claude Sonnet 4",   "tag": "best"},
        }
    },
    "openai": {
        "label": "OpenAI", "icon": "⚡",
        "key_prefix": "sk-",
        "key_url": "https://platform.openai.com/api-keys",
        "models": {
            "gpt-4o-mini":   {"label": "GPT-4o Mini",  "tag": "cheapest"},
            "gpt-4o":        {"label": "GPT-4o",        "tag": "best"},
            "gpt-4-turbo":   {"label": "GPT-4 Turbo",  "tag": "balanced"},
            "o1-mini":       {"label": "o1 Mini",       "tag": "reasoning"},
        }
    },
    "google": {
        "label": "Google", "icon": "🔷",
        "key_prefix": "AI",
        "key_url": "https://aistudio.google.com/app/apikey",
        "models": {
            "gemini-1.5-flash":   {"label": "Gemini 1.5 Flash",   "tag": "cheapest"},
            "gemini-1.5-pro":     {"label": "Gemini 1.5 Pro",     "tag": "best"},
            "gemini-2.0-flash":   {"label": "Gemini 2.0 Flash",   "tag": "balanced"},
        }
    },
    "mistral": {
        "label": "Mistral", "icon": "🌪️",
        "key_prefix": "",
        "key_url": "https://console.mistral.ai/api-keys",
        "models": {
            "mistral-small-latest":  {"label": "Mistral Small",  "tag": "cheapest"},
            "mistral-large-latest":  {"label": "Mistral Large",  "tag": "best"},
            "codestral-latest":      {"label": "Codestral",      "tag": "code"},
        }
    },
    "cohere": {
        "label": "Cohere", "icon": "🌊",
        "key_prefix": "",
        "key_url": "https://dashboard.cohere.com/api-keys",
        "models": {
            "command-r":       {"label": "Command R",       "tag": "cheapest"},
            "command-r-plus":  {"label": "Command R+",      "tag": "best"},
        }
    },
    "groq": {
        "label": "Groq", "icon": "🚀",
        "key_prefix": "gsk_",
        "key_url": "https://console.groq.com/keys",
        "models": {
            "llama-3.1-8b-instant":   {"label": "Llama 3.1 8B",   "tag": "cheapest"},
            "llama-3.3-70b-versatile":{"label": "Llama 3.3 70B",  "tag": "best"},
            "mixtral-8x7b-32768":     {"label": "Mixtral 8x7B",   "tag": "balanced"},
        }
    },
}

class QueryRequest(BaseModel):
    question: str
    api_key: str
    provider: str
    model: str

class Source(BaseModel):
    index: int
    file: str
    page: str
    snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.get("/")
def root():
    return {"status": "ok", "message": "DocMind API is running"}

@app.get("/providers")
def get_providers():
    return PROVIDERS

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents); tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
    finally:
        os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_documents(pages)
    col = get_collection()
    col.add(
        documents=[c.page_content for c in chunks],
        metadatas=[{"source_file": file.filename, "page": str(c.metadata.get("page", "?"))} for c in chunks],
        ids=[f"{file.filename}-{i}" for i in range(len(chunks))],
    )
    return {"filename": file.filename, "chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if req.provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {req.provider}")
    try:
        col     = get_collection()
        results = col.query(query_texts=[req.question], n_results=4)
    except Exception as e:
        raise HTTPException(500, f"Vector search failed: {str(e)}")

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    context_parts, sources = [], []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        page = meta.get("page", "?"); src = meta.get("source_file", "unknown")
        context_parts.append(f"[Source {i} | File: {src} | Page: {page}]\n{doc}")
        sources.append(Source(index=i, file=src, page=page, snippet=doc[:200]+"...", relevance_score=round(1-float(dist),3)))

    context  = "\n\n---\n\n".join(context_parts)
    system   = "You are a precise document assistant. Answer using ONLY the provided excerpts. Cite sources (e.g. 'According to Source 2...'). If not in excerpts, say so. Be concise but complete."
    user_msg = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {req.question}"

    try:
        if req.provider == "anthropic":
            import anthropic as _a
            r = _a.Anthropic(api_key=req.api_key).messages.create(
                model=req.model, max_tokens=1024, system=system,
                messages=[{"role":"user","content":user_msg}])
            answer = r.content[0].text

        elif req.provider == "openai":
            from openai import OpenAI
            r = OpenAI(api_key=req.api_key).chat.completions.create(
                model=req.model, max_tokens=1024,
                messages=[{"role":"system","content":system},{"role":"user","content":user_msg}])
            answer = r.choices[0].message.content

        elif req.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=req.api_key)
            answer = genai.GenerativeModel(req.model, system_instruction=system).generate_content(user_msg).text

        elif req.provider == "mistral":
            from mistralai import Mistral
            r = Mistral(api_key=req.api_key).chat.complete(
                model=req.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user_msg}])
            answer = r.choices[0].message.content

        elif req.provider == "cohere":
            import cohere
            r = cohere.ClientV2(api_key=req.api_key).chat(
                model=req.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user_msg}])
            answer = r.message.content[0].text

        elif req.provider == "groq":
            from groq import Groq
            r = Groq(api_key=req.api_key).chat.completions.create(
                model=req.model, max_tokens=1024,
                messages=[{"role":"system","content":system},{"role":"user","content":user_msg}])
            answer = r.choices[0].message.content

        else:
            raise HTTPException(400, "Unsupported provider")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, f"API error: {str(e)}")

    return QueryResponse(answer=answer, sources=sources)