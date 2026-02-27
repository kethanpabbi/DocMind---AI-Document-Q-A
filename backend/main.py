from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import tempfile
import anthropic

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = "./chroma_db"
COLLECTION = "documents"

# Use chromadb's built-in embedding (no torch needed)
_ef = embedding_functions.DefaultEmbeddingFunction()
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=_ef,
    )


# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    api_key: str

class Source(BaseModel):
    index: int
    file: str
    page: int | str
    snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "DocMind API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    col = get_collection()
    col.add(
        documents=[c.page_content for c in chunks],
        metadatas=[{
            "source_file": file.filename,
            "page": str(c.metadata.get("page", "?")),
        } for c in chunks],
        ids=[f"{file.filename}-{i}" for i in range(len(chunks))],
    )

    return {"filename": file.filename, "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.api_key.startswith("sk-ant-"):
        raise HTTPException(400, "Invalid Anthropic API key.")

    try:
        col = get_collection()
        results = col.query(query_texts=[req.question], n_results=4)
    except Exception as e:
        raise HTTPException(500, f"Vector search failed: {str(e)}")

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        page = meta.get("page", "?")
        src  = meta.get("source_file", "unknown")
        context_parts.append(f"[Source {i} | File: {src} | Page: {page}]\n{doc}")
        sources.append(Source(
            index=i,
            file=src,
            page=page,
            snippet=doc[:200] + "...",
            relevance_score=round(1 - float(dist), 3),
        ))

    context = "\n\n---\n\n".join(context_parts)

    try:
        client = anthropic.Anthropic(api_key=req.api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system="""You are a precise document assistant.
Answer the user's question using ONLY the provided document excerpts.
Always cite which source(s) you used (e.g. "According to Source 2...").
If the answer is not in the excerpts, say so clearly — do not invent information.
Be concise but complete.""",
            messages=[{"role": "user", "content": f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {req.question}"}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(401, "Invalid API key.")
    except Exception as e:
        raise HTTPException(500, str(e))

    return QueryResponse(answer=response.content[0].text, sources=sources)