from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import tempfile
import anthropic

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="documents",
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
    return {"status": "ok", "message": "Document Q&A API is running"}


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
    for chunk in chunks:
        chunk.metadata["source_file"] = file.filename

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name="documents",
    )
    vectorstore.persist()

    return {"filename": file.filename, "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.api_key.startswith("sk-ant-"):
        raise HTTPException(400, "Invalid Anthropic API key.")

    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(req.question, k=4)
    except Exception as e:
        raise HTTPException(500, f"Vector search failed: {str(e)}")

    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        src  = doc.metadata.get("source_file", "unknown")
        display_page = page + 1 if isinstance(page, int) else page
        context_parts.append(
            f"[Source {i} | File: {src} | Page: {display_page}]\n{doc.page_content}"
        )
        sources.append(Source(
            index=i,
            file=src,
            page=display_page,
            snippet=doc.page_content[:200] + "...",
            relevance_score=round(1 - float(score), 3),
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
        raise HTTPException(401, "Invalid API key — check your Anthropic key.")
    except anthropic.BadRequestError as e:
        raise HTTPException(402, str(e))

    return QueryResponse(answer=response.content[0].text, sources=sources)
