# DocMind — AI Document Q&A

Upload PDFs, ask questions, get accurate answers with source citations.
Supports 6 AI providers and multiple models per provider.

**Live Demo:** [doc-mind-ai-document-q-a.vercel.app](https://doc-mind-ai-document-q-a.vercel.app)

**Medium Blog:** [Read the full writeup on Medium](https://medium.com/@kethanpabbi/i-built-docmind-anai-document-q-a-system-that-supports-6-llm-providers-heres-how-de0182856340)

---

## Features

- 📄 Upload one or multiple PDFs
- 🤖 Choose from 6 AI providers and multiple models
- 💬 Ask questions in natural language
- 📎 Get answers with page-level source citations
- 🔑 Users bring their own API key — zero cost to you
- ⚡ Fast retrieval via ChromaDB vector search

---

## Supported Providers & Models

| Provider     | Models                                                               |
|--------------|----------------------------------------------------------------------|
| 🤖 Anthropic | Claude Haiku 3.5 (cheapest), Claude Sonnet 4 (best)                  |
| ⚡ OpenAI    | GPT-4o Mini (cheapest), GPT-4o (best), GPT-4 Turbo, o1 Mini          |
| 🔷 Google    | Gemini 1.5 Flash (cheapest), Gemini 1.5 Pro (best), Gemini 2.0 Flash |
| 🌪️ Mistral   | Mistral Small (cheapest), Mistral Large (best), Codestral            |
| 🌊 Cohere    | Command R (cheapest), Command R+ (best)                              |
| 🚀 Groq      | Llama 3.1 8B (cheapest), Llama 3.3 70B (best), Mixtral 8x7B          |

---

## Tech Stack

**Backend:** FastAPI · ChromaDB · LangChain · pypdf · Python 3.11
**Frontend:** Vanilla HTML/CSS/JS (no framework)
**Deployment:** Railway (backend) · Vercel (frontend)

---

## Project Structure

```plaintext
docmind/
├── backend/
│   ├── main.py            # FastAPI app — upload, query, provider routing
│   ├── requirements.txt   # Python dependencies
│   └── Procfile           # Railway start command
├── frontend/
│   ├── index.html         # Full UI — provider/model selector, chat
│   └── vercel.json        # Vercel deployment config
└── README.md
```

---

## How It Works

```plaintext
PDF upload → text extraction → chunking (1000 chars, 150 overlap)
          → ChromaDB embeddings → stored in vector DB

Question → embed query → retrieve top 4 chunks
         → send to chosen AI provider → answer with source citations
```

### Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff'}}}%%
flowchart TD
    %% Define Styles
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px,color:#ff0000;
    classDef frontend fill:#ffffff,stroke:#000000,stroke-width:2px,color:#ff0000;
    classDef backend fill:#ffffff,stroke:#000000,stroke-width:2px,color:#ff0000;
    classDef db fill:#ffffff,stroke:#000000,stroke-width:2px,color:#ff0000;
    classDef llm fill:#ffffff,stroke:#000000,stroke-width:2px,color:#ff0000;
    linkStyle default stroke:#000000,stroke-width:2px,color:#ff0000;

    %% Subgraphs
    subgraph Frontend ["Frontend (Vanilla HTML/JS)"]
        UI[User Interface]
        Config[API Key & Model Selection]
    end
    class Frontend frontend;

    subgraph Backend ["Backend (FastAPI)"]
        API[API Router /main.py]
        
        subgraph Ingestion ["PDF Ingestion"]
            Parse[PyPDFLoader<br/>Extract Text]
            Chunk[LangChain<br/>Text Splitter]
        end
        
        subgraph Query ["Query Processing"]
            Search[ChromaDB<br/>Similarity Search]
            PromptGen[Prompt Construction<br/>Inject Top 4 Contexts]
            Router[Provider Router]
        end
    end
    class Backend backend;

    VectorDB[(ChromaDB<br/>Local Vector DB)]
    class VectorDB db;

    subgraph ExternalProviders ["External LLM APIs"]
        Anthropic[Anthropic API]
        OpenAI[OpenAI API]
        Google[Google Gemini API]
        Mistral[Mistral API]
        Cohere[Cohere API]
        Groq[Groq API]
    end
    class ExternalProviders llm;

    %% Edges
    UI -- "1. Upload PDF (.pdf)" --> API
    API -- "/upload" --> Parse
    Parse --> Chunk
    Chunk -- "Embed & Store" --> VectorDB

    Config -. "User specifies Key/Model" .-> UI
    UI -- "2. Ask Question" --> API
    
    API -- "/query" --> Search
    Search -- "Retrieve relevant chunks" --> VectorDB
    VectorDB -- "Top 4 Contexts" --> Search
    Search --> PromptGen
    PromptGen --> Router
    
    Router -- "Model + API Key" --> Anthropic
    Router -- "Model + API Key" --> OpenAI
    Router -- "Model + API Key" --> Google
    Router -- "Model + API Key" --> Mistral
    Router -- "Model + API Key" --> Cohere
    Router -- "Model + API Key" --> Groq
    
    Anthropic -. "Answer" .-> Router
    OpenAI -. "Answer" .-> Router
    Google -. "Answer" .-> Router
    Mistral -. "Answer" .-> Router
    Cohere -. "Answer" .-> Router
    Groq -. "Answer" .-> Router
    
    Router -- "Return Answer & Sources" --> UI
```

---

## Deployment

### Backend → Railway

1. Push repo to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Set Root Directory to `backend`
4. Railway auto-detects Python and runs the Procfile
5. Go to Settings → Networking → Generate Domain → set port to `8080`
6. Copy your Railway URL

### Frontend → Vercel

1. Open `frontend/index.html`
2. Replace `API_BASE` with your Railway URL:

   ```js
   const API_BASE = "https://your-app.up.railway.app";
   ```

3. Go to vercel.com → New Project → Import from GitHub
4. Set Root Directory to `frontend`
5. Deploy — get your live Vercel URL

---

## Local Development

```bash
cd backend
pip3.11 install -r requirements.txt
uvicorn main:app --reload
```

In `frontend/index.html` set:

```js
const API_BASE = "http://localhost:8000";
```

Then open `frontend/index.html` directly in your browser.

---

## Getting API Keys

| Provider  | Link                                     |
|-----------|------------------------------------------|
| Anthropic | <https://console.anthropic.com>          |
| OpenAI    | <https://platform.openai.com/api-keys>   |
| Google    | <https://aistudio.google.com/app/apikey> |
| Mistral   | <https://console.mistral.ai/api-keys>    |
| Cohere    | <https://dashboard.cohere.com/api-keys>  |
| Groq      | <https://console.groq.com/keys>          |
