# DocMind — AI Document Q&A

Upload PDFs, ask questions, get answers with source citations.
**Stack:** FastAPI · ChromaDB · sentence-transformers · Claude API · Vanilla JS

## Structure
```
document-qa-v2/
├── backend/       → FastAPI (deploy to Railway)
│   ├── main.py
│   ├── requirements.txt
│   └── Procfile
└── frontend/      → Static HTML (deploy to Vercel)
    ├── index.html
    └── vercel.json
```

## Deploy

### Backend → Railway
1. Go to railway.app → New Project → Deploy from GitHub
2. Point to the `/backend` folder
3. Railway auto-detects the Procfile and runs the server
4. Copy the Railway URL (e.g. `https://docmind.up.railway.app`)

### Frontend → Vercel
1. Open `frontend/index.html`
2. Replace `YOUR_RAILWAY_URL_HERE` with your Railway URL
3. Go to vercel.com → New Project → Deploy from GitHub
4. Point to the `/frontend` folder
5. Done — get your Vercel URL

## Local Dev
```bash
cd backend
pip3.11 install -r requirements.txt
uvicorn main:app --reload
# open frontend/index.html in browser
# set API_BASE = "http://localhost:8000"
```
