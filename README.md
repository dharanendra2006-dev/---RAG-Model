---
title: Voice RAG - HH Goa 2026
emoji: 🎙️
colorFrom: indigo
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# HH Goa 2026 — Task 2: Voice-Enabled RAG (Vaani-style)

## Day 1 status (Aug 14 milestone)
- [x] Repo scaffold
- [x] Config + typed schemas
- [x] Dataset config verification script
- [x] Streaming subsample + defensive flatten
- [x] 5 chunk views (fixed, sentence, semantic, contextual, metadata-aware)
- [x] FAISS index build per strategy
- [x] Retrieval smoke test

## Run order (Day 1)
```bash
cd backend && pip install -r requirements.txt --break-system-packages
cp .env.example .env   # fill in keys later, not needed for Day 1

cd ../ingestion
python 00_check_configs.py          # confirm hf_config in backend/config.py
python 01_subsample.py              # -> data/passages_hi.parquet
python 03_build_chunk_table.py      # -> data/chunks_all_views.parquet
python 04_build_index.py            # -> data/indexes/*.index + *.parquet

cd ../retrieval
python 00_smoke_test.py "your question here"
```

If `00_smoke_test.py` prints ranked passages with scores, Day 1 milestone is done:
**text question -> retrieved passages.**

## Next (Day 2 — Aug 15)
- BM25 index (`retrieval/bm25.py`)
- Score fusion (`retrieval/fusion.py`)
- Retrieval quality test against `is_selected_gt`

## Deployment (live link)

Target: Hugging Face Spaces, Docker SDK. Chosen over Render/Railway
free tiers because this stack (torch + sentence-transformers + faiss)
needs more RAM than their free 512MB-1GB usually allows without
OOM-killing the container - HF's free CPU Basic tier gives 16GB.

The `Dockerfile` at repo root builds one container that serves both
the API (`/api/*`) and the frontend (`/`) - see `backend/api/main.py`.
**This has not been build-tested** (no Docker available in the
environment that wrote it) - the first real build will happen on HF's
servers. If it fails, copy the build log here and it can be fixed
directly rather than guessed at blind.

### One-time setup

```powershell
# from repo root
git lfs install
git lfs track "*.index" "*.pkl" "*.parquet"
git add .gitattributes
git add .
git commit -m "Add Docker deploy, API, STT, frontend"
```

1. Create a Space at huggingface.co/new-space - SDK: **Docker**, visibility: your choice.
2. In the Space's **Settings -> Variables and secrets**, add as *secrets* (not public variables):
   - `ANTHROPIC_API_KEY`
   - `ELEVENLABS_API_KEY`
   - (add `SARVAM_API_KEY` too if you switch `stt_provider`)
3. Push this repo to the Space's git remote (shown on the Space's page after creation, looks like `https://huggingface.co/spaces/<username>/<space-name>`):

```powershell
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

4. Watch the **Build logs** tab on the Space page. First build will take a while (torch + faiss + model weights). Once it says "Running", the Space's URL is your live link.

### If the build fails or the app 500s on the deployed Space
Paste the build log (or, once running, the same JSON error the `/api/query` exception handler now returns - see `backend/api/main.py`) and it'll get fixed directly, the same way the local 500 was diagnosed earlier in this conversation.

