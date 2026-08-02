# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status: legacy

This is the **original self-hosted version** of FSP Finder — a single GPU box running Docker Compose
(FastAPI + Celery + Redis). It has been superseded by an AWS rewrite (App Runner + ECS/GPU tasks +
S3 + DynamoDB), a snapshot of which lives in the sibling `fsp-finder-source` repo. The live site at
fspfinder.com runs the AWS version, not this one.

Treat this as reference/archival code. Before changing anything here, confirm the change is actually
wanted *here* rather than in the cloud version — the two have diverged (this one still has the LLM
detection path, Genius lookups, Celery/Redis, and no seed control; the cloud one dropped the LLM,
added seeding, per-word confidence, and 30-day shareable job links).

## Commands

Requires an NVIDIA GPU (>12 GB VRAM), Docker Desktop (+ WSL2 on Windows), and Git LFS for the LoRA
weights in `lora_config/`. A `.env` file in the repo root is **required** — `docker-compose.yml`
declares it via `env_file`, so compose fails without it. Keys: `SENDGRID_API_KEY`,
`GENIUS_API_TOKEN`, `HF_TOKEN` (the last is also a build arg, used to pull gated Gemma weights).

```bash
docker-compose up -d --build
```

```bash
docker-compose down
```

App is served at `http://localhost:8000` (the nginx service that used to front it on :80/:443 is
commented out in the compose file).

Because compose bind-mounts `.:/app` over the image, **Python edits do not need a rebuild** — just
restart the affected service. Rebuild only for `requirements.txt`, `.Dockerfile`, or model changes
(the rebuild re-runs `download_models.py`, which pulls Whisper+LoRA, Demucs `mdx_extra`, Gemma 2 9B,
and Silero VAD into the `model_cache` volume — slow and multi-GB).

```bash
docker-compose restart worker
```

```bash
docker-compose logs -f worker
```

There is no test suite, no linter, and no CI. `git lfs install && git lfs pull` if `lora_config/`
came down as pointer files.

## Architecture

Four compose services sharing four named volumes; **all inter-process state goes through Redis and
the shared filesystem**, never direct calls.

- **web** — `app.py`, FastAPI/uvicorn on :8000. Accepts uploads, enqueues Celery tasks, serves
  Jinja2 pages. Does no ML, but *does* run the final audio render inline (`apply_censoring`).
- **worker** — `celery_worker.py`, `celery -A celery_worker.celery_app worker -P solo`, with the
  NVIDIA device reserved. `-P solo` is deliberate: one process, so the module-level `WHISPER_MODEL`
  and `LLM_CHAIN` globals stay warm across jobs instead of reloading multi-GB weights per task.
- **beat** — Celery beat, runs `cleanup_stale_files` hourly to delete `/job_artifacts` dirs and
  `uploads/` files older than 2 hours.
- **redis** — two logical DBs used for different things (see below).

Volumes: `model_cache` → `/model-cache` (all model weights), `job_artifacts` → `/job_artifacts`
(per-run temp dirs), `app_data` → `/app/uploads`, `training_data` → `/app/training_data`. Web and
worker **must** share `job_artifacts` and `model_cache` — the analysis result passes absolute
container paths (`vocals_path`, `no_vocals_path`, `temp_dir`) from worker to web.

### Redis usage — two DBs, easy to confuse

- **db 0** — Celery broker *and* result backend. `AsyncResult(job_id)` reads task outcome from here.
- **db 1** — hand-rolled state (`redis_client` in both `app.py` and `celery_worker.py`):
  `status_{job_id}` → `"queued"`/`"processing"`, and `{job_id}` → the full analysis JSON.

### The `/results` side effect

`GET /results` is not a pure read. It calls `task_result.get()` and **caches the result into db 1
under the bare `job_id`** — and `POST /finalize` reads only that key. So the results page must have
been loaded before finalize can work, and once the key expires, finalize 404s even though the Celery
result may still exist in db 0. Any change to the results flow has to preserve that write.

### Job lifecycle

`POST /analyze` writes each upload to `uploads/` and calls `analysis_task.delay(...)`, returning
Celery task ids as `job_ids`. The browser holds a blocking "Processing…" overlay and polls
`/status/{id}` every 5s until all complete, then navigates to `/results?job_ids=a,b,c`. Closing the
tab loses the job — there is no persistence beyond Redis TTLs and no job list.

### ML pipeline (`ml_logic.py`)

Demucs `mdx_extra` two-stem split (`--shifts 2`) → `whisper_timestamped` on the vocals stem using a
LoRA-merged fine-tune of `openai/whisper-medium.en` at `/model-cache/whisper-medium-ft`, optional
silero VAD → `process_untranscribed_gaps` re-transcribes every ≥5s non-silent (>-30 dBFS) gap at
higher beam/best-of → `apply_censoring_logic` flags words → (later, on finalize) `apply_censoring`
attenuates flagged spans by 60 dB in the vocals stem and overlays it back onto the instrumental.

Flagging is lexical: substring hits against `default_curse_words`, exact hits against
`singular_curse_words` plus the user's comma-separated additions, and three two-word phrase rules
("god dam\*", "mother fuck\*", "jerk off") that flag both tokens. The optional LLM pass (Gemma 2 9B
via LangChain) only *adds* phrases on top of that — `backup_censoring` always runs.

### Word IDs

Every word carries `id = (segment_index, word_index)`. Gap-recovered segments deliberately use
**negative** segment indices (`i = -start_ms`, decrementing) so they can't collide with first-pass
ids. These tuples are the entire contract between the transcript HTML, `initial_explicit_ids`, and
`apply_censoring` — reordering or renumbering segments anywhere breaks censoring silently.

### Frontend

No build step. Each template is a standalone document (no base template) pulling remote
`sakura.css` from unpkg plus local `static/style.css`, with a gtag snippet repeated per page.
Server→client data crosses as a `data-results` JSON blob on a hidden div; censor state is just a
`.censored` class on `.word-toggle` spans, serialized into a hidden input on submit.

## Known quirks in this code (present, load-bearing, or just broken)

- `apply_censoring_logic` has `ids_to_mute.extend(new_ids)` indented **inside** the
  `for word_index in explicit_ids` loop, in both the LLM and non-LLM branches, so flagged ids are
  emitted repeatedly (1+2+3+… per line). Harmless downstream — `apply_censoring` dedupes via a set,
  and the frontend uses `.some()` — but it bloats the payload. The cloud rewrite fixed it.
- **The LLM path cannot load as written**: `ml_logic.LLM_MODEL_ID` is `/model-cache/gemma-2-9b-it`
  while `download_models.cache_gemma_model` saves to `/model-cache/google/gemma-2-9b-it`. The
  `llm_detection` checkbox in `index.html` is `disabled` anyway.
- Redis TTLs disagree: `/analyze` and `/results` use `setex(..., 3600, ...)`, but
  `REDIS_EXPIRATION_SECONDS = 900` — so hitting `/keep-alive` *shortens* a fresh job's TTL from an
  hour to 15 minutes. The frontend's `SESSION_DURATION_MIN = 15` matches the 900.
- `/finalize` sets `media_type='audio/mpeg' if format == 'mp3' else 'audio/wav'`, so FLAC downloads
  are labeled WAV.
- `/finalize` never cleans up the rendered file or temp dir (the cleanup code is commented out); the
  hourly beat task is what eventually reclaims it.
- `.gitignore` lists `train_data/`, but the code writes to `training_data/` — that directory is
  untracked-but-not-ignored, and `/finalize` writes a JSON per download into it.
- `nginx.conf` is gitignored and currently empty; the nginx service is commented out in
  `docker-compose.yml`. TLS was handled by a host-level win-acme setup, not by anything in this repo.
- `.Dockerfile` is a non-standard filename, referenced explicitly by every compose service.
