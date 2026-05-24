# OpenTrainer Studio Control Server

This is the MCP-like control layer and React frontend for OpenTrainer Studio.
It exposes AI profile, GPU target, training, scaling, data, system, and C++
service bridge actions as JSON tools.

It intentionally uses only the Python standard library so it can run before a
full web stack is chosen.

## Run over stdio

```bash
python -m ml_mcp_server.server --stdio
```

Send JSON lines:

```json
{"id":1,"method":"tools/list","params":{}}
{"id":2,"method":"tools/call","params":{"name":"system.report","arguments":{}}}
```

## Run over HTTP API

```bash
python -m ml_mcp_server.server --http --host 127.0.0.1 --port 8765
```

Endpoints:

- `GET /health`
- `GET /tools`
- `GET /experiments`
- `POST /call`

Example:

```bash
curl -s http://127.0.0.1:8765/tools
curl -s http://127.0.0.1:8765/call \
  -H 'content-type: application/json' \
  -d '{"name":"scaling.recommend","arguments":{"compute_budget":1e18}}'
```

If `frontend/dist` exists, the same server also serves the React production UI
at `http://127.0.0.1:8765`.

## Build and run the React frontend

Build once:

```bash
cd frontend
npm install
npm run build
```

Then run the combined API + UI server:

```bash
cd ..
python3 -m ml_mcp_server.server --http --host 127.0.0.1 --port 8765
```

For frontend development, run Vite separately:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies API calls to the Python server.

## Tool Groups

- `experiment.*`: create/list/status experiment records.
- `ai.*`: create/list/update/delete managed AI profiles and assign GPU targets.
- `gpu.*`: create/list local, rented, or remote GPU targets.
- `gpu.check`: verify a remote control endpoint by calling its `/health`.
- `system.report`: inspect local Torch/GPU capability through A2.
- `scaling.recommend`: ask A3 for model/training budget recommendations.
- `tokenizer.train`: run A1 tokenizer training.
- `training.*`: start/status/stop/metrics for A1 training runs.
- `data.filter_text`: run A4 quality/language/PII filters on a text string.
- `data.ingest_document`: inspect/extract text/PDF documents for training data.
- `cpp.*`: health/generate/register bridge for `cpp-ai-service`.

## Suggested User Flow

1. Open `Train an AI`.
2. Create one AI profile by giving it a name and purpose.
3. Add a data path, or use `Add Data` to inspect text/PDF inputs first.
4. Choose `Local Machine` for a smoke test or connect an external GPU server.
5. Run a tiny recipe first, then scale up after the setup works.
6. Use `Advanced` only for raw tool calls, experiment debugging, and C++ service
   integration.

## Notes

Long-running training is launched as a subprocess and tracked by PID. Metrics are
read from the experiment run directory when `metrics.jsonl` exists.
