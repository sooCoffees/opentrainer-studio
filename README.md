# OpenTrainer Studio

OpenTrainer Studio is a beginner-friendly local workspace for training small
language models from personal data. It builds on the Stanford CS336 assignment
structure, then adds a React UI and an MCP-like JSON tool layer so training can
be driven from a browser, an agent, or another service.

The core idea is simple: anyone should be able to create an AI profile, attach
data, choose where training runs, and start with a tiny test before scaling to a
rented GPU.

## What This Project Does

- Creates separate AI profiles for different model ideas.
- Ingests text and document paths for future training data pipelines.
- Tracks local or external GPU targets.
- Exposes training, data, scaling, system, and C++ service actions as JSON tools.
- Provides a React frontend at `http://127.0.0.1:8765`.
- Keeps the original CS336-style assignment modules available for deeper work.

## Fixed User Flow

1. Open `Train an AI`.
2. Create one AI by giving it a name and purpose.
3. Add a data path, or go to `Add Data` to inspect PDFs/text first.
4. Choose a training computer:
   - `Local Machine` for smoke tests and UI validation.
   - External GPU server when you rent or own a stronger machine.
5. Run a tiny test recipe first.
6. Scale the recipe only after the tiny run works.
7. Use `Advanced` only for raw tools, experiment debugging, and C++ service
   integration.

## Quick Start

Build the frontend:

```bash
cd ml-mcp-server/frontend
npm install
npm run build
```

Run the combined API and UI server:

```bash
cd ..
python3 -m ml_mcp_server.server --http --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## External GPU Setup

On the GPU server, clone this project and run:

```bash
cd ml-mcp-server
python3 -m ml_mcp_server.server --http --host 0.0.0.0 --port 8765
```

Then in OpenTrainer Studio:

1. Go to `Train an AI`.
2. Use `Optional: Connect External GPU`.
3. Paste the server URL, for example `http://GPU_SERVER_IP:8765`.
4. Save it and use `Check GPU`.

Use SSH tunnels, VPNs, or firewall rules for real deployments. Do not expose the
training server publicly without authentication.

## Project Structure

| Path | Purpose |
| --- | --- |
| `ml-mcp-server` | MCP-like control layer and React frontend |
| `assignment1-basics` | Tokenizer, Transformer LM, optimizer, checkpointing, generation |
| `assignment2-systems` | Attention, profiling, CUDA/Triton-oriented systems work |
| `assignment3-scaling` | Scaling-law and budget planning utilities |
| `assignment4-data` | Extraction, filtering, PII masking, dedup, quality checks |
| `assignment5-alignment` | SFT, DPO, GRPO/RL-style alignment utilities |

## JSON Tool Layer

The local server exposes tools through `POST /call`:

```bash
curl -s http://127.0.0.1:8765/call \
  -H 'content-type: application/json' \
  -d '{"name":"system.report","arguments":{}}'
```

Main tool groups:

- `ai.*`: create, list, update, delete, and assign GPU targets for AI profiles.
- `gpu.*`: create/list/check local or external training machines.
- `data.*`: filter text and inspect/extract document inputs.
- `training.*`: start, stop, inspect status, and read metrics.
- `scaling.recommend`: generate budget recommendations.
- `tokenizer.train`: run tokenizer training.
- `cpp.*`: bridge to a C++ AI service.

## Current Status

- React UI is implemented and served by the Python control server.
- AI profile management, GPU target management, document intake, and delete
  actions are wired through JSON tools.
- Local CPU validation has been done across the CS336 modules during
  development.
- CUDA/Triton and large training runs still need validation on a rented GPU.

## Roadmap

- Add a one-click tiny training run button in the main UI.
- Add auth before remote GPU endpoints are exposed outside a trusted network.
- Add OCR backend installation profiles for scanned PDFs.
- Stream live training metrics into the React UI.
- Add preset recipes for small, medium, and rented-GPU training.
- Add export paths for serving through the C++ AI service.
