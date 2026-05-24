import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const titles = {
  overview: "Home",
  ai: "Train an AI",
  data: "Add Data",
  advanced: "Advanced",
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function parseJson(value) {
  return JSON.parse(value || "{}");
}

function slugify(value) {
  return (value || "my-ai")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "my-ai";
}

function formatNumber(value, digits = 3) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
  const number = Number(value);
  if (Math.abs(number) >= 1000) return Math.round(number).toLocaleString();
  return number.toFixed(digits).replace(/\.?0+$/, "");
}

function metricLoss(metric) {
  if (!metric) return null;
  return metric.valid_loss ?? metric.train_loss ?? metric.loss ?? null;
}

function metricSpeed(metric) {
  return metric?.tokens_per_sec ?? null;
}

function Sidebar({ activeView, health, onSelect }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="mark">ML</div>
        <div>
          <h1>AI Trainer</h1>
          <p>Simple model training</p>
        </div>
      </div>
      <nav className="nav" aria-label="Primary">
        {Object.entries(titles).map(([key, label]) => (
          <button
            className={`nav-item ${activeView === key ? "active" : ""}`}
            key={key}
            onClick={() => onSelect(key)}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="status-block">
        <span className={`status-dot ${health.ok ? "ok" : ""}`} />
        <span>{health.text}</span>
      </div>
    </aside>
  );
}

function Topbar({ title, onRefresh, onSystemReport }) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Local workspace</p>
        <h2>{title}</h2>
      </div>
      <div className="actions">
        <button className="button ghost" onClick={onRefresh}>
          Refresh
        </button>
        <button className="button primary" onClick={onSystemReport}>
          System report
        </button>
      </div>
    </header>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <div className="metric-trace" aria-hidden="true" />
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
      </div>
      {children}
    </section>
  );
}

function Overview({ tools, experiments, systemReport, latest, onView, onTool }) {
  const device = systemReport?.device?.device || "Unknown";
  const cuda = systemReport?.device?.cuda_available ? "Available" : "Unavailable";
  const rails = [52, 78, 44, 64, 86, 58, 72, 48, 82, 61, 69, 90, 55, 76, 43, 67];
  return (
    <section className="view active">
      <div className="metrics-grid">
        <Metric label="Tools" value={tools.length} />
        <Metric label="Experiments" value={experiments.length} />
        <Metric label="Device" value={device} />
        <Metric label="CUDA" value={cuda} />
      </div>

      <section className="signal-panel" aria-label="Training signal preview">
        <div>
          <p className="eyebrow">Simple training workspace</p>
          <h3>Train your own AI without command-line steps</h3>
          <p>
            Create one AI, attach your files, choose where it should train, then run a small test
            before scaling up.
          </p>
        </div>
        <div className="signal-rails" aria-hidden="true">
          {rails.map((height, index) => (
            <span
              className="signal-bar"
              key={`${height}-${index}`}
              style={{ "--bar-height": `${height}%`, "--delay": `${index * 90}ms` }}
            />
          ))}
        </div>
        <div className="signal-status">
          <span className="pulse-ring" />
          <span>API online</span>
        </div>
      </section>

      <div className="split">
        <Panel title="Quick Actions">
          <div className="quick-grid">
            <button className="quick-action" onClick={() => onView("ai")}>
              Start a new AI
            </button>
            <button className="quick-action" onClick={() => onView("ai")}>
              Choose training computer
            </button>
            <button className="quick-action" onClick={() => onView("data")}>
              Add PDF or text data
            </button>
            <button className="quick-action" onClick={() => onTool("system.report", {})}>
              Check this machine
            </button>
            <button className="quick-action" onClick={() => onView("advanced")}>
              Advanced tools
            </button>
          </div>
        </Panel>

        <Panel title="Latest Output">
          <div className="output-frame">
            <div className="terminal-dots">
              <span />
              <span />
              <span />
            </div>
            <pre className="output">{pretty(latest)}</pre>
          </div>
        </Panel>
      </div>
    </section>
  );
}

function AIManager({
  aiProfiles,
  gpuTargets,
  onCreateAI,
  onCreateGpu,
  onAssignGpu,
  onCheckGpu,
  onDeleteAI,
}) {
  const [aiForm, setAiForm] = useState({
    id: "research-assistant",
    name: "Research Assistant",
    purpose: "Answer questions from my own PDFs, notes, and project documents.",
    gpu_target_id: "local-mps",
    base_model: "cs336-transformer",
    dataset_path: "",
    config_path: "",
    notes: "",
  });
  const [gpuForm, setGpuForm] = useState({
    id: "gpu-rental-01",
    name: "Rental GPU 01",
    kind: "remote",
    endpoint: "http://your-gpu-host:8765",
    device: "cuda",
    ssh_host: "ubuntu@your-gpu-host",
    notes: "Run this ML control server on the GPU machine, then paste its endpoint here.",
  });

  function updateAiField(key, value) {
    setAiForm((current) => ({ ...current, [key]: value }));
  }

  function updateGpuField(key, value) {
    setGpuForm((current) => ({ ...current, [key]: value }));
  }

  function targetName(id) {
    return gpuTargets.find((target) => target.id === id)?.name || "Unassigned";
  }

  return (
    <section className="view active">
      <div className="workflow-strip">
        {[
          ["1", "Create AI", "Name the assistant and describe what it should learn."],
          ["2", "Add data", "Point it at text, JSONL, or extracted PDF/OCR data."],
          ["3", "Choose GPU", "Use this computer, a rented box, or another server."],
          ["4", "Pick config", "Use a small debug config first, then scale up."],
          ["5", "Train", "Start training and watch metrics/checkpoints."],
        ].map(([step, title, body]) => (
          <div className="workflow-step" key={step}>
            <span>{step}</span>
            <strong>{title}</strong>
            <small>{body}</small>
          </div>
        ))}
      </div>

      <section className="ai-hero">
        <div>
          <p className="eyebrow">AI instance control</p>
          <h3>One AI, one purpose, one training track</h3>
          <p>
            Start by creating an AI profile. After that, attach data, choose where it trains,
            generate a config, then start a training run.
          </p>
        </div>
        <div className="gpu-strip">
          {gpuTargets.map((target) => (
            <div className="gpu-chip" key={target.id}>
              <span className={target.kind === "local" ? "chip-dot local" : "chip-dot remote"} />
              <strong>{target.name}</strong>
              <small>{target.device || target.kind}</small>
            </div>
          ))}
        </div>
      </section>

      <div className="two-col wide-left">
        <Panel title="Step 1: Create an AI">
          <div className="helper-card">
            <strong>What is this?</strong>
            <span>
              This is not training yet. It creates a workspace for one AI so its data, GPU,
              configs, and experiments do not get mixed with other AIs.
            </span>
          </div>
          <div className="form-grid">
            <label>
              AI ID
              <input value={aiForm.id} onChange={(event) => updateAiField("id", event.target.value)} />
            </label>
            <label>
              Name
              <input value={aiForm.name} onChange={(event) => updateAiField("name", event.target.value)} />
            </label>
          </div>
          <label>
            What should this AI learn?
            <textarea
              rows={4}
              value={aiForm.purpose}
              onChange={(event) => updateAiField("purpose", event.target.value)}
            />
          </label>
          <div className="form-grid">
            <label>
              Where should it train?
              <select
                value={aiForm.gpu_target_id}
                onChange={(event) => updateAiField("gpu_target_id", event.target.value)}
              >
                <option value="">Unassigned</option>
                {gpuTargets.map((target) => (
                  <option value={target.id} key={target.id}>
                    {target.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Starting model
              <input
                value={aiForm.base_model}
                onChange={(event) => updateAiField("base_model", event.target.value)}
              />
            </label>
          </div>
          <div className="form-grid">
            <label>
              Dataset path
              <input
                value={aiForm.dataset_path}
                onChange={(event) => updateAiField("dataset_path", event.target.value)}
                placeholder="/path/to/data.jsonl"
              />
            </label>
            <label>
              Training config path
              <input
                value={aiForm.config_path}
                onChange={(event) => updateAiField("config_path", event.target.value)}
                placeholder="assignment1-basics/configs/scale_1e18.json"
              />
            </label>
          </div>
          <button className="button primary wide" onClick={() => onCreateAI(aiForm)}>
            Create AI Profile
          </button>
        </Panel>

        <Panel title="Your AIs">
          <div className="ai-grid">
            {aiProfiles.length === 0 ? (
              <div className="row">
                <div className="row-meta">No AI profiles yet.</div>
              </div>
            ) : (
              aiProfiles.map((profile) => (
                <div className="ai-card" key={profile.id}>
                  <div className="ai-card-head">
                    <div>
                      <div className="row-title">{profile.name}</div>
                      <div className="row-meta">{profile.id}</div>
                    </div>
                    <span className="status-pill">{profile.status}</span>
                  </div>
                  <p>{profile.purpose || "No purpose set."}</p>
                  <div className="ai-meta-grid">
                    <span>Compute</span>
                    <strong>{targetName(profile.gpu_target_id)}</strong>
                    <span>Experiment</span>
                    <strong>{profile.experiment_id || "none"}</strong>
                    <span>Dataset</span>
                    <strong>{profile.dataset_path || "unset"}</strong>
                    <span>Next</span>
                    <strong>
                      {!profile.dataset_path
                        ? "Add a dataset path"
                        : !profile.config_path
                          ? "Choose or generate a config"
                          : "Ready for training"}
                    </strong>
                  </div>
                  <label>
                    Switch training compute
                    <select
                      value={profile.gpu_target_id || ""}
                      onChange={(event) => onAssignGpu(profile.id, event.target.value)}
                    >
                      <option value="">Unassigned</option>
                      {gpuTargets.map((target) => (
                        <option value={target.id} key={target.id}>
                          {target.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="card-actions">
                    <button className="button danger" onClick={() => onDeleteAI(profile)}>
                      Delete AI profile
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </Panel>
      </div>

      <div className="two-col gpu-section">
        <Panel title="Step 3: Connect training compute">
          <div className="helper-card">
            <strong>External GPU setup</strong>
            <span>
              Rent or open a server, clone this workspace there, run the ML control server on
              that machine, expose port 8765 securely, then paste its URL below.
            </span>
          </div>
          <div className="form-grid">
            <label>
              GPU ID
              <input value={gpuForm.id} onChange={(event) => updateGpuField("id", event.target.value)} />
            </label>
            <label>
              Name
              <input value={gpuForm.name} onChange={(event) => updateGpuField("name", event.target.value)} />
            </label>
          </div>
          <div className="form-grid">
            <label>
              Kind
              <select value={gpuForm.kind} onChange={(event) => updateGpuField("kind", event.target.value)}>
                <option value="local">Local</option>
                <option value="remote">Remote</option>
                <option value="rented">Rented</option>
              </select>
            </label>
            <label>
              Device
              <input value={gpuForm.device} onChange={(event) => updateGpuField("device", event.target.value)} />
            </label>
          </div>
          <label>
            Control endpoint
            <input value={gpuForm.endpoint} onChange={(event) => updateGpuField("endpoint", event.target.value)} />
          </label>
          <label>
            SSH host
            <input value={gpuForm.ssh_host} onChange={(event) => updateGpuField("ssh_host", event.target.value)} />
          </label>
          <button className="button primary wide" onClick={() => onCreateGpu(gpuForm)}>
            Add GPU Target
          </button>
        </Panel>

        <Panel title="Training compute targets">
          <div className="gpu-target-list">
            {gpuTargets.map((target) => (
              <div className="row gpu-row" key={target.id}>
                <div>
                  <div className="row-title">{target.name}</div>
                  <div className="row-meta">{target.id}</div>
                </div>
                <div className="gpu-row-meta">
                  <span>{target.kind}</span>
                  <strong>{target.device || "unknown"}</strong>
                  <small>{target.endpoint || target.ssh_host || target.notes}</small>
                  <button
                    className="mini-button"
                    onClick={() => onCheckGpu(target.id, target.endpoint)}
                  >
                    Check connection
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </section>
  );
}

function SimpleTraining({
  aiProfiles,
  gpuTargets,
  trainingByAi,
  onCreateAI,
  onUpdateAI,
  onCreateGpu,
  onAssignGpu,
  onCheckGpu,
  onStartTinyTest,
  onRefreshTraining,
  onDeleteAI,
  onChatAI,
}) {
  const firstTarget = gpuTargets[0]?.id || "";
  const [form, setForm] = useState({
    name: "My Study Assistant",
    purpose: "Learn from my PDFs, notes, and documents so it can answer my questions.",
    dataset_path: "",
    gpu_target_id: firstTarget,
    base_model: "cs336-transformer",
    config_path: "assignment1-basics/configs/debug_fixture.json",
  });
  const [remote, setRemote] = useState({
    name: "My GPU Server",
    endpoint: "http://your-gpu-server:8765",
    ssh_host: "",
  });
  const [selectedAiId, setSelectedAiId] = useState(aiProfiles[0]?.id || "");
  const [materialPath, setMaterialPath] = useState("");
  const [chatBaseUrl, setChatBaseUrl] = useState("http://127.0.0.1:8080");
  const [chatDraft, setChatDraft] = useState("What did you learn from my training materials?");
  const [chatByAi, setChatByAi] = useState({});
  const [chatBusy, setChatBusy] = useState(false);

  const selectedProfile =
    aiProfiles.find((profile) => profile.id === selectedAiId) || aiProfiles[0] || null;
  const selectedTraining = selectedProfile ? trainingByAi[selectedProfile.id] || {} : {};
  const selectedChat = selectedProfile ? chatByAi[selectedProfile.id] || [] : [];

  useEffect(() => {
    if (!form.gpu_target_id && firstTarget) {
      setForm((current) => ({ ...current, gpu_target_id: firstTarget }));
    }
  }, [firstTarget, form.gpu_target_id]);

  useEffect(() => {
    if (!selectedAiId && aiProfiles.length > 0) {
      setSelectedAiId(aiProfiles[0].id);
    }
  }, [aiProfiles, selectedAiId]);

  useEffect(() => {
    if (selectedProfile) {
      setMaterialPath(selectedProfile.dataset_path || "");
    } else {
      setMaterialPath("");
    }
  }, [selectedProfile?.id, selectedProfile?.dataset_path]);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateRemote(key, value) {
    setRemote((current) => ({ ...current, [key]: value }));
  }

  function targetName(id) {
    return gpuTargets.find((target) => target.id === id)?.name || "Not chosen yet";
  }

  function targetLabel(target) {
    if (target.kind === "local") return `${target.name} - this computer`;
    return `${target.name} - external GPU`;
  }

  async function createProfile() {
    const id = `${slugify(form.name)}-${Date.now().toString().slice(-5)}`;
    const result = await onCreateAI({
      id,
      name: form.name,
      purpose: form.purpose,
      gpu_target_id: form.gpu_target_id,
      base_model: form.base_model,
      dataset_path: form.dataset_path,
      config_path: form.config_path,
      notes: "Created from the simple training wizard.",
    });
    const createdId = result?.ai_profile?.id || result?.json?.ai_profile?.id || id;
    setSelectedAiId(createdId);
    setMaterialPath(form.dataset_path || "");
  }

  function addRemoteTarget() {
    const id = `${slugify(remote.name)}-${Date.now().toString().slice(-5)}`;
    onCreateGpu({
      id,
      name: remote.name,
      kind: "remote",
      endpoint: remote.endpoint,
      device: "cuda",
      ssh_host: remote.ssh_host,
      notes: "External GPU server for training.",
    });
  }

  function trainingState(profile) {
    return trainingByAi[profile.id] || {};
  }

  async function saveMaterials() {
    if (!selectedProfile) return;
    await onUpdateAI(selectedProfile.id, { dataset_path: materialPath });
  }

  function extractAssistantText(result) {
    if (result?.error) return result.error;
    return (
      result?.text ||
      result?.response ||
      result?.completion ||
      result?.output ||
      result?.message ||
      result?.json?.text ||
      result?.json?.response ||
      pretty(result)
    );
  }

  async function sendChat() {
    if (!selectedProfile || !chatDraft.trim() || chatBusy) return;
    const prompt = chatDraft.trim();
    setChatDraft("");
    setChatBusy(true);
    setChatByAi((current) => ({
      ...current,
      [selectedProfile.id]: [...(current[selectedProfile.id] || []), { role: "user", text: prompt }],
    }));
    const result = await onChatAI(selectedProfile, prompt, chatBaseUrl);
    setChatByAi((current) => ({
      ...current,
      [selectedProfile.id]: [
        ...(current[selectedProfile.id] || []),
        {
          role: result?.error ? "system" : "assistant",
          text: extractAssistantText(result),
        },
      ],
    }));
    setChatBusy(false);
  }

  function TrainingChart({ metrics }) {
    const points = (metrics || []).filter((metric) => metricLoss(metric) !== null).slice(-18);
    if (points.length === 0) {
      return <div className="chart-empty">No metrics yet. Start a tiny test to see loss.</div>;
    }
    const losses = points.map((metric) => Number(metricLoss(metric)));
    const maxLoss = Math.max(...losses);
    const minLoss = Math.min(...losses);
    const spread = Math.max(maxLoss - minLoss, 1e-6);
    return (
      <div className="loss-bars" aria-label="Loss trend">
        {points.map((metric, index) => {
          const loss = Number(metricLoss(metric));
          const height = 20 + ((maxLoss - loss) / spread) * 64;
          return (
            <span
              key={`${metric.iter ?? index}-${index}`}
              title={`iter ${metric.iter ?? "?"}: loss ${formatNumber(loss)}`}
              style={{ "--bar-height": `${height}%` }}
            />
          );
        })}
      </div>
    );
  }

  return (
    <section className="view active">
      <section className="simple-hero">
        <div>
          <p className="eyebrow">For non-engineers</p>
          <h3>Create an AI in three steps</h3>
          <p>
            Give it a name, tell it what to learn, point it at your data. Everything technical is
            optional until you are ready to scale.
          </p>
        </div>
        <div className="simple-flow" aria-label="Simple training flow">
          <span>Create AI</span>
          <span>Open AI space</span>
          <span>Add data</span>
          <span>Train</span>
        </div>
      </section>

      <div className="two-col wide-left">
        <Panel title="Create Your AI">
          <div className="helper-card plain">
            <strong>You only need these three things first.</strong>
            <span>
              Creation only gives the AI a name and goal. After that, it opens its own training
              space where you add data and run tests.
            </span>
          </div>
          <label>
            What do you want to call it?
            <input value={form.name} onChange={(event) => updateForm("name", event.target.value)} />
          </label>
          <label>
            What should it learn or help with?
            <textarea
              rows={4}
              value={form.purpose}
              onChange={(event) => updateForm("purpose", event.target.value)}
            />
          </label>
          <label>
            Training files, optional for now.
            <input
              value={form.dataset_path}
              onChange={(event) => updateForm("dataset_path", event.target.value)}
              placeholder="/path/to/your/text-or-jsonl-data"
            />
          </label>
          <label>
            Where should training run?
            <select
              value={form.gpu_target_id}
              onChange={(event) => updateForm("gpu_target_id", event.target.value)}
            >
              <option value="">Choose later</option>
              {gpuTargets.map((target) => (
                <option value={target.id} key={target.id}>
                  {targetLabel(target)}
                </option>
              ))}
            </select>
          </label>

          <details className="advanced-details">
            <summary>Advanced settings</summary>
            <div className="form-grid">
              <label>
                Starting model
                <input
                  value={form.base_model}
                  onChange={(event) => updateForm("base_model", event.target.value)}
                />
              </label>
              <label>
                Training recipe
                <input
                  value={form.config_path}
                  onChange={(event) => updateForm("config_path", event.target.value)}
                />
              </label>
            </div>
          </details>

          <button className="button primary wide" onClick={createProfile}>
            Create AI
          </button>
        </Panel>

        <Panel title="Your AI Spaces">
          <div className="ai-selector-list">
            {aiProfiles.length === 0 ? (
              <div className="empty-state">
                <strong>No AI yet</strong>
                <span>Create one first. A separate training space will appear here.</span>
              </div>
            ) : (
              aiProfiles.map((profile) => {
                const state = trainingState(profile);
                return (
                  <button
                    className={`selector-card ${selectedProfile?.id === profile.id ? "active" : ""}`}
                    key={profile.id}
                    onClick={() => setSelectedAiId(profile.id)}
                    type="button"
                  >
                    <span>
                      <strong>{profile.name}</strong>
                      <small>{profile.dataset_path ? "Data added" : "Needs training data"}</small>
                    </span>
                    <em>{state.running ? "training" : profile.status || "draft"}</em>
                  </button>
                );
              })
            )}
          </div>
        </Panel>
      </div>

      <section className="training-workspace" aria-label="Selected AI training workspace">
        {selectedProfile ? (
          <>
            <div className="workspace-head">
              <div>
                <p className="eyebrow">AI Training Space</p>
                <h3>{selectedProfile.name}</h3>
                <p>{selectedProfile.purpose || "No goal added yet."}</p>
              </div>
              <span className="status-pill">{selectedTraining.running ? "training" : selectedProfile.status}</span>
            </div>

            <div className="workspace-grid">
              <div className="material-box">
                <h4>1. Add Training Materials</h4>
                <p>
                  Paste the folder or file path for PDFs, notes, text, or JSONL. This belongs only to
                  this AI.
                </p>
                <label>
                  Data path
                  <input
                    value={materialPath}
                    onChange={(event) => setMaterialPath(event.target.value)}
                    placeholder="/path/to/my-ai-training-data"
                  />
                </label>
                <button className="button primary wide" onClick={saveMaterials}>
                  Save Materials
                </button>
              </div>

              <div className="material-box">
                <h4>2. Choose Training Computer</h4>
                <p>Start with any available machine. Switch to an external GPU when the test works.</p>
                <label>
                  Training computer
                  <select
                    value={selectedProfile.gpu_target_id || ""}
                    onChange={(event) => onAssignGpu(selectedProfile.id, event.target.value)}
                  >
                    <option value="">Choose later</option>
                    {gpuTargets.map((target) => (
                      <option value={target.id} key={target.id}>
                        {targetLabel(target)}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedProfile.gpu_target_id && (
                  <button className="button ghost wide" onClick={() => onCheckGpu(selectedProfile.gpu_target_id)}>
                    Check Connection
                  </button>
                )}
              </div>

              <div className="material-box train-box">
                <h4>3. Run First Training Test</h4>
                <p>
                  The tiny test is a fast safety check. If loss appears and the run finishes, the AI
                  is ready for bigger training.
                </p>
                <div className="simple-meta compact">
                  <div>
                    <span>Files</span>
                    <strong>{selectedProfile.dataset_path || "Not added yet"}</strong>
                  </div>
                  <div>
                    <span>Training computer</span>
                    <strong>{targetName(selectedProfile.gpu_target_id)}</strong>
                  </div>
                  <div>
                    <span>Next action</span>
                    <strong>
                      {selectedTraining.running
                        ? "Wait for this test to finish"
                        : selectedTraining.metrics?.length
                          ? "Review result or scale up"
                          : "Start tiny training test"}
                    </strong>
                  </div>
                </div>
                <div className="workspace-actions">
                  <button className="button primary" onClick={() => onStartTinyTest(selectedProfile)}>
                    Start Tiny Test
                  </button>
                  <button className="button ghost" onClick={() => onRefreshTraining(selectedProfile)}>
                    Refresh Metrics
                  </button>
                  <button className="button danger" onClick={() => onDeleteAI(selectedProfile)}>
                    Delete AI
                  </button>
                </div>
              </div>
            </div>

            <div className="training-snapshot workspace-snapshot">
              <div className="snapshot-head">
                <div>
                  <span>Training status</span>
                  <strong>
                    {selectedTraining.running
                      ? "Running"
                      : selectedTraining.experiment?.status || selectedProfile.status}
                  </strong>
                </div>
                <div>
                  <span>Latest loss</span>
                  <strong>{formatNumber(metricLoss(selectedTraining.latest_metric))}</strong>
                </div>
                <div>
                  <span>Tokens/sec</span>
                  <strong>{formatNumber(metricSpeed(selectedTraining.latest_metric), 1)}</strong>
                </div>
              </div>
              <TrainingChart metrics={selectedTraining.metrics} />
            </div>

            <div className="chat-workspace">
              <div className="chat-head">
                <div>
                  <h4>4. Chat With This AI</h4>
                  <p>
                    This tests the model through your C++ AI service. Run training first, register
                    the trained model in the service, then ask questions here.
                  </p>
                </div>
                <span className="status-pill">model: {selectedProfile.id}</span>
              </div>
              <label>
                C++ AI service URL
                <input value={chatBaseUrl} onChange={(event) => setChatBaseUrl(event.target.value)} />
              </label>
              <div className="chat-window" aria-label="AI chat messages">
                {selectedChat.length === 0 ? (
                  <div className="chat-empty">
                    No chat yet. Ask a question after the model is registered in your service.
                  </div>
                ) : (
                  selectedChat.map((message, index) => (
                    <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                      <strong>{message.role === "user" ? "You" : message.role === "system" ? "System" : "AI"}</strong>
                      <p>{message.text}</p>
                    </div>
                  ))
                )}
              </div>
              <div className="chat-compose">
                <textarea
                  rows={3}
                  value={chatDraft}
                  onChange={(event) => setChatDraft(event.target.value)}
                  placeholder="Ask this AI something..."
                />
                <button className="button primary" onClick={sendChat} disabled={chatBusy}>
                  {chatBusy ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <strong>Create an AI first</strong>
            <span>After creation, this area becomes that AI's dedicated training workspace.</span>
          </div>
        )}
      </section>

      <div className="two-col gpu-section">
        <Panel title="Optional: Connect External GPU">
          <div className="helper-card plain">
            <strong>When do you need this?</strong>
            <span>
              Use this only when your own computer is too slow. On the rented server, run this
              training server, then paste its web address here.
            </span>
          </div>
          <label>
            Server name
            <input value={remote.name} onChange={(event) => updateRemote("name", event.target.value)} />
          </label>
          <label>
            Server URL
            <input
              value={remote.endpoint}
              onChange={(event) => updateRemote("endpoint", event.target.value)}
              placeholder="http://gpu-server-ip:8765"
            />
          </label>
          <label>
            SSH login, optional
            <input
              value={remote.ssh_host}
              onChange={(event) => updateRemote("ssh_host", event.target.value)}
              placeholder="ubuntu@gpu-server-ip"
            />
          </label>
          <div className="copy-line">
            python3 -m ml_mcp_server.server --http --host 0.0.0.0 --port 8765
          </div>
          <button className="button primary wide" onClick={addRemoteTarget}>
            Save External GPU
          </button>
        </Panel>
      </div>
    </section>
  );
}

function Experiments({ experiments, onCreate }) {
  const [id, setId] = useState("tiny-run-001");
  const [name, setName] = useState("Tiny debug run");
  const [metadata, setMetadata] = useState("{}");

  return (
    <section className="view active">
      <div className="two-col">
        <Panel title="Create Experiment">
          <label>
            Experiment ID
            <input value={id} onChange={(event) => setId(event.target.value)} />
          </label>
          <label>
            Name
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            Metadata JSON
            <textarea rows={5} value={metadata} onChange={(event) => setMetadata(event.target.value)} />
          </label>
          <button className="button primary wide" onClick={() => onCreate(id, name, parseJson(metadata))}>
            Create
          </button>
        </Panel>

        <Panel title="Experiments">
          <div className="list">
            {experiments.length === 0 ? (
              <div className="row">
                <div className="row-meta">No experiments yet.</div>
              </div>
            ) : (
              experiments.map((experiment) => (
                <div className="row" key={experiment.id}>
                  <div className="row-title">{experiment.name}</div>
                  <div className="row-meta">id: {experiment.id}</div>
                  <div className="row-meta">status: {experiment.status}</div>
                  <div className="row-meta">run_dir: {experiment.run_dir || ""}</div>
                </div>
              ))
            )}
          </div>
        </Panel>
      </div>
    </section>
  );
}

function ToolRunner({ tools, onCall }) {
  const [selected, setSelected] = useState("");
  const [args, setArgs] = useState("{}");

  const selectedTool = useMemo(
    () => tools.find((tool) => tool.name === selected) || tools[0],
    [selected, tools],
  );

  useEffect(() => {
    if (!selected && tools.length) {
      setSelected(tools[0].name);
    }
  }, [selected, tools]);

  useEffect(() => {
    if (!selectedTool) return;
    const nextArgs = {};
    for (const key of selectedTool.input_schema?.required || []) {
      nextArgs[key] = "";
    }
    setArgs(pretty(nextArgs));
  }, [selectedTool]);

  return (
    <section className="view active">
      <div className="two-col wide-left">
        <Panel title="Tool Runner">
          <label>
            Tool
            <select value={selectedTool?.name || ""} onChange={(event) => setSelected(event.target.value)}>
              {tools.map((tool) => (
                <option value={tool.name} key={tool.name}>
                  {tool.name}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">{selectedTool?.description || ""}</p>
          <label>
            Arguments JSON
            <textarea rows={12} value={args} onChange={(event) => setArgs(event.target.value)} />
          </label>
          <button className="button primary wide" onClick={() => onCall(selectedTool.name, parseJson(args))}>
            Call Tool
          </button>
        </Panel>

        <Panel title="Available Tools">
          <div className="tool-list">
            {tools.map((tool) => (
              <button className="row tool-row" key={tool.name} onClick={() => setSelected(tool.name)}>
                <div className="row-title">{tool.name}</div>
                <div className="row-meta">{tool.description}</div>
              </button>
            ))}
          </div>
        </Panel>
      </div>
    </section>
  );
}

function KeyValues({ payload }) {
  if (!payload) return <div className="row-meta">No result yet.</div>;
  return (
    <div className="kv">
      {Object.entries(payload).map(([key, value]) => (
        <div className="kv-row" key={key}>
          <div className="kv-key">{key}</div>
          <div className="kv-value">{typeof value === "string" ? value : pretty(value)}</div>
        </div>
      ))}
    </div>
  );
}

function DataView({ onFilter, result, onIngestDocument, documentResult }) {
  const [text, setText] = useState(
    "Hello email me at test@example.com. This is a clean English paragraph with useful words.",
  );
  const [documentPath, setDocumentPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [mode, setMode] = useState("auto");
  return (
    <section className="view active">
      <div className="workflow-strip">
        <div className="workflow-step">
          <span>1</span>
          <strong>Extract</strong>
          <small>Turn PDFs, OCR output, or text files into plain text/JSONL.</small>
        </div>
        <div className="workflow-step">
          <span>2</span>
          <strong>Clean</strong>
          <small>Mask private data and check language/quality/toxicity.</small>
        </div>
        <div className="workflow-step">
          <span>3</span>
          <strong>Attach</strong>
          <small>Use the resulting path as the dataset for an AI profile.</small>
        </div>
      </div>

      <div className="two-col wide-left">
        <Panel title="PDF / OCR Intake">
          <div className="helper-card">
            <strong>How document ingestion works</strong>
            <span>
              Put the file on the same machine as this server, then paste its path. Text files
              extract now. PDFs need a PDF backend; scanned PDFs need OCR on this machine or a
              connected external worker.
            </span>
          </div>
          <label>
            Document path
            <input
              value={documentPath}
              onChange={(event) => setDocumentPath(event.target.value)}
              placeholder="/Users/you/data/manual.pdf"
            />
          </label>
          <div className="form-grid">
            <label>
              Mode
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="auto">Auto</option>
                <option value="pdf-text">PDF text</option>
                <option value="ocr">OCR scanned PDF</option>
              </select>
            </label>
            <label>
              Output path
              <input
                value={outputPath}
                onChange={(event) => setOutputPath(event.target.value)}
                placeholder="/Users/you/data/extracted.txt"
              />
            </label>
          </div>
          <button
            className="button primary wide"
            onClick={() => onIngestDocument({ path: documentPath, mode, output_path: outputPath })}
          >
            Inspect Document
          </button>
        </Panel>

        <Panel title="Document Result">
          <KeyValues payload={documentResult} />
        </Panel>
      </div>

      <div className="two-col wide-left data-section">
        <Panel title="Data Filter">
          <label>
            Text
            <textarea rows={12} value={text} onChange={(event) => setText(event.target.value)} />
          </label>
          <button className="button primary wide" onClick={() => onFilter(text)}>
            Filter Text
          </button>
        </Panel>
        <Panel title="Filter Result">
          <KeyValues payload={result} />
        </Panel>
      </div>
    </section>
  );
}

function CppView({ onCall, output }) {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8080");
  const [prompt, setPrompt] = useState("Hello from CS336 control.");
  return (
    <section className="view active">
      <div className="two-col">
        <Panel title="C++ Service Bridge">
          <label>
            Base URL
            <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>
          <label>
            Prompt
            <textarea rows={6} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          </label>
          <div className="button-row">
            <button className="button ghost" onClick={() => onCall("cpp.health", { base_url: baseUrl })}>
              Health
            </button>
            <button
              className="button primary"
              onClick={() => onCall("cpp.generate", { base_url: baseUrl, prompt })}
            >
              Generate
            </button>
          </div>
        </Panel>
        <Panel title="Bridge Output">
          <div className="output-frame">
            <div className="terminal-dots">
              <span />
              <span />
              <span />
            </div>
            <pre className="output">{pretty(output)}</pre>
          </div>
        </Panel>
      </div>
    </section>
  );
}

function AdvancedView({ experiments, tools, latest, cppOutput, onCreateExperiment, onCall, onCallCpp }) {
  return (
    <section className="view active">
      <div className="helper-card plain advanced-note">
        <strong>Advanced area</strong>
        <span>
          These controls are for debugging, custom experiments, and service integration. Most users
          can stay on Train an AI and Add Data.
        </span>
      </div>
      <div className="advanced-stack">
        <details className="advanced-section">
          <summary>Experiments</summary>
          <Experiments experiments={experiments} onCreate={onCreateExperiment} />
        </details>
        <details className="advanced-section">
          <summary>Raw tool runner</summary>
          <ToolRunner tools={tools} onCall={onCall} />
        </details>
        <details className="advanced-section">
          <summary>C++ AI service bridge</summary>
          <CppView onCall={onCallCpp} output={cppOutput} />
        </details>
        <details className="advanced-section">
          <summary>Latest raw output</summary>
          <div className="output-frame">
            <div className="terminal-dots">
              <span />
              <span />
              <span />
            </div>
            <pre className="output">{pretty(latest)}</pre>
          </div>
        </details>
      </div>
    </section>
  );
}

function App() {
  const [activeView, setActiveView] = useState("overview");
  const [health, setHealth] = useState({ ok: false, text: "Checking server" });
  const [tools, setTools] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [aiProfiles, setAiProfiles] = useState([]);
  const [gpuTargets, setGpuTargets] = useState([]);
  const [latest, setLatest] = useState({});
  const [systemReport, setSystemReport] = useState(null);
  const [dataResult, setDataResult] = useState(null);
  const [documentResult, setDocumentResult] = useState(null);
  const [cppOutput, setCppOutput] = useState({});
  const [trainingByAi, setTrainingByAi] = useState({});

  useEffect(() => {
    function handlePointerMove(event) {
      document.documentElement.style.setProperty("--mouse-x", `${event.clientX}px`);
      document.documentElement.style.setProperty("--mouse-y", `${event.clientY}px`);
    }
    window.addEventListener("pointermove", handlePointerMove);
    return () => window.removeEventListener("pointermove", handlePointerMove);
  }, []);

  async function refreshAll() {
    try {
      const healthPayload = await getJson("/health");
      setHealth({ ok: true, text: healthPayload.service });
    } catch (error) {
      setHealth({ ok: false, text: error.message });
    }

    const toolsPayload = await getJson("/tools");
    setTools(toolsPayload.tools || []);
    const experimentsPayload = await getJson("/experiments");
    setExperiments(experimentsPayload.experiments || []);
    const aiPayload = await getJson("/ai");
    setAiProfiles(aiPayload.ai_profiles || []);
    const gpuPayload = await getJson("/gpu-targets");
    setGpuTargets(gpuPayload.gpu_targets || []);
  }

  async function callTool(name, argumentsObject = {}) {
    try {
      const result = await postJson("/call", { name, arguments: argumentsObject });
      const nextLatest = { tool: name, result };
      setLatest(nextLatest);
      if (name === "system.report") {
        setSystemReport(result.json || null);
      }
      return result;
    } catch (error) {
      const nextLatest = { tool: name, error: error.message };
      setLatest(nextLatest);
      return nextLatest;
    }
  }

  async function createExperiment(id, name, metadata) {
    await callTool("experiment.create", { id, name, metadata });
    await refreshAll();
  }

  async function createAI(form) {
    const result = await callTool("ai.create", form);
    await refreshAll();
    return result;
  }

  async function updateAI(id, patch) {
    const result = await callTool("ai.update", { id, patch });
    await refreshAll();
    return result;
  }

  async function createGpu(form) {
    await callTool("gpu.create", form);
    await refreshAll();
  }

  async function assignGpu(aiId, gpuTargetId) {
    await callTool("ai.assign_gpu", { ai_id: aiId, gpu_target_id: gpuTargetId });
    await refreshAll();
  }

  async function startTinyTest(profile) {
    const result = await callTool("ai.start_tiny_test", { id: profile.id });
    if (!result.error) {
      setTrainingByAi((current) => ({ ...current, [profile.id]: result }));
    }
    await refreshAll();
  }

  async function refreshTraining(profile) {
    const result = await callTool("ai.training_snapshot", { id: profile.id, limit: 60 });
    if (!result.error) {
      setTrainingByAi((current) => ({ ...current, [profile.id]: result }));
    }
    await refreshAll();
  }

  async function deleteAI(profile) {
    const confirmed = window.confirm(
      `Delete AI profile "${profile.name}"?\n\nThis keeps its experiment record and logs.`,
    );
    if (!confirmed) return;
    await callTool("ai.delete", { id: profile.id });
    await refreshAll();
  }

  async function filterText(text) {
    const result = await callTool("data.filter_text", { text });
    setDataResult(result);
  }

  async function ingestDocument(args) {
    const result = await callTool("data.ingest_document", args);
    setDocumentResult(result);
  }

  async function callCpp(name, args) {
    const result = await callTool(name, args);
    setCppOutput(result);
    return result;
  }

  async function chatAI(profile, prompt, baseUrl) {
    return callCpp("cpp.generate", {
      base_url: baseUrl,
      model: profile.id,
      prompt,
    });
  }

  async function checkGpu(gpuTargetId, endpoint) {
    await callTool("gpu.check", { gpu_target_id: gpuTargetId, endpoint });
  }

  useEffect(() => {
    refreshAll().then(() => callTool("system.report", {}));
  }, []);

  useEffect(() => {
    const runningProfiles = aiProfiles.filter((profile) => trainingByAi[profile.id]?.running);
    if (runningProfiles.length === 0) return undefined;
    const timer = window.setInterval(() => {
      for (const profile of runningProfiles) {
        refreshTraining(profile);
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [aiProfiles, trainingByAi]);

  return (
    <div className="shell">
      <Sidebar activeView={activeView} health={health} onSelect={setActiveView} />
      <main className="main">
        <Topbar
          title={titles[activeView]}
          onRefresh={refreshAll}
          onSystemReport={() => callTool("system.report", {})}
        />

        {activeView === "overview" && (
          <Overview
            tools={tools}
            experiments={experiments}
            systemReport={systemReport}
            latest={latest}
            onView={setActiveView}
            onTool={callTool}
          />
        )}
        {activeView === "ai" && (
          <SimpleTraining
            aiProfiles={aiProfiles}
            gpuTargets={gpuTargets}
            trainingByAi={trainingByAi}
            onCreateAI={createAI}
            onUpdateAI={updateAI}
            onCreateGpu={createGpu}
            onAssignGpu={assignGpu}
            onCheckGpu={checkGpu}
            onStartTinyTest={startTinyTest}
            onRefreshTraining={refreshTraining}
            onDeleteAI={deleteAI}
            onChatAI={chatAI}
          />
        )}
        {activeView === "data" && (
          <DataView
            onFilter={filterText}
            result={dataResult}
            onIngestDocument={ingestDocument}
            documentResult={documentResult}
          />
        )}
        {activeView === "advanced" && (
          <AdvancedView
            experiments={experiments}
            tools={tools}
            latest={latest}
            cppOutput={cppOutput}
            onCreateExperiment={createExperiment}
            onCall={callTool}
            onCallCpp={callCpp}
          />
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
