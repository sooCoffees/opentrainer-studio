import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const titles = {
  overview: "Studio",
  ai: "My AIs",
  data: "Knowledge Tools",
  advanced: "Developer",
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
  if (!response.ok) {
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
          <h1>OpenTrainer</h1>
          <p>Personal AI builder</p>
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
          <p className="eyebrow">Personal AI workspace</p>
          <h3>Create an AI, give it files, ask it questions, then train when ready</h3>
          <p>
            Files become instant knowledge first. Training is an optional upgrade after the AI can
            already answer from your materials.
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
              Create or open an AI
            </button>
            <button className="quick-action" onClick={() => onView("ai")}>
              Add knowledge files
            </button>
            <button className="quick-action" onClick={() => onView("data")}>
              Inspect PDF or text
            </button>
            <button className="quick-action" onClick={() => onTool("system.report", {})}>
              Check this machine
            </button>
            <button className="quick-action" onClick={() => onView("advanced")}>
              Developer tools
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
  onCheckChatService,
  onRegisterModel,
  onUploadKnowledge,
  onListKnowledge,
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
  const [chatMode, setChatMode] = useState("data");
  const [modelPath, setModelPath] = useState("");
  const [modelBackend, setModelBackend] = useState("python");
  const [knowledgeByAi, setKnowledgeByAi] = useState({});
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  const selectedProfile =
    aiProfiles.find((profile) => profile.id === selectedAiId) || aiProfiles[0] || null;
  const selectedTraining = selectedProfile ? trainingByAi[selectedProfile.id] || {} : {};
  const selectedChat = selectedProfile ? chatByAi[selectedProfile.id] || [] : [];
  const selectedKnowledge = selectedProfile ? knowledgeByAi[selectedProfile.id] || [] : [];

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
      setModelPath(selectedProfile.metadata?.model_path || selectedProfile.metadata?.checkpoint_path || "");
      refreshKnowledge(selectedProfile);
    } else {
      setMaterialPath("");
      setModelPath("");
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
    await refreshKnowledge(selectedProfile);
  }

  async function refreshKnowledge(profile = selectedProfile) {
    if (!profile) return;
    const result = await onListKnowledge(profile.id);
    if (!result.error && result.files) {
      setKnowledgeByAi((current) => ({ ...current, [profile.id]: result.files }));
    }
  }

  async function uploadKnowledgeFiles(event) {
    if (!selectedProfile) return;
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    setUploadBusy(true);
    setUploadMessage("");
    try {
      let uploaded = 0;
      for (const file of files) {
        const content = await file.text();
        const result = await onUploadKnowledge(selectedProfile.id, file.name, content);
        if (result?.ok) {
          uploaded += 1;
          setMaterialPath(result.ai_profile?.dataset_path || materialPath);
        } else {
          setUploadMessage(result?.error || "Some files could not be uploaded.");
        }
      }
      await refreshKnowledge(selectedProfile);
      if (uploaded > 0) {
        setUploadMessage(`${uploaded} file${uploaded === 1 ? "" : "s"} added to this AI.`);
      }
    } finally {
      setUploadBusy(false);
      event.target.value = "";
    }
  }

  function extractAssistantText(result) {
    if (result?.ok === false) {
      return [result.error, result.next_step].filter(Boolean).join("\n\n");
    }
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
    const result = await onChatAI(selectedProfile, prompt, chatMode);
    setChatByAi((current) => ({
      ...current,
      [selectedProfile.id]: [
        ...(current[selectedProfile.id] || []),
        {
          role: result?.error ? "system" : "assistant",
          text: extractAssistantText(result),
          sources: result?.sources || [],
        },
      ],
    }));
    setChatBusy(false);
  }

  async function checkChatService() {
    if (!selectedProfile) return;
    setChatBusy(true);
    const result = await onCheckChatService(chatBaseUrl);
    setChatByAi((current) => ({
      ...current,
      [selectedProfile.id]: [
        ...(current[selectedProfile.id] || []),
        {
          role: result?.ok === false || result?.error ? "system" : "assistant",
          text:
            result?.ok === false || result?.error
              ? extractAssistantText(result)
              : `Service is online.\n\n${pretty(result)}`,
        },
      ],
    }));
    setChatBusy(false);
  }

  async function registerCurrentModel() {
    if (!selectedProfile || chatBusy) return;
    if (!modelPath.trim()) {
      setChatByAi((current) => ({
        ...current,
        [selectedProfile.id]: [
          ...(current[selectedProfile.id] || []),
          {
            role: "system",
            text: "Add the trained model or checkpoint path first, then register it in C++ AI Service.",
          },
        ],
      }));
      return;
    }
    setChatBusy(true);
    const result = await onRegisterModel(selectedProfile, chatBaseUrl, modelPath.trim(), modelBackend);
    setChatByAi((current) => ({
      ...current,
      [selectedProfile.id]: [
        ...(current[selectedProfile.id] || []),
        {
          role: result?.ok === false || result?.error ? "system" : "assistant",
          text:
            result?.ok === false || result?.error
              ? extractAssistantText(result)
              : `Registered this AI in C++ AI Service as model "${selectedProfile.id}".\n\n${pretty(result)}`,
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
          <p className="eyebrow">Main workflow</p>
          <h3>Build a personal AI from your own files</h3>
          <p>
            Create the AI, attach knowledge, ask questions immediately, then train or publish only
            when the answers are useful.
          </p>
        </div>
        <div className="simple-flow" aria-label="Simple training flow">
          <span>Create AI</span>
          <span>Add knowledge</span>
          <span>Ask now</span>
          <span>Train later</span>
        </div>
      </section>

      <div className="two-col wide-left">
        <Panel title="Create A Personal AI">
          <div className="helper-card plain">
            <strong>Start with identity and purpose.</strong>
            <span>
              Creation does not train a model yet. It opens a workspace where this AI can read
              files, answer from them, and later be trained or published.
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
            Knowledge files, optional for now.
            <input
              value={form.dataset_path}
              onChange={(event) => updateForm("dataset_path", event.target.value)}
              placeholder="/path/to/your/text-or-jsonl-data"
            />
          </label>
          <label>
            Training computer, optional for later.
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
                <span>Create one first. Its knowledge, chat, training, and publish controls appear here.</span>
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
                      <small>{profile.dataset_path ? "Knowledge attached" : "Needs knowledge files"}</small>
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
                <p className="eyebrow">Selected AI</p>
                <h3>{selectedProfile.name}</h3>
                <p>{selectedProfile.purpose || "No goal added yet."}</p>
              </div>
              <span className="status-pill">{selectedTraining.running ? "training" : selectedProfile.status}</span>
            </div>

            <div className="builder-flow" aria-label="AI builder stages">
              <div>
                <span>1</span>
                <strong>Knowledge</strong>
                <small>Attach files this AI can read now.</small>
              </div>
              <div>
                <span>2</span>
                <strong>Chat</strong>
                <small>Ask from files before training.</small>
              </div>
              <div>
                <span>3</span>
                <strong>Train</strong>
                <small>Upgrade model weights when needed.</small>
              </div>
              <div>
                <span>4</span>
                <strong>Publish</strong>
                <small>Connect C++ service, RAG, or MCP.</small>
              </div>
            </div>

            <div className="workspace-grid">
              <div className="material-box">
                <h4>1. Add Knowledge</h4>
                <p>
                  Paste a folder or file path for PDFs, notes, text, or JSONL. This becomes usable
                  immediately in chat.
                </p>
                <label className="file-drop">
                  <span>{uploadBusy ? "Adding files..." : "Choose knowledge files"}</span>
                  <small>.txt, .md, .jsonl, .csv</small>
                  <input
                    type="file"
                    multiple
                    accept=".txt,.md,.jsonl,.csv,text/*"
                    onChange={uploadKnowledgeFiles}
                    disabled={uploadBusy}
                  />
                </label>
                {uploadMessage && <div className="inline-note">{uploadMessage}</div>}
                <label>
                  Knowledge path
                  <input
                    value={materialPath}
                    onChange={(event) => setMaterialPath(event.target.value)}
                    placeholder="/path/to/my-ai-training-data"
                  />
                </label>
                <button className="button primary wide" onClick={saveMaterials}>
                  Save Knowledge
                </button>
              </div>

              <div className="material-box">
                <h4>2. Prepare Training Computer</h4>
                <p>Only needed when you want to train model weights. File chat works before this.</p>
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
                <h4>3. Check Training Environment</h4>
                <p>
                  Tiny test proves this machine can run the training stack. It is not real learning
                  from your files yet.
                </p>
                <div className="simple-meta compact">
                  <div>
                    <span>Knowledge</span>
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
                          ? "Environment works"
                          : "Run environment check"}
                    </strong>
                  </div>
                </div>
                <div className="workspace-actions">
                  <button className="button primary" onClick={() => onStartTinyTest(selectedProfile)}>
                    Check Training Environment
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

            <div className="knowledge-library">
              <div className="chat-head">
                <div>
                  <h4>Knowledge Library</h4>
                  <p>Files this AI can use immediately in Instant knowledge mode.</p>
                </div>
                <button className="button ghost" onClick={() => refreshKnowledge(selectedProfile)}>
                  Refresh Files
                </button>
              </div>
              {selectedKnowledge.length === 0 ? (
                <div className="empty-state">
                  <strong>No readable files attached</strong>
                  <span>Choose a text knowledge file above or paste a folder path.</span>
                </div>
              ) : (
                <div className="knowledge-list">
                  {selectedKnowledge.map((file) => (
                    <div className="knowledge-item" key={file.path}>
                      <div>
                        <strong>{file.name}</strong>
                        <span>{file.type} · {file.characters.toLocaleString()} characters</span>
                      </div>
                      <p>{file.preview}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="chat-workspace">
              <div className="chat-head">
                <div>
                  <h4>4. Chat With This AI</h4>
                  <p>
                    Ask from knowledge files first. Switch to raw checkpoint only when you are
                    checking whether training actually changed the model.
                  </p>
                </div>
                <span className="status-pill">{chatMode === "data" ? "data preview" : "raw checkpoint"}</span>
              </div>
              <label>
                Chat mode
                <select value={chatMode} onChange={(event) => setChatMode(event.target.value)}>
                  <option value="data">Instant knowledge from files</option>
                  <option value="checkpoint">Raw checkpoint output</option>
                </select>
              </label>
              <div className="chat-window" aria-label="AI chat messages">
                {selectedChat.length === 0 ? (
                  <div className="chat-empty">
                    No chat yet. Add a knowledge file, keep Instant knowledge selected, then ask
                    something like "你是谁".
                  </div>
                ) : (
                  selectedChat.map((message, index) => (
                    <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                      <strong>{message.role === "user" ? "You" : message.role === "system" ? "System" : "AI"}</strong>
                      <p>{message.text}</p>
                      {message.sources?.length > 0 && (
                        <div className="source-list">
                          {message.sources.map((source, sourceIndex) => (
                            <div className="source-item" key={`${source.path}-${sourceIndex}`}>
                              <span>{source.title || source.path}</span>
                              <small>{source.path}</small>
                            </div>
                          ))}
                        </div>
                      )}
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

            <div className="serve-workspace">
              <div className="chat-head">
                <div>
                  <h4>5. Optional: Add To C++ AI Service</h4>
                  <p>
                    Publish only after the AI answers well enough locally. This is for RAG, MCP, or
                    external apps.
                  </p>
                </div>
                <span className="status-pill">model: {selectedProfile.id}</span>
              </div>
              <label>
                C++ AI service URL
                <input value={chatBaseUrl} onChange={(event) => setChatBaseUrl(event.target.value)} />
              </label>
              <div className="form-grid">
                <label>
                  Trained model or checkpoint path
                  <input
                    value={modelPath}
                    onChange={(event) => setModelPath(event.target.value)}
                    placeholder="/path/to/trained-model-or-checkpoint"
                  />
                </label>
                <label>
                  Serving backend
                  <select value={modelBackend} onChange={(event) => setModelBackend(event.target.value)}>
                    <option value="python">Python checkpoint</option>
                    <option value="gguf">GGUF</option>
                    <option value="torch">Torch</option>
                    <option value="onnx">ONNX</option>
                    <option value="custom">Custom</option>
                  </select>
                </label>
              </div>
              <div className="chat-service-actions">
                <button className="button ghost" onClick={checkChatService} disabled={chatBusy}>
                  Check Service
                </button>
                <button className="button primary" onClick={registerCurrentModel} disabled={chatBusy}>
                  Register This AI
                </button>
                <span>Use this after local answers are worth serving outside this app.</span>
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

  async function chatAI(profile, prompt, mode) {
    if (mode === "data") {
      return callTool("ai.answer_from_data", {
        id: profile.id,
        prompt,
      });
    }
    return callTool("ai.generate_local", {
      id: profile.id,
      prompt,
      max_new_tokens: 80,
    });
  }

  async function checkChatService(baseUrl) {
    return callCpp("cpp.health", { base_url: baseUrl });
  }

  async function registerModel(profile, baseUrl, path, backend) {
    return callCpp("cpp.register_model", {
      base_url: baseUrl,
      model_id: profile.id,
      path,
      backend,
    });
  }

  async function uploadKnowledge(aiId, filename, content) {
    const result = await callTool("ai.add_knowledge_file", { id: aiId, filename, content });
    await refreshAll();
    return result;
  }

  async function listKnowledge(aiId) {
    return callTool("ai.knowledge", { id: aiId });
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
            onCheckChatService={checkChatService}
            onRegisterModel={registerModel}
            onUploadKnowledge={uploadKnowledge}
            onListKnowledge={listKnowledge}
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
