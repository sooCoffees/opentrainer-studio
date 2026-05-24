const state = {
  tools: [],
  experiments: [],
  latest: {},
};

const titles = {
  overview: "Overview",
  experiments: "Experiments",
  tools: "Tools",
  data: "Data",
  cpp: "C++ Service",
};

function $(id) {
  return document.getElementById(id);
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function callTool(name, argumentsObject = {}) {
  const response = await fetch("/call", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, arguments: argumentsObject }),
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  state.latest = { tool: name, result: payload };
  $("latestOutput").textContent = pretty(state.latest);
  return payload;
}

function switchView(name) {
  document.querySelectorAll(".view").forEach((node) => node.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((node) => node.classList.remove("active"));
  $(name).classList.add("active");
  document.querySelector(`[data-view="${name}"]`)?.classList.add("active");
  $("viewTitle").textContent = titles[name] || name;
}

function renderTools() {
  $("toolCount").textContent = state.tools.length;
  $("toolSelect").innerHTML = state.tools
    .map((tool) => `<option value="${tool.name}">${tool.name}</option>`)
    .join("");
  $("toolList").innerHTML = state.tools
    .map(
      (tool) => `
        <div class="row">
          <div class="row-title">${tool.name}</div>
          <div class="row-meta">${tool.description}</div>
        </div>
      `,
    )
    .join("");
  updateToolDescription();
}

function renderExperiments() {
  $("experimentCount").textContent = state.experiments.length;
  if (!state.experiments.length) {
    $("experimentList").innerHTML = `<div class="row"><div class="row-meta">No experiments yet.</div></div>`;
    return;
  }
  $("experimentList").innerHTML = state.experiments
    .map(
      (experiment) => `
        <div class="row">
          <div class="row-title">${experiment.name}</div>
          <div class="row-meta">id: ${experiment.id}</div>
          <div class="row-meta">status: ${experiment.status}</div>
          <div class="row-meta">run_dir: ${experiment.run_dir || ""}</div>
        </div>
      `,
    )
    .join("");
}

function renderKeyValues(targetId, payload) {
  const target = $(targetId);
  target.innerHTML = Object.entries(payload)
    .map(
      ([key, value]) => `
        <div class="kv-row">
          <div class="kv-key">${key}</div>
          <div class="kv-value">${typeof value === "string" ? value : pretty(value)}</div>
        </div>
      `,
    )
    .join("");
}

function updateToolDescription() {
  const selected = $("toolSelect").value;
  const tool = state.tools.find((item) => item.name === selected);
  $("toolDescription").textContent = tool?.description || "";
  if (tool) {
    const args = {};
    for (const key of tool.input_schema.required || []) {
      args[key] = "";
    }
    $("toolArgs").value = pretty(args);
  }
}

async function refreshAll() {
  try {
    const health = await getJson("/health");
    $("healthDot").classList.add("ok");
    $("healthText").textContent = health.service;
  } catch (error) {
    $("healthDot").classList.remove("ok");
    $("healthText").textContent = error.message;
  }

  const toolsPayload = await getJson("/tools");
  state.tools = toolsPayload.tools || [];
  renderTools();

  const experimentsPayload = await getJson("/experiments");
  state.experiments = experimentsPayload.experiments || [];
  renderExperiments();
}

async function runSystemReport() {
  const payload = await callTool("system.report", {});
  const report = payload.json || {};
  $("deviceName").textContent = report.device?.device || "Unknown";
  $("cudaState").textContent = report.device?.cuda_available ? "Available" : "Unavailable";
}

function parseJsonField(id) {
  try {
    return JSON.parse($(id).value || "{}");
  } catch (error) {
    throw new Error(`Invalid JSON in ${id}: ${error.message}`);
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelectorAll("[data-view-target]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.viewTarget));
  });
  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.addEventListener("click", async () => {
      await callTool(button.dataset.tool, {});
    });
  });

  $("refreshBtn").addEventListener("click", refreshAll);
  $("systemReportBtn").addEventListener("click", runSystemReport);
  $("toolSelect").addEventListener("change", updateToolDescription);

  $("callToolBtn").addEventListener("click", async () => {
    const result = await callTool($("toolSelect").value, parseJsonField("toolArgs"));
    $("latestOutput").textContent = pretty(result);
  });

  $("createExperimentBtn").addEventListener("click", async () => {
    const metadata = parseJsonField("experimentMetadata");
    await callTool("experiment.create", {
      id: $("experimentId").value.trim(),
      name: $("experimentName").value.trim(),
      metadata,
    });
    await refreshAll();
  });

  $("filterTextBtn").addEventListener("click", async () => {
    const result = await callTool("data.filter_text", { text: $("dataText").value });
    renderKeyValues("dataResult", result);
  });

  $("cppHealthBtn").addEventListener("click", async () => {
    try {
      const result = await callTool("cpp.health", { base_url: $("cppBaseUrl").value.trim() });
      $("cppOutput").textContent = pretty(result);
    } catch (error) {
      $("cppOutput").textContent = pretty({ error: error.message });
    }
  });

  $("cppGenerateBtn").addEventListener("click", async () => {
    try {
      const result = await callTool("cpp.generate", {
        base_url: $("cppBaseUrl").value.trim(),
        prompt: $("cppPrompt").value,
      });
      $("cppOutput").textContent = pretty(result);
    } catch (error) {
      $("cppOutput").textContent = pretty({ error: error.message });
    }
  });
}

bindEvents();
refreshAll().then(runSystemReport).catch((error) => {
  $("latestOutput").textContent = pretty({ error: error.message });
});

