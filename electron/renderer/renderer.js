const elements = {
  pythonPath: document.getElementById('pythonPath'),
  envPath: document.getElementById('envPath'),
  modelsPath: document.getElementById('modelsPath'),
  modelSelect: document.getElementById('modelSelect'),
  deviceSelect: document.getElementById('deviceSelect'),
  torchDType: document.getElementById('torchDType'),
  quantType: document.getElementById('quantType'),
  port: document.getElementById('port'),
  identityPath: document.getElementById('identityPath'),
  warmupTokensInterval: document.getElementById('warmupTokensInterval'),
  saveSettings: document.getElementById('saveSettings'),
  startServer: document.getElementById('startServer'),
  stopServer: document.getElementById('stopServer'),
  logViewer: document.getElementById('logViewer')
};

function gatherSettings() {
  return {
    pythonPath: elements.pythonPath.value.trim(),
    envPath: elements.envPath.value.trim(),
    modelsPath: elements.modelsPath.value.trim(),
    selectedModel: elements.modelSelect.value,
    device: elements.deviceSelect.value,
    torchDType: elements.torchDType.value.trim(),
    quantType: elements.quantType.value.trim(),
    port: Number(elements.port.value) || 31331,
    identityPath: elements.identityPath.value.trim(),
    warmupTokensInterval: Number(elements.warmupTokensInterval.value) || 0
  };
}

function populateModels(models) {
  elements.modelSelect.innerHTML = '';
  models.forEach((model) => {
    const option = document.createElement('option');
    option.value = model;
    option.innerText = model;
    elements.modelSelect.appendChild(option);
  });
}

function populateDevices(devices) {
  elements.deviceSelect.innerHTML = '';
  devices.forEach((device) => {
    const option = document.createElement('option');
    option.value = device;
    option.innerText = device.toUpperCase();
    elements.deviceSelect.appendChild(option);
  });
}

let logLines = [];
const MAX_LOG_LINES = 1000;

function appendLog(message) {
  const newLines = message.split('\n');
  logLines.push(...newLines);
  if (logLines.length > MAX_LOG_LINES) {
    logLines = logLines.slice(logLines.length - MAX_LOG_LINES);
  }
  elements.logViewer.textContent = logLines.join('\n');
  elements.logViewer.scrollTop = elements.logViewer.scrollHeight;
}

async function bootstrap() {
  const settings = await window.electronAPI.getSettings();
  window.currentModels = settings.models || [];
  populateModels(window.currentModels);
  populateDevices(['cpu', 'mps', 'cuda']);

  elements.pythonPath.value = settings.pythonPath || '';
  elements.envPath.value = settings.envPath || '';
  elements.modelsPath.value = settings.modelsPath || '';
  elements.modelSelect.value = settings.selectedModel || window.currentModels[0];
  elements.deviceSelect.value = settings.device || 'cpu';
  elements.torchDType.value = settings.torchDType || 'float16';
  elements.quantType.value = settings.quantType || 'none';
  elements.port.value = settings.port ?? 31331;
  elements.identityPath.value = settings.identityPath || '';
  elements.warmupTokensInterval.value = settings.warmupTokensInterval ?? 0;
}

elements.saveSettings.addEventListener('click', async () => {
  await window.electronAPI.saveSettings(gatherSettings());
  appendLog('Settings saved.\n');
});

elements.startServer.addEventListener('click', async () => {
  const state = await window.electronAPI.startServer();
  appendLog(`Start requested -> ${state}\n`);
});

elements.stopServer.addEventListener('click', async () => {
  const state = await window.electronAPI.stopServer();
  appendLog(`Stop requested -> ${state}\n`);
});

window.electronAPI.onLogUpdate(({ stream, message }) => {
  appendLog(`[${stream}] ${message}`);
});

bootstrap();
