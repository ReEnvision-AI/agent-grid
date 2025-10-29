const { app, Menu, Tray, dialog, BrowserWindow, ipcMain, shell, nativeImage, powerSaveBlocker, powerMonitor } = require('electron');

// Quit if not on Apple Silicon, which is required for the bundled Python runtime.
if (process.platform === 'darwin' && process.arch !== 'arm64') {
  dialog.showErrorBox(
    'Unsupported Architecture',
    'This application is designed exclusively for Apple Silicon (M-series) Macs and cannot run on Intel-based Macs.'
  );
  app.quit();
}
const path = require('path');
const fs = require('fs');
const log = require('electron-log');
const { spawn } = require('child_process');

const { ensurePythonRuntime } = require('./runtime/bootstrap');
const defaults = require('./config/defaults.json');

let tray;
let preferencesWindow;
let serverProcess = null;
let serverState = 'stopped';
let runtimeState = 'pending';
let runtimeErrorMessage = null;
let runtimeMetadata = null;
let powerSaveBlockerId = null;

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  log.warn('[app] Another Agent Grid instance detected. Exiting this instance.');
  app.quit();
} else {
  app.on('second-instance', () => {
    log.info('[app] Second launch detected — existing instance already running.');
  });
}

let store;

async function initStore() {
  const { default: Store } = await import('electron-store');
  store = new Store({ defaults });
}

function getStore() {
  if (!store) {
    throw new Error('Settings store not initialized');
  }
  return store;
}
const MODELS = [
  'Qwen/Qwen2.5-Coder-32B-Instruct',
  'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5',
  'nvidia/NVIDIA-Nemotron-Nano-12B-v2',
  'unsloth/gpt-oss-20b-BF16'
];

function createTray() {
  updateTrayIcon();
  buildTrayMenu();
}

function preventSleep() {
  if (!powerSaveBlockerId) {
    powerSaveBlockerId = powerSaveBlocker.start('prevent-app-suspension');
    log.info('[sleep] 🛡️ Sleep prevention ENABLED - System will stay awake while server is running');
    log.info('[sleep] 💡 Note: Your Mac will not sleep automatically while Agent Grid server is active');
  } else {
    log.info('[sleep] ℹ️ Sleep prevention already active');
  }
}

function allowSleep() {
  if (powerSaveBlockerId) {
    powerSaveBlocker.stop(powerSaveBlockerId);
    powerSaveBlockerId = null;
    log.info('[sleep] Sleep prevention DISABLED - System can now sleep normally');
    log.info('[sleep] Your Mac will resume normal sleep behavior');
  } else {
    log.info('[sleep] Sleep prevention already disabled');
  }
}

// Power state management functions
function saveServerStateBeforeSleep() {
  const settingsStore = getStore();
  const currentState = {
    serverState: serverState,
    powerState: powerMonitor.isOnBatteryPower() ? 'battery' : 'ac',
    timestamp: Date.now(),
    serverProcessId: serverProcess?.pid || null
  };

  settingsStore.set('serverStateBeforeSleep', currentState.serverState);
  settingsStore.set('powerStateBeforeSleep', currentState.powerState);
  settingsStore.set('sleepTimestamp', currentState.timestamp);

  log.info('[power] 💾 Server state saved before sleep:', currentState);
  return currentState;
}

function getSavedServerState() {
  const settingsStore = getStore();
  return {
    serverState: settingsStore.get('serverStateBeforeSleep', 'stopped'),
    powerState: settingsStore.get('powerStateBeforeSleep', 'unknown'),
    timestamp: settingsStore.get('sleepTimestamp', null)
  };
}

function shouldRestoreServerOnWake() {
  const settings = readSettings();
  const savedState = getSavedServerState();
  const currentPowerState = powerMonitor.isOnBatteryPower() ? 'battery' : 'ac';

  // Check user preference
  const restorePreference = settings.restoreServerOnWake || 'always';

  if (restorePreference === 'never') {
    log.info('[power] Auto-restore disabled by user preference');
    return false;
  }

  if (restorePreference === 'ac-only' && currentPowerState === 'battery') {
    log.info('[power] Auto-restore skipped: on battery power and user set "AC only"');
    return false;
  }

  // Only restore if server was actually running
  if (savedState.serverState !== 'running') {
    log.info('[power] ⏸Auto-restore skipped: server was not running before sleep');
    return false;
  }

  // Check if this is a rapid sleep/wake cycle (less than 30 seconds)
  if (savedState.timestamp && (Date.now() - savedState.timestamp) < 30000) {
    log.info('[power] ⚡ Auto-restore skipped: rapid sleep/wake cycle detected');
    return false;
  }

  log.info('[power] Auto-restore conditions met');
  return true;
}

async function restoreServerAfterWake() {
  log.info('[power] Starting server restoration after wake...');

  // Wait a bit for system to fully wake up
  await new Promise(resolve => setTimeout(resolve, 2000));

  if (!shouldRestoreServerOnWake()) {
    return;
  }

  try {
    // Check if runtime is ready
    if (runtimeState !== 'ready') {
      log.warn('[power] Runtime not ready, postponing restoration');
      return;
    }

    // Check if server is already running
    if (serverState === 'running') {
      log.info('[power] Server already running after wake');
      return;
    }

    log.info('[power] Restoring server after wake...');
    startServer();

    // Show notification
    if (process.platform === 'darwin') {
      const { Notification } = require('electron');
      new Notification({
        title: 'Agent Grid',
        body: 'Server automatically restored after wake',
        silent: true
      }).show();
    }

  } catch (error) {
    log.error('[power] Failed to restore server after wake:', error);
  }
}

function handleSystemSuspend() {
  log.info('[power] System entering sleep mode');

  // Save current server state
  const savedState = saveServerStateBeforeSleep();

  // Stop server if it's running (unless user prefers to keep it running on AC)
  const settings = readSettings();
  const keepRunning = settings.keepServerRunningOnLidClose && !powerMonitor.isOnBatteryPower();

  if (serverState === 'running' && !keepRunning) {
    log.info('[power] Stopping server before sleep');
    stopServer();
  } else if (serverState === 'running' && keepRunning) {
    log.info('[power] ⚡ Server continues running (AC power + user preference)');
  }
}

function handleSystemResume() {
  log.info('[power] System resuming from sleep');

  // Start restoration process with a delay
  setTimeout(() => {
    restoreServerAfterWake();
  }, 3000);
}

function handlePowerSourceChange() {
  const currentPowerState = powerMonitor.isOnBatteryPower() ? 'battery' : 'ac';
  log.info(`[power] Power source changed to: ${currentPowerState}`);

  // If server is running and we're now on battery, notify user
  if (serverState === 'running' && currentPowerState === 'battery') {
    log.info('[power] Server now running on battery power');

    // Show notification
    if (process.platform === 'darwin') {
      const { Notification } = require('electron');
      new Notification({
        title: 'Agent Grid',
        body: 'Server running on battery power',
        subtitle: 'Consider connecting to AC power for optimal performance',
        silent: true
      }).show();
    }
  }
}

// Initialize power event listeners
function initializePowerMonitoring() {
  log.info('[power] Initializing power monitoring system');

  // System sleep/wake events
  powerMonitor.on('suspend', handleSystemSuspend);
  powerMonitor.on('resume', handleSystemResume);

  // Power source change events
  powerMonitor.on('on-ac', () => {
    log.info('[power] Connected to AC power');
    handlePowerSourceChange();
  });

  powerMonitor.on('on-battery', () => {
    log.info('[power] Switched to battery power');
    handlePowerSourceChange();
  });

  // Log initial power state
  const initialPowerState = powerMonitor.isOnBatteryPower() ? 'battery' : 'ac';
  log.info(`[power] Initial power state: ${initialPowerState}`);
}

function updateTrayIcon() {
  let icon;

  try {
    // Choose icon based on server state
    let iconName = 'tray.png';
    if (serverState === 'running') {
      iconName = 'tray-active.png';
    } else if (serverState === 'starting') {
      iconName = 'tray-starting.png';
    }

    // Try to load custom tray icon
    icon = nativeImage.createFromPath(__dirname + '/assets/icons/' + iconName);

    // Set template mode for better visibility in light/dark mode
    icon.setTemplateImage(true);

    // Fallback to default icon if state-specific icon not found
    if (!icon || icon.isEmpty()) {
      icon = nativeImage.createFromPath(__dirname + '/assets/icons/tray.png');
      if (icon && !icon.isEmpty()) {
        icon.setTemplateImage(true);
      }
    }

    // Final fallback to system icon
    if (!icon || icon.isEmpty()) {
      icon = nativeImage.createFromNamedImage('NSTouchBarPlayTemplate') || nativeImage.createEmpty();
    }
  } catch (error) {
    console.log('Failed to load custom tray icon, using system icon:', error.message);
    icon = nativeImage.createFromNamedImage('NSTouchBarPlayTemplate') || nativeImage.createEmpty();
  }

  if (tray) {
    tray.setImage(icon);
  } else {
    tray = new Tray(icon);
  }

  // Update tooltip based on server state and sleep prevention
  const tooltip = serverState === 'running' && powerSaveBlockerId
    ? 'Agent Grid - Server Running (Sleep Prevention Active)'
    : 'Agent Grid';
  tray.setToolTip(tooltip);
}

function buildTrayMenu() {
  const menuItems = [
    { label: stateLabel(), enabled: false },
    { type: 'separator' },
    { label: 'Start Server…', click: startServer, enabled: runtimeState === 'ready' && serverState === 'stopped' },
    { label: 'Stop Server', click: stopServer, enabled: serverState === 'running' || serverState === 'starting' },
    { type: 'separator' }
  ];

  // Add sleep prevention indicator when server is running
  if (serverState === 'running' && powerSaveBlockerId) {
    menuItems.push({ label: '☁️ Sleep Prevention Active', enabled: false });
    menuItems.push({ type: 'separator' });
  }

  menuItems.push(
    { label: 'Preferences…', click: showPreferences },
    { label: 'View Logs', click: openLogs },
    { type: 'separator' },
    { label: 'Quit Agent Grid', click: quitApp }
  );

  const menu = Menu.buildFromTemplate(menuItems);
  tray.setContextMenu(menu);
}

function stateLabel() {
  if (runtimeState === 'initializing') {
    return 'Status: Installing runtime…';
  }
  if (runtimeState === 'failed') {
    return 'Status: Runtime failed';
  }
  switch (serverState) {
    case 'running':
      return 'Status: Running';
    case 'starting':
      return 'Status: Starting…';
    case 'failed':
      return 'Status: Failed';
    default:
      return 'Status: Stopped';
  }
}

function showPreferences() {
  if (preferencesWindow) {
    preferencesWindow.focus();
    return;
  }
  preferencesWindow = new BrowserWindow({
    width: 560,
    height: 560,
    resizable: false,
    maximizable: false,
    show: false,
    skipTaskbar: true,
    title: 'Agent Grid Preferences',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  });
  preferencesWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  preferencesWindow.once('ready-to-show', () => {
    preferencesWindow?.show();
  });
  preferencesWindow.on('closed', () => {
    preferencesWindow = null;
  });
}

function openLogs() {
  const logDir = path.join(app.getPath('userData'), 'logs');
  if (!fs.existsSync(logDir)) {
    dialog.showMessageBox({
      type: 'info',
      message: 'No logs yet',
      detail: 'Logs will appear after the server runs.'
    });
    return;
  }
  shell.openPath(logDir);
}

function quitApp() {
  stopServer();
  app.quit();
}

function startServer() {
  if (runtimeState === 'initializing') {
    dialog.showMessageBox({
      type: 'info',
      message: 'Setting up Python runtime',
      detail: 'The Agent Grid runtime is still installing. Try again once installation finishes.'
    });
    return;
  }
  if (runtimeState !== 'ready') {
    dialog.showErrorBox('Runtime unavailable', runtimeErrorMessage || 'The Agent Grid Python runtime failed to install. Repair it from Preferences.');
    return;
  }
  if (serverProcess) {
    dialog.showErrorBox('Server already running', 'Stop the current server before starting a new one.');
    return;
  }
  const settings = readSettings();
  const pythonPath = settings.pythonPath?.trim();
  if (!pythonPath) {
    dialog.showErrorBox('Python runtime unavailable', 'The embedded Python runtime is missing. Repair it from Preferences.');
    return;
  }
  if (pythonPath === '/usr/bin/python3') {
    dialog.showErrorBox('Unsupported Python', 'macOS stub /usr/bin/python3 cannot be used. Select your virtualenv python in Preferences.');
    showPreferences();
    return;
  }
  if (!fs.existsSync(pythonPath)) {
    dialog.showErrorBox('Python not found', `No executable found at ${pythonPath}`);
    showPreferences();
    return;
  }

  try {
    const configPath = writeRuntimeConfig(settings);
    serverState = 'starting';
    updateTrayIcon();
    buildTrayMenu();
    const env = { ...process.env, ...buildEnvironment(settings) };
    const args = ['-m', 'agentgrid.launcher.server', configPath];
    serverProcess = spawn(pythonPath, args, {
      cwd: app.getPath('home'),
      env,
      detached: true
    });
    ensureLogDirectory();
    const stdoutStream = fs.createWriteStream(path.join(app.getPath('userData'), 'logs', 'stdout.log'), { flags: 'a' });
    const stderrStream = fs.createWriteStream(path.join(app.getPath('userData'), 'logs', 'stderr.log'), { flags: 'a' });
    serverProcess.stdout.pipe(stdoutStream);
    serverProcess.stderr.pipe(stderrStream);

    serverProcess.stdout.on('data', handleStdout);
    serverProcess.stderr.on('data', handleStderr);

    serverProcess.on('exit', (code) => {
      serverProcess = null;
      serverState = code === 0 ? 'stopped' : 'failed';
      const stateText = code === 0 ? 'stopped' : 'failed';
      log.info(`[server] 🛑 Server has ${stateText} - disabling sleep prevention`);
      allowSleep();
      updateTrayIcon();
      buildTrayMenu();
    });
  } catch (error) {
    serverProcess = null;
    serverState = 'failed';
    log.error('[server] ❌ Server failed to start - disabling sleep prevention', error);
    dialog.showErrorBox('Failed to start server', String(error));
    allowSleep();
    updateTrayIcon();
    buildTrayMenu();
  }
}

function stopServer() {
  if (!serverProcess || !serverProcess.pid) {
    serverState = 'stopped';
    log.info('[server] ⏹️ Server already stopped - ensuring sleep prevention is disabled');
    allowSleep();
    updateTrayIcon();
    buildTrayMenu();
    return;
  }

  // Kill the entire process group by sending a signal to -PGID.
  // The `detached: true` option in `spawn` ensures this is a new process group.
  log.info(`Stopping server process group ${serverProcess.pid}`);
  try {
    process.kill(-serverProcess.pid, 'SIGTERM');
  } catch (err) {
    log.error(`Error sending SIGTERM to process group ${serverProcess.pid}:`, err);
    // Fallback for safety if the group kill fails
    try {
      serverProcess.kill('SIGTERM');
    } catch (err2) {
      log.error(`Fallback SIGTERM failed for process ${serverProcess.pid}:`, err2);
    }
  }

  // Set a timeout to forcefully kill the process if it doesn't terminate
  setTimeout(() => {
    if (serverProcess && serverProcess.pid) {
      log.warn(`Server process group ${serverProcess.pid} did not terminate gracefully, sending SIGKILL.`);
      try {
        process.kill(-serverProcess.pid, 'SIGKILL');
      } catch (err) {
        // This can happen if the process group is already gone.
        log.warn(`Failed to send SIGKILL to process group ${serverProcess.pid}:`, err.message);
      }
    }
  }, 4000); // 4-second grace period
}

function readSettings() {
  const settingsStore = getStore();
  const identityRaw = settingsStore.get('identityPath');
  const identityPath = normalizeIdentityPath(identityRaw);
  if (identityPath !== (identityRaw || '')) {
    settingsStore.set('identityPath', identityPath);
  }
  return {
    pythonPath: settingsStore.get('pythonPath') || '',
    envPath: settingsStore.get('envPath'),
    modelsPath: settingsStore.get('modelsPath'),
    selectedModel: settingsStore.get('selectedModel'),
    device: settingsStore.get('device'),
    torchDType: settingsStore.get('torchDType'),
    quantType: settingsStore.get('quantType'),
    port: settingsStore.get('port'),
    identityPath,
    warmupTokensInterval: settingsStore.get('warmupTokensInterval'),
    dhtPrefix: settingsStore.get('dhtPrefix'),
    throughput: settingsStore.get('throughput')
  };
}

function writeRuntimeConfig(settings) {
  const identityPath = normalizeIdentityPath(settings.identityPath);
  const config = {
    converted_model_name_or_path: settings.selectedModel || MODELS[0],
    torch_dtype: settings.torchDType || 'float16',
    device: settings.device || 'cpu',
    port: settings.port || 31331,
    token: process.env.HF_TOKEN || null,
    warmup_tokens_interval: settings.warmupTokensInterval || 0,
    quant_type: settings.quantType || 'none',
    attn_cache_tokens: 128000,
    dht_prefix: normalizeDhtPrefix(settings.dhtPrefix),
    throughput: normalizeThroughput(settings.throughput)
  };

  if (identityPath) {
    config.identity_path = identityPath;
  }

  const dir = path.join(app.getPath('userData'), 'runtime');
  fs.mkdirSync(dir, { recursive: true });
  const configPath = path.join(dir, 'config.json');
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  return configPath;
}


function normalizeDhtPrefix(prefix) {
  const value = prefix?.trim();
  return value ? value : null;
}

function normalizeThroughput(input) {
  if (typeof input === 'number') {
    return input;
  }
  if (input === undefined || input === null) {
    return 'auto';
  }
  const value = String(input).trim();
  if (value === '') {
    return 'auto';
  }
  const normalized = value.toLowerCase();
  if (normalized === 'auto' || normalized === 'eval' || normalized === 'dry_run') {
    return normalized;
  }
  const asNumber = Number(value);
  if (!Number.isNaN(asNumber) && Number.isFinite(asNumber)) {
    return asNumber;
  }
  return 'auto';
}

function normalizeIdentityPath(value) {
  if (!value) {
    return '';
  }
  const trimmed = String(value).trim();
  if (!trimmed || trimmed === './dev.id') {
    return '';
  }
  return trimmed;
}

function buildEnvironment(settings) {
  const env = { ...process.env };
  env.AGENTGRID_ENV_PATH = path.resolve(settings.envPath || '.env');
  env.AGENTGRID_MODELS_FILE = path.resolve(settings.modelsPath || 'models');
  env.ATTN_CACHE_TOKENS = '128000';
  return env;
}

function ensureLogDirectory() {
  const dir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(dir, { recursive: true });
}

function handleStdout(data) {
  const text = data.toString();
  log.info(text.trim());
  if (text.includes('Running a server on') || text.includes('Started')) {
    serverState = 'running';
    log.info('[server] 🚀 Server is now running - enabling sleep prevention');
    preventSleep();
    updateTrayIcon();
    buildTrayMenu();
  }
  if (preferencesWindow) {
    preferencesWindow.webContents.send('log-update', { stream: 'stdout', message: text });
  }
}

function handleStderr(data) {
  const text = data.toString();

  // Check if this is an INFO message (not actually an error)
  if (text.includes('[INFO]')) {
    log.info(text.trim()); // Log as info instead of error

    // Check for server start message
    if (text.includes('Started')) {
      serverState = 'running';
      log.info('[server] 🚀 Server is now running (detected from stderr) - enabling sleep prevention');
      preventSleep();
      updateTrayIcon();
      buildTrayMenu();
    }
  } else {
    log.error(text.trim()); // Log actual errors as errors
  }

  if (preferencesWindow) {
    preferencesWindow.webContents.send('log-update', { stream: 'stderr', message: text });
  }
}

app.whenReady().then(async () => {
  log.info('[app] 🚀 Agent Grid is starting up...');
  log.info('[sleep] 💤 Sleep prevention system initialized - will activate when server starts');

  try {
    await initStore();
  } catch (error) {
    dialog.showErrorBox('Failed to initialize settings', String(error));
    app.quit();
    return;
  }

  // Initialize power monitoring
  initializePowerMonitoring();

  createTray();
  if (process.platform === 'darwin') {
    try {
      app.dock?.hide?.();
    } catch (error) {
      log.warn('[ui] Failed to hide dock icon', error);
    }
    if (typeof app.setActivationPolicy === 'function') {
      app.setActivationPolicy('accessory');
    }
  }
  runtimeState = 'initializing';
  buildTrayMenu();
  try {
    const projectRoot = app.isPackaged
      ? path.resolve(process.resourcesPath)
      : path.resolve(__dirname, '..');

    log.info(`[runtime] Starting runtime setup. isPackaged: ${app.isPackaged}, projectRoot: ${projectRoot}`);

    runtimeMetadata = await ensurePythonRuntime({
      app,
      log,
      store: getStore(),
      projectRoot: projectRoot
    });
    runtimeState = 'ready';
    runtimeErrorMessage = null;
    log.info('[runtime] Python runtime ready', runtimeMetadata);
  } catch (error) {
    runtimeState = 'failed';
    runtimeErrorMessage = String(error);
    log.error('[runtime] Failed to provision Python runtime', error);
    dialog.showErrorBox('Failed to install Python runtime', runtimeErrorMessage);
  }
  buildTrayMenu();
  ipcMain.handle('get-settings', () => ({
    ...readSettings(),
    models: MODELS,
    runtimeState,
    runtimeMetadata
  }));
  ipcMain.handle('save-settings', (_event, payload) => {
    const settingsStore = getStore();
    Object.entries(payload).forEach(([key, value]) => {
      settingsStore.set(key, value);
    });
    return true;
  });
  ipcMain.handle('start-server', () => {
    startServer();
    return serverState;
  });
  ipcMain.handle('stop-server', () => {
    stopServer();
    return serverState;
  });
  ipcMain.handle('repair-runtime', async () => {
    runtimeState = 'initializing';
    runtimeErrorMessage = null;
    buildTrayMenu();
    try {
      const runtimeRoot = path.join(app.getPath('userData'), 'python-runtime');
      fs.rmSync(runtimeRoot, { recursive: true, force: true });
    } catch (error) {
      log.warn('[runtime] Failed to remove existing runtime directory during repair', error);
    }
    try {
      runtimeMetadata = await ensurePythonRuntime({
        app,
        log,
        store: getStore(),
        projectRoot: path.resolve(__dirname, '..')
      });
      runtimeState = 'ready';
      buildTrayMenu();
      return { runtimeState, runtimeMetadata };
    } catch (error) {
      runtimeState = 'failed';
      runtimeErrorMessage = String(error);
      buildTrayMenu();
      log.error('[runtime] Repair failed', error);
      throw error;
    }
  });
});

app.on('window-all-closed', (event) => {
  event.preventDefault();
});

app.on('before-quit', (event) => {
  // Always clean up power save blocker on quit
  log.info('[app] 🏠 Application quitting - cleaning up sleep prevention');
  allowSleep();

  if (serverState === 'stopped' || !serverProcess) {
    // Server is not running, quit immediately.
    log.info('[app] 👋 Goodbye! Agent Grid is shutting down.');
    return;
  }

  // Server is running, so we need to stop it gracefully.
  event.preventDefault(); // Prevent the app from quitting.

  log.info('Quit requested, stopping server first...');

  // The 'exit' handler on serverProcess will set it to null.
  // We can add a one-time listener to quit the app after that.
  serverProcess.once('exit', () => {
    log.info('Server stopped, quitting application.');
    app.quit(); // Now quit for real.
  });

  // Initiate the stop sequence.
  stopServer();
});
