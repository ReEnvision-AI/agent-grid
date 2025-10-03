const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 11;
const METADATA_VERSION = 2;
const BUNDLED_PYTHON_DIR = 'python';

function runCommand(command, args, { cwd, env, log } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      log?.info(`[runtime] ${command} stdout: ${text.trim()}`);
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      log?.warn(`[runtime] ${command} stderr: ${text.trim()}`);
    });

    child.once('error', (error) => {
      reject(error);
    });

    child.once('close', (code) => {
      if (code !== 0) {
        const error = new Error(`Command failed: ${command} ${args.join(' ')} (exit code ${code})`);
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function getPlatformKey() {
  const platform = process.platform;
  const arch = process.arch;
  if (platform === 'darwin') {
    if (arch === 'arm64') {
      return 'darwin-arm64';
    }
    if (arch === 'x64') {
      return 'darwin-x64';
    }
  }
  if (platform === 'linux') {
    if (arch === 'arm64') {
      return 'linux-arm64';
    }
    if (arch === 'x64') {
      return 'linux-x64';
    }
  }
  if (platform === 'win32') {
    if (arch === 'x64') {
      return 'win32-x64';
    }
    if (arch === 'ia32') {
      return 'win32-ia32';
    }
  }
  return `${platform}-${arch}`;
}

function computeFileDigest(filePath) {
  try {
    const hash = crypto.createHash('sha256');
    hash.update(fs.readFileSync(filePath));
    return hash.digest('hex');
  } catch (error) {
    return null;
  }
}

function resolveBundledPython(log) {
  const platformKey = getPlatformKey();
  const candidates = [];

  const resourcesPath = process.resourcesPath;
  if (resourcesPath) {
    candidates.push(path.join(resourcesPath, BUNDLED_PYTHON_DIR, platformKey));
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  candidates.push(path.join(projectRoot, 'electron', 'python-runtime', platformKey));
  candidates.push(path.join(projectRoot, 'python-runtime', platformKey));

  for (const candidate of candidates) {
    const prebuiltVenvPath = path.join(candidate, 'venv');
    if (fs.existsSync(prebuiltVenvPath)) {
      const venvPython = resolveVenvPython(prebuiltVenvPath);
      if (venvPython) {
        const signature = computeFileDigest(path.join(prebuiltVenvPath, 'pyvenv.cfg'));
        log?.info(`[runtime] Using bundled prebuilt Python venv from ${prebuiltVenvPath}`);
        return {
          root: candidate,
          pythonPath: venvPython,
          prebuiltVenv: true,
          prebuiltVenvPath,
          prebuiltSignature: signature
        };
      }
    }

    const pythonPath = resolveVenvPython(candidate);
    if (pythonPath) {
      log?.info(`[runtime] Using bundled Python runtime from ${candidate}`);
      return { root: candidate, pythonPath, prebuiltVenv: false };
    }
  }

  return null;
}

async function findBootstrapPython(log) {
  const bundled = resolveBundledPython(log);
  if (bundled) {
    const { stdout: versionOut } = await runCommand(bundled.pythonPath, ['-c', 'import platform; print(platform.python_version())']);
    return {
      executable: bundled.pythonPath,
      version: versionOut.trim(),
      bundled: true,
      bundledRoot: bundled.root,
      prebuiltVenv: Boolean(bundled.prebuiltVenv),
      prebuiltVenvPath: bundled.prebuiltVenvPath || null,
      prebuiltSignature: bundled.prebuiltSignature || null
    };
  }
  const candidates = [];
  if (process.env.AGENTGRID_BOOTSTRAP_PYTHON) {
    candidates.push(process.env.AGENTGRID_BOOTSTRAP_PYTHON);
  }
  candidates.push('python3', 'python');

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    try {
      const { stdout: executableOut } = await runCommand(candidate, ['-c', 'import sys; print(sys.executable)']);
      const executable = executableOut.trim();
      const { stdout: versionOut } = await runCommand(executable, ['-c', 'import platform; print(platform.python_version())']);
      const version = versionOut.trim();
      if (meetsMinimumVersion(version)) {
        log?.info(`[runtime] Using bootstrap Python at ${executable} (${version})`);
        return { executable, version, bundled: false };
      }
      log?.warn(`[runtime] Ignoring ${executable} due to unsupported version ${version}`);
    } catch (error) {
      log?.warn(`[runtime] Failed probing Python candidate "${candidate}": ${error}`);
    }
  }

  throw new Error('No suitable Python interpreter found. Install Python 3.11+ or set AGENTGRID_BOOTSTRAP_PYTHON.');
}

function meetsMinimumVersion(versionText) {
  const [majorStr, minorStr] = versionText.split('.');
  const major = Number(majorStr);
  const minor = Number(minorStr);
  if (Number.isNaN(major) || Number.isNaN(minor)) {
    return false;
  }
  if (major > MIN_PYTHON_MAJOR) {
    return true;
  }
  if (major === MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR) {
    return true;
  }
  return false;
}

function resolveVenvPython(venvPath) {
  const binDir = process.platform === 'win32' ? path.join(venvPath, 'Scripts') : path.join(venvPath, 'bin');
  const candidates = process.platform === 'win32'
    ? ['python.exe', 'python']
    : ['python3', 'python'];
  for (const name of candidates) {
    const fullPath = path.join(binDir, name);
    if (fs.existsSync(fullPath)) {
      return fullPath;
    }
  }
  return null;
}

function readAgentGridVersion(projectRoot, log) {
  try {
    const versionPath = path.join(projectRoot, 'src', 'agentgrid', 'VERSION');
    return fs.readFileSync(versionPath, 'utf8').trim();
  } catch (error) {
    log?.warn(`[runtime] Unable to read agentgrid version: ${error}`);
    return null;
  }
}

function determineExtras() {
  switch (process.platform) {
    case 'darwin':
      return 'inference';
    case 'linux':
      return 'inference,gpu';
    default:
      return 'inference';
  }
}

function loadMetadata(metadataPath, log) {
  if (!fs.existsSync(metadataPath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  } catch (error) {
    log?.warn(`[runtime] Failed reading metadata, ignoring cache: ${error}`);
    return null;
  }
}

function writeMetadata(metadataPath, metadata) {
  fs.mkdirSync(path.dirname(metadataPath), { recursive: true });
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
}

async function installDependencies(pythonExecutable, projectRoot, extras, log) {
  const pipEnv = {
    PIP_DISABLE_PIP_VERSION_CHECK: '1',
    PYTHONWARNINGS: 'ignore'
  };
  await runCommand(pythonExecutable, ['-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], {
    env: pipEnv,
    log
  });

  const spec = extras ? `.[${extras}]` : '.';
  await runCommand(pythonExecutable, ['-m', 'pip', 'install', spec], {
    cwd: projectRoot,
    env: pipEnv,
    log
  });
}

function runtimeMetadataUpToDate({ metadata, venvPath, extras, agentGridVersion, expectedPrebuiltSignature }, log) {
  if (!metadata) {
    return false;
  }
  if (metadata.version !== METADATA_VERSION) {
    log?.info('[runtime] Metadata version mismatch; reinstall required');
    return false;
  }
  if (metadata.installMethod === 'prebuilt') {
    if (!expectedPrebuiltSignature || metadata.prebuiltVenvSignature !== expectedPrebuiltSignature) {
      log?.info('[runtime] Bundled venv signature changed; reinstall required');
      return false;
    }
  } else {
    if (metadata.extras !== extras) {
      log?.info('[runtime] Extras selection changed; reinstall required');
      return false;
    }
    if (agentGridVersion && metadata.agentGridVersion !== agentGridVersion) {
      log?.info('[runtime] Agent Grid version changed; reinstall required');
      return false;
    }
  }
  if (!fs.existsSync(metadata.pythonPath)) {
    log?.info('[runtime] Cached pythonPath is missing; reinstall required');
    return false;
  }
  const venvPython = resolveVenvPython(venvPath);
  if (!venvPython) {
    log?.info('[runtime] Venv python missing; reinstall required');
    return false;
  }
  return true;
}

function copyDirectory(src, dest) {
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
}

async function ensurePythonRuntime({ app, log, store, projectRoot }) {
  const runtimeRoot = path.join(app.getPath('userData'), 'python-runtime');
  const metadataPath = path.join(runtimeRoot, 'runtime.json');
  const venvPath = path.join(runtimeRoot, 'venv');
  const extras = determineExtras();
  const agentGridVersion = readAgentGridVersion(projectRoot, log);

  const bootstrap = await findBootstrapPython(log);
  const metadata = loadMetadata(metadataPath, log);
  const expectedPrebuiltSignature = bootstrap.prebuiltSignature || null;

  if (runtimeMetadataUpToDate({ metadata, venvPath, extras, agentGridVersion, expectedPrebuiltSignature }, log)) {
    log?.info('[runtime] Existing Python runtime is up to date');
    store.set('pythonPath', metadata.pythonPath);
    return metadata;
  }

  const { executable, version } = bootstrap;
  fs.mkdirSync(runtimeRoot, { recursive: true });

  let runtimeMetadata;

  if (bootstrap.bundled && bootstrap.prebuiltVenv && bootstrap.prebuiltVenvPath) {
    log?.info('[runtime] Installing bundled virtual environment');
    copyDirectory(bootstrap.prebuiltVenvPath, venvPath);
    const venvPython = resolveVenvPython(venvPath);
    if (!venvPython) {
      throw new Error('Bundled virtual environment is missing its Python executable.');
    }
    runtimeMetadata = {
      version: METADATA_VERSION,
      pythonPath: venvPython,
      pythonVersion: version,
      agentGridVersion: agentGridVersion || null,
      extras,
      createdAt: new Date().toISOString(),
      basePython: executable,
      basePythonVersion: version,
      platform: `${process.platform}-${os.arch()}`,
      bundled: true,
      bundledRoot: bootstrap.bundledRoot || null,
      installMethod: 'prebuilt',
      prebuiltVenvSignature: bootstrap.prebuiltSignature || null
    };
  } else {
    log?.info(`[runtime] Creating virtual environment at ${venvPath}`);
    fs.rmSync(venvPath, { recursive: true, force: true });

    await runCommand(executable, ['-m', 'venv', venvPath], { log });

    const venvPython = resolveVenvPython(venvPath);
    if (!venvPython) {
      throw new Error('Failed to locate Python executable inside virtual environment.');
    }

    await installDependencies(venvPython, projectRoot, extras, log);

    const installedVersionOutput = await runCommand(venvPython, ['-c', 'import platform; print(platform.python_version())']);
    const venvVersion = installedVersionOutput.stdout.trim();

    runtimeMetadata = {
      version: METADATA_VERSION,
      pythonPath: venvPython,
      pythonVersion: venvVersion,
      agentGridVersion: agentGridVersion || null,
      extras,
      createdAt: new Date().toISOString(),
      basePython: executable,
      basePythonVersion: version,
      platform: `${process.platform}-${os.arch()}`,
      bundled: Boolean(bootstrap.bundled),
      bundledRoot: bootstrap.bundledRoot || null,
      installMethod: 'pip',
      prebuiltVenvSignature: null
    };
  }

  writeMetadata(metadataPath, runtimeMetadata);
  store.set('pythonPath', runtimeMetadata.pythonPath);
  return runtimeMetadata;
}

module.exports = {
  ensurePythonRuntime
};
