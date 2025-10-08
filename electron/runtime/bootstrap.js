const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 11;
const METADATA_VERSION = 3;
const BUNDLED_PYTHON_DIRS = ['python', 'python-runtime'];

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

function getArchiveFormat(archivePath) {
  if (archivePath.endsWith('.tar.gz') || archivePath.endsWith('.tgz')) {
    return 'tar.gz';
  }
  if (archivePath.endsWith('.zip')) {
    return 'zip';
  }
  if (archivePath.endsWith('.tar')) {
    return 'tar';
  }
  return 'unknown';
}

function findBundledArchive(candidateRoot, platformKey) {
  const names = [
    `${platformKey}.tar.gz`,
    `${platformKey}.tgz`,
    `${platformKey}.tar`,
    `${platformKey}.zip`
  ];
  for (const name of names) {
    const fullPath = path.join(candidateRoot, name);
    if (fs.existsSync(fullPath)) {
      return fullPath;
    }
  }
  // fallback: scan directory for archive starting with platform key
  try {
    const entries = fs.readdirSync(candidateRoot);
    for (const entry of entries) {
      if (!entry.startsWith(platformKey)) {
        continue;
      }
      const ext = getArchiveFormat(entry);
      if (ext === 'unknown') {
        continue;
      }
      const fullPath = path.join(candidateRoot, entry);
      if (fs.existsSync(fullPath)) {
        return fullPath;
      }
    }
  } catch (error) {
    // ignore
  }
  return null;
}

function resolveBundledPython(log) {
  const platformKey = getPlatformKey();
  const baseCandidates = [];

  const resourcesPath = process.resourcesPath;
  if (resourcesPath) {
    for (const dir of BUNDLED_PYTHON_DIRS) {
      baseCandidates.push(path.join(resourcesPath, dir));
    }
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  for (const dir of ['electron/python-runtime', 'python-runtime', 'python']) {
    baseCandidates.push(path.join(projectRoot, dir));
  }

  for (const base of baseCandidates) {
    const platformDir = path.join(base, platformKey);
    log?.info(`[runtime] Probing bundled runtime directory: ${platformDir}`);
    if (fs.existsSync(platformDir) && fs.statSync(platformDir).isDirectory()) {
      const prebuiltVenvPath = path.join(platformDir, 'venv');
      const venvPython = resolveVenvPython(prebuiltVenvPath);
      if (venvPython) {
        const signature = computeFileDigest(path.join(prebuiltVenvPath, 'pyvenv.cfg'));
        log?.info(`[runtime] Using bundled prebuilt Python venv from ${prebuiltVenvPath}`);
        return {
          root: platformDir,
          pythonPath: venvPython,
          prebuiltVenv: true,
          prebuiltVenvPath,
          prebuiltSignature: signature,
          archivePath: null,
          archiveFormat: null
        };
      }

      const pythonPath = resolveVenvPython(platformDir);
      if (pythonPath) {
        log?.info(`[runtime] Using bundled Python runtime from ${platformDir}`);
        return {
          root: platformDir,
          pythonPath,
          prebuiltVenv: false,
          prebuiltVenvPath: null,
          prebuiltSignature: null,
          archivePath: null,
          archiveFormat: null
        };
      }
    }

    const archivePath = findBundledArchive(base, platformKey);
    if (archivePath) {
      const signature = computeFileDigest(archivePath);
      log?.info(`[runtime] Found bundled runtime archive at ${archivePath}`);
      return {
        root: base,
        pythonPath: null,
        prebuiltVenv: false,
        prebuiltVenvPath: null,
        prebuiltSignature: signature,
        archivePath,
        archiveFormat: getArchiveFormat(archivePath)
      };
    }
  }

  if (resourcesPath) {
    try {
      const entries = fs.readdirSync(resourcesPath);
      log?.warn(`[runtime] Available entries under resourcesPath (${resourcesPath}): ${entries.join(', ')}`);
    } catch (error) {
      log?.warn(`[runtime] Failed listing resourcesPath ${resourcesPath}: ${error}`);
    }
  }
  return null;
}

async function findBootstrapPython(log) {
  const bundled = resolveBundledPython(log);
  if (bundled) {
    let pythonVersion = null;
    if (bundled.pythonPath) {
      try {
        const { stdout: versionOut } = await runCommand(bundled.pythonPath, ['-c', 'import platform; print(platform.python_version())']);
        pythonVersion = versionOut.trim();
      } catch (error) {
        log?.warn(`[runtime] Unable to determine version of bundled interpreter: ${error}`);
        pythonVersion = null;
      }
    }
    return {
      executable: bundled.pythonPath || null,
      version: pythonVersion,
      bundled: true,
      bundledRoot: bundled.root,
      prebuiltVenv: Boolean(bundled.prebuiltVenv),
      prebuiltVenvPath: bundled.prebuiltVenvPath || null,
      prebuiltSignature: bundled.prebuiltSignature || null,
      archivePath: bundled.archivePath || null,
      archiveFormat: bundled.archiveFormat || null
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

function runtimeMetadataUpToDate({ metadata, venvPath, extras, agentGridVersion, expectedPrebuiltSignature }, log) {
  if (!metadata) {
    return false;
  }
  if (metadata.version !== METADATA_VERSION) {
    log?.info('[runtime] Metadata version mismatch; reinstall required');
    return false;
  }
  if (metadata.extras !== extras) {
    log?.info('[runtime] Extras selection changed; reinstall required');
    return false;
  }
  if (agentGridVersion && metadata.agentGridVersion !== agentGridVersion) {
    log?.info('[runtime] Agent Grid version changed; reinstall required');
    return false;
  }
  if (metadata.installMethod === 'prebuilt' || metadata.installMethod === 'prebuilt-archive') {
    if (!expectedPrebuiltSignature || metadata.prebuiltSignature !== expectedPrebuiltSignature) {
      log?.info('[runtime] Bundled runtime signature changed; reinstall required');
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

function rewritePyvenvCfg(venvPath, baseDir, basePython, versionedPython, log) {
  const cfgPath = path.join(venvPath, 'pyvenv.cfg');
  if (!fs.existsSync(cfgPath)) {
    log?.warn(`[runtime] pyvenv.cfg missing at ${cfgPath}`);
    return;
  }
  try {
    const replacements = {
      home: baseDir,
      executable: versionedPython || basePython,
      command: `${basePython} -m venv ${venvPath}`
    };
    const lines = fs.readFileSync(cfgPath, 'utf8').split(/\r?\n/);
    const updated = lines.map((line) => {
      const [key, ...rest] = line.split('=');
      if (!key || rest.length === 0) {
        return line;
      }
      const normalizedKey = key.trim();
      if (replacements[normalizedKey]) {
        return `${normalizedKey} = ${replacements[normalizedKey]}`;
      }
      return line;
    });
    fs.writeFileSync(cfgPath, `${updated.join('\n')}\n`);
  } catch (error) {
    log?.warn(`[runtime] Failed to rewrite pyvenv.cfg at ${cfgPath}: ${error}`);
  }
}

function rewriteVenvShebangs(binDir, pythonPath, log) {
  let entries = [];
  try {
    entries = fs.readdirSync(binDir, { withFileTypes: true });
  } catch (error) {
    log?.warn(`[runtime] Failed listing ${binDir} for shebang rewrite: ${error}`);
    return;
  }
  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    const fullPath = path.join(binDir, entry.name);
    try {
      const content = fs.readFileSync(fullPath);
      if (content.length < 2 || content[0] !== 0x23 || content[1] !== 0x21) {
        continue;
      }
      const text = content.toString('utf8');
      const newlineIndex = text.indexOf('\n');
      if (newlineIndex === -1) {
        continue;
      }
      const shebang = text.slice(0, newlineIndex);
      const rest = text.slice(newlineIndex + 1);
      const desiredShebang = `#!${pythonPath}`;
      if (shebang === desiredShebang) {
        continue;
      }
      fs.writeFileSync(fullPath, `${desiredShebang}\n${rest}`);
    } catch (error) {
      log?.warn(`[runtime] Failed rewriting shebang for ${fullPath}: ${error}`);
    }
  }
}

function relocateBundledVenv(runtimeRoot, log) {
  const venvPath = path.join(runtimeRoot, 'venv');
  const binDir = path.join(venvPath, 'bin');
  const baseDir = path.join(runtimeRoot, 'bin');
  if (!fs.existsSync(venvPath)) {
    throw new Error(`Bundled runtime is missing virtual environment at ${venvPath}`);
  }
  if (!fs.existsSync(baseDir)) {
    throw new Error(`Bundled runtime is missing base interpreter directory at ${baseDir}`);
  }
  let basePython = path.join(baseDir, 'python3');
  if (!fs.existsSync(basePython)) {
    basePython = path.join(baseDir, 'python');
  }
  if (!fs.existsSync(basePython)) {
    throw new Error(`Bundled runtime is missing python executable in ${baseDir}`);
  }
  let versionedPython = null;
  try {
    for (const entry of fs.readdirSync(baseDir)) {
      if (/^python3\.\d+$/.test(entry)) {
        versionedPython = path.join(baseDir, entry);
        break;
      }
    }
  } catch (error) {
    log?.warn(`[runtime] Failed scanning ${baseDir} for versioned interpreter: ${error}`);
  }

  const linkSpecs = [
    { name: 'python', target: basePython },
    { name: 'python3', target: basePython }
  ];
  if (versionedPython) {
    linkSpecs.push({ name: path.basename(versionedPython), target: versionedPython });
  }

  for (const { name, target } of linkSpecs) {
    const dest = path.join(binDir, name);
    try {
      fs.rmSync(dest, { force: true });
      if (process.platform === 'win32') {
        fs.copyFileSync(target, dest);
      } else {
        fs.symlinkSync(target, dest);
      }
    } catch (error) {
      log?.warn(`[runtime] Failed updating symlink ${dest} -> ${target}: ${error}`);
    }
  }

  rewritePyvenvCfg(venvPath, baseDir, basePython, versionedPython, log);
  const venvPython = path.join(binDir, 'python3');
  rewriteVenvShebangs(binDir, venvPython, log);
  return venvPython;
}

async function extractBundledArchive(archivePath, destinationDir, log) {
  const format = getArchiveFormat(archivePath);
  if (format === 'unknown') {
    throw new Error(`Unsupported runtime archive format for ${archivePath}`);
  }

  fs.rmSync(destinationDir, { recursive: true, force: true });
  fs.mkdirSync(destinationDir, { recursive: true });

  log?.info(`[runtime] Extracting bundled runtime archive (${format}) to ${destinationDir}`);
  if (format === 'tar.gz' || format === 'tgz') {
    await runCommand('tar', ['-xzf', archivePath, '-C', destinationDir], { log });
  } else if (format === 'tar') {
    await runCommand('tar', ['-xf', archivePath, '-C', destinationDir], { log });
  } else if (format === 'zip') {
    await runCommand('unzip', ['-oq', archivePath, '-d', destinationDir], { log });
  } else {
    throw new Error(`Unsupported runtime archive format "${format}" for ${archivePath}`);
  }
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

  fs.mkdirSync(runtimeRoot, { recursive: true });

  let runtimeMetadata;

  if (bootstrap.bundled && bootstrap.prebuiltVenv && bootstrap.prebuiltVenvPath) {
    log?.info('[runtime] Installing bundled virtual environment');
    if (!bootstrap.bundledRoot) {
      throw new Error('Bundled runtime missing root directory reference.');
    }
    copyDirectory(bootstrap.bundledRoot, runtimeRoot);
    const venvPython = relocateBundledVenv(runtimeRoot, log);
    const installedVersionOutput = await runCommand(venvPython, ['-c', 'import platform; print(platform.python_version())']);
    const venvVersion = installedVersionOutput.stdout.trim();
    runtimeMetadata = {
      version: METADATA_VERSION,
      pythonPath: venvPython,
      pythonVersion: venvVersion,
      agentGridVersion: agentGridVersion || null,
      extras,
      createdAt: new Date().toISOString(),
      basePython: venvPython,
      basePythonVersion: venvVersion,
      platform: `${process.platform}-${os.arch()}`,
      bundled: true,
      bundledRoot: bootstrap.bundledRoot || null,
      installMethod: 'prebuilt',
      prebuiltSignature: bootstrap.prebuiltSignature || null,
      archiveFormat: null
    };
  } else if (bootstrap.bundled && bootstrap.archivePath) {
    log?.info('[runtime] Extracting bundled runtime archive');
    await extractBundledArchive(bootstrap.archivePath, runtimeRoot, log);
    const venvPython = relocateBundledVenv(runtimeRoot, log);
    const installedVersionOutput = await runCommand(venvPython, ['-c', 'import platform; print(platform.python_version())']);
    const venvVersion = installedVersionOutput.stdout.trim();
    runtimeMetadata = {
      version: METADATA_VERSION,
      pythonPath: venvPython,
      pythonVersion: venvVersion,
      agentGridVersion: agentGridVersion || null,
      extras,
      createdAt: new Date().toISOString(),
      basePython: venvPython,
      basePythonVersion: venvVersion,
      platform: `${process.platform}-${os.arch()}`,
      bundled: true,
      bundledRoot: bootstrap.bundledRoot || null,
      installMethod: 'prebuilt-archive',
      prebuiltSignature: bootstrap.prebuiltSignature || null,
      archiveFormat: bootstrap.archiveFormat || null
    };
  } else {
    throw new Error('Bundled Python runtime is missing. Re-run setup_python.sh to generate it before packaging the app.');
  }

  writeMetadata(metadataPath, runtimeMetadata);
  store.set('pythonPath', runtimeMetadata.pythonPath);
  return runtimeMetadata;
}

module.exports = {
  ensurePythonRuntime
};
