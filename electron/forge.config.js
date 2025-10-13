require('dotenv').config();
const fs = require('node:fs');
const path = require('node:path');
const fsp = fs.promises;
const { promisify } = require('node:util');
const { execFile } = require('node:child_process');
const execFileAsync = promisify(execFile);

const projectRoot = __dirname;

function collectPythonRuntimeArchives() {
  const runtimeRoot = path.join(projectRoot, 'python-runtime');
  if (!fs.existsSync(runtimeRoot)) {
    return [];
  }

  const archives = [];
  for (const entry of fs.readdirSync(runtimeRoot, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      // Skip unpacked runtimes; they contain unsigned binaries that break notarization.
      continue;
    }
    if (/\.(tar\.gz|tgz|tar|zip)$/i.test(entry.name)) {
      archives.push({
        from: path.join(runtimeRoot, entry.name),
        to: path.join('python-runtime', entry.name)
      });
    }
  }
  return archives;
}

const pythonRuntimeResources = collectPythonRuntimeArchives();
const pythonRuntimeResourcePaths = pythonRuntimeResources.map((resource) => resource.from);
if (pythonRuntimeResources.length === 0) {
  console.warn('electron/forge.config.js: No Python runtime archives found. Re-run ./electron/setup_python.sh before packaging.');
}

async function resolveMacResourcesRoot(stagingPath) {
  const entries = await fsp.readdir(stagingPath, { withFileTypes: true });
  const appBundle = entries.find((entry) => entry.isDirectory() && entry.name.endsWith('.app'));
  if (!appBundle) {
    throw new Error(`Could not locate a .app bundle within ${stagingPath}`);
  }
  return path.join(stagingPath, appBundle.name, 'Contents', 'Resources');
}

function ensurePythonRuntimeDir(resources) {
  if (resources.length === 0) {
    return [];
  }

  return [
    (stagingPath, electronVersion, platform, arch, done) => {
      resolveMacResourcesRoot(stagingPath)
        .then((resourcesRoot) =>
          fsp.mkdir(path.join(resourcesRoot, 'python-runtime'), { recursive: true }),
        )
        .then(() => done())
        .catch(done);
    },
  ];
}

function relocatePythonRuntimeArchives(resources) {
  if (resources.length === 0) {
    return [];
  }

  return [
    (stagingPath, electronVersion, platform, arch, done) => {
      resolveMacResourcesRoot(stagingPath)
        .then(async (resourcesRoot) => {
          for (const resource of resources) {
            const sourcePath = path.join(resourcesRoot, path.basename(resource.from));
            const destinationPath = path.join(resourcesRoot, resource.to);
            await fsp.mkdir(path.dirname(destinationPath), { recursive: true });
            await fsp.rename(sourcePath, destinationPath);
          }
        })
        .then(() => done())
        .catch(done);
    },
  ];
}

async function stapleNotarizedArtifacts(makeResults) {
  for (const result of makeResults) {
    for (const artifact of result.artifacts) {
      if (artifact.endsWith('.dmg') || artifact.endsWith('.pkg')) {
        await execFileAsync('xcrun', ['stapler', 'staple', artifact]);
      }
    }
  }
}

/** @type {import('@electron-forge/shared-types').ForgeConfig} */
module.exports = {
  packagerConfig: {
    overwrite: true,
    appBundleId: 'ai.reenvision.agentgrid',
    appCategoryType: 'public.app-category.developer-tools',
    asar: true,
    icon: path.join(projectRoot, 'assets', 'icons', 'icon'),
    extraResource: pythonRuntimeResourcePaths,
    beforeCopyExtraResources: ensurePythonRuntimeDir(pythonRuntimeResources),
    afterCopyExtraResources: relocatePythonRuntimeArchives(pythonRuntimeResources),
    extendInfo: {
      LSUIElement: '1',
    },
    osxSign:
      {
        optionsForFile: (filePath) => {
          return {
            entitlements: './build/entitlements.mac.plist'
          };
        }
      },
    osxNotarize:
      {
        tool: 'notarytool',
        appleId: process.env.APPLE_ID,
        appleIdPassword: process.env.APPLE_PASSWORD,
        teamId: process.env.APPLE_TEAM_ID
      }
  },
  hooks: {
    async afterMake(forgeConfig, makeResults) {
      if (!process.env.APPLE_ID || !process.env.APPLE_PASSWORD || !process.env.APPLE_TEAM_ID) {
        return makeResults;
      }

      await stapleNotarizedArtifacts(makeResults);
      return makeResults;
    },
  },
  makers: [
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    {
      name: '@electron-forge/maker-dmg',
      config: {
        icon: './assets/icons/icon.icns',
      }
    },
  ],
};
