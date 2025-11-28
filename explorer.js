
import * as THREE from 'three';

// Auto-detect API URL (same origin in production)
const API_URL = window.location.origin;

// Idle opacity config (user adjustable)
let idleAlpha = 0.3;   // starting opacity for idle points (0–1)
let trailDecay = 8;    // how many points stay in the bright trail

// State
let latents = [];
let projection = [];
let audioInfo = null;
let hoveredIdx = null;
let selectedIdx = null;
let isPlaying = false;
let audioContext = null;
let currentSource = null;

// Three.js objects
let scene, camera, renderer, points, material;
let spherical = { theta: 0, phi: Math.PI / 2, radius: 4 };
let isDragging = false;
let previousMouse = { x: 0, y: 0 };
let autoRotate = true;

// Playback state
let playbackAudio = null;
let playbackStartTime = 0;
let isFullPlaying = false;
let playheadLine = null;
let glowPoints = null;
let baseColors = null;
let trailLine = null;
let trailLineColors = null;

// Line mode state
let lineMode = false;
let rainbowLine = null;

// Theme / color palette
let theme = {
  background: '#08080c',
  // default gradient from blue → pink → orange
  pointPalette: ['#6366f1', '#ec4899', '#f97316']
};

function getPointColorFromPalette(t) {
  const palette = theme.pointPalette;
  if (!palette || palette.length === 0) {
    // fallback to old HSL behavior
    return new THREE.Color().setHSL(t * 0.7 + 0.55, 0.75, 0.55);
  }

  if (palette.length === 1) {
    return new THREE.Color(palette[0]);
  }

  // interpolate between palette colors
  const scaled = t * (palette.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;

  const c0 = new THREE.Color(palette[idx]);
  const c1 = new THREE.Color(palette[Math.min(idx + 1, palette.length - 1)]);

  return c0.lerp(c1, frac);
}

function applyThemeToScene() {
  // background
  if (scene) {
    scene.background = new THREE.Color(theme.background);
  }

  // points
  if (points && points.geometry && points.geometry.attributes.color && projection.length) {
    const colorsAttr = points.geometry.attributes.color;
    const colors = colorsAttr.array;
    const n = projection.length;

    for (let i = 0; i < n; i++) {
      const t = n <= 1 ? 0 : i / (n - 1);
      const color = getPointColorFromPalette(t);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    colorsAttr.needsUpdate = true;
    baseColors = new Float32Array(colors);
  }
}

// Public API: user can call this from outside
function setUserTheme(options) {
  if (options.background) {
    theme.background = options.background;
  }
  if (Array.isArray(options.pointPalette) && options.pointPalette.length > 0) {
    theme.pointPalette = options.pointPalette;
  }
  applyThemeToScene();
}

// expose on window so you can do window.setLatentTheme(...) in your own UI
window.setLatentTheme = setUserTheme;


// Cluster visualization state
let clusterK = 8;
let clusterCentroids = [];
let clusterMeshes = [];
let clusterTriggerTimes = [];
let clusterAssignments = [];
let rainbowLineColors = null;
let connectionLines = null;

// DOM elements
const container = document.getElementById('canvas-container');
const statusEl = document.getElementById('status');
const serverStatusEl = document.getElementById('server-status');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const infoPanel = document.getElementById('info-panel');
const infoIdx = document.getElementById('info-idx');
const infoTime = document.getElementById('info-time');
const infoPlaying = document.getElementById('info-playing');
const footerText = document.getElementById('footer-text');
const playBtn = document.getElementById('play-btn');
const lineToggle = document.getElementById('line-toggle');

const bgColorInput = document.getElementById('bg-color-input');
const ptColorInput1 = document.getElementById('pt-color-1');
const ptColorInput2 = document.getElementById('pt-color-2');
const ptColorInput3 = document.getElementById('pt-color-3');
const paletteApplyBtn = document.getElementById('palette-apply-btn');
const stylingBtn = document.getElementById('styling-btn');
const paletteEditor = document.getElementById('palette-editor');
const idleAlphaSlider = document.getElementById('idle-alpha-slider');
const idleAlphaValue  = document.getElementById('idle-alpha-value');
const decaySlider     = document.getElementById('decay-slider');
const decayValue      = document.getElementById('decay-value');


if (stylingBtn && paletteEditor) {
  stylingBtn.addEventListener('click', () => {
    const isVisible = paletteEditor.style.display === 'flex';
    paletteEditor.style.display = isVisible ? 'none' : 'flex';
  });
}
if (paletteApplyBtn) {
  paletteApplyBtn.addEventListener('click', () => {
    const bg = bgColorInput?.value || theme.background;
    const palette = [
      ptColorInput1?.value,
      ptColorInput2?.value,
      ptColorInput3?.value
    ].filter(Boolean);

    window.setLatentTheme({
      background: bg,
      pointPalette: palette
    });
  });
}
if (idleAlphaSlider && idleAlphaValue) {
  idleAlphaSlider.addEventListener('input', () => {
    idleAlpha = parseInt(idleAlphaSlider.value, 10) / 100;
    idleAlphaValue.textContent = idleAlpha.toFixed(2);

    // If nothing is playing, immediately apply to all points
    if (points && !isFullPlaying) {
      const alphaAttr = points.geometry.getAttribute('alpha');
      if (alphaAttr) {
        const alphas = alphaAttr.array;
        for (let i = 0; i < alphas.length; i++) {
          alphas[i] = idleAlpha;
        }
        alphaAttr.needsUpdate = true;
      }
    }
  });
}

if (decaySlider && decayValue) {
  decaySlider.addEventListener('input', () => {
    trailDecay = parseInt(decaySlider.value, 10);
    decayValue.textContent = trailDecay.toString();
    // No immediate geometry update needed; it will affect next playback frames
  });
}


async function handleEncodeStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = JSON.parse(line.slice(6));

      if (data.stage === 'cached') {
        statusEl.textContent = 'Loading from cache...';
      } else if (data.stage === 'loading_vae') {
        statusEl.textContent = 'Loading VAE...';
      } else if (data.stage === 'encoding') {
        statusEl.textContent = 'Encoding audio...';
      } else if (data.stage === 'unloading_vae') {
        statusEl.textContent = 'Freeing VRAM...';
      } else if (data.stage === 'umap') {
        statusEl.textContent = `Running UMAP on ${data.num_latents} latents...`;
        audioInfo = { numLatents: data.num_latents, samplesPerLatent: 2048 };
        projection = Array.from({ length: data.num_latents }, () => [
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2
        ]);
        initThreeJS();
      } else if (data.stage === 'done') {
        latents = data.latents;
        audioInfo = {
          duration: data.duration_seconds,
          numLatents: data.num_latents,
          samplesPerLatent: data.samples_per_latent
        };

        if (!points) {
          projection = data.projection;
          initThreeJS();
          buildClusters();
        } else {
          animateToProjection(data.projection);
        }

        statusEl.textContent = `${data.num_latents} latents • ${data.duration_seconds.toFixed(1)}s • Click to play`;
        footerText.textContent = `${data.num_latents} points`;
        playBtn.style.display = 'inline-block';
        lineToggle.style.display = 'inline-block';
        document.getElementById('cluster-control').style.display = 'flex';
        document.getElementById('resynth-btn').style.display = 'inline-block';
      } else if (data.stage === 'error') {
        throw new Error(data.error);
      }
    }
  }
}

// Check server
async function checkServer() {
  try {
    const r = await fetch(`${API_URL}/health`);
    const data = await r.json();
    serverStatusEl.textContent = data.vae_loaded ? 'VAE loaded' : 'Ready';
    serverStatusEl.classList.remove('offline');
  } catch {
    serverStatusEl.textContent = 'Server offline';
    serverStatusEl.classList.add('offline');
  }
}

const uploadLabelText = document.getElementById('upload-label-text');
const youtubeToggleBtn = document.getElementById('youtube-toggle-btn');
const youtubeForm = document.getElementById('youtube-form');
const youtubeUrlInput = document.getElementById('youtube-url-input');
const youtubeSubmitBtn = document.getElementById('youtube-submit-btn');
async function processYouTubeUrl(url) {
  uploadBtn.style.pointerEvents = 'none';
  youtubeToggleBtn.style.pointerEvents = 'none';
  youtubeSubmitBtn.disabled = true;
  youtubeSubmitBtn.textContent = 'Fetching...';
  statusEl.textContent = 'Downloading from YouTube...';

  if (isFullPlaying) stopFullPlayback();
  playbackAudio = null;

  try {
    const response = await fetch(`${API_URL}/encode_youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'YouTube fetch failed');
    }

    await handleEncodeStream(response);

  } catch (err) {
    console.error(err);
    statusEl.textContent = `YouTube error: ${err.message}`;
  } finally {
    uploadBtn.style.pointerEvents = 'auto';
    youtubeToggleBtn.style.pointerEvents = 'auto';
    youtubeSubmitBtn.disabled = false;
    youtubeSubmitBtn.textContent = 'Fetch';
  }
}

// Process uploaded file with streaming UMAP
async function processFile(file) {
  uploadLabelText.textContent = 'Processing...';
  uploadBtn.style.pointerEvents = 'none';
  statusEl.textContent = `Uploading ${file.name}...`;

  if (isFullPlaying) stopFullPlayback();
  playbackAudio = null;

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/encode_stream`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(await response.text());

    await handleEncodeStream(response);

  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    uploadLabelText.textContent = 'Upload Audio';
    uploadBtn.style.pointerEvents = 'auto';
  }
}

youtubeToggleBtn.addEventListener('click', () => {
  const isVisible = youtubeForm.style.display === 'flex';
  youtubeForm.style.display = isVisible ? 'none' : 'flex';
  if (!isVisible) {
    youtubeUrlInput.focus();
  }
});

youtubeSubmitBtn.addEventListener('click', () => {
  const url = youtubeUrlInput.value.trim();
  if (!url) {
    statusEl.textContent = 'Please paste a YouTube URL';
    return;
  }
  processYouTubeUrl(url);
});

youtubeUrlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const url = youtubeUrlInput.value.trim();
    if (!url) {
      statusEl.textContent = 'Please paste a YouTube URL';
      return;
    }
    processYouTubeUrl(url);
  }
});


// Animate from current positions to target projection
let animationId = null;
function animateToProjection(targetProjection) {
  if (animationId) cancelAnimationFrame(animationId);
  if (!points) return;

  const startPositions = [...points.geometry.attributes.position.array];
  const startTime = performance.now();
  const duration = 1500;

  function animate() {
    const elapsed = performance.now() - startTime;
    const t = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);

    const positions = points.geometry.attributes.position.array;
    targetProjection.forEach((p, i) => {
      positions[i * 3] = startPositions[i * 3] + (p[0] * 2 - startPositions[i * 3]) * ease;
      positions[i * 3 + 1] = startPositions[i * 3 + 1] + (p[1] * 2 - startPositions[i * 3 + 1]) * ease;
      positions[i * 3 + 2] = startPositions[i * 3 + 2] + (p[2] * 2 - startPositions[i * 3 + 2]) * ease;
    });
    points.geometry.attributes.position.needsUpdate = true;

    if (connectionLines) {
      const linePos = connectionLines.geometry.attributes.position.array;
      for (let i = 0; i < targetProjection.length - 1; i++) {
        const p0 = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
        const p1 = [positions[(i + 1) * 3], positions[(i + 1) * 3 + 1], positions[(i + 1) * 3 + 2]];
        linePos[i * 6] = p0[0]; linePos[i * 6 + 1] = p0[1]; linePos[i * 6 + 2] = p0[2];
        linePos[i * 6 + 3] = p1[0]; linePos[i * 6 + 4] = p1[1]; linePos[i * 6 + 5] = p1[2];
      }
      connectionLines.geometry.attributes.position.needsUpdate = true;
    }

    if (t < 1) {
      animationId = requestAnimationFrame(animate);
    } else {
      projection = targetProjection;
      animationId = null;
      buildClusters();
    }
  }
  animate();
}

// File upload via button
fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) processFile(file);
  fileInput.value = ''; // Reset so same file can be re-selected
});

// Resynth
const resynthInput = document.getElementById('resynth-input');
const resynthBtn = document.getElementById('resynth-btn');
let resynthAudio = null;
let isResynthMode = false;

resynthInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  if (isFullPlaying) stopFullPlayback();

  resynthBtn.style.opacity = '0.5';
  resynthBtn.style.pointerEvents = 'none';
  statusEl.textContent = 'Resynthesizing...';

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/resynth`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Resynth failed');
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);

    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = `resynth_${file.name}`;
    a.click();

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    const arrayBuffer = await (await fetch(audioUrl)).arrayBuffer();
    resynthAudio = await audioContext.decodeAudioData(arrayBuffer);
    isResynthMode = true;

    statusEl.textContent = `Resynth ready! Press Play or Space`;
    await startPlaybackAt(0);

  } catch (err) {
    statusEl.textContent = `Resynth error: ${err.message}`;
    console.error(err);
  } finally {
    resynthBtn.style.opacity = '1';
    resynthBtn.style.pointerEvents = 'auto';
    resynthInput.value = '';
  }
});

// Drag and drop
document.body.addEventListener('dragover', (e) => {
  e.preventDefault();
  document.body.style.background = '#12121a';
});
document.body.addEventListener('dragleave', (e) => {
  e.preventDefault();
  document.body.style.background = '#08080c';
});
document.body.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.style.background = '#08080c';
  const file = e.dataTransfer.files?.[0];
  if (file && file.type.startsWith('audio/')) processFile(file);
});

// Play latent
async function playLatent(idx) {
  if (idx === null || idx >= audioInfo?.numLatents) return;
  if (isFullPlaying) return;

  if (currentSource) {
    currentSource.stop();
    currentSource = null;
  }

  isPlaying = true;
  selectedIdx = idx;
  infoPlaying.style.display = 'block';

  try {
    const response = await fetch(`${API_URL}/play`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: idx })
    });

    if (!response.ok) throw new Error('Play failed');

    const arrayBuffer = await response.arrayBuffer();

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.onended = () => {
      isPlaying = false;
      selectedIdx = null;
      infoPlaying.style.display = 'none';
    };
    source.start();
    currentSource = source;

  } catch (err) {
    console.error('Playback error:', err);
    isPlaying = false;
    selectedIdx = null;
    infoPlaying.style.display = 'none';
  }
}

// Line mode toggle
lineToggle.addEventListener('click', toggleLineMode);

function toggleLineMode() {
  lineMode = !lineMode;
  lineToggle.textContent = lineMode ? '⚫ Points' : '🌈 Line';
  lineToggle.style.background = lineMode ? '#a855f7' : '#6366f1';

  if (!scene || !projection.length) return;

  if (lineMode) {
    if (points) points.visible = false;
    if (connectionLines) connectionLines.visible = false;
    if (!rainbowLine) createRainbowLine();
    rainbowLine.visible = true;
  } else {
    if (points) points.visible = true;
    if (connectionLines) connectionLines.visible = true;
    if (rainbowLine) rainbowLine.visible = false;
  }
}

function createRainbowLine() {
  if (!projection.length) return;

  const linePositions = [];
  const lineColors = [];

  for (let i = 0; i < projection.length; i++) {
    const p = projection[i];
    linePositions.push(p[0] * 2, p[1] * 2, p[2] * 2);
    const t = i / projection.length;
    const color = new THREE.Color().setHSL(t, 0.9, 0.55);
    lineColors.push(color.r, color.g, color.b);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

  rainbowLineColors = new Float32Array(lineColors);

  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 2,
    transparent: true,
    opacity: 1
  });

  rainbowLine = new THREE.Line(geom, mat);
  rainbowLine.visible = false;
  scene.add(rainbowLine);
}

// K-means clustering
const kSlider = document.getElementById('k-slider');
const kValue = document.getElementById('k-value');

kSlider.addEventListener('input', (e) => {
  clusterK = parseInt(e.target.value);
  kValue.textContent = clusterK;
  buildClusters();
});

function buildClusters() {
  if (!projection.length || !scene) return;

  clusterMeshes.forEach(m => scene.remove(m));
  clusterMeshes = [];
  clusterTriggerTimes = new Array(clusterK).fill(0);

  const pts = projection.map(p => [p[0] * 2, p[1] * 2, p[2] * 2]);

  // K-means++ initialization
  let centroids = [];
  centroids.push([...pts[Math.floor(Math.random() * pts.length)]]);

  for (let i = 1; i < clusterK && i < pts.length; i++) {
    const dists = pts.map(p => {
      let minD = Infinity;
      centroids.forEach(c => {
        const d = (p[0]-c[0])**2 + (p[1]-c[1])**2 + (p[2]-c[2])**2;
        if (d < minD) minD = d;
      });
      return minD;
    });

    const totalDist = dists.reduce((a, b) => a + b, 0);
    let r = Math.random() * totalDist;
    let chosen = 0;
    for (let j = 0; j < dists.length; j++) {
      r -= dists[j];
      if (r <= 0) { chosen = j; break; }
    }
    centroids.push([...pts[chosen]]);
  }

  // K-means iterations
  for (let iter = 0; iter < 50; iter++) {
    clusterAssignments = pts.map(p => {
      let minDist = Infinity, minIdx = 0;
      centroids.forEach((c, i) => {
        const d = (p[0]-c[0])**2 + (p[1]-c[1])**2 + (p[2]-c[2])**2;
        if (d < minDist) { minDist = d; minIdx = i; }
      });
      return minIdx;
    });

    const newCentroids = centroids.map(() => [0, 0, 0]);
    const counts = new Array(clusterK).fill(0);
    pts.forEach((p, i) => {
      const c = clusterAssignments[i];
      newCentroids[c][0] += p[0];
      newCentroids[c][1] += p[1];
      newCentroids[c][2] += p[2];
      counts[c]++;
    });

    let converged = true;
    const newCentroidsFinal = newCentroids.map((c, i) => {
      if (counts[i] === 0) return centroids[i];
      const nc = [c[0]/counts[i], c[1]/counts[i], c[2]/counts[i]];
      const moved = (nc[0]-centroids[i][0])**2 + (nc[1]-centroids[i][1])**2 + (nc[2]-centroids[i][2])**2;
      if (moved > 0.0001) converged = false;
      return nc;
    });
    centroids = newCentroidsFinal;
    if (converged) break;
  }

  clusterCentroids = centroids;

  // Glow texture
  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = 128;
  glowCanvas.height = 128;
  const ctx = glowCanvas.getContext('2d');
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.3, 'rgba(255,255,255,0.5)');
  gradient.addColorStop(0.6, 'rgba(255,255,255,0.15)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const glowTexture = new THREE.CanvasTexture(glowCanvas);

  // Create sprites
  for (let k = 0; k < clusterK; k++) {
    const clusterPts = pts.filter((_, i) => clusterAssignments[i] === k);
    if (clusterPts.length < 2) continue;

    const t = k / clusterK;
    const color = new THREE.Color().setHSL(t, 0.7, 0.5);

    const center = centroids[k];
    let maxDist = 0;
    clusterPts.forEach(p => {
      const d = Math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2);
      if (d > maxDist) maxDist = d;
    });

    const spriteSize = maxDist * 2.5;

    const mat = new THREE.SpriteMaterial({
      map: glowTexture,
      color: color,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const sprite = new THREE.Sprite(mat);
    sprite.position.set(center[0], center[1], center[2]);
    sprite.scale.set(spriteSize, spriteSize, 1);
    sprite.userData = { baseOpacity: 0.35, clusterIdx: k };
    scene.add(sprite);
    clusterMeshes.push(sprite);
  }
}

function updateClusterVisualization(currentCluster) {
  const now = performance.now();

  if (currentCluster >= 0 && currentCluster < clusterK) {
    clusterTriggerTimes[currentCluster] = now;
  }

  clusterMeshes.forEach(mesh => {
    const k = mesh.userData.clusterIdx;
    const baseOpacity = mesh.userData.baseOpacity;
    const timeSinceTrigger = now - clusterTriggerTimes[k];
    const fadeDuration = 80;

    let opacity;
    if (timeSinceTrigger < fadeDuration) {
      const t = timeSinceTrigger / fadeDuration;
      const fade = t * t * t;
      opacity = 1.0 - fade * (1.0 - baseOpacity);
    } else {
      opacity = baseOpacity;
    }

    mesh.material.opacity = opacity;
  });
}

// Playbar elements
const playbarContainer = document.getElementById('playbar-container');
const playbar = document.getElementById('playbar');
const playbarProgress = document.getElementById('playbar-progress');
const playbarHandle = document.getElementById('playbar-handle');
const playbarCurrent = document.getElementById('playbar-current');
const playbarDuration = document.getElementById('playbar-duration');
const volumeSlider = document.getElementById('volume-slider');
const volumeIcon = document.getElementById('volume-icon');

let gainNode = null;
let currentVolume = 1.0;

volumeSlider.addEventListener('input', (e) => {
  currentVolume = e.target.value / 100;
  if (gainNode) gainNode.gain.value = currentVolume;
  volumeIcon.textContent = currentVolume === 0 ? '🔇' : currentVolume < 0.5 ? '🔉' : '🔊';
});

volumeIcon.addEventListener('click', () => {
  if (currentVolume > 0) {
    volumeSlider.value = 0;
    currentVolume = 0;
  } else {
    volumeSlider.value = 100;
    currentVolume = 1;
  }
  if (gainNode) gainNode.gain.value = currentVolume;
  volumeIcon.textContent = currentVolume === 0 ? '🔇' : '🔊';
});

playBtn.addEventListener('click', toggleFullPlayback);

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

async function toggleFullPlayback() {
  if (isFullPlaying) {
    stopFullPlayback();
    return;
  }
  await startPlaybackAt(0);
}

async function startPlaybackAt(offsetSeconds) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (!gainNode) {
    gainNode = audioContext.createGain();
    gainNode.connect(audioContext.destination);
    gainNode.gain.value = currentVolume;
  }

  if (currentSource) {
    currentSource.onended = null;
    currentSource.stop();
    currentSource = null;
  }

  playBtn.textContent = 'Loading...';

  try {
    let audioBuffer;
    if (isResynthMode && resynthAudio) {
      audioBuffer = resynthAudio;
    } else {
      if (!playbackAudio) {
        const response = await fetch(`${API_URL}/audio_full`);
        if (!response.ok) throw new Error('Failed to load audio');
        const arrayBuffer = await response.arrayBuffer();
        playbackAudio = await audioContext.decodeAudioData(arrayBuffer);
      }
      audioBuffer = playbackAudio;
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(gainNode);
    source.onended = stopFullPlayback;

    const duration = audioBuffer.duration;
    offsetSeconds = Math.max(0, Math.min(offsetSeconds, duration - 0.1));

    playbackStartTime = audioContext.currentTime - offsetSeconds;
    source.start(0, offsetSeconds);
    currentSource = source;
    isFullPlaying = true;
    autoRotate = true;

    playBtn.textContent = '⏹ Stop';
    playBtn.style.background = '#ef4444';

    playbarContainer.classList.add('enabled');
    playbarDuration.textContent = formatTime(duration);

  } catch (err) {
    console.error('Playback error:', err);
    playBtn.textContent = '▶ Play';
  }
}

function getCurrentAudioBuffer() {
  return (isResynthMode && resynthAudio) ? resynthAudio : playbackAudio;
}

playbar.addEventListener('click', (e) => {
  const audio = getCurrentAudioBuffer();
  if (!audio) return;
  const rect = playbar.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = x / rect.width;
  const seekTime = pct * audio.duration;
  if (isFullPlaying) startPlaybackAt(seekTime);
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;

  const audio = getCurrentAudioBuffer();

  if (e.code === 'Space') {
    e.preventDefault();
    if (audioInfo || resynthAudio) toggleFullPlayback();
  } else if (e.code === 'ArrowLeft' && isFullPlaying && audio) {
    e.preventDefault();
    const elapsed = audioContext.currentTime - playbackStartTime;
    startPlaybackAt(Math.max(0, elapsed - 5));
  } else if (e.code === 'ArrowRight' && isFullPlaying && audio) {
    e.preventDefault();
    const elapsed = audioContext.currentTime - playbackStartTime;
    startPlaybackAt(Math.min(audio.duration, elapsed + 5));
  }
});

function stopFullPlayback() {
  if (currentSource) {
    currentSource.stop();
    currentSource = null;
  }
  isFullPlaying = false;
  isResynthMode = false;
  playBtn.textContent = '▶ Play';
  playBtn.style.background = '#22c55e';

  if (points && baseColors) {
    const colorAttr = points.geometry.getAttribute('color');
    const colors = colorAttr.array;
    for (let i = 0; i < baseColors.length; i++) {
      colors[i] = baseColors[i];
    }
    colorAttr.needsUpdate = true;

    const alphaAttr = points.geometry.getAttribute('alpha');
    if (alphaAttr) {
      const alphas = alphaAttr.array;
      for (let i = 0; i < alphas.length; i++) {
        alphas[i] = idleAlpha;
      }
      alphaAttr.needsUpdate = true;
    }
  }

  if (rainbowLine && rainbowLineColors) {
    const colors = rainbowLine.geometry.attributes.color.array;
    for (let i = 0; i < rainbowLineColors.length; i++) colors[i] = rainbowLineColors[i];
    rainbowLine.geometry.attributes.color.needsUpdate = true;
  }

  if (glowPoints && scene) { scene.remove(glowPoints); glowPoints = null; }
  if (trailLine && scene) { scene.remove(trailLine); trailLine = null; }

  playbarContainer.classList.remove('enabled');
  playbarProgress.style.width = '0%';
  playbarHandle.style.left = '0%';
  playbarCurrent.textContent = '0:00';
}


function updatePlaybackVisualization() {
    if (!isFullPlaying || !audioContext || !projection.length) return;

    const elapsed = audioContext.currentTime - playbackStartTime;
    const samplePos = elapsed * 44100;
    const latentIdx = Math.floor(samplePos / (audioInfo?.samplesPerLatent || 2048));
    const progress = (samplePos % (audioInfo?.samplesPerLatent || 2048)) / (audioInfo?.samplesPerLatent || 2048);

    const audio = getCurrentAudioBuffer();
    if (audio) {
        const pct = (elapsed / audio.duration) * 100;
        playbarProgress.style.width = `${Math.min(100, pct)}%`;
        playbarHandle.style.left = `${Math.min(100, pct)}%`;
        playbarCurrent.textContent = formatTime(elapsed);
    }

    if (latentIdx >= projection.length) return;

    if (lineMode && rainbowLine && rainbowLineColors) {
        const colors = rainbowLine.geometry.attributes.color.array;
        const trailLength = 20;

        for (let i = 0; i < projection.length; i++) {
            const dist = latentIdx - i;
            if (dist >= 0 && dist < trailLength) {
                const fade = 1 - dist / trailLength;
                const brightness = 0.3 + fade * 0.7;
                colors[i * 3] = rainbowLineColors[i * 3] * brightness;
                colors[i * 3 + 1] = rainbowLineColors[i * 3 + 1] * brightness;
                colors[i * 3 + 2] = rainbowLineColors[i * 3 + 2] * brightness;
            } else if (dist >= trailLength) {
                colors[i * 3] = rainbowLineColors[i * 3] * 0.3;
                colors[i * 3 + 1] = rainbowLineColors[i * 3 + 1] * 0.3;
                colors[i * 3 + 2] = rainbowLineColors[i * 3 + 2] * 0.3;
            } else {
                colors[i * 3] = rainbowLineColors[i * 3] * 0.08;
                colors[i * 3 + 1] = rainbowLineColors[i * 3 + 1] * 0.08;
                colors[i * 3 + 2] = rainbowLineColors[i * 3 + 2] * 0.08;
            }
        }
        rainbowLine.geometry.attributes.color.needsUpdate = true;
  } else if (points && baseColors) {
    const colorAttr = points.geometry.getAttribute('color');
    const alphaAttr = points.geometry.getAttribute('alpha');
    if (!colorAttr || !alphaAttr) return;

    const colors = colorAttr.array;
    const alphas = alphaAttr.array;

    const trailLength = Math.max(1, trailDecay); // number of points in decay trail

    for (let i = 0; i < projection.length; i++) {
      const dist = latentIdx - i; // 0 = current, 1 = previous, etc.

      if (dist < 0) {
        // future points: just idle
        colors[i * 3]     = baseColors[i * 3];
        colors[i * 3 + 1] = baseColors[i * 3 + 1];
        colors[i * 3 + 2] = baseColors[i * 3 + 2];
        alphas[i] = idleAlpha;
        continue;
      }

      // dist >= 0 → we are at or behind the playhead

      // 0 at current point, 1 at end of trail (or beyond)
      const t = Math.min(dist / trailLength, 1.0);
      const glow = 1.0 - t;       // 1 → current, 0 → far past
      const glowSq = glow * glow; // smoother falloff

      // color glow
      colors[i * 3]     = baseColors[i * 3]     + (1 - baseColors[i * 3])     * glowSq;
      colors[i * 3 + 1] = baseColors[i * 3 + 1] + (1 - baseColors[i * 3 + 1]) * glowSq;
      colors[i * 3 + 2] = baseColors[i * 3 + 2] + (0.3 - baseColors[i * 3 + 2]) * glowSq;

      // alpha: lerp from idleAlpha → 1.0
      alphas[i] = idleAlpha + (1.0 - idleAlpha) * glowSq;
    }

    colorAttr.needsUpdate = true;
    alphaAttr.needsUpdate = true;
  }




    // Trail line
    const trailLineLength = 25;
    const trailStart = Math.max(0, latentIdx - trailLineLength);

    if (!trailLine && projection.length > 1) {
        const maxVerts = projection.length;
        const positions = new Float32Array(maxVerts * 3);
        const lineColors = new Float32Array(maxVerts * 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geom.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
        geom.setDrawRange(0, 0);
        const mat = new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 0.9});
        trailLine = new THREE.Line(geom, mat);
        scene.add(trailLine);
    }

    if (trailLine && latentIdx > 0) {
        const positions = trailLine.geometry.attributes.position.array;
        const lineColors = trailLine.geometry.attributes.color.array;
        const numVerts = Math.min(latentIdx - trailStart + 1, latentIdx + 1);

        for (let i = 0; i < numVerts; i++) {
            const idx = trailStart + i;
            if (idx >= projection.length) break;
            const p = projection[idx];
            positions[i * 3] = p[0] * 2;
            positions[i * 3 + 1] = p[1] * 2;
            positions[i * 3 + 2] = p[2] * 2;

            const fade = i / (numVerts - 1 || 1);
            const brightness = 0.2 + fade * 0.8;
            const t = idx / projection.length;
            const color = new THREE.Color().setHSL(t, 0.85, 0.5 * brightness + 0.3);
            lineColors[i * 3] = color.r;
            lineColors[i * 3 + 1] = color.g;
            lineColors[i * 3 + 2] = color.b;
        }

        trailLine.geometry.attributes.position.needsUpdate = true;
        trailLine.geometry.attributes.color.needsUpdate = true;
        trailLine.geometry.setDrawRange(0, numVerts);
    }


    // Playhead
    if (latentIdx < projection.length - 1) {
        const p0 = projection[latentIdx];
        const p1 = projection[latentIdx + 1];
        const x = (p0[0] + (p1[0] - p0[0]) * progress) * 2;
        const y = (p0[1] + (p1[1] - p0[1]) * progress) * 2;
        const z = (p0[2] + (p1[2] - p0[2]) * progress) * 2;

        if (!glowPoints) {
            glowPoints = new THREE.Group();

            const coreGeom = new THREE.SphereGeometry(0.08, 16, 16);
            const coreMat = new THREE.MeshBasicMaterial({color: 0xffffff});
            const core = new THREE.Mesh(coreGeom, coreMat);
            core.name = 'core';
            glowPoints.add(core);

            const haloGeom = new THREE.SphereGeometry(0.18, 16, 16);
            const haloMat = new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.35});
            const halo = new THREE.Mesh(haloGeom, haloMat);
            halo.name = 'halo';
            glowPoints.add(halo);

            scene.add(glowPoints);
        }
        glowPoints.position.set(x, y, z);

        const t = latentIdx / projection.length;
        const core = glowPoints.getObjectByName('core');
        const halo = glowPoints.getObjectByName('halo');
        if (core) core.material.color.setHSL(t, 0.9, 0.85);
        if (halo) halo.material.color.setHSL(t, 0.9, 0.6);
    }

    // Clusters
    if (clusterAssignments.length > 0 && latentIdx < clusterAssignments.length) {
        updateClusterVisualization(clusterAssignments[latentIdx]);
    }
}

// Three.js
function initThreeJS() {
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(theme.background);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 4;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Points
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(projection.length * 3);
    const colors = new Float32Array(projection.length * 3);
    const n = projection.length;
    const alphas = new Float32Array(projection.length);
    projection.forEach((p, i) => {
        positions[i * 3] = p[0] * 2;
        positions[i * 3 + 1] = p[1] * 2;
        positions[i * 3 + 2] = p[2] * 2;

        const t = n <= 1 ? 0 : i / (n - 1);
        const color = getPointColorFromPalette(t);
        const min_alpha = 0.1;

        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;

        alphas[i] = idleAlpha;
    });

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('alpha',    new THREE.BufferAttribute(alphas, 1)); // NEW

    baseColors = new Float32Array(colors);

    material = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      vertexColors: true,
      uniforms: {
        pointSize: { value: 0.06 }
      },
      vertexShader: `
        precision mediump float;
    
        attribute float alpha;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float pointSize;
    
        void main() {
          vColor = color;
          vAlpha = alpha;
    
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPosition;
    
          // perspective-correct point size, with clamp so it never vanishes
          float z = -mvPosition.z;
          z = max(z, 0.1);
          gl_PointSize = pointSize * (300.0 / z);
        }
      `,
      fragmentShader: `
        precision mediump float;
    
        varying vec3 vColor;
        varying float vAlpha;
    
        void main() {
          // circular point sprite
          vec2 c = gl_PointCoord - vec2(0.5);
          float d = length(c);
          if (d > 0.5) discard;
    

          // gl_FragColor = vec4(vColor, 1.0); // debug
          gl_FragColor = vec4(vColor, vAlpha);
        }
      `
    });

    points = new THREE.Points(geometry, material);
    scene.add(points);

    // Connection lines
    const linePositions = [];
    for (let i = 0; i < projection.length - 1; i++) {
    linePositions.push(
      projection[i][0] * 2, projection[i][1] * 2, projection[i][2] * 2,
      projection[i + 1][0] * 2, projection[i + 1][1] * 2, projection[i + 1][2] * 2
    );
    }
    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x333344, transparent: true, opacity: 0.3 });
    connectionLines = new THREE.LineSegments(lineGeom, lineMat);
    scene.add(connectionLines);

    rainbowLine = null;
    rainbowLineColors = null;

    // Raycaster
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.12;
    const mouse = new THREE.Vector2();

    // Drag to pan
    let dragMode = 'rotate'; // 'rotate' | 'pan'
    let target = new THREE.Vector3(0, 0, 0);

    function updateCamera() {
      camera.position.x = target.x + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      camera.position.y = target.y + spherical.radius * Math.cos(spherical.phi);
      camera.position.z = target.z + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      camera.lookAt(target);
    }

    container.addEventListener('mousedown', (e) => {
      if (e.button === 0) { // left button
        isDragging = true;
        autoRotate = false;
        dragMode = e.ctrlKey ? 'pan' : 'rotate';
        previousMouse = { x: e.clientX, y: e.clientY };
      }
    });

    container.addEventListener('mouseup', () => { isDragging = false; });

    container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    if (isDragging) {
      const deltaX = e.clientX - previousMouse.x;
      const deltaY = e.clientY - previousMouse.y;

      if (dragMode === 'rotate') {
        spherical.theta -= deltaX * 0.008;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - deltaY * 0.008));
      } else if (dragMode === 'pan') {
        const panSpeed = 0.002 * spherical.radius;

        // forward (camera view direction), up, right vectors
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        const up = camera.up.clone().normalize();
        const right = new THREE.Vector3().crossVectors(forward, up).normalize();

        const moveX = -deltaX * panSpeed;
        const moveY = deltaY * panSpeed;

        // move target in camera plane
        target.add(right.multiplyScalar(moveX));
        target.add(up.multiplyScalar(moveY));
      }

      previousMouse = { x: e.clientX, y: e.clientY };
      updateCamera();
    }


    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);

    if (intersects.length > 0) {
      hoveredIdx = intersects[0].index;
      infoPanel.classList.add('visible');
      infoIdx.textContent = `Latent #${hoveredIdx}`;
      infoTime.textContent = `Time: ${((hoveredIdx * audioInfo.samplesPerLatent) / 44100).toFixed(3)}s`;
      container.style.cursor = 'pointer';
    } else {
      hoveredIdx = null;
      infoPanel.classList.remove('visible');
      container.style.cursor = isDragging ? 'grabbing' : 'grab';
    }
    });

    container.addEventListener('wheel', (e) => {
    e.preventDefault();
    spherical.radius = Math.max(1.5, Math.min(12, spherical.radius + e.deltaY * 0.005));
    updateCamera();
    }, { passive: false });

    container.addEventListener('click', () => {
    if (hoveredIdx !== null) playLatent(hoveredIdx);
    });

    updateCamera();

    function animate() {
    requestAnimationFrame(animate);
    if (autoRotate) {
      scene.rotation.y += 0.002;
      // updateCamera();
    }
    updatePlaybackVisualization();
    renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    });
}

// Init
checkServer();
