let lastPrediction = "";
let pollInterval   = null;
let webcamRunning  = false;

// ===== Page Switching =====
function switchPage(pageId, clickedBtn) {
  // hide all pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(pageId).classList.add('active');

  // if going to live page — start webcam
  if (pageId === 'page2') {
    startWebcam();
  } else {
    stopWebcam();
  }
}

// ===== Webcam =====
function startWebcam() {
  if (webcamRunning) return;
  webcamRunning = true;
  const webcam = document.getElementById('webcam');
  webcam.src   = '/predict_live?' + Date.now();
  startPolling();
}

async function stopWebcam() {
  if (!webcamRunning) return;
  webcamRunning = false;
  document.getElementById('webcam').src = '';
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  try { await fetch('/stop_webcam'); } catch(e) {}
  document.getElementById('resultLive').innerText  = '---';
  document.getElementById('sentenceBox').innerText = '---';
}

// ===== Polling =====
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    if (!webcamRunning) { clearInterval(pollInterval); return; }
    try {
      const res  = await fetch('/get_prediction');
      const data = await res.json();
      if (data.prediction) {
        const conf = data.confidence
          ? ' (' + (data.confidence * 100).toFixed(0) + '%)'
          : '';
        document.getElementById('resultLive').innerText = data.prediction + conf;
      }
      if (data.sentence !== undefined) {
        document.getElementById('sentenceBox').innerText = data.sentence || '---';
      }
    } catch(e) { console.warn('Poll error:', e); }
  }, 500);
}

// ===== Sentence Controls =====
async function clearSentence() {
  await fetch('/clear_sentence');
  document.getElementById('sentenceBox').innerText = '---';
}

async function deleteLast() {
  const res  = await fetch('/delete_last');
  const data = await res.json();
  document.getElementById('sentenceBox').innerText = data.sentence || '---';
}

async function speakSentence() {
  await fetch('/speak_sentence');
}

// ===== Drop Zone =====
const dropZone   = document.getElementById('dropZone');
const mediaUpload= document.getElementById('mediaUpload');
const preview    = document.getElementById('preview');
const loader     = document.getElementById('loader');
const resultBox  = document.getElementById('result');

dropZone.addEventListener('click', () => mediaUpload.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.background = '#bae6fd';
});

dropZone.addEventListener('dragleave', () => {
  dropZone.style.background = '';
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.background = '';
  handleMedia(e.dataTransfer.files[0]);
});

mediaUpload.addEventListener('change', function() {
  handleMedia(this.files[0]);
});

function handleMedia(file) {
  if (!file) return;
  preview.innerHTML = '';
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = url;
    img.style.maxWidth = '100%';
    preview.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const video = document.createElement('video');
    video.src = url;
    video.controls = true;
    video.style.maxWidth = '100%';
    preview.appendChild(video);
  }
}

// ===== Predict (Upload) =====
document.getElementById('predictBtn').addEventListener('click', async () => {
  const file = mediaUpload.files[0];
  if (!file) { alert('Please upload a file first!'); return; }

  loader.classList.remove('hidden');
  lastPrediction = '';

  try {
    const formData = new FormData();
    formData.append('file', file);
    const endpoint = file.type.startsWith('image/') ? '/predict_image' : '/predict_video';
    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data     = await response.json();

    if (data.error) {
      resultBox.innerText = 'Error: ' + data.error;
    } else {
      resultBox.innerText = data.prediction + ' (' + Number(data.confidence).toFixed(2) + ')';
    }
  } catch(err) {
    alert('Error contacting server!');
  }

  loader.classList.add('hidden');
});