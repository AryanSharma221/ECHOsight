import * as faceapi from '@vladmandic/face-api';
import Tesseract from 'tesseract.js';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';

// ---- DOM ELEMENTS ----
const video = document.getElementById('webcam');
const liveOverlay = document.getElementById('overlay');
const staticOverlay = document.getElementById('staticOverlay');

// Dashboard Elements
const tabLive = document.getElementById('tabLive');
const tabMemory = document.getElementById('tabMemory');
const modeLive = document.getElementById('liveMode');
const modeMemory = document.getElementById('memoryMode');
const memoryWebcam = document.getElementById('memoryWebcam');

const toggleAudio = document.getElementById('toggleAudio');
const toggleObjDet = document.getElementById('toggleObjDet');
const toggleFaceDet = document.getElementById('toggleFaceDet');
const toggleTextDet = document.getElementById('toggleTextDet');

const btnHush = document.getElementById('btnHush');
const sysStatus = document.getElementById('sysStatus');
const resultsPanel = document.getElementById('resultsPanel');

// Face Identity Elements
const faceName = document.getElementById('faceName');
const btnCaptureCamera = document.getElementById('btnCaptureCamera');
const faceUpload = document.getElementById('faceUpload');
const facesDatabase = document.getElementById('facesDatabase');

// ---- STATE ----
let modelsLoaded = false;
let objModel = null;
let labeledFaceDescriptors = [], faceMatcher = null;
let activeMode = 'live'; // 'live' or 'memory'
let audioQueue = [], isSpeaking = false, lastSpokenSet = new Set(), lastSpokenText = "";

let textConsecutiveFrames = 0;
let lastTextAlertTime = 0;
let isTextCurrentlyVisible = false;

// Ensure scroll animations work
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.scroll-reveal').forEach(el => observer.observe(el));
}

// FAQ Logic
document.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
        const item = btn.parentElement;
        item.classList.toggle('active');
    });
});

// Mobile Nav Logic
const hamburger = document.getElementById('hamburgerMenu');
const navLinks = document.getElementById('navLinks');
if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('open');
        navLinks.classList.toggle('active');
    });
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('open');
            navLinks.classList.remove('active');
        });
    });
}

// ---- CORE INIT ----
async function initSystem() {
    initScrollReveal();
    sysStatus.innerText = "Initializing Hardware...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } });
        video.srcObject = stream;
        memoryWebcam.srcObject = stream;

        sysStatus.innerText = "Syncing Neural Models...";

        const faceURL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(faceURL),
            faceapi.nets.faceLandmark68Net.loadFromUri(faceURL),
            faceapi.nets.faceRecognitionNet.loadFromUri(faceURL),
            cocoSsd.load({ base: 'lite_mobilenet_v2' }).then(m => objModel = m)
        ]);

        modelsLoaded = true;
        sysStatus.innerText = "System Ready";
        sysStatus.classList.add('ready');

        startLiveDetectionLoop();
    } catch (e) {
        console.error(e);
        sysStatus.innerText = "Camera Access Required";
        sysStatus.style.color = "#ff5555";
    }
}

// ---- AUDIO ENGINE ----
function processAudioQueue() {
    if (isSpeaking || audioQueue.length === 0 || !window.speechSynthesis) return;
    isSpeaking = true;
    const utterance = new SpeechSynthesisUtterance(audioQueue.shift());

    // Find best english voice
    const voices = window.speechSynthesis.getVoices();
    const premium = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Aria') || v.name.includes('Natural')));
    if (premium) utterance.voice = premium;

    utterance.rate = 1.05;
    utterance.onend = () => { isSpeaking = false; processAudioQueue(); };
    window.speechSynthesis.speak(utterance);
}

function enqueueAudio(text, type) {
    if (!toggleAudio.checked) return;

    // De-dupe identical text blocks instantly (e.g., rapid text triggering)
    if (text === lastSpokenText && (Date.now() - lastTextAlertTime < 2000)) return;
    lastSpokenText = text;

    // We no longer suppress strings aggressively for 15s because individual systems 
    // now mathematically throttle themselves via IOU Track Cooldowns & Distance Deltas.

    // Absolute priority queue for faces
    if (type === 'face') {
        audioQueue.unshift(text);
    } else {
        audioQueue.push(text);
    }

    // Cap maximum queue length to prevent massive TTS backlogs if the system gets overloaded
    if (audioQueue.length > 5) {
        audioQueue = audioQueue.slice(0, 5);
    }

    processAudioQueue();
}

btnHush.addEventListener('click', () => {
    audioQueue = []; if (window.speechSynthesis) window.speechSynthesis.cancel();
    isSpeaking = false;
});

// ---- UI CARD RENDERER ----
function calculateDistance(boxWidth, refWidth) {
    // Monocular bounding box focal length distance approximation
    // Assuming an average face/object width in physical space
    const ratio = boxWidth / refWidth;
    let distanceMeters = (0.28 / ratio).toFixed(1);

    if (distanceMeters > 5) return "5+ meters";
    return `${distanceMeters} meters`;
}

function createResultCard(label, scoreFormatted, type, colorHex = "#a1a1a1") {
    // Fill width based on score if it's a percentage, max it if not.
    let fillPct = scoreFormatted.includes('%') ? scoreFormatted : "100%";

    const div = document.createElement('div');
    div.className = 'result-card';
    div.innerHTML = `
        <div class="card-top">
            <span class="c-label">${label}</span>
            <span class="c-score">${scoreFormatted}</span>
        </div>
        <div class="c-type" style="color: ${colorHex}">${type}</div>
        <div class="card-meter">
            <div class="card-fill" style="width: ${fillPct}; background: ${colorHex}"></div>
        </div>
    `;
    return div;
}

function renderPredictions(predictions) {
    resultsPanel.innerHTML = '';
    if (predictions.length === 0) {
        resultsPanel.innerHTML = '<div class="empty-state">No entities detected in frame.</div>';
        return;
    }
    predictions.forEach(p => resultsPanel.appendChild(p));
}

// ---- LIVE CAMERA LOOP ----
let lastTextScan = 0;
let isPipelineRunning = false;

// Real-Time Bounding Box Tracker & Embedding Averaging Memory
let faceTrackUIDCounter = 0;
let liveTrackedFaces = new Map(); // { box, descriptors: [], framesAlive: 0, lastSeenNow: 0, lockedIdentity: null, lastAnnouncedTime: 0, lastDistanceRaw: 0, lastReportedName: null }
let objTrackUIDCounter = 0;
let liveTrackedObjs = new Map(); // { class, box, framesAlive: 0, lastSeenNow: 0, lastAnnouncedTime: 0, lastDistanceRaw: 0 }
let LIVE_TRACK_TIMEOUT = 1000; // ms to keep track alive if entity drops out of frame

// Intersection Over Union (IOU) helper for pairing faces across frames
function getIOU(box1, box2) {
    const xA = Math.max(box1.x, box2.x);
    const yA = Math.max(box1.y, box2.y);
    const xB = Math.min(box1.x + box1.width, box2.x + box2.width);
    const yB = Math.min(box1.y + box1.height, box2.y + box2.height);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;

    return interArea / (box1Area + box2Area - interArea);
}

async function startLiveDetectionLoop() {
    setInterval(async () => {
        // Enforce strict ordered sequential locking
        if (!modelsLoaded || activeMode !== 'live' || video.paused || isPipelineRunning) return;
        isPipelineRunning = true;

        const width = video.videoWidth; const height = video.videoHeight;
        if (width === 0) { isPipelineRunning = false; return; }
        if (width !== liveOverlay.width) faceapi.matchDimensions(liveOverlay, { width, height });

        const ctx = liveOverlay.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        const now = Date.now();
        let domCards = [];
        let frameAudioEvents = [];

        // 1 & 2. Face Detection & Recognition (Highest Priority, runs FIRST)
        if (toggleFaceDet.checked) {
            try {
                // We enforce FaceAlignment via .withFaceLandmarks() automatically under the hood natively
                const facePreds = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.45 })).withFaceLandmarks().withFaceDescriptors();
                const resizedFaces = faceapi.resizeResults(facePreds, { width, height });

                let activeTrackUids = new Set();

                resizedFaces.forEach(f => {
                    const dist = calculateDistance(f.detection.box.width, width);
                    let name = "Unknown Person";
                    let conf = (f.detection.score * 100).toFixed(0) + "%";
                    let matchedLabel = 'unknown';

                    // 1. Assign Face to an existing Tracking ID based on Box Overlap (IOU)
                    let matchedUID = null;
                    let bestIOU = 0;
                    for (let [uid, track] of liveTrackedFaces.entries()) {
                        const iou = getIOU(f.detection.box, track);
                        if (iou > 0.3 && iou > bestIOU) { // Lowered overlap threshold for rapid head movements
                            bestIOU = iou;
                            matchedUID = uid;
                        }
                    }

                    if (!matchedUID) {
                        matchedUID = ++faceTrackUIDCounter;
                        liveTrackedFaces.set(matchedUID, { framesAlive: 0, descriptors: [], lockedIdentity: null, lockedConf: null, lastAnnouncedTime: 0, lastDistanceRaw: null, lastReportedName: null });
                    }

                    activeTrackUids.add(matchedUID);

                    // 2. Update Track State
                    let track = liveTrackedFaces.get(matchedUID);
                    track.x = f.detection.box.x; track.y = f.detection.box.y;
                    track.width = f.detection.box.width; track.height = f.detection.box.height;
                    track.lastSeenNow = Date.now();

                    // 3. Keep a rotating window of the last 3 Frame Descriptors (Embedding Array)
                    if (track.descriptors.length >= 3) track.descriptors.shift();
                    track.descriptors.push(f.descriptor);
                    track.framesAlive++;

                    // 4. Compute Averaged Embedding across 3 Frames once accrued buffer
                    if (faceMatcher && track.descriptors.length === 3) {
                        // Dynamically compute the Mean Average Tensor across 128 dimensions from 3 arrays
                        const avgEmbedding = new Float32Array(128);
                        for (let i = 0; i < 128; i++) {
                            let sum = 0;
                            for (let d = 0; d < 3; d++) {
                                sum += track.descriptors[d][i];
                            }
                            avgEmbedding[i] = sum / 3;
                        }

                        // Run the mathematically stabilized averaged tensor against the registry
                        const match = faceMatcher.findBestMatch(avgEmbedding);

                        // If it's technically Unknown but scored close, force strict evaluation since we smoothed the tensor
                        if (match.label !== 'unknown' && match.distance <= 0.55) {
                            track.lockedIdentity = match.label;
                            const similarity = Math.max(75, Math.floor((1 - match.distance + 0.3) * 100)); // Map UX logic
                            track.lockedConf = similarity + "% match";
                        } else {
                            track.lockedIdentity = 'unknown';
                            track.lockedConf = (f.detection.score * 100).toFixed(0) + "%";
                        }
                    }

                    // 5. Apply the tracked logic to the DOM and apply Throttled TTS Rules
                    let reportedName = track.lockedIdentity === 'unknown' || !track.lockedIdentity ? "Unknown person" : track.lockedIdentity;
                    name = track.lockedIdentity ? track.lockedIdentity : "Unknown Person";
                    if (track.lockedConf) conf = track.lockedConf;

                    let currentDistRaw = 0.28 / (f.detection.box.width / width);
                    let shouldAnnounce = false;
                    let bypassCooldown = false;

                    // TTS Rule 1: Initial Recognition (Wait 3 frames for Tensor Smoothing so we don't say "Unknown")
                    if (track.framesAlive === 3) shouldAnnounce = true;
                    // TTS Rule 2: 2-Second Heartbeat
                    if (track.lastAnnouncedTime !== 0 && Date.now() - track.lastAnnouncedTime >= 2000) shouldAnnounce = true;
                    // TTS Rule 3: Math.abs distance changed >= 1.0 meter
                    if (track.lastDistanceRaw !== null && Math.abs(currentDistRaw - track.lastDistanceRaw) >= 1.0) shouldAnnounce = true;
                    // TTS Rule 4: Identity Shifted 
                    if (track.lastReportedName && track.lastReportedName !== reportedName) {
                        shouldAnnounce = true;
                        bypassCooldown = true; // CRITICAL: If identity changes, IMMEDIATELY announce it regardless of cooldown
                    }

                    // Strict Cooldown Gate: Do NOT announce anything if < 2000ms have passed. NO EXCEPTIONS.
                    // (Except for the very first announcement, OR an Identity Shift)
                    if (!bypassCooldown && track.framesAlive > 3 && track.lastAnnouncedTime !== 0 && Date.now() - track.lastAnnouncedTime < 2000) {
                        shouldAnnounce = false;
                    }

                    if (shouldAnnounce) {
                        let prefix = reportedName;
                        if (track.framesAlive <= 3 && reportedName === "Unknown person") prefix = "New person";
                        else if (track.framesAlive <= 3 && reportedName !== "Unknown person") prefix = `New person, ${reportedName},`;

                        // If we are overriding an existing track with a new identity
                        if (bypassCooldown && track.framesAlive > 3) prefix = `${reportedName}`;

                        frameAudioEvents.push({ type: 'face', text: `${prefix} detected ${dist} ahead` });
                        track.lastAnnouncedTime = Date.now();
                        track.lastDistanceRaw = currentDistRaw;
                        track.lastReportedName = reportedName;
                    }

                    domCards.push(createResultCard(`${name} [${dist}]`, conf, "BIOMETRIC", "#ffaa00"));
                    ctx.strokeStyle = "rgba(255, 170, 0, 0.8)"; ctx.lineWidth = 2;
                    ctx.strokeRect(f.detection.box.x, f.detection.box.y, f.detection.box.width, f.detection.box.height);
                    ctx.fillStyle = "#fff"; ctx.font = "16px Inter";
                    ctx.fillText(name, f.detection.box.x, f.detection.box.y > 20 ? f.detection.box.y - 6 : f.detection.box.y + 20);
                });

                // Garbage Collect lost Face Tracks after 1 second
                for (let [uid, track] of liveTrackedFaces.entries()) {
                    if (Date.now() - track.lastSeenNow > LIVE_TRACK_TIMEOUT) {
                        liveTrackedFaces.delete(uid);
                    }
                }

            } catch (e) { }
        }

        // 3. Specific Object Detection (Runs strictly AFTER faces finish evaluating)
        if (toggleObjDet.checked) {
            try {
                const objPreds = await objModel.detect(video, 20, 0.40);
                const nowObjRaw = Date.now();

                objPreds.forEach(p => {
                    if (p.class === 'refrigerator' || p.class === 'remote') return;
                    if (p.score < 0.40) return; // Keep multi-class limits generous to detect background objects

                    let label = p.class;
                    if (label === 'cell phone') label = 'smartphone';

                    const box = { x: p.bbox[0], y: p.bbox[1], width: p.bbox[2], height: p.bbox[3] };

                    // CRITICAL FIX: The Object Detector (CocoSSD) frequently detects bodies as "person".
                    // If the Face Detector already claimed this spatial region, suppress the Object Detector 
                    // from announcing "person" a second time, which forces the Audio queue to get stuck.
                    if (label === 'person') {
                        let faceOverlapFound = false;
                        for (let [uid, faceTrack] of liveTrackedFaces.entries()) {
                            // Check if the face center coordinates are completely inside the person bounding box
                            const faceCenterX = faceTrack.x + faceTrack.width / 2;
                            const faceCenterY = faceTrack.y + faceTrack.height / 2;
                            if (faceCenterX >= box.x && faceCenterX <= box.x + box.width &&
                                faceCenterY >= box.y && faceCenterY <= box.y + box.height) {
                                faceOverlapFound = true;
                            }
                        }
                        if (faceOverlapFound) return; // Skip "person" object detection, let Face Tracker handle it
                    }

                    // Assign Object to existing Tracking ID via IOU
                    let matchedUID = null;
                    let bestIOU = 0;
                    for (let [uid, track] of liveTrackedObjs.entries()) {
                        if (track.class === label) {
                            const iou = getIOU(box, track.box);
                            if (iou > 0.4 && iou > bestIOU) {
                                bestIOU = iou;
                                matchedUID = uid;
                            }
                        }
                    }

                    if (!matchedUID) {
                        matchedUID = ++objTrackUIDCounter;
                        liveTrackedObjs.set(matchedUID, { class: label, framesAlive: 0, lastAnnouncedTime: 0, lastDistanceRaw: null });
                    }

                    let track = liveTrackedObjs.get(matchedUID);
                    track.box = box;
                    track.lastSeenNow = nowObjRaw;
                    track.framesAlive++;

                    const currentDistRaw = 0.28 / (box.width / width);
                    const dist = calculateDistance(box.width, width);

                    let shouldAnnounce = false;
                    if (track.framesAlive === 1) shouldAnnounce = true;
                    if (track.lastAnnouncedTime !== 0 && Date.now() - track.lastAnnouncedTime >= 2000) shouldAnnounce = true;
                    if (track.lastDistanceRaw !== null && Math.abs(currentDistRaw - track.lastDistanceRaw) >= 1.0) shouldAnnounce = true;

                    // Strict Cooldown Gate (No Exceptions after first frame)
                    if (track.framesAlive > 1 && track.lastAnnouncedTime !== 0 && Date.now() - track.lastAnnouncedTime < 2000) {
                        shouldAnnounce = false;
                    }

                    if (shouldAnnounce) {
                        frameAudioEvents.push({ type: 'obj', text: `${label} detected ${dist} ahead` });
                        track.lastAnnouncedTime = Date.now();
                        track.lastDistanceRaw = currentDistRaw;
                    }

                    const pct = (p.score * 100).toFixed(0) + "%";
                    domCards.push(createResultCard(`${label} [${dist}]`, pct, "OBJECT", "#10b981"));

                    ctx.strokeStyle = "rgba(16, 185, 129, 0.8)"; ctx.lineWidth = 2;
                    ctx.strokeRect(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]);
                });

                // Garbage Collect lost Object Tracks after 1 second
                for (let [uid, track] of liveTrackedObjs.entries()) {
                    if (nowObjRaw - track.lastSeenNow > LIVE_TRACK_TIMEOUT) {
                        liveTrackedObjs.delete(uid);
                    }
                }
            } catch (e) { }
        }

        // 4. Text Detection 
        if (toggleTextDet.checked) {
            // UI Update
            if (isTextCurrentlyVisible) {
                domCards.push(createResultCard("Text Present", "Confirmed", "TEXT AWARENESS", "#ffffff"));
            } else {
                domCards.push(createResultCard("No text detected", "-", "TEXT AWARENESS", "#555555"));
            }

            if (now - lastTextScan > 1000) {
                lastTextScan = now;
                const off = document.createElement('canvas'); off.width = width; off.height = height;
                const offCtx = off.getContext('2d', { willReadFrequently: true });

                // Fast Canvas-Level Preprocessing (Blur + Contrast)
                offCtx.filter = 'contrast(200%) grayscale(100%) blur(0.5px)';
                offCtx.drawImage(video, 0, 0, width, height);
                offCtx.filter = 'none';

                try {
                    // Manual Adaptive Thresholding (Pixel iteration)
                    const imgData = offCtx.getImageData(0, 0, width, height);
                    const d = imgData.data;
                    for (let i = 0; i < d.length; i += 4) {
                        const r = d[i], g = d[i + 1], b = d[i + 2];
                        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                        const thresh = gray > 120 ? 255 : 0; // Binarization
                        d[i] = d[i + 1] = d[i + 2] = thresh;
                    }
                    offCtx.putImageData(imgData, 0, 0);
                } catch (e) { }

                // Fire & forget Tesseract so it doesn't block the next frame's Object/Face pipeline
                Tesseract.recognize(off, 'eng').then(({ data }) => {
                    const textStr = data.text.replace(/[\n\r\s]/g, '').trim();
                    if (data.confidence >= 60 && textStr.length >= 3 && /[a-zA-Z0-9]/.test(textStr)) {
                        textConsecutiveFrames++;
                        if (textConsecutiveFrames >= 3) {
                            isTextCurrentlyVisible = true;
                            if (Date.now() - lastTextAlertTime > 5000) {
                                frameAudioEvents.push({ type: 'text', text: "Text ahead" });
                                lastTextAlertTime = Date.now();
                            }
                        }
                    } else {
                        textConsecutiveFrames = 0;
                        isTextCurrentlyVisible = false;
                    }
                }).catch(e => {
                    textConsecutiveFrames = 0;
                    isTextCurrentlyVisible = false;
                });
            }
        }

        // Render entire stack of Cards to Sidebar simultaneously
        if (domCards.length > 0) renderPredictions(domCards);
        else renderPredictions([]); // clear if nothing detected

        // ---- GLOBAL AUDIO EVENT BATCHING ----
        // 1. Sort by Priority (Faces > Objects > Text)
        // 2. Combine into a single flowing utterance
        if (frameAudioEvents.length > 0) {
            frameAudioEvents.sort((a, b) => {
                const ranks = { 'face': 1, 'obj': 2, 'text': 3 };
                return ranks[a.type] - ranks[b.type];
            });
            const sentence = frameAudioEvents.map(e => e.text).join('. ');
            enqueueAudio(sentence, 'combined');
        }

        isPipelineRunning = false;
    }, 100);
}

// ---- DASHBOARD TOGGLES ----
tabLive.addEventListener('click', () => setMode('live'));
tabMemory.addEventListener('click', () => setMode('memory'));

function setMode(mode) {
    activeMode = mode;
    if (mode === 'live') {
        tabLive.classList.add('active'); tabMemory.classList.remove('active');
        modeLive.classList.add('active'); modeMemory.classList.remove('active');
    } else {
        tabMemory.classList.add('active'); tabLive.classList.remove('active');
        modeMemory.classList.add('active'); modeLive.classList.remove('active');
        resultsPanel.innerHTML = '<div class="empty-state">Upload an image to process.</div>';
        renderFaceDatabase(); // Render the face database when switching to memory mode
        memoryWebcam.play().catch(e => { });
    }
}

// ---- IDENTITY MEMORY BANK ----
function renderFaceDatabase() {
    facesDatabase.innerHTML = '';
    if (labeledFaceDescriptors.length === 0) {
        facesDatabase.innerHTML = '<div class="empty-state">Memory bank empty.</div>';
        return;
    }
    labeledFaceDescriptors.forEach((desc, index) => {
        const div = document.createElement('div');
        div.style.cssText = "background: var(--bg-tertiary); border: 1px solid var(--border); padding: 12px; border-radius: 8px; display: flex; flex-direction: column; transition: all 0.2s; align-items: center;";
        div.onmouseover = () => div.style.borderColor = "var(--text-muted)";
        div.onmouseout = () => div.style.borderColor = "var(--border)";

        div.innerHTML = `
            <img src="${desc.uiImage || '/vite.svg'}" style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; margin-bottom: 12px; border: 1px solid var(--border);">
            <div style="font-size: 0.95rem; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; width: 100%; text-align: center; margin-bottom: 12px; color: var(--text-primary);">${desc.label}</div>
            <button class="btn-dangerous remove-face" data-index="${index}" style="padding: 6px 12px; font-size: 0.8rem; border-radius: 4px; border: 1px solid rgba(255,85,85,0.3); background: rgba(255,85,85,0.1); color: #ff5555; cursor: pointer; width: 100%; transition: 0.2s;">Forget</button>
        `;
        facesDatabase.appendChild(div);
    });

    document.querySelectorAll('.remove-face').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = e.target.getAttribute('data-index');
            labeledFaceDescriptors.splice(idx, 1);
            if (labeledFaceDescriptors.length > 0) faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.55);
            else faceMatcher = null;
            renderFaceDatabase();
        });
    });
}

async function registerFace(imageSource, label, uiImage) {
    sysStatus.innerText = "Extracting Features...";
    const detection = await faceapi.detectSingleFace(imageSource).withFaceLandmarks().withFaceDescriptor();
    if (!detection) {
        alert("No face detected in the image/feed. Please try again.");
    } else {
        // N-Shot Support: Append to existing descriptors instead of overwriting
        let existingIndex = labeledFaceDescriptors.findIndex(fd => fd.label === label);
        if (existingIndex !== -1) {
            labeledFaceDescriptors[existingIndex].descriptors.push(detection.descriptor);
            alert(`Added new facial vector snap to existing identity: ${label}!`);
        } else {
            const sd = new faceapi.LabeledFaceDescriptors(label, [detection.descriptor]);
            sd.uiImage = uiImage; // Save the first image as the display profile
            labeledFaceDescriptors.push(sd);
            alert(`Successfully registered new identity: ${label}!`);
        }

        faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.55);
        faceName.value = '';
        renderFaceDatabase();
    }
    sysStatus.innerText = "System Ready";
}

// Capture from Webcam
btnCaptureCamera.addEventListener('click', () => {
    const name = faceName.value.trim();
    if (!name) return alert("Please enter a name first.");
    if (!modelsLoaded) return alert("Models still loading.");

    btnCaptureCamera.disabled = true;
    btnCaptureCamera.innerText = "Scanning Face...";
    sysStatus.innerText = "Training Memory (0/10) - Move head slightly!";

    // Create UI Image from the very first frame
    const canvas = document.createElement('canvas');
    canvas.width = memoryWebcam.videoWidth; canvas.height = memoryWebcam.videoHeight;
    canvas.getContext('2d').drawImage(memoryWebcam, 0, 0);
    const uiImage = canvas.toDataURL('image/jpeg');

    let captureCount = 0;
    let newDescriptors = [];

    const euclideanDistance = (arr1, arr2) => {
        let sum = 0;
        for (let i = 0; i < arr1.length; i++) sum += Math.pow(arr1[i] - arr2[i], 2);
        return Math.sqrt(sum);
    };

    const trainingInterval = setInterval(async () => {
        captureCount++;

        let instruction = "Move head slightly!";
        const p = newDescriptors.length;
        if (p < 4) instruction = "Look straight at the camera";
        else if (p < 8) instruction = "Turn your head slightly left";
        else if (p < 12) instruction = "Turn your head slightly right";
        else if (p < 16) instruction = "Tilt your head slightly up";
        else instruction = "Tilt your head slightly down";

        sysStatus.innerText = `Training Identity (${p}/20) - ${instruction}`;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = memoryWebcam.videoWidth; tempCanvas.height = memoryWebcam.videoHeight;
        tempCanvas.getContext('2d').drawImage(memoryWebcam, 0, 0);

        try {
            const detection = await faceapi.detectSingleFace(tempCanvas).withFaceLandmarks().withFaceDescriptor();
            if (detection) {
                // Uniqueness Filter: Check if Face Angle/Lighting actually changed
                let isUnique = true;
                if (newDescriptors.length > 0) {
                    const dist = euclideanDistance(detection.descriptor, newDescriptors[newDescriptors.length - 1]);
                    if (dist < 0.05) isUnique = false; // Identical geometric frame, ignore
                }

                if (isUnique) newDescriptors.push(detection.descriptor);
            }
        } catch (e) { }

        // Finish when we successfully capture 20 unique geometric angles, or time out after 80 attempts (~16 seconds)
        if (newDescriptors.length >= 20 || captureCount >= 80) {
            clearInterval(trainingInterval);

            if (newDescriptors.length === 0) {
                alert("No faces detected during training! Please ensure you are visible and try again.");
            } else {
                let existingIndex = labeledFaceDescriptors.findIndex(fd => fd.label === name);
                if (existingIndex !== -1) {
                    labeledFaceDescriptors[existingIndex].descriptors.push(...newDescriptors);
                    alert(`Successfully added ${newDescriptors.length} dynamic structural vectors to ${name}!`);
                } else {
                    const sd = new faceapi.LabeledFaceDescriptors(name, newDescriptors);
                    sd.uiImage = uiImage;
                    labeledFaceDescriptors.push(sd);
                    alert(`Successfully registered Identity: ${name} with ${newDescriptors.length} dynamic structural vectors!`);
                }

                faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.55);
                faceName.value = '';
                renderFaceDatabase();
            }

            btnCaptureCamera.disabled = false;
            btnCaptureCamera.innerText = "Snap from Camera";
            sysStatus.innerText = "System Ready";
        }
    }, 200); // Check camera every 200ms
});

// Capture from Uploaded Photo
faceUpload.addEventListener('change', async (e) => {
    const name = faceName.value.trim();
    if (!name) {
        faceUpload.value = "";
        return alert("Please enter a name first.");
    }
    if (!modelsLoaded) return alert("Models still loading.");

    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = async () => {
            await registerFace(img, name, url);
            faceUpload.value = "";
        };
        img.src = url;
    }
});

initSystem();
