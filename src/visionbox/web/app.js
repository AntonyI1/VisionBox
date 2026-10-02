(() => {
    const STATUS_INTERVAL = 2000;
    const CAMERAS_INTERVAL = 2000;
    const PAGE_SIZE = 50;

    let currentView = 'live';
    let cameras = [];          // [{name, fps, recording, state, connected, ...}]
    let cameraNames = [];      // sorted list
    let eventsOffset = 0;
    let eventsTotal = 0;
    let eventsFilter = '';
    let modalEventId = null;

    // ---------- Shared helpers ----------

    function fmtUptime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return h + 'h ' + m + 'm';
        return m + 'm';
    }

    function fmtTime(iso) {
        if (!iso) return '--';
        const s = /[Z+-]\d{0,4}$/.test(iso) ? iso : iso + 'Z';
        const d = new Date(s);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
            ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function metaRow(label, value) {
        return '<div class="meta-row"><span class="meta-label">' + label +
            '</span><span class="meta-value">' + value + '</span></div>';
    }

    function option(value, text) {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = text;
        return opt;
    }

    function populateCamSelect(sel, includeAll, currentVal) {
        const prev = currentVal !== undefined ? currentVal : sel.value;
        sel.innerHTML = '';
        sel.appendChild(option('', includeAll ? 'All cameras' : 'Select camera...'));
        cameraNames.forEach(n => sel.appendChild(option(n, n)));
        if (prev && cameraNames.includes(prev)) sel.value = prev;
    }

    // ---------- Navigation ----------

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            if (view === currentView) return;
            document.querySelector('.nav-btn.active').classList.remove('active');
            btn.classList.add('active');
            document.querySelector('.view.active').classList.remove('active');
            document.getElementById('view-' + view).classList.add('active');
            currentView = view;
            onViewChange(view);
        });
    });

    function onViewChange(view) {
        closeFocus();
        if (view !== 'live') stopGridStreams();
        if (view === 'live') {
            renderLiveGrid();
        } else if (view === 'events') {
            populateCamSelect(document.getElementById('events-cam-filter'), true, eventsFilter);
            resetEvents();
            loadEvents();
        } else if (view === 'zones') {
            populateCamSelect(document.getElementById('zones-cam-select'), false);
            if (zonesCurrentCam) loadZonesView();
        } else if (view === 'review') {
            populateCamSelect(document.getElementById('review-cam-select'), true, reviewCam);
            loadReviewClasses();   // '' = all cameras (default)
        } else if (view === 'training') {
            loadTrainingClasses();
        }
    }

    // ---------- Global status polling ----------

    function pollStatus() {
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status-cams').textContent =
                    data.connected_cameras + '/' + data.cameras + ' cameras';
                document.getElementById('status-recording').innerHTML =
                    data.recording_cameras > 0
                        ? '<span class="rec-dot"></span>' + data.recording_cameras + ' recording'
                        : '<span class="rec-dot idle"></span>idle';
                document.getElementById('status-events').textContent = data.event_count + ' events';
                if (data.storage) {
                    document.getElementById('status-storage').textContent =
                        data.storage.recordings_human + ' / ' + data.storage.disk_total_human;
                }
                document.getElementById('status-uptime').textContent = fmtUptime(data.uptime);
            })
            .catch(() => {});
    }

    function pollCameras() {
        fetch('/api/cameras')
            .then(r => r.json())
            .then(data => {
                const newNames = data.map(c => c.name).sort();
                const namesChanged =
                    cameraNames.length !== newNames.length ||
                    cameraNames.some((n, i) => n !== newNames[i]);
                cameras = data;
                cameraNames = newNames;
                if (namesChanged && currentView === 'live') renderLiveGrid();
                if (currentView === 'live') updateLiveStats();
                updateFocusStats();
            })
            .catch(() => {});
    }

    let statusTimer = null, camerasTimer = null;

    function startPolling() {
        if (statusTimer === null) statusTimer = setInterval(pollStatus, STATUS_INTERVAL);
        if (camerasTimer === null) camerasTimer = setInterval(pollCameras, CAMERAS_INTERVAL);
    }

    function stopPolling() {
        clearInterval(statusTimer); statusTimer = null;
        clearInterval(camerasTimer); camerasTimer = null;
    }

    function onVisibility() {
        if (document.hidden) {
            stopPolling();
            stopGridStreams();
            if (focusedCam) focusImg.src = '';
        } else {
            startPolling();
            pollStatus();
            pollCameras();
            if (focusedCam) {
                focusImg.src = '/api/cameras/' + encodeURIComponent(focusedCam) + '/stream';
            } else if (gridShouldStream()) {
                startGridStreams();
            }
        }
    }
    document.addEventListener('visibilitychange', onVisibility);

    pollStatus();
    pollCameras();
    if (!document.hidden) startPolling();

    // ---------- Live grid ----------

    const liveGrid = document.getElementById('live-grid');
    const liveEmpty = document.getElementById('live-empty');
    const tileRefs = new Map();

    // Each open /stream holds a server worker thread; only stream a tile while it can be seen.
    function gridShouldStream() {
        return currentView === 'live' && !focusedCam && !document.hidden;
    }

    function startGridStreams() {
        tileRefs.forEach((r, name) => {
            const url = '/api/cameras/' + encodeURIComponent(name) + '/stream';
            if (r.img.getAttribute('src') !== url) r.img.src = url;
        });
    }

    function stopGridStreams() {
        tileRefs.forEach(r => { r.img.src = ''; });
    }

    function renderLiveGrid() {
        liveGrid.innerHTML = '';
        tileRefs.clear();
        if (cameras.length === 0) {
            liveEmpty.style.display = 'block';
            return;
        }
        liveEmpty.style.display = 'none';
        const cols = cameras.length <= 1 ? 1 : cameras.length <= 4 ? 2 : 3;
        liveGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        const frag = document.createDocumentFragment();
        cameras.forEach(cam => {
            const tile = document.createElement('div');
            tile.className = 'live-tile';
            tile.dataset.cam = cam.name;
            tile.innerHTML =
                '<div class="tile-stream-wrap">' +
                '<img class="tile-stream" alt="">' +
                '<div class="tile-offline-msg">Offline</div>' +
                '</div>' +
                '<div class="tile-bar">' +
                '<span class="tile-dot"></span>' +
                '<span class="tile-name">' + cam.name + '</span>' +
                '<span class="tile-fps"></span>' +
                '</div>';
            tile.addEventListener('click', () => openFocus(cam.name));
            tileRefs.set(cam.name, {
                img: tile.querySelector('.tile-stream'),
                dot: tile.querySelector('.tile-dot'),
                fps: tile.querySelector('.tile-fps'),
                wrap: tile.querySelector('.tile-stream-wrap'),
            });
            frag.appendChild(tile);
        });
        liveGrid.appendChild(frag);
        if (gridShouldStream()) startGridStreams();
        updateLiveStats();
    }

    function updateLiveStats() {
        cameras.forEach(cam => {
            const r = tileRefs.get(cam.name);
            if (!r) return;
            r.dot.className = 'tile-dot ' + (cam.recording ? 'recording'
                : cam.connected ? 'connected' : 'offline');
            r.fps.textContent = cam.connected ? '' : (cam.last_error || 'offline');
            r.wrap.classList.toggle('disconnected', !cam.connected);
        });
    }

    // Focus mode (single-camera full view)
    const focusEl = document.getElementById('live-focus');
    const focusImg = document.getElementById('focus-stream');
    const focusName = document.getElementById('focus-name');
    const focusStats = document.getElementById('focus-stats');
    let focusedCam = null;

    function openFocus(name) {
        stopGridStreams();
        focusedCam = name;
        focusImg.src = '/api/cameras/' + encodeURIComponent(name) + '/stream';
        focusName.textContent = name;
        focusEl.style.display = 'flex';
        updateFocusStats();
    }

    function updateFocusStats() {
        if (!focusedCam) return;
        const cam = cameras.find(c => c.name === focusedCam);
        if (!cam) return;
        focusStats.textContent = !cam.connected ? 'offline'
            : cam.recording ? 'recording' : '';
    }

    function closeFocus() {
        if (!focusedCam) return;
        focusedCam = null;
        focusImg.src = '';
        focusEl.style.display = 'none';
        if (gridShouldStream()) startGridStreams();
    }
    document.getElementById('focus-close').addEventListener('click', closeFocus);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && focusedCam) closeFocus();
    });

    // ---------- Events ----------

    const eventsGrid = document.getElementById('events-grid');
    const eventsEmpty = document.getElementById('events-empty');
    const loadMoreBtn = document.getElementById('load-more');
    const eventsFilterEl = document.getElementById('events-cam-filter');

    eventsFilterEl.addEventListener('change', () => {
        eventsFilter = eventsFilterEl.value;
        resetEvents();
        loadEvents();
    });

    function resetEvents() {
        eventsOffset = 0;
        eventsTotal = 0;
        eventsGrid.innerHTML = '';
        loadMoreBtn.style.display = 'none';
        eventsEmpty.style.display = 'none';
    }

    function loadEvents() {
        const camQ = eventsFilter ? '&camera=' + encodeURIComponent(eventsFilter) : '';
        fetch('/api/events?limit=' + PAGE_SIZE + '&offset=' + eventsOffset + camQ)
            .then(r => r.json())
            .then(data => {
                eventsTotal = data.total;
                if (data.events.length === 0 && eventsOffset === 0) {
                    eventsEmpty.style.display = 'block';
                    loadMoreBtn.style.display = 'none';
                    return;
                }
                eventsEmpty.style.display = 'none';
                const frag = document.createDocumentFragment();
                data.events.forEach(ev => frag.appendChild(createCard(ev)));
                eventsGrid.appendChild(frag);
                eventsOffset += data.events.length;
                loadMoreBtn.style.display = eventsOffset < eventsTotal ? 'block' : 'none';
            })
            .catch(() => {});
    }

    loadMoreBtn.addEventListener('click', loadEvents);

    function createCard(ev) {
        const card = document.createElement('div');
        card.className = 'event-card';
        card.dataset.eventId = ev.event_id;
        const thumbEl = ev.thumbnail
            ? '<img class="thumb" src="/api/events/' + ev.event_id + '/thumbnail" loading="lazy" alt="">'
            : '<div class="thumb-placeholder">No thumbnail</div>';
        const time = fmtTime(ev.start_time);
        const dur = ev.duration ? ev.duration.toFixed(1) + 's' : '--';
        const dets = ev.detection_count || 0;
        const label = ev.top_label || '';
        const cam = ev.camera || '';
        card.innerHTML = thumbEl +
            '<div class="card-info">' +
            '<div class="card-time">' + time + '</div>' +
            '<div class="card-details">' +
            (cam ? '<span class="card-cam">' + cam + '</span>' : '') +
            '<span>' + dur + '</span>' +
            '<span>' + dets + ' det</span>' +
            '</div>' +
            (label ? '<span class="card-label">' + label + '</span>' : '') +
            '</div>';
        card.addEventListener('click', () => openModal(ev.event_id));
        return card;
    }

    // ---------- Event modal ----------

    const modalOverlay = document.getElementById('modal-overlay');
    const modalTitle = document.getElementById('modal-title');
    const modalVideo = document.getElementById('modal-video');
    const modalMeta = document.getElementById('modal-meta');

    function openModal(eventId) {
        modalEventId = eventId;
        fetch('/api/events/' + eventId)
            .then(r => r.json())
            .then(ev => {
                modalTitle.textContent = (ev.camera || 'Event') + '  ·  ' + (ev.event_id || '');
                const clipType = ev.clean_clip ? 'clean' : 'annotated';
                setClipSource(eventId, clipType);
                document.querySelectorAll('.clip-btn').forEach(b => {
                    b.classList.toggle('active', b.dataset.type === clipType);
                });
                modalMeta.innerHTML =
                    metaRow('Camera', ev.camera || '--') +
                    metaRow('Start', fmtTime(ev.start_time)) +
                    metaRow('End', fmtTime(ev.end_time)) +
                    metaRow('Duration', (ev.duration || 0).toFixed(1) + 's') +
                    metaRow('Detections', ev.detection_count || 0) +
                    metaRow('Top Label', ev.top_label || '--') +
                    metaRow('Clean Clip', ev.clean_clip ? 'Yes' : 'No');
                modalOverlay.style.display = 'flex';
            })
            .catch(() => {});
    }

    function setClipSource(eventId, type) {
        modalVideo.src = '/api/events/' + eventId + '/clip/' + type;
        modalVideo.load();
    }

    document.querySelectorAll('.clip-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!modalEventId) return;
            document.querySelector('.clip-btn.active').classList.remove('active');
            btn.classList.add('active');
            setClipSource(modalEventId, btn.dataset.type);
        });
    });

    document.getElementById('modal-close').addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', e => {
        if (e.target === modalOverlay) closeModal();
    });

    function closeModal() {
        modalOverlay.style.display = 'none';
        modalVideo.pause();
        modalVideo.src = '';
        modalEventId = null;
    }

    document.getElementById('modal-delete').addEventListener('click', () => {
        if (!modalEventId) return;
        if (!confirm('Delete this event and its recordings?')) return;
        fetch('/api/events/' + modalEventId, { method: 'DELETE' })
            .then(r => r.json())
            .then(() => {
                const card = eventsGrid.querySelector(
                    '[data-event-id="' + modalEventId + '"]'
                );
                if (card) card.remove();
                closeModal();
                eventsTotal--;
            })
            .catch(() => {});
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeModal();
    });

    // ---------- Zones (per camera) ----------

    const zonesCamSelect = document.getElementById('zones-cam-select');
    const zoneCanvas = document.getElementById('zone-canvas');
    const zoneCtx = zoneCanvas.getContext('2d');
    const zoneList = document.getElementById('zone-list');
    const zoneForm = document.getElementById('zone-form');
    const zoneNameInput = document.getElementById('zone-name');
    const zoneSaveBtn = document.getElementById('zone-save-btn');
    const zoneHint = document.getElementById('zone-hint');
    const zoneEmpty = document.getElementById('zone-empty');

    let zones = [];
    let drawingPoints = [];
    let isDrawing = false;
    let snapshotImg = null;
    let zoneType = 'include';
    let zonesCurrentCam = '';
    let zoneFramePending = false;
    let zoneCursor = null;

    zonesCamSelect.addEventListener('change', () => {
        zonesCurrentCam = zonesCamSelect.value;
        loadZonesView();
    });

    function loadZonesView() {
        if (!zonesCurrentCam) {
            zoneEmpty.style.display = 'block';
            zoneCanvas.style.display = 'none';
            return;
        }
        const img = new Image();
        img.onload = function () {
            snapshotImg = img;
            zoneCanvas.width = img.naturalWidth;
            zoneCanvas.height = img.naturalHeight;
            zoneCanvas.style.display = 'block';
            zoneEmpty.style.display = 'none';
            fetch('/api/cameras/' + encodeURIComponent(zonesCurrentCam) + '/zones')
                .then(r => r.json())
                .then(data => {
                    zones = data;
                    renderZoneList();
                    drawZoneCanvas();
                });
        };
        img.onerror = function () {
            zoneEmpty.textContent = 'Camera not connected — cannot get snapshot.';
            zoneEmpty.style.display = 'block';
            zoneCanvas.style.display = 'none';
        };
        img.src = '/api/cameras/' + encodeURIComponent(zonesCurrentCam) + '/snapshot?' + Date.now();
    }

    function drawZoneCanvas() {
        if (!snapshotImg) return;
        const w = zoneCanvas.width, h = zoneCanvas.height;
        zoneCtx.clearRect(0, 0, w, h);
        zoneCtx.drawImage(snapshotImg, 0, 0, w, h);
        zones.forEach(z => {
            const pts = z.points.map(p => [p[0] * w, p[1] * h]);
            const color = z.type === 'include' ? 'rgba(123,165,94,' : 'rgba(221,92,70,';
            zoneCtx.beginPath();
            pts.forEach((p, i) => i === 0 ? zoneCtx.moveTo(p[0], p[1]) : zoneCtx.lineTo(p[0], p[1]));
            zoneCtx.closePath();
            zoneCtx.fillStyle = color + '0.25)';
            zoneCtx.fill();
            zoneCtx.strokeStyle = color + '0.9)';
            zoneCtx.lineWidth = 2;
            zoneCtx.stroke();
            const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
            const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
            zoneCtx.font = '14px sans-serif';
            zoneCtx.fillStyle = '#fff';
            zoneCtx.textAlign = 'center';
            zoneCtx.fillText(z.name, cx, cy);
        });
        if (drawingPoints.length > 0) {
            const pts = drawingPoints.map(p => [p[0] * w, p[1] * h]);
            const color = zoneType === 'include' ? 'rgba(123,165,94,' : 'rgba(221,92,70,';
            zoneCtx.beginPath();
            pts.forEach((p, i) => i === 0 ? zoneCtx.moveTo(p[0], p[1]) : zoneCtx.lineTo(p[0], p[1]));
            zoneCtx.strokeStyle = color + '0.9)';
            zoneCtx.lineWidth = 2;
            zoneCtx.setLineDash([6, 4]);
            zoneCtx.stroke();
            zoneCtx.setLineDash([]);
            pts.forEach(p => {
                zoneCtx.beginPath();
                zoneCtx.arc(p[0], p[1], 5, 0, Math.PI * 2);
                zoneCtx.fillStyle = color + '0.9)';
                zoneCtx.fill();
            });
        }
    }

    function canvasCoords(e) {
        const rect = zoneCanvas.getBoundingClientRect();
        return [
            (e.clientX - rect.left) / rect.width,
            (e.clientY - rect.top) / rect.height,
        ];
    }

    zoneCanvas.addEventListener('click', e => {
        if (!isDrawing) return;
        const [nx, ny] = canvasCoords(e);
        if (drawingPoints.length >= 3) {
            const [fx, fy] = drawingPoints[0];
            const w = zoneCanvas.width, h = zoneCanvas.height;
            const dist = Math.hypot((nx - fx) * w, (ny - fy) * h);
            if (dist < 15) {
                zoneSaveBtn.disabled = false;
                isDrawing = false;
                zoneHint.style.display = 'none';
                drawZoneCanvas();
                return;
            }
        }
        drawingPoints.push([nx, ny]);
        drawZoneCanvas();
    });

    zoneCanvas.addEventListener('mousemove', e => {
        if (!isDrawing || drawingPoints.length === 0) return;
        zoneCursor = canvasCoords(e);
        if (zoneFramePending) return;
        zoneFramePending = true;
        requestAnimationFrame(() => {
            zoneFramePending = false;
            if (!isDrawing || drawingPoints.length === 0 || !zoneCursor) return;
            drawZoneCanvas();
            const [nx, ny] = zoneCursor;
            const w = zoneCanvas.width, h = zoneCanvas.height;
            const last = drawingPoints[drawingPoints.length - 1];
            zoneCtx.beginPath();
            zoneCtx.moveTo(last[0] * w, last[1] * h);
            zoneCtx.lineTo(nx * w, ny * h);
            zoneCtx.strokeStyle = 'rgba(255,255,255,0.5)';
            zoneCtx.lineWidth = 1;
            zoneCtx.setLineDash([4, 4]);
            zoneCtx.stroke();
            zoneCtx.setLineDash([]);
        });
    });

    document.getElementById('zone-add-btn').addEventListener('click', () => {
        if (!zonesCurrentCam) return;
        isDrawing = true;
        drawingPoints = [];
        zoneNameInput.value = '';
        zoneSaveBtn.disabled = true;
        zoneForm.style.display = 'flex';
        zoneHint.style.display = 'block';
        document.getElementById('zone-add-btn').style.display = 'none';
    });

    document.getElementById('zone-cancel-btn').addEventListener('click', cancelZoneDrawing);

    function cancelZoneDrawing() {
        isDrawing = false;
        drawingPoints = [];
        zoneForm.style.display = 'none';
        zoneHint.style.display = 'none';
        document.getElementById('zone-add-btn').style.display = 'block';
        drawZoneCanvas();
    }

    document.querySelectorAll('.zone-type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelector('.zone-type-btn.active').classList.remove('active');
            btn.classList.add('active');
            zoneType = btn.dataset.type;
            drawZoneCanvas();
        });
    });

    zoneSaveBtn.addEventListener('click', () => {
        const name = zoneNameInput.value.trim();
        if (!name || drawingPoints.length < 3 || !zonesCurrentCam) return;
        fetch('/api/cameras/' + encodeURIComponent(zonesCurrentCam) + '/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, type: zoneType, points: drawingPoints }),
        })
        .then(r => r.json())
        .then(() => {
            zones.push({ name, type: zoneType, points: drawingPoints.slice() });
            cancelZoneDrawing();
            renderZoneList();
            drawZoneCanvas();
        });
    });

    function renderZoneList() {
        zoneList.innerHTML = '';
        zones.forEach(z => {
            const item = document.createElement('div');
            item.className = 'zone-item';
            item.innerHTML =
                '<span class="zone-badge ' + z.type + '">' + z.type + '</span>' +
                '<span class="zone-name">' + z.name + '</span>' +
                '<button class="zone-del-btn" title="Delete">&times;</button>';
            item.querySelector('.zone-del-btn').addEventListener('click', () => {
                fetch('/api/cameras/' + encodeURIComponent(zonesCurrentCam) +
                      '/zones/' + encodeURIComponent(z.name), { method: 'DELETE' })
                    .then(() => {
                        zones = zones.filter(zz => zz.name !== z.name);
                        renderZoneList();
                        drawZoneCanvas();
                    });
            });
            zoneList.appendChild(item);
        });
    }

    // ---------- Review (per camera, per class) ----------

    const reviewCamSelect = document.getElementById('review-cam-select');
    const reviewClassSelect = document.getElementById('review-class-select');
    const reviewImage = document.getElementById('review-image');
    const reviewEmpty = document.getElementById('review-empty');
    const reviewMeta = document.getElementById('review-meta');
    const reviewProgress = document.getElementById('review-progress');
    let reviewCam = '';        // '' = all cameras
    let reviewClass = '';
    let reviewCropCam = '';     // camera that owns the crop currently shown
    let reviewOffset = 0;
    let reviewTotal = 0;

    reviewCamSelect.addEventListener('change', () => {
        reviewCam = reviewCamSelect.value;
        reviewClass = '';
        reviewClassSelect.innerHTML = '<option value="">Select class...</option>';
        clearReview();
        loadReviewClasses();
    });

    function loadReviewClasses() {
        const url = reviewCam
            ? '/api/cameras/' + encodeURIComponent(reviewCam) + '/review/classes'
            : '/api/review/classes';
        fetch(url)
            .then(r => r.json())
            .then(classes => {
                reviewClassSelect.innerHTML = '<option value="">Select class...</option>';
                classes.forEach(c => reviewClassSelect.appendChild(option(c.name, c.name + ' (' + c.count + ')')));
                if (reviewClass) reviewClassSelect.value = reviewClass;
            });
    }

    reviewClassSelect.addEventListener('change', () => {
        reviewClass = reviewClassSelect.value;
        reviewOffset = 0;
        if (reviewClass) loadReviewCrop();
        else clearReview();
    });

    function loadReviewCrop() {
        if (!reviewClass) return;
        const url = (reviewCam
            ? '/api/cameras/' + encodeURIComponent(reviewCam) + '/review/' + encodeURIComponent(reviewClass)
            : '/api/review/' + encodeURIComponent(reviewClass)) + '?offset=' + reviewOffset;
        fetch(url)
            .then(r => r.json())
            .then(data => {
                reviewTotal = data.total;
                if (!data.crop || data.total === 0) {
                    clearReview();
                    reviewProgress.textContent = 'No crops';
                    loadReviewClasses();
                    return;
                }
                reviewOffset = data.offset;
                const crop = data.crop;
                reviewCropCam = crop.camera || reviewCam;   // owns image/approve/reject
                reviewImage.src = '/api/cameras/' + encodeURIComponent(reviewCropCam) +
                    '/review/' + encodeURIComponent(reviewClass) +
                    '/' + encodeURIComponent(crop.filename) + '/image';
                reviewImage.dataset.filename = crop.filename;
                reviewImage.style.display = 'block';
                reviewEmpty.style.display = 'none';
                reviewProgress.textContent = (reviewOffset + 1) + ' of ' + reviewTotal;
                let meta = '';
                if (!reviewCam && crop.camera) meta += metaRow('Camera', crop.camera);
                if (crop.track_id != null) meta += metaRow('Track', '#' + crop.track_id);
                if (crop.confidence != null) meta += metaRow('Confidence', Math.round(crop.confidence * 100) + '%');
                if (crop.timestamp) meta += metaRow('Time', fmtTime(crop.timestamp));
                reviewMeta.innerHTML = meta;
            });
    }

    function clearReview() {
        reviewImage.style.display = 'none';
        reviewImage.src = '';
        reviewImage.dataset.filename = '';
        reviewCropCam = '';
        reviewEmpty.style.display = 'block';
        reviewMeta.innerHTML = '';
        reviewProgress.textContent = '';
    }

    function reviewAction(action) {
        const filename = reviewImage.dataset.filename;
        const cam = reviewCropCam || reviewCam;
        if (!filename || !reviewClass || !cam) return;
        const url = '/api/cameras/' + encodeURIComponent(cam) +
                    '/review/' + encodeURIComponent(reviewClass) +
                    '/' + encodeURIComponent(filename) + '/' + action;
        fetch(url, { method: 'POST' })
            .then(r => r.json())
            .then(() => {
                reviewTotal--;
                if (reviewTotal <= 0) {
                    clearReview();
                    reviewProgress.textContent = 'No crops';
                    loadReviewClasses();
                    return;
                }
                if (reviewOffset >= reviewTotal) reviewOffset = reviewTotal - 1;
                loadReviewCrop();
            });
    }

    document.getElementById('review-approve').addEventListener('click', () => reviewAction('approve'));
    document.getElementById('review-reject').addEventListener('click', () => reviewAction('reject'));
    document.getElementById('review-skip').addEventListener('click', () => {
        if (reviewOffset < reviewTotal - 1) { reviewOffset++; loadReviewCrop(); }
    });

    document.addEventListener('keydown', e => {
        if (currentView !== 'review') return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'a' || e.key === 'A') reviewAction('approve');
        else if (e.key === 'r' || e.key === 'R') reviewAction('reject');
        else if (e.key === 'ArrowRight') {
            if (reviewOffset < reviewTotal - 1) { reviewOffset++; loadReviewCrop(); }
        } else if (e.key === 'ArrowLeft') {
            if (reviewOffset > 0) { reviewOffset--; loadReviewCrop(); }
        }
    });

    // ---------- Training (global) ----------

    const trainingClassSelect = document.getElementById('training-class-select');
    const trainingImage = document.getElementById('training-image');
    const trainingEmpty = document.getElementById('training-empty');
    const trainingMeta = document.getElementById('training-meta');
    const trainingProgress = document.getElementById('training-progress');
    let trainingClass = '';
    let trainingOffset = 0;
    let trainingTotal = 0;

    function loadTrainingClasses() {
        fetch('/api/training/classes')
            .then(r => r.json())
            .then(classes => {
                trainingClassSelect.innerHTML = '<option value="">Select class...</option>';
                classes.forEach(c => trainingClassSelect.appendChild(option(c.name, c.name + ' (' + c.count + ')')));
                if (trainingClass) trainingClassSelect.value = trainingClass;
            });
    }

    trainingClassSelect.addEventListener('change', () => {
        trainingClass = trainingClassSelect.value;
        trainingOffset = 0;
        if (trainingClass) loadTrainingImage();
        else clearTraining();
    });

    function loadTrainingImage() {
        if (!trainingClass) return;
        fetch('/api/training/' + encodeURIComponent(trainingClass) + '?offset=' + trainingOffset)
            .then(r => r.json())
            .then(data => {
                trainingTotal = data.total;
                if (!data.image || data.total === 0) {
                    clearTraining();
                    trainingProgress.textContent = 'No images';
                    loadTrainingClasses();
                    return;
                }
                trainingOffset = data.offset;
                const img = data.image;
                trainingImage.src = '/api/training/' + encodeURIComponent(trainingClass) +
                    '/' + encodeURIComponent(img.filename) + '/image';
                trainingImage.dataset.filename = img.filename;
                trainingImage.style.display = 'block';
                trainingEmpty.style.display = 'none';
                trainingProgress.textContent = (trainingOffset + 1) + ' of ' + trainingTotal;
                let meta = '';
                if (img.track_id != null) meta += metaRow('Track', '#' + img.track_id);
                if (img.confidence != null) meta += metaRow('Confidence', Math.round(img.confidence * 100) + '%');
                if (img.timestamp) meta += metaRow('Time', fmtTime(img.timestamp));
                trainingMeta.innerHTML = meta;
            });
    }

    function clearTraining() {
        trainingImage.style.display = 'none';
        trainingImage.src = '';
        trainingImage.dataset.filename = '';
        trainingEmpty.style.display = 'block';
        trainingMeta.innerHTML = '';
        trainingProgress.textContent = '';
    }

    function deleteTrainingImage() {
        const filename = trainingImage.dataset.filename;
        if (!filename || !trainingClass) return;
        if (!confirm('Permanently delete this training image?')) return;
        fetch('/api/training/' + encodeURIComponent(trainingClass) + '/' +
            encodeURIComponent(filename), { method: 'DELETE' })
            .then(r => r.json())
            .then(() => {
                trainingTotal--;
                if (trainingTotal <= 0) {
                    clearTraining();
                    trainingProgress.textContent = 'No images';
                    loadTrainingClasses();
                    return;
                }
                if (trainingOffset >= trainingTotal) trainingOffset = trainingTotal - 1;
                loadTrainingImage();
            });
    }

    document.getElementById('training-delete').addEventListener('click', deleteTrainingImage);
    document.getElementById('training-prev').addEventListener('click', () => {
        if (trainingOffset > 0) { trainingOffset--; loadTrainingImage(); }
    });
    document.getElementById('training-next').addEventListener('click', () => {
        if (trainingOffset < trainingTotal - 1) { trainingOffset++; loadTrainingImage(); }
    });

    document.addEventListener('keydown', e => {
        if (currentView !== 'training') return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'd' || e.key === 'D') deleteTrainingImage();
        else if (e.key === 'ArrowRight') {
            if (trainingOffset < trainingTotal - 1) { trainingOffset++; loadTrainingImage(); }
        } else if (e.key === 'ArrowLeft') {
            if (trainingOffset > 0) { trainingOffset--; loadTrainingImage(); }
        }
    });

    // ---------- Service worker (PWA) ----------
    // Only activates on secure contexts (HTTPS / localhost); register() rejects elsewhere.

    if ('serviceWorker' in navigator) {
        const hadController = !!navigator.serviceWorker.controller;
        let reloaded = false;
        navigator.serviceWorker.addEventListener('controllerchange', () => {
            // An updated worker claimed the page (skipWaiting) — reload once for a fresh shell.
            if (!hadController || reloaded) return;
            reloaded = true;
            location.reload();
        });
        navigator.serviceWorker.register('/sw.js').catch(() => {});
    }

    // Initial render
    onViewChange('live');
})();
