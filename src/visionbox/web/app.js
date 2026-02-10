(() => {
    const POLL_INTERVAL = 2000;
    const PAGE_SIZE = 50;

    let currentView = 'live';
    let eventsOffset = 0;
    let eventsTotal = 0;
    let modalEventId = null;

    // Zone editor state
    let zones = [];
    let drawingPoints = [];
    let isDrawing = false;
    let snapshotImg = null;
    let zoneType = 'include';

    // DOM refs
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const statusFps = document.getElementById('status-fps');
    const statusEvents = document.getElementById('status-events');
    const statusStorage = document.getElementById('status-storage');
    const statusUptime = document.getElementById('status-uptime');
    const eventsGrid = document.getElementById('events-grid');
    const eventsEmpty = document.getElementById('events-empty');
    const loadMoreBtn = document.getElementById('load-more');
    const modalOverlay = document.getElementById('modal-overlay');
    const modalTitle = document.getElementById('modal-title');
    const modalVideo = document.getElementById('modal-video');
    const modalMeta = document.getElementById('modal-meta');
    const liveStream = document.getElementById('live-stream');

    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            if (view === currentView) return;

            document.querySelector('.nav-btn.active').classList.remove('active');
            btn.classList.add('active');

            document.querySelector('.view.active').classList.remove('active');
            document.getElementById('view-' + view).classList.add('active');

            currentView = view;

            if (view === 'live') {
                liveStream.src = '/api/stream';
            } else if (view === 'events') {
                liveStream.src = '';
                eventsOffset = 0;
                eventsGrid.innerHTML = '';
                loadEvents();
            } else if (view === 'zones') {
                liveStream.src = '';
                loadZonesView();
            } else if (view === 'review') {
                liveStream.src = '';
                loadReviewClasses();
            } else if (view === 'training') {
                liveStream.src = '';
                loadTrainingClasses();
            }
        });
    });

    // Status polling
    function pollStatus() {
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                statusDot.className = 'dot ' + data.state;
                statusText.textContent = data.state.toUpperCase();
                statusFps.textContent = data.fps + ' FPS';
                statusEvents.textContent = data.event_count + ' events';
                if (data.storage) {
                    statusStorage.textContent = data.storage.recordings_human +
                        ' / ' + data.storage.disk_total_human;
                }
                statusUptime.textContent = formatUptime(data.uptime);
            })
            .catch(() => {});
    }

    function formatUptime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return h + 'h ' + m + 'm';
        return m + 'm';
    }

    setInterval(pollStatus, POLL_INTERVAL);
    pollStatus();

    // Events
    function loadEvents() {
        fetch('/api/events?limit=' + PAGE_SIZE + '&offset=' + eventsOffset)
            .then(r => r.json())
            .then(data => {
                eventsTotal = data.total;

                if (data.events.length === 0 && eventsOffset === 0) {
                    eventsEmpty.style.display = 'block';
                    loadMoreBtn.style.display = 'none';
                    return;
                }

                eventsEmpty.style.display = 'none';
                data.events.forEach(ev => eventsGrid.appendChild(createCard(ev)));
                eventsOffset += data.events.length;
                loadMoreBtn.style.display =
                    eventsOffset < eventsTotal ? 'block' : 'none';
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

        const time = formatTime(ev.start_time);
        const dur = ev.duration ? ev.duration.toFixed(1) + 's' : '--';
        const dets = ev.detection_count || 0;
        const label = ev.top_label || '';

        card.innerHTML = thumbEl +
            '<div class="card-info">' +
            '<div class="card-time">' + time + '</div>' +
            '<div class="card-details">' +
            '<span>' + dur + '</span>' +
            '<span>' + dets + ' detections</span>' +
            '</div>' +
            (label ? '<span class="card-label">' + label + '</span>' : '') +
            '</div>';

        card.addEventListener('click', () => openModal(ev.event_id));
        return card;
    }

    function formatTime(iso) {
        if (!iso) return '--';
        const s = /[Z+-]\d{0,4}$/.test(iso) ? iso : iso + 'Z';
        const d = new Date(s);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
            ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    // Modal
    function openModal(eventId) {
        modalEventId = eventId;
        fetch('/api/events/' + eventId)
            .then(r => r.json())
            .then(ev => {
                modalTitle.textContent = 'Event ' + ev.event_id;

                const clipType = ev.clean_clip ? 'clean' : 'annotated';
                setClipSource(eventId, clipType);

                document.querySelectorAll('.clip-btn').forEach(b => {
                    b.classList.toggle('active', b.dataset.type === clipType);
                });

                modalMeta.innerHTML = metaRow('Start', formatTime(ev.start_time)) +
                    metaRow('End', formatTime(ev.end_time)) +
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

    function metaRow(label, value) {
        return '<div class="meta-row"><span class="meta-label">' + label +
            '</span><span class="meta-value">' + value + '</span></div>';
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
                statusEvents.textContent = eventsTotal + ' events';
            })
            .catch(() => {});
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeModal();
    });

    // Review
    const reviewClassSelect = document.getElementById('review-class-select');
    const reviewImage = document.getElementById('review-image');
    const reviewEmpty = document.getElementById('review-empty');
    const reviewMeta = document.getElementById('review-meta');
    const reviewProgress = document.getElementById('review-progress');
    let reviewClass = '';
    let reviewOffset = 0;
    let reviewTotal = 0;

    function loadReviewClasses() {
        fetch('/api/review/classes')
            .then(r => r.json())
            .then(classes => {
                reviewClassSelect.innerHTML = '<option value="">Select class...</option>';
                classes.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.name;
                    opt.textContent = c.name + ' (' + c.count + ')';
                    reviewClassSelect.appendChild(opt);
                });
                if (reviewClass) {
                    reviewClassSelect.value = reviewClass;
                }
            });
    }

    reviewClassSelect.addEventListener('change', () => {
        reviewClass = reviewClassSelect.value;
        reviewOffset = 0;
        if (reviewClass) {
            loadReviewCrop();
        } else {
            clearReview();
        }
    });

    function loadReviewCrop() {
        if (!reviewClass) return;
        fetch('/api/review/' + encodeURIComponent(reviewClass) + '?offset=' + reviewOffset)
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
                reviewImage.src = '/api/review/' + encodeURIComponent(reviewClass) +
                    '/' + encodeURIComponent(crop.filename) + '/image';
                reviewImage.dataset.filename = crop.filename;
                reviewImage.style.display = 'block';
                reviewEmpty.style.display = 'none';
                reviewProgress.textContent = (reviewOffset + 1) + ' of ' + reviewTotal;

                let meta = '';
                if (crop.track_id != null) meta += metaRow('Track', '#' + crop.track_id);
                if (crop.confidence != null) meta += metaRow('Confidence', Math.round(crop.confidence * 100) + '%');
                if (crop.timestamp) meta += metaRow('Time', formatTime(crop.timestamp));
                reviewMeta.innerHTML = meta;
            });
    }

    function clearReview() {
        reviewImage.style.display = 'none';
        reviewImage.src = '';
        reviewImage.dataset.filename = '';
        reviewEmpty.style.display = 'block';
        reviewMeta.innerHTML = '';
        reviewProgress.textContent = '';
    }

    function reviewAction(action) {
        const filename = reviewImage.dataset.filename;
        if (!filename || !reviewClass) return;

        fetch('/api/review/' + encodeURIComponent(reviewClass) + '/' +
            encodeURIComponent(filename) + '/' + action, { method: 'POST' })
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
        if (reviewOffset < reviewTotal - 1) {
            reviewOffset++;
            loadReviewCrop();
        }
    });

    document.addEventListener('keydown', e => {
        if (currentView !== 'review') return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'a' || e.key === 'A') reviewAction('approve');
        else if (e.key === 'r' || e.key === 'R') reviewAction('reject');
        else if (e.key === 'ArrowRight') {
            if (reviewOffset < reviewTotal - 1) {
                reviewOffset++;
                loadReviewCrop();
            }
        } else if (e.key === 'ArrowLeft') {
            if (reviewOffset > 0) {
                reviewOffset--;
                loadReviewCrop();
            }
        }
    });

    // Training
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
                classes.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.name;
                    opt.textContent = c.name + ' (' + c.count + ')';
                    trainingClassSelect.appendChild(opt);
                });
                if (trainingClass) {
                    trainingClassSelect.value = trainingClass;
                }
            });
    }

    trainingClassSelect.addEventListener('change', () => {
        trainingClass = trainingClassSelect.value;
        trainingOffset = 0;
        if (trainingClass) {
            loadTrainingImage();
        } else {
            clearTraining();
        }
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
                if (img.timestamp) meta += metaRow('Time', formatTime(img.timestamp));
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
        if (trainingOffset > 0) {
            trainingOffset--;
            loadTrainingImage();
        }
    });
    document.getElementById('training-next').addEventListener('click', () => {
        if (trainingOffset < trainingTotal - 1) {
            trainingOffset++;
            loadTrainingImage();
        }
    });

    document.addEventListener('keydown', e => {
        if (currentView !== 'training') return;
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'd' || e.key === 'D') deleteTrainingImage();
        else if (e.key === 'ArrowRight') {
            if (trainingOffset < trainingTotal - 1) {
                trainingOffset++;
                loadTrainingImage();
            }
        } else if (e.key === 'ArrowLeft') {
            if (trainingOffset > 0) {
                trainingOffset--;
                loadTrainingImage();
            }
        }
    });

    // Zone editor
    const zoneCanvas = document.getElementById('zone-canvas');
    const zoneCtx = zoneCanvas.getContext('2d');
    const zoneList = document.getElementById('zone-list');
    const zoneForm = document.getElementById('zone-form');
    const zoneNameInput = document.getElementById('zone-name');
    const zoneSaveBtn = document.getElementById('zone-save-btn');
    const zoneHint = document.getElementById('zone-hint');

    function loadZonesView() {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function () {
            snapshotImg = img;
            zoneCanvas.width = img.naturalWidth;
            zoneCanvas.height = img.naturalHeight;
            fetch('/api/zones').then(r => r.json()).then(data => {
                zones = data;
                renderZoneList();
                drawZoneCanvas();
            });
        };
        img.src = '/api/snapshot?' + Date.now();
    }

    function drawZoneCanvas() {
        if (!snapshotImg) return;
        const w = zoneCanvas.width, h = zoneCanvas.height;
        zoneCtx.clearRect(0, 0, w, h);
        zoneCtx.drawImage(snapshotImg, 0, 0, w, h);

        zones.forEach(z => {
            const pts = z.points.map(p => [p[0] * w, p[1] * h]);
            const color = z.type === 'include' ? 'rgba(67,160,71,' : 'rgba(229,57,53,';
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
            const color = zoneType === 'include' ? 'rgba(67,160,71,' : 'rgba(229,57,53,';
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
        drawZoneCanvas();
        const [nx, ny] = canvasCoords(e);
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

    document.getElementById('zone-add-btn').addEventListener('click', () => {
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
        if (!name || drawingPoints.length < 3) return;

        fetch('/api/zones', {
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
                fetch('/api/zones/' + encodeURIComponent(z.name), { method: 'DELETE' })
                    .then(() => {
                        zones = zones.filter(zz => zz.name !== z.name);
                        renderZoneList();
                        drawZoneCanvas();
                    });
            });
            zoneList.appendChild(item);
        });
    }
})();
