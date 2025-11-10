// Main application logic

let currentPage = 1;
let currentFilters = {};
let selectedAlertId = null;
let burstPreviewState = {
  date: '',
  direction: 'both',
  camera: ''
};
let burstSpotlightSelectedCard = null;
let burstSpotlightCurrentImage = null;
let burstSpotlightCurrentBurst = null;

function getAlertIdFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get('alert') || params.get('alertId') || null;
}

function updateAlertIdInUrl(alertId) {
  const url = new URL(window.location.href);
  if (alertId) {
    url.searchParams.set('alert', alertId);
  } else {
    url.searchParams.delete('alert');
    url.searchParams.delete('alertId');
  }
  const search = url.searchParams.toString();
  const nextUrl = `${url.pathname}${search ? `?${search}` : ''}${url.hash}`;
  window.history.replaceState({}, '', nextUrl);
}

function hideAlertModal(options = {}) {
  const modal = document.getElementById('alertModal');
  if (modal) modal.style.display = 'none';
  selectedAlertId = null;
  document.querySelectorAll('.alert-item').forEach(item => item.classList.remove('selected'));
  if (!options.skipUrlUpdate) updateAlertIdInUrl(null);
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  try {
    await api.healthCheck();
    document.getElementById('status').textContent = 'Connected';
    document.getElementById('status').style.background = '#e8f5e9';
    document.getElementById('status').style.color = '#2e7d32';
  } catch (error) {
    document.getElementById('status').textContent = 'Connection Error';
    document.getElementById('status').style.background = '#ffebee';
    document.getElementById('status').style.color = '#c62828';
    showError('Failed to connect to server');
  }

  // Set default 24h range and apply filters
  setDefaultDateRange24h();
  applyDefaultFilters();
  await loadAlerts();
  setupEventListeners();
  setupBurstPreviewControls();
  await handleDeepLinkFromUrl({ initial: true });
}

function setupEventListeners() {
  // Search controls
  document.getElementById('searchBtn').addEventListener('click', applySearch);
  document.getElementById('clearBtn').addEventListener('click', clearSearch);

  // Allow search on Enter key
  document.getElementById('searchInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      applySearch();
    }
  });

  // Alert modal controls
  const alertModal = document.getElementById('alertModal');
  const closeAlertModalBtn = document.getElementById('closeAlertModal');
  closeAlertModalBtn.addEventListener('click', () => {
    hideAlertModal();
  });

  // Modal controls (old image modal)
  const modal = document.getElementById('imageModal');
  const closeBtn = document.querySelector('#imageModal .close');
  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
  window.addEventListener('click', (event) => {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
    if (event.target === alertModal) {
      hideAlertModal();
    }
  });

  window.addEventListener('popstate', () => {
    handleDeepLinkFromUrl({ triggeredByHistory: true });
  });
}

async function loadAlerts(page = 1) {
  const container = document.getElementById('alertsContainer');
  container.innerHTML = '<div class="loading">Loading alerts...</div>';
  try {
    const filters = { ...currentFilters, page, limit: 20 };
    const data = await api.getAlerts(filters);

    // Show warning if backend indicates date filter mismatch
    renderDateFilterWarning(data.meta);

    if (data.alerts.length === 0) {
      container.innerHTML = '<p class="placeholder">No alerts found for selected filters</p>';
      return;
    }

    container.innerHTML = '';
    data.alerts.forEach(alert => container.appendChild(createAlertElement(alert)));
    renderPagination(data.pagination);
    currentPage = page;
  } catch (error) {
    console.error('Error loading alerts:', error);
    container.innerHTML = `<div class="error">Failed to load alerts: ${error.message}</div>`;
  }
}

function createAlertElement(alert) {
  const div = document.createElement('div');
  div.className = 'alert-item';
  div.dataset.alertId = alert._id;
  if (selectedAlertId === alert._id) {
    div.classList.add('selected');
  }
  const timestamp = new Date(alert.start_time).toLocaleString();
  const statusClass = `status-${alert.status || 'new'}`;
  div.innerHTML = `
    <div class="alert-item-content">
      <span class="status-badge ${statusClass}">${alert.status || 'new'}</span>
      <span class="alert-id-text">${alert.id || alert._id}</span>
      <span class="alert-meta-item">📷 ${alert.peripheral_id || 'Unknown'}</span>
      <span class="alert-meta-item">🕒 ${timestamp}</span>
    </div>
  `;
  div.addEventListener('click', () => selectAlert(alert._id, div));
  return div;
}

function renderPagination(pagination) {
  const container = document.getElementById('pagination');

  container.innerHTML = `
    <button id="prevPage" ${pagination.page <= 1 ? 'disabled' : ''}>Previous</button>
    <span>Page ${pagination.page} of ${pagination.pages}</span>
    <button id="nextPage" ${pagination.page >= pagination.pages ? 'disabled' : ''}>Next</button>
  `;

  document.getElementById('prevPage')?.addEventListener('click', () => {
    loadAlerts(currentPage - 1);
  });

  document.getElementById('nextPage')?.addEventListener('click', () => {
    loadAlerts(currentPage + 1);
  });
}

function setupBurstPreviewControls() {
  const dateInput = document.getElementById('burstDate');
  const cameraInput = document.getElementById('burstCamera');
  const loadBtn = document.getElementById('burstLoadBtn');
  const directionRadios = document.querySelectorAll('input[name="burstDirection"]');
  if (!dateInput || !cameraInput || !loadBtn || !directionRadios.length) return;

  const todayIso = new Date().toISOString().slice(0, 10);
  if (!burstPreviewState.date) {
    burstPreviewState.date = todayIso;
  }
  if (!dateInput.value) {
    dateInput.value = burstPreviewState.date;
  }

  dateInput.addEventListener('change', () => {
    burstPreviewState.date = dateInput.value;
  });

  cameraInput.addEventListener('input', (event) => {
    burstPreviewState.camera = event.target.value;
  });

  directionRadios.forEach((radio) => {
    radio.addEventListener('change', () => {
      if (radio.checked) {
        burstPreviewState.direction = radio.value;
      }
    });
    if (radio.checked) {
      burstPreviewState.direction = radio.value;
    }
  });

  loadBtn.addEventListener('click', async (event) => {
    event.preventDefault();
    await loadBurstPreviews();
  });

  // Initial load with defaults
  loadBurstPreviews();
}

function formatBurstTimestamp(value) {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

async function loadBurstPreviews() {
  const grid = document.getElementById('burstPreviewGrid');
  const statusEl = document.getElementById('burstPreviewStatus');
  if (!grid || !statusEl) return;

  const date = burstPreviewState.date;
  const direction = burstPreviewState.direction || 'entry';
  const camera = (burstPreviewState.camera || '').trim();

  if (!date) {
    statusEl.textContent = 'Select a date to load burst previews.';
    grid.innerHTML = '<p class="placeholder">No date selected.</p>';
    return;
  }

  statusEl.textContent = 'Loading burst previews...';
  grid.innerHTML = '<div class="loading">Loading bursts...</div>';

  try {
    const response = await api.getBursts({ date, direction, camera });
    const bursts = Array.isArray(response?.bursts) ? response.bursts : [];
    renderBurstPreviewGrid(bursts);

    const directionLabel = direction === 'both' ? 'entry & exit' : direction;
    const cameraLabel = camera ? ` · Camera filter: ${camera}` : '';
    statusEl.textContent = `${bursts.length} burst${bursts.length === 1 ? '' : 's'} for ${directionLabel} on ${date}${cameraLabel}`;
  } catch (error) {
    console.error('Failed to load bursts:', error);
    statusEl.textContent = 'Failed to load burst previews.';
    grid.innerHTML = `<div class="error">Unable to load bursts: ${error.message}</div>`;
  }
}

function renderBurstPreviewGrid(bursts) {
  const grid = document.getElementById('burstPreviewGrid');
  if (!grid) return;

  if (!bursts.length) {
    grid.innerHTML = '<p class="placeholder">No burst cutouts found for the selected filters.</p>';
    showBurstSpotlight(null, null);
    return;
  }

  grid.innerHTML = '';
  let firstSelection = null;
  bursts.forEach((burst) => {
    const card = document.createElement('div');
    card.className = 'burst-card';
    card.addEventListener('click', () => showBurstSpotlight(burst, card));

    const header = document.createElement('div');
    header.className = 'burst-card-header';

    const meta = document.createElement('div');
    meta.className = 'burst-meta';
    const cameraLabel = burst.peripheral?.id || burst.peripheral?.name || 'Unknown camera';
    const detectionLabel = burst.detectionId || 'No detection id';
    meta.innerHTML = `
      <span class="burst-id">${cameraLabel}</span>
      <span class="burst-sub">${detectionLabel} · n=${burst.burstCount}</span>
    `;

    header.appendChild(meta);
    card.appendChild(header);

    const strip = document.createElement('div');
    strip.className = 'burst-thumb-strip';
    const images = Array.isArray(burst.burstImages) ? burst.burstImages : [];
    if (images.length === 0 && burst.cutoutImage) images.push(burst.cutoutImage);
    images.forEach((src, index) => {
      const img = document.createElement('img');
      img.src = src;
      img.alt = `burst-${index}`;
      img.loading = 'lazy';
      strip.appendChild(img);
    });
    card.appendChild(strip);

    const footer = document.createElement('div');
    footer.className = 'burst-card-footer';
    footer.innerHTML = `
      <span>${(burst.direction || 'unknown').toUpperCase()}</span>
      <span>${formatBurstTimestamp(burst.timestamp)}</span>
    `;
    card.appendChild(footer);

    grid.appendChild(card);
    if (!firstSelection) {
      firstSelection = { burst, card };
    }
  });

  if (firstSelection) {
    showBurstSpotlight(firstSelection.burst, firstSelection.card);
  }
}

function showBurstSpotlight(burst, card) {
  const viewer = document.getElementById('burstPreviewViewer');
  if (!viewer) return;

  if (burstSpotlightSelectedCard) {
    burstSpotlightSelectedCard.classList.remove('selected');
  }
  if (card) {
    card.classList.add('selected');
    burstSpotlightSelectedCard = card;
  } else {
    burstSpotlightSelectedCard = null;
  }

  if (!burst) {
    viewer.innerHTML = '<p class="placeholder">Select a burst to preview.</p>';
    burstSpotlightCurrentImage = null;
    burstSpotlightCurrentBurst = null;
    return;
  }

  const images = Array.isArray(burst.burstImages) && burst.burstImages.length
    ? [...burst.burstImages]
    : (burst.cutoutImage ? [burst.cutoutImage] : []);

  if (!images.length) {
    viewer.innerHTML = '<div class="burst-spotlight-empty">No images available for this burst.</div>';
    burstSpotlightCurrentImage = null;
    burstSpotlightCurrentBurst = null;
    return;
  }

  viewer.innerHTML = '';

  const mediaWrapper = document.createElement('div');
  mediaWrapper.className = 'burst-spotlight-media';
  const mainImg = document.createElement('img');
  mainImg.src = images[0];
  mainImg.alt = 'Burst spotlight';
  mainImg.loading = 'lazy';
  mediaWrapper.appendChild(mainImg);
  burstSpotlightCurrentImage = images[0];
  burstSpotlightCurrentBurst = burst;

  const details = document.createElement('div');
  details.className = 'burst-spotlight-details';

  const meta = document.createElement('div');
  meta.className = 'burst-spotlight-meta';
  const cameraLabel = burst.peripheral?.id || burst.peripheral?.name || 'Unknown camera';
  const detectionLabel = burst.detectionId || 'No detection id';
  const directionLabel = (burst.direction || 'unknown').toUpperCase();
  meta.innerHTML = `
    <span><strong>Camera:</strong> ${cameraLabel}</span>
    <span><strong>Detection:</strong> ${detectionLabel}</span>
    <span><strong>Direction:</strong> ${directionLabel}</span>
    <span><strong>Frames:</strong> ${burst.burstCount}</span>
    <span><strong>Timestamp:</strong> ${formatBurstTimestamp(burst.timestamp)}</span>
  `;

  details.appendChild(meta);

  if (images.length > 1) {
    const thumbs = document.createElement('div');
    thumbs.className = 'burst-spotlight-thumbs';

    images.forEach((src, index) => {
      const thumb = document.createElement('img');
      thumb.src = src;
      thumb.alt = `frame-${index}`;
      thumb.loading = 'lazy';
      thumb.addEventListener('click', () => {
        mainImg.src = src;
        burstSpotlightCurrentImage = src;
      });
      thumbs.appendChild(thumb);
    });

    details.appendChild(thumbs);
  }

  const actions = document.createElement('div');
  actions.className = 'burst-spotlight-actions';
  const openAlertBtn = document.createElement('button');
  openAlertBtn.textContent = 'Open Alert';
  openAlertBtn.addEventListener('click', () => openBurstAlert(burst));
  actions.appendChild(openAlertBtn);
  const viewImageBtn = document.createElement('button');
  viewImageBtn.textContent = 'View Image';
  viewImageBtn.classList.add('secondary');
  viewImageBtn.addEventListener('click', () => openBurstImageModal(burstSpotlightCurrentImage || images[0], burst));
  actions.appendChild(viewImageBtn);
  details.appendChild(actions);

  viewer.appendChild(mediaWrapper);
  viewer.appendChild(details);
}

async function openBurstAlert(burst) {
  if (!burst || !burst.detectionId) {
    const statusEl = document.getElementById('burstPreviewStatus');
    if (statusEl) statusEl.textContent = 'No detection ID available for this burst.';
    return;
  }

  const detectionId = burst.detectionId;
  const statusEl = document.getElementById('burstPreviewStatus');

  const alertItems = Array.from(document.querySelectorAll('.alert-item'));
  const existing = alertItems.find(item => {
    const text = item.querySelector('.alert-id-text')?.textContent?.trim();
    return text === detectionId;
  });

  if (existing) {
    await selectAlert(existing.dataset.alertId, existing);
    if (statusEl) statusEl.textContent = `Opened alert ${detectionId}`;
    return;
  }

  try {
    const result = await api.getAlerts({ search: detectionId, limit: 1, page: 1 });
    const match = result?.alerts?.[0];
    if (match && match._id) {
      await selectAlert(match._id, null);
      if (statusEl) statusEl.textContent = `Opened alert ${detectionId}`;
      return;
    }
    if (statusEl) statusEl.textContent = `No alert found for ${detectionId}`;
  } catch (error) {
    console.error('Failed to open alert for burst:', error);
    if (statusEl) statusEl.textContent = `Failed to open alert: ${error.message}`;
  }
}

function openBurstImageModal(imageUrl, burst) {
  if (!imageUrl) return;
  const modal = document.getElementById('imageModal');
  const canvas = document.getElementById('imageCanvas');
  const metadataDiv = document.getElementById('imageMetadata');
  if (!modal || !canvas || !metadataDiv) return;

  canvasUtils.loadImage(imageUrl)
    .then((img) => {
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      const safeUrl = encodeURI(imageUrl);
      metadataDiv.innerHTML = `
        <div><strong>Image:</strong> <a href="${safeUrl}" target="_blank" rel="noopener">${safeUrl}</a></div>
        <div><strong>Camera:</strong> ${burst?.peripheral?.id || burst?.peripheral?.name || 'Unknown'}</div>
        <div><strong>Direction:</strong> ${(burst?.direction || 'unknown').toUpperCase()}</div>
        <div><strong>Frames:</strong> ${burst?.burstCount ?? 'N/A'}</div>
        <div><strong>Timestamp:</strong> ${formatBurstTimestamp(burst?.timestamp)}</div>
      `;
      modal.style.display = 'block';
    })
    .catch((error) => {
      console.error('Failed to load burst image for viewer:', error);
    });
}

async function selectAlert(alertId, elementRef, options = {}) {
  selectedAlertId = alertId;
  // Update selection in list
  document.querySelectorAll('.alert-item').forEach(item => item.classList.remove('selected'));
  if (elementRef) elementRef.classList.add('selected');
  if (!options.skipUrlUpdate) {
    updateAlertIdInUrl(alertId);
  }
  const modal = document.getElementById('alertModal');
  const modalContent = document.getElementById('alertModalContent');
  modal.style.display = 'block';
  modalContent.innerHTML = '<div class="loading">Loading alert details...</div>';
  try {
    const alert = await api.getAlert(alertId, { enrich: true });
    renderAlertModal(alert);
  } catch (error) {
    console.error('Error loading alert details:', error);
    modalContent.innerHTML = `<div class="error">Failed to load alert details: ${error.message}</div>`;
  }
}

async function handleDeepLinkFromUrl(options = {}) {
  const alertIdFromUrl = getAlertIdFromUrl();
  if (alertIdFromUrl) {
    if (alertIdFromUrl !== selectedAlertId) {
      const listItem = document.querySelector(`.alert-item[data-alert-id="${alertIdFromUrl}"]`) || null;
      try {
        await selectAlert(alertIdFromUrl, listItem, { skipUrlUpdate: true, openingFromUrl: true });
      } catch (error) {
        console.error('Failed to open alert from URL:', error);
      }
    }
  } else if (selectedAlertId && options.triggeredByHistory) {
    hideAlertModal({ skipUrlUpdate: true });
  }
}

function buildLineCrossingDetails(alert) {
  const enrichment = alert.enrichment || {};
  const direction = enrichment.lineDirection || enrichment.parsed?.direction || 'N/A';
  const lineId = enrichment.lineId || 'N/A';
  const cameraId = alert.peripheral_id || 'N/A';
  const deviceId = alert.from_device || 'N/A';
  const startTime = alert.start_time ? new Date(alert.start_time).toLocaleString() : 'N/A';
  const endTime = alert.end_time ? new Date(alert.end_time).toLocaleString() : 'N/A';
  const detectionId = alert.detection_id || enrichment.parsed?.detectionId || 'N/A';
  const classifiers = enrichment.classifiers || [];
  const classifierHtml = classifiers.length ? classifiers.map(c => `<div class='classifier-chip'><span>${c.label}</span><span>${(c.score*100).toFixed(1)}%</span></div>`).join('') : '<div class="classifier-none">None</div>';
  const pose = enrichment.pose || null;
  const poseHtml = pose ? `<span>${pose.coords.length} keypoints / ${pose.skeleton.length} edges</span>` : '<span class="muted">No pose</span>';
  let multiDirHtml = '';
  if (enrichment.multiDirections) {
    const dirs = Object.keys(enrichment.multiDirections).filter(Boolean);
    if (dirs.length > 1) multiDirHtml = `<div class="multi-dir-bar">${dirs.map(d => `<span class="dir-pill" data-dir="${d}">${d}</span>`).join('')}</div>`;
  }
  return `
    <div class="line-details-panel">
      <div class="line-details-header">
        <h3 class="h-section">Line Crossing Details</h3>
        ${multiDirHtml}
      </div>
      <div class="line-details-row">
        <div class="info-card ld-card"><h3>Direction</h3><div class="value">${direction}</div></div>
        <div class="info-card ld-card"><h3>Line ID</h3><div class="value">${lineId}</div></div>
        <div class="info-card ld-card"><h3>Camera ID</h3><div class="value">${cameraId}</div></div>
        <div class="info-card ld-card"><h3>Device ID</h3><div class="value">${deviceId}</div></div>
        <div class="info-card ld-card"><h3>Start Time</h3><div class="value">${startTime}</div></div>
        <div class="info-card ld-card"><h3>End Time</h3><div class="value">${endTime}</div></div>
        <div class="info-card ld-card"><h3>Detection ID</h3><div class="value mono">${detectionId}</div></div>
        <div class="info-card ld-card wide"><h3>Classifiers</h3><div class="value classifier-chips">${classifierHtml}</div></div>
        <div class="info-card ld-card"><h3>Pose</h3><div class="value pose-val">${poseHtml}</div></div>
      </div>
    </div>`;
}

async function renderAlertModal(alert) {
  const container = document.getElementById('alertModalContent');
  const enrichmentDetailsHtml = buildLineCrossingDetails(alert);
  const enrichment = alert.enrichment || {};
  const rawBurst = enrichment.burstImages || [];
  const cutoutImage = enrichment.cutoutImage || null;
  const burstImages = cutoutImage ? [cutoutImage, ...rawBurst.filter(u => u !== cutoutImage)] : rawBurst;
  container.innerHTML = `
    <div class="alert-modal-header">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <h2>Alert Details</h2>
        <span class="status-badge status-${alert.status || 'new'}">${alert.status || 'new'}</span>
        <span class="status-badge" id="annotationsStatusBadge" style="background:#999;color:white;">Checking...</span>
      </div>
      <div class="alert-id-text">${alert.id || alert._id}</div>
    </div>
    ${enrichmentDetailsHtml}
    <div class="annotations-filter" id="annotationsFilter">
      <h3 class="h-section">Annotation Filters</h3>
      <div class="filter-grid">
        <div class="filter-group wide" id="fg-labels">
          <label for="filterLabels">Labels (comma)</label>
          <input id="filterLabels" type="text" placeholder="e.g. person,car" />
        </div>
        <div class="filter-group wide" id="fg-ids">
          <label for="filterIds">Detection IDs (comma)</label>
          <input id="filterIds" type="text" placeholder="e.g. 123,abc" />
        </div>
        <div class="filter-group tight" id="fg-types">
          <label>Types</label>
          <div class="types-group">
            <label><input type="checkbox" id="toggleBoxes" checked /> Boxes</label>
            <label><input type="checkbox" id="togglePaths" checked /> Paths</label>
          </div>
        </div>
        <div class="filter-actions">
          <button id="applyAnnotationFilters">Apply</button>
          <button id="resetAnnotationFilters">Reset</button>
        </div>
      </div>
    </div>
    <div class="timeline-controls" id="timelineControls" style="display:none;">
      <div class="timeline-bar">
        <span class="timeline-title">Frame Timeline</span>
        <span class="timeline-value" id="timelineValue">All frames</span>
      </div>
      <input type="range" id="timelineSlider" min="0" max="0" value="0" step="1" />
      <div class="timeline-hint" id="timelineHint">Drag to reveal detections over time</div>
    </div>
    <div class="alert-modal-media" id="alertMediaContainer">
      <div class="loading">Loading media...</div>
      ${alert.videoUrl ? '<div class="media-hint">Click image to play video</div>' : ''}
    </div>
    <div class="burst-gallery">
      <h3 class="h-section">Burst Images</h3>
      ${burstImages.length ? `<div class="burst-strip" id="burstStrip">${burstImages.map((u,i)=>`<img src='${u}' data-index='${i}' class='burst-thumb ${i===0 && cutoutImage? 'burst-cutout-first':''}' loading='lazy' alt='burst ${i}'/>`).join('')}</div>` : '<div class="burst-strip empty">No burst images</div>'}
    </div>
    <div id="annotationsStatsContainer"></div>`;
  console.log('[Timeline] Rendered modal markup', {
    hasTimelineControls: !!document.getElementById('timelineControls')
  });
  if (alert.imageUrl) loadAlertImageInModal(alert);
  setupBurstGalleryInteractions();
}

function setupBurstGalleryInteractions() {
  const strip = document.getElementById('burstStrip');
  if (!strip) return;
  strip.querySelectorAll('img.burst-thumb').forEach(img => { img.style.cursor = 'zoom-in'; const preload = new Image(); preload.src = img.src; });
}

async function loadAlertImageInModal(alert) {
  const container = document.getElementById('alertMediaContainer');

  try {
    console.log('Loading image from URL:', alert.imageUrl);

    // Load image
    const image = await canvasUtils.loadImage(alert.imageUrl);
    console.log('Image loaded successfully:', image.width, 'x', image.height);

    // Load metadata if available
    let metadata = null;
    let annotationsData = null;
    let hasDetections = false;

    if (alert.metadataUrl) {
      try {
        console.log('Loading metadata from URL:', alert.metadataUrl);
        annotationsData = await api.getMetadata(alert.metadataUrl);
        console.log('Annotations data loaded:', annotationsData);

        // Process annotations to extract detections
        metadata = processAnnotations(annotationsData);

        // Check if detections are present
        hasDetections = annotationsData?.data?.detections?.length > 0;

        // Update annotations status badge
        updateAnnotationsStatusBadge(hasDetections);
      } catch (error) {
        console.warn('Failed to load metadata:', error);
        // Update badge to show no annotations file
        updateAnnotationsStatusBadge(null);
      }
    } else {
      // No metadata URL provided
      updateAnnotationsStatusBadge(null);
    }

    // Create canvas with image and bounding boxes
    const canvas = document.createElement('canvas');
    canvasUtils.drawBoundingBoxes(canvas, image, metadata, annotationsData);

    container.innerHTML = '';
    container.appendChild(canvas);

    // Add hint if video is available
    if (alert.videoUrl) {
      const hint = document.createElement('div');
      hint.className = 'media-hint';
      hint.textContent = 'Click image to play video';
      container.appendChild(hint);

      // Add click handler to switch to video while keeping overlay
      canvas.addEventListener('click', () => {
        switchToVideo(alert, container, canvas, metadata, annotationsData);
      });
    }

    // Display annotations statistics below (bottom of modal) instead of inside media container
    if (annotationsData) {
      const statsDiv = renderAnnotationsStats(annotationsData);
      const statsContainer = document.getElementById('annotationsStatsContainer');
      if (statsContainer) {
        // Ensure existing content replaced (if reloading)
        statsContainer.innerHTML = '';
        statsContainer.appendChild(statsDiv);
      } else {
        // Fallback: append inside media container if placeholder missing
        container.appendChild(statsDiv);
      }
    }

    // After appending canvas, wire filter controls
    setupAnnotationFilterControls(canvas, image, metadata, annotationsData);
  } catch (error) {
    console.error('Error loading image:', error);
    console.error('Image URL was:', alert.imageUrl);

    // Update badge to show error
    updateAnnotationsStatusBadge(null);

    // Show user-friendly error message
    const errorMessage = error.message.includes('File not found') || error.message.includes('404')
      ? 'Media file not available in storage'
      : 'Failed to load image';

    container.innerHTML = `
      <div class="error" style="padding: 30px; text-align: center; background: #f9f9f9; border-radius: 8px;">
        <div style="font-size: 64px; margin-bottom: 15px; opacity: 0.5;">📷</div>
        <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #666;">${errorMessage}</div>
        <div style="font-size: 13px; color: #999; margin-bottom: 5px;">
          Path: <code style="background: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px;">${alert.video_snapshot || 'N/A'}</code>
        </div>
        <div style="font-size: 12px; color: #aaa; margin-top: 10px;">
          The media file may have been deleted, never uploaded, or the path may be incorrect.
        </div>
      </div>
    `;
  }
}

function switchToVideo(alert, container, canvas, metadata, annotationsData) {
  if (!alert.videoUrl) return;

  // Keep existing canvas dimensions
  const origW = canvas.width;
  const origH = canvas.height;

  // Create wrapper to overlay canvas atop video
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.style.position = 'relative';
  wrapper.style.width = '100%';
  wrapper.style.maxHeight = '70vh';
  wrapper.style.background = '#000';
  wrapper.style.overflow = 'hidden';

  const video = document.createElement('video');
  video.controls = true;
  video.autoplay = true;
  video.style.width = '100%';
  video.style.height = '100%';
  video.style.objectFit = 'contain';
  video.style.display = 'block';

  const source = document.createElement('source');
  source.src = alert.videoUrl;
  source.type = 'video/mp4';
  video.appendChild(source);
  wrapper.appendChild(video);

  // Prepare overlay canvas
  canvas.width = origW;
  canvas.height = origH;
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.pointerEvents = 'none';
  canvas.style.objectFit = 'contain';
  canvas.style.zIndex = '2';
  wrapper.appendChild(canvas);

  container.appendChild(wrapper);

  // Redraw overlay only (no base image) with existing filters (defaults)
  const overlayOptions = { overlayOnly: true, showBoxes: true, showPaths: true };
  const timelineSlider = document.getElementById('timelineSlider');
  if (timelineSlider) {
    const parsed = Number.parseInt(timelineSlider.value, 10);
    if (!Number.isNaN(parsed)) {
      overlayOptions.maxFrameIndex = parsed;
    }
  }
  canvasUtils.drawBoundingBoxes(canvas, null, metadata, annotationsData, overlayOptions);

  // Optional: overlay toggle button
  const toggleBtn = document.createElement('button');
  toggleBtn.textContent = 'Toggle Overlay';
  toggleBtn.style.position = 'absolute';
  toggleBtn.style.top = '10px';
  toggleBtn.style.right = '10px';
  toggleBtn.style.zIndex = '3';
  toggleBtn.style.background = 'rgba(0,0,0,0.6)';
  toggleBtn.style.color = '#fff';
  toggleBtn.style.border = 'none';
  toggleBtn.style.padding = '6px 10px';
  toggleBtn.style.borderRadius = '4px';
  toggleBtn.style.cursor = 'pointer';
  wrapper.appendChild(toggleBtn);
  let overlayVisible = true;
  toggleBtn.onclick = () => { overlayVisible = !overlayVisible; canvas.style.display = overlayVisible ? 'block' : 'none'; };

  // Hint re-added
  const hint = document.createElement('div');
  hint.className = 'media-hint';
  hint.textContent = 'Video playing with overlay';
  wrapper.appendChild(hint);

  // Handle video load error
  video.addEventListener('error', () => {
    container.innerHTML = `
      <div class="error" style="padding: 30px; text-align: center;">
        <div style="font-size: 64px; margin-bottom: 15px; opacity: 0.5;">🎥</div>
        <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #666;">Video file not available</div>
        <div style="font-size: 13px; color: #999;">The video file could not be loaded.</div>
      </div>
    `;
  });
}

// Update annotations status badge
function updateAnnotationsStatusBadge(hasDetections) {
  const badge = document.getElementById('annotationsStatusBadge');
  if (!badge) return;

  if (hasDetections === true) {
    // Detections found
    badge.textContent = '✓ Detections Found';
    badge.style.background = '#4CAF50';
    badge.style.color = 'white';
  } else if (hasDetections === false) {
    // No detections (but annotations file exists)
    badge.textContent = '⚠ No Detections';
    badge.style.background = '#FF9800';
    badge.style.color = 'white';
  } else {
    // No annotations file or error loading
    badge.textContent = '✗ No Annotations';
    badge.style.background = '#757575';
    badge.style.color = 'white';
  }
}

// Process annotations data structure to extract detections
function processAnnotations(annotationsData) {
  if (!annotationsData || !annotationsData.data) {
    return null;
  }

  const data = annotationsData.data;
  const detections = data.detections || [];

  // Convert detections to the format expected by canvas utils
  if (detections.length === 0) {
    return null;
  }

  return {
    detections: detections,
    metadata: data.metadata || {},
    imageWidth: null, // Will be determined by canvas
    imageHeight: null
  };
}

// Render annotations statistics
function renderAnnotationsStats(annotationsData) {
  const statsDiv = document.createElement('div');
  statsDiv.className = 'annotations-stats';

  if (!annotationsData || !annotationsData.data) {
    statsDiv.innerHTML = `
      <h3>📊 Annotations Data</h3>
      <p style="color:#999; margin:0;">No annotations data available</p>
    `;
    return statsDiv;
  }

  const data = annotationsData.data;
  const detections = data.detections || [];
  const metadata = data.metadata || {};
  const hasDetections = detections.length > 0;
  const stats = calculateDetectionStats(detections);
  const timeRange = metadata.start_time && metadata.end_time ? calculateDuration(metadata.start_time, metadata.end_time) : null;

  const labelBreakdownHtml = hasDetections ? Array.from(stats.labelCounts.entries())
    .sort((a,b)=>b[1]-a[1])
    .map(([label,count]) => `<div class="label-chip"><span style="font-weight:bold;color:${getLabelColor(label)};">${label}</span><span style="color:#666;">${count}</span></div>`).join('') : '';

  statsDiv.innerHTML = `
    <h3>📊 Annotations Statistics ${hasDetections ? '<span style="background:#4CAF50;color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;">Detections Found</span>' : '<span style="background:#FF9800;color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;">No Detections</span>'}</h3>

    <div class="stats-grid">
      <div class="stats-card">
        <div class="label">Total Detections</div>
        <div class="value" style="color:${hasDetections ? '#4CAF50' : '#999'};">${detections.length}</div>
      </div>
      ${timeRange ? `<div class="stats-card">
        <div class="label">Duration</div>
        <div class="value" style="font-size:18px;color:#2196F3;">${timeRange}</div>
      </div>` : ''}
      ${stats.uniqueLabels.size > 0 ? `<div class="stats-card">
        <div class="label">Unique Labels</div>
        <div class="value" style="color:#FF9800;">${stats.uniqueLabels.size}</div>
      </div>` : ''}
      ${stats.uniqueObjects.size > 0 ? `<div class="stats-card">
        <div class="label">Tracked Objects</div>
        <div class="value" style="color:#9C27B0;">${stats.uniqueObjects.size}</div>
      </div>` : ''}
    </div>

    ${hasDetections ? `
    <div class="breakdown">
      <div style="font-size:13px;font-weight:bold;color:#333;margin-bottom:10px;">Detection Breakdown by Label</div>
      <div class="label-chips">${labelBreakdownHtml}</div>
    </div>
    ${stats.avgScore !== null ? `<div class="score">
      <div style="font-size:13px;font-weight:bold;color:#333;margin-bottom:8px;">Average Confidence Score</div>
      <div class="avg-bar-wrapper">
        <div class="avg-bar-track"><div class="avg-bar-fill" style="width:${(stats.avgScore*100).toFixed(1)}%;"></div></div>
        <div style="font-size:16px;font-weight:bold;color:#4CAF50;min-width:60px;">${(stats.avgScore*100).toFixed(1)}%</div>
      </div>
    </div>` : ''}
    ` : `
      <div class="warning-box">
        <div style="font-weight:bold;margin-bottom:5px;">⚠️ No Detections Found</div>
        <div style="font-size:13px;">The annotations file was loaded successfully but contains no detection data.</div>
      </div>
    `}
    ${metadata.start_time ? `<div class="metadata-range">Metadata time range: ${new Date(metadata.start_time).toLocaleString()} - ${metadata.end_time ? new Date(metadata.end_time).toLocaleString() : 'N/A'}</div>` : ''}
  `;

  return statsDiv;
}

// Calculate detection statistics
function calculateDetectionStats(detections) {
  const stats = {
    uniqueLabels: new Set(),
    uniqueObjects: new Set(),
    labelCounts: new Map(),
    avgScore: null,
    timeRange: {
      earliest: null,
      latest: null
    }
  };

  if (detections.length === 0) {
    return stats;
  }

  let totalScore = 0;
  let scoreCount = 0;

  detections.forEach(detection => {
    // Track unique labels
    if (detection.label) {
      stats.uniqueLabels.add(detection.label);
      stats.labelCounts.set(
        detection.label,
        (stats.labelCounts.get(detection.label) || 0) + 1
      );
    }

    // Track unique object IDs
    if (detection.id) {
      stats.uniqueObjects.add(detection.id);
    }

    // Calculate average score
    if (typeof detection.score === 'number') {
      totalScore += detection.score;
      scoreCount++;
    }

    // Track time range
    if (detection.timestamp) {
      const time = new Date(detection.timestamp);
      if (!stats.timeRange.earliest || time < new Date(stats.timeRange.earliest)) {
        stats.timeRange.earliest = detection.timestamp;
      }
      if (!stats.timeRange.latest || time > new Date(stats.timeRange.latest)) {
        stats.timeRange.latest = detection.timestamp;
      }
    }
  });

  if (scoreCount > 0) {
    stats.avgScore = totalScore / scoreCount;
  }

  return stats;
}

// Calculate duration between two timestamps
function calculateDuration(startTime, endTime) {
  const start = new Date(startTime);
  const end = new Date(endTime);
  const durationMs = end - start;

  if (durationMs < 0) return 'N/A';

  const seconds = Math.floor(durationMs / 1000);
  if (seconds < 60) return `${seconds}s`;

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

// Get color for label
function getLabelColor(label) {
  const colors = {
    'person': '#FF5722',
    'face_cover': '#9C27B0',
    'vehicle': '#2196F3',
    'car': '#2196F3',
    'truck': '#3F51B5',
    'bicycle': '#4CAF50',
    'motorcycle': '#FF9800'
  };
  return colors[label?.toLowerCase()] || '#666';
}


async function renderAlertDetail(alert) {
  const container = document.getElementById('detailContent');
  const timestamp = new Date(alert.start_time).toLocaleString();
  const endTime = alert.end_time ? new Date(alert.end_time).toLocaleString() : 'N/A';

  container.innerHTML = `
    <div class="detail-section">
      <h3>Alert ID</h3>
      <div style="font-family: 'Monaco', 'Courier New', monospace; font-size: 14px; color: #1976D2; padding: 10px; background: #f5f5f5; border-radius: 4px; word-break: break-all;">
        ${alert.id || alert._id}
      </div>
    </div>
    
    <div class="detail-section">
      <h3>Information</h3>
      <div class="detail-meta">
        <div class="detail-meta-item">
          <strong>Camera ID</strong>
          ${alert.peripheral_id || 'N/A'}
        </div>
        <div class="detail-meta-item">
          <strong>Device ID</strong>
          ${alert.from_device || 'N/A'}
        </div>
        <div class="detail-meta-item">
          <strong>Start Time</strong>
          ${timestamp}
        </div>
        <div class="detail-meta-item">
          <strong>End Time</strong>
          ${endTime}
        </div>
        <div class="detail-meta-item">
          <strong>Status</strong>
          <span class="status-badge status-${alert.status || 'new'}">${alert.status || 'new'}</span>
        </div>
        <div class="detail-meta-item">
          <strong>Detection ID</strong>
          ${alert.detection_id || 'N/A'}
        </div>
        <div class="detail-meta-item">
          <strong>Label</strong>
          ${alert.label || 'N/A'}
        </div>
        <div class="detail-meta-item">
          <strong>Duration</strong>
          ${alert.duration || 0}s
        </div>
      </div>
    </div>
    
    ${alert.llm_classification && alert.llm_classification.summary ? `
    <div class="detail-section">
      <h3>AI Summary</h3>
      <div style="padding: 10px; background: #f9f9f9; border-radius: 4px; line-height: 1.6;">
        ${alert.llm_classification.summary}
      </div>
      ${alert.llm_classification.count_humans ? `
        <div style="margin-top: 10px; font-size: 14px; color: #666;">
          👥 ${alert.llm_classification.count_humans} human(s) detected
          ${alert.llm_classification.frames_processed ? ` | 🎬 ${alert.llm_classification.frames_processed} frames processed` : ''}
        </div>
      ` : ''}
    </div>
    ` : ''}
    
    <div class="detail-section">
      <h3>Image Snapshot</h3>
      <div class="detail-image" id="detailImageContainer">
        <div class="loading">Loading image...</div>
      </div>
    </div>
    
    ${alert.videoUrl ? `
    <div class="detail-section">
      <h3>Video</h3>
      <video controls style="max-width: 100%; border-radius: 4px;">
        <source src="${alert.videoUrl}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
    ` : ''}
  `;

  // Load and display image with bounding boxes
  if (alert.imageUrl) {
    loadAlertImage(alert);
  }
}

async function loadAlertImage(alert) {
  const container = document.getElementById('detailImageContainer');

  try {
    console.log('Loading image from URL:', alert.imageUrl);

    // Load image
    const image = await canvasUtils.loadImage(alert.imageUrl);
    console.log('Image loaded successfully:', image.width, 'x', image.height);

    // Load metadata if available
    let metadata = null;
    if (alert.metadataUrl) {
      try {
        console.log('Loading metadata from URL:', alert.metadataUrl);
        metadata = await api.getMetadata(alert.metadataUrl);
        console.log('Metadata loaded:', metadata);
      } catch (error) {
        console.warn('Failed to load metadata:', error);
      }
    }

    // Create canvas with image and bounding boxes
    const canvas = document.createElement('canvas');
    canvas.style.maxWidth = '100%';
    canvas.style.cursor = 'pointer';

    canvasUtils.drawBoundingBoxes(canvas, image, metadata);

    container.innerHTML = '';
    container.appendChild(canvas);

    // Add click handler to view full size
    canvas.addEventListener('click', () => {
      showImageModal(alert, image, metadata);
    });

    // Add button to view full size
    const button = document.createElement('button');
    button.className = 'view-full';
    button.textContent = 'View Full Size';
    button.addEventListener('click', () => {
      showImageModal(alert, image, metadata);
    });
    container.appendChild(button);

  } catch (error) {
    console.error('Error loading image:', error);
    console.error('Image URL was:', alert.imageUrl);

    // Show user-friendly error message
    const errorMessage = error.message.includes('File not found') || error.message.includes('404')
      ? 'Media file not available in storage'
      : 'Failed to load image';

    container.innerHTML = `
      <div class="error" style="padding: 30px; text-align: center; background: #f9f9f9; border-radius: 8px;">
        <div style="font-size: 64px; margin-bottom: 15px; opacity: 0.5;">📷</div>
        <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #666;">${errorMessage}</div>
        <div style="font-size: 13px; color: #999; margin-bottom: 5px;">
          Path: <code style="background: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px;">${alert.video_snapshot || 'N/A'}</code>
        </div>
        <div style="font-size: 12px; color: #aaa; margin-top: 10px;">
          The media file may have been deleted, never uploaded, or the path may be incorrect.
        </div>
      </div>
    `;
  }
}

function showImageModal(alert, image, metadata) {
  const modal = document.getElementById('imageModal');
  const canvas = document.getElementById('imageCanvas');
  const metadataContainer = document.getElementById('imageMetadata');

  // Draw image on modal canvas
  canvasUtils.drawBoundingBoxes(canvas, image, metadata);

  // Display metadata
  let metadataHtml = '<h3>Detection Details</h3>';
  if (metadata && metadata.boxes && metadata.boxes.length > 0) {
    metadataHtml += '<ul>';
    metadata.boxes.forEach((box, index) => {
      metadataHtml += `
        <li>
          <strong>Object ${index + 1}:</strong> 
          ${box.class || 'unknown'} 
          (${(box.confidence * 100).toFixed(1)}% confidence) - 
          Position: (${Math.round(box.x)}, ${Math.round(box.y)}) 
          Size: ${Math.round(box.width)}x${Math.round(box.height)}
        </li>
      `;
    });
    metadataHtml += '</ul>';
  } else {
    metadataHtml += '<p>No bounding box data available</p>';
  }

  metadataContainer.innerHTML = metadataHtml;
  modal.style.display = 'block';
}

function applySearch() {
  const searchValue = document.getElementById('searchInput').value.trim();
  const startDate = document.getElementById('startDate').value;
  const endDate = document.getElementById('endDate').value;

  currentFilters = {};
  if (searchValue) {
    currentFilters.search = searchValue;
  }

  // Add date filters if set
  if (startDate) {
    currentFilters.startDate = new Date(startDate).toISOString();
  }
  if (endDate) {
    currentFilters.endDate = new Date(endDate).toISOString();
  }

  loadAlerts(1);
}

function clearSearch() {
  document.getElementById('searchInput').value = '';
  setDefaultDateRange24h();
  applyDefaultFilters();
  loadAlerts(1);
}

// Set default date range (last 24 hours)
function setDefaultDateRange24h() {
  const now = new Date();
  const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  const format = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}T${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  document.getElementById('startDate').value = format(oneDayAgo);
  document.getElementById('endDate').value = format(now);
}

// Apply default filters to currentFilters (used when setting default date range)
function applyDefaultFilters() {
  const startDate = document.getElementById('startDate').value;
  const endDate = document.getElementById('endDate').value;

  currentFilters = {};

  if (startDate) {
    currentFilters.startDate = new Date(startDate).toISOString();
  }
  if (endDate) {
    currentFilters.endDate = new Date(endDate).toISOString();
  }
}

function renderDateFilterWarning(meta) {
  let bar = document.getElementById('dateFilterWarningBar');
  if (!meta) {
    if (bar) bar.remove();
    return;
  }
  if (meta.dateFilterApplied && meta.dateFilterWarning) {
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'dateFilterWarningBar';
      bar.style.cssText = 'background:#fff3cd;color:#856404;padding:10px 15px;border:1px solid #ffe8a1;border-radius:6px;margin-bottom:12px;font-size:13px;';
      document.querySelector('.alerts-list').insertBefore(bar, document.querySelector('.alerts-list').firstChild.nextSibling);
    }
    bar.innerHTML = `⚠ No alerts match the selected date range. There are alerts outside this range. (start_time stored as ${meta.startTimeType})`;
  } else if (bar) {
    bar.remove();
  }
}

function showError(message) {
  const container = document.getElementById('alertsContainer');
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.textContent = message;
  container.prepend(errorDiv);

  setTimeout(() => errorDiv.remove(), 5000);
}

function setupAnnotationFilterControls(canvas, image, metadata, annotationsData) {
  const filterPanel = document.getElementById('annotationsFilter');
  if (!filterPanel) return;
  const mediaContainer = document.getElementById('alertMediaContainer');
  const timelineControls = document.getElementById('timelineControls');
  const timelineSlider = document.getElementById('timelineSlider');
  const timelineValue = document.getElementById('timelineValue');
  const timelineHint = document.getElementById('timelineHint');
  console.log('[Timeline] Elements lookup', {
    hasFilterPanel: !!filterPanel,
    hasTimelineControls: !!timelineControls,
    hasSlider: !!timelineSlider,
    hasTimelineValue: !!timelineValue
  });

  const isVideoActive = () => !!mediaContainer.querySelector('video');

  // Hide panel if no annotations data
  if (!annotationsData || !annotationsData.data || !Array.isArray(annotationsData.data.detections) || annotationsData.data.detections.length === 0) {
    filterPanel.style.display = 'none';
    if (timelineControls) {
      timelineControls.style.display = 'none';
    }
    console.log('[Timeline] No detections found, hiding controls.', { annotationsDataExists: !!annotationsData });
    return;
  } else {
    filterPanel.style.display = 'block';
  }

  const labelsInput = document.getElementById('filterLabels');
  const idsInput = document.getElementById('filterIds');
  const boxesCheckbox = document.getElementById('toggleBoxes');
  const pathsCheckbox = document.getElementById('togglePaths');
  const applyBtn = document.getElementById('applyAnnotationFilters');
  const resetBtn = document.getElementById('resetAnnotationFilters');

  if (!applyBtn || !resetBtn) return; // safety

  const detections = annotationsData.data.detections;
  console.log('[Timeline] Initializing timeline controls', {
    detectionCount: detections.length,
    hasTimelineControls: !!timelineControls,
    hasSlider: !!timelineSlider
  });

  const formatFrameLabel = (rawTs) => {
    if (!rawTs) return null;
    const ts = Date.parse(rawTs);
    if (Number.isNaN(ts)) return rawTs;
    const dt = new Date(ts);
    const time = dt.toLocaleTimeString([], { hour12: false });
    const millis = String(dt.getMilliseconds()).padStart(3, '0');
    return `${time}.${millis}`;
  };

  const buildTimelineFrames = () => {
    if (!Array.isArray(detections) || detections.length === 0) return [];
    const decorated = detections.map((d, idx) => {
      const parsed = d.timestamp ? Date.parse(d.timestamp) : NaN;
      return {
        detection: d,
        idx,
        ts: Number.isNaN(parsed) ? null : parsed
      };
    }).sort((a, b) => {
      if (a.ts !== null && b.ts !== null) return a.ts - b.ts || a.idx - b.idx;
      if (a.ts !== null) return -1;
      if (b.ts !== null) return 1;
      return a.idx - b.idx;
    });

    const frames = [];
    let currentKey = null;
    decorated.forEach((item) => {
      const key = item.ts !== null ? `ts-${item.ts}` : `idx-${item.idx}`;
      if (key !== currentKey) {
        currentKey = key;
        frames.push({
          key,
          timestamp: item.detection.timestamp || null,
          label: item.detection.timestamp ? formatFrameLabel(item.detection.timestamp) : `Detection ${item.idx + 1}`
        });
      }
      const frameIndex = frames.length - 1;
      Object.defineProperty(item.detection, '__frameIndex', {
        value: frameIndex,
        enumerable: false,
        configurable: true,
        writable: true
      });
    });
    console.log('[Timeline] Frames derived from detections', {
      detections: detections.length,
      frameBuckets: frames.length
    });
    return frames;
  };

  const frames = buildTimelineFrames();
  let activeFrameIndex = frames.length > 0 ? frames.length - 1 : null;
  console.log('[Timeline] Frames built', { frameCount: frames.length, activeFrameIndex });

  const updateTimelineDisplay = () => {
    if (!timelineControls || !timelineSlider || !timelineValue) return;
    if (frames.length <= 1 || typeof activeFrameIndex !== 'number') {
      timelineValue.textContent = frames.length > 0 ? `All frames (${frames.length})` : 'All frames';
      if (timelineHint) timelineHint.style.display = 'none';
      console.log('[Timeline] Display updated: showing all frames', {
        frameCount: frames.length,
        activeFrameIndex
      });
      return;
    }
    if (timelineHint) timelineHint.style.display = 'block';
    if (activeFrameIndex >= frames.length - 1) {
      timelineValue.textContent = `All frames (${frames.length})`;
      console.log('[Timeline] Display at last frame', { activeFrameIndex, frameCount: frames.length });
      return;
    }
    const frame = frames[activeFrameIndex];
    const label = frame?.label ? ` • ${frame.label}` : '';
    timelineValue.textContent = `Frame ${activeFrameIndex + 1}/${frames.length}${label}`;
    console.log('[Timeline] Display partial', { activeFrameIndex, frameCount: frames.length, label: frame?.label });
  };

  if (timelineControls && timelineSlider && timelineValue) {
    timelineControls.style.display = 'flex';
    timelineSlider.min = '0';
    timelineSlider.max = String(Math.max(frames.length - 1, 0));
    timelineSlider.step = '1';
    timelineSlider.value = String(typeof activeFrameIndex === 'number' ? activeFrameIndex : 0);
    timelineSlider.disabled = frames.length <= 1;
    console.log('[Timeline] Controls ready', {
      sliderMin: timelineSlider.min,
      sliderMax: timelineSlider.max,
      sliderValue: timelineSlider.value,
      sliderDisabled: timelineSlider.disabled
    });
    updateTimelineDisplay();
  }

  const redraw = () => {
    const rawLabels = labelsInput.value.trim();
    const rawIds = idsInput.value.trim();
    const labels = rawLabels ? rawLabels.split(',').map(s => s.trim().toLowerCase()).filter(Boolean) : null;
    const ids = rawIds ? rawIds.split(',').map(s => s.trim()).filter(Boolean) : null;
    const showBoxes = boxesCheckbox.checked;
    const showPaths = pathsCheckbox.checked;
    const options = { labels, ids, showBoxes, showPaths, overlayOnly: isVideoActive() };
    if (typeof activeFrameIndex === 'number') {
      options.maxFrameIndex = activeFrameIndex;
    }
    canvasUtils.drawBoundingBoxes(canvas, isVideoActive() ? null : image, metadata, annotationsData, options);
  };

  applyBtn.onclick = (e) => { e.preventDefault(); redraw(); };
  resetBtn.onclick = (e) => {
    e.preventDefault();
    labelsInput.value = '';
    idsInput.value = '';
    boxesCheckbox.checked = true;
    pathsCheckbox.checked = true;
    if (timelineSlider && frames.length > 0) {
      activeFrameIndex = frames.length - 1;
      timelineSlider.value = String(activeFrameIndex);
      updateTimelineDisplay();
    }
    const options = { showBoxes: true, showPaths: true, overlayOnly: isVideoActive() };
    if (typeof activeFrameIndex === 'number') options.maxFrameIndex = activeFrameIndex;
    canvasUtils.drawBoundingBoxes(canvas, isVideoActive() ? null : image, metadata, annotationsData, options);
  };

  if (timelineSlider && frames.length > 1) {
    const onScrubChange = (value) => {
      const parsed = Number.parseInt(value, 10);
      if (Number.isNaN(parsed)) return;
      activeFrameIndex = Math.min(Math.max(parsed, 0), frames.length - 1);
      console.log('[Timeline] Scrub change', { newIndex: activeFrameIndex, framesCount: frames.length });
      updateTimelineDisplay();
      redraw();
    };
    timelineSlider.addEventListener('input', (event) => onScrubChange(event.target.value));
    timelineSlider.addEventListener('change', (event) => onScrubChange(event.target.value));
  }
}
