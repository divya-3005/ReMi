/**
 * app.js — ReMi frontend application logic
 *
 * Architecture:
 * - State is a single plain object; UI functions read from it, never from the DOM.
 * - Citation markers are <button> elements injected from parsed Markdown.
 * - The ONLY animation is the source highlight fade (200ms ease-out).
 * - All API calls go through apiFetch() for consistent error handling.
 */

'use strict';

/* ── Configuration ────────────────────────────────────────────── */
let API_BASE = window.API_BASE || '';
if (!API_BASE) {
  // If running locally but NOT on the FastAPI port (8000), point to localhost:8000
  if (['localhost', '127.0.0.1'].includes(window.location.hostname) && window.location.port !== '8000') {
    API_BASE = 'http://localhost:8000';
  } else if (window.location.protocol === 'file:') {
    API_BASE = 'http://localhost:8000';
  } else {
    API_BASE = ''; // Use relative paths for Docker/Render deployments
  }
}

/* ── Application State ────────────────────────────────────────── */
let state = {
  documents: [],      // DocumentInfo[]
  report: null,       // ResearchResponse | null
  activeCitation: null, // footnote_id | null
  isResearching: false,
};

/* ── DOM References ───────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const dom = {
  docList:          $('doc-list'),
  docEmptyState:    $('doc-empty-state'),
  fileInput:        $('file-input'),
  uploadStatus:     $('upload-status'),
  queryInput:       $('query-input'),
  researchBtn:      $('research-btn'),
  reportArea:       $('report-area'),
  workflowTrace:    $('workflow-trace'),
  evaluationPanel:  $('evaluation-panel'),
  sourcePrompt:     $('source-prompt'),
  sourceViewer:     $('source-viewer'),
  sourceMeta:       $('source-meta'),
  sourceText:       $('source-text'),
  sourceSheet:      $('source-sheet'),
  sourceSheetClose: $('source-sheet-close'),
  sourceMetaMobile: $('source-meta-mobile'),
  sourceTextMobile: $('source-text-mobile'),
  sourceBackdrop:   $('source-sheet-backdrop'),
  toast:            $('toast'),
};

/* ── API Helpers ──────────────────────────────────────────────── */
async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const msg = data?.detail?.message || data?.detail || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return data;
  } catch (err) {
    if (err.name === 'TypeError') {
      throw new Error('Cannot reach the ReMi API. Is the server running on ' + API_BASE + '?');
    }
    throw err;
  }
}

/* ── Toast ────────────────────────────────────────────────────── */
let toastTimeout;
function showToast(message, duration = 3500) {
  clearTimeout(toastTimeout);
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  dom.toast.classList.remove('toast-hidden');
  toastTimeout = setTimeout(() => {
    dom.toast.classList.add('toast-hidden');
    setTimeout(() => { dom.toast.hidden = true; }, 150);
  }, duration);
}

/* ── Document List ────────────────────────────────────────────── */
async function loadDocuments() {
  try {
    const docs = await apiFetch('/documents');
    state.documents = docs;
    renderDocumentList();
  } catch {
    // Silently fail on initial load — API may not be up yet
  }
}

function renderDocumentList() {
  const docs = state.documents;

  if (docs.length === 0) {
    dom.docList.innerHTML = '';
    if (!dom.docEmptyState.parentNode) dom.docList.appendChild(dom.docEmptyState);
    return;
  }

  dom.docList.innerHTML = '';
  docs.forEach(doc => {
    const card = document.createElement('div');
    card.className = 'doc-card';
    card.setAttribute('role', 'listitem');
    card.innerHTML = `
      <span class="doc-filename">${escHtml(doc.filename)}</span>
      <span class="doc-meta">${doc.num_pages} pp · ${doc.chunk_count} chunks</span>
    `;
    dom.docList.appendChild(card);
  });
}

/* ── File Upload ──────────────────────────────────────────────── */
dom.fileInput.addEventListener('change', async () => {
  const file = dom.fileInput.files[0];
  if (!file) return;

  dom.uploadStatus.textContent = `Uploading ${file.name}…`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const result = await apiFetch('/documents/upload', {
      method: 'POST',
      body: formData,
    });
    dom.uploadStatus.textContent = `✓ ${result.chunks_indexed} chunks indexed`;
    await loadDocuments();
    showToast(`${result.filename} indexed — ${result.chunks_indexed} chunks ready.`);
  } catch (err) {
    dom.uploadStatus.textContent = `Upload failed.`;
    showToast(`Upload failed: ${err.message}`, 5000);
  } finally {
    dom.fileInput.value = '';
  }
});

/* ── Research ─────────────────────────────────────────────────── */
dom.researchBtn.addEventListener('click', runResearch);
dom.queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) runResearch();
});

async function runResearch() {
  const query = dom.queryInput.value.trim();
  if (!query || state.isResearching) return;

  state.isResearching = true;
  state.activeCitation = null;
  dom.researchBtn.disabled = true;

  // Loading state (skeleton loader)
  dom.reportArea.innerHTML = `
    <div class="report-loading" aria-label="Researching...">
      <div class="skeleton-line skeleton-title skeleton-line w-40"></div>
      <div class="skeleton-block" style="margin-bottom: 2rem;">
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-90"></div>
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-80"></div>
      </div>
      <div class="skeleton-title skeleton-line w-60"></div>
      <div class="skeleton-block">
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-60"></div>
      </div>
    </div>
  `;
  dom.workflowTrace.hidden = true;
  dom.evaluationPanel.hidden = true;
  resetSourceViewer();

  try {
    const result = await apiFetch('/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    state.report = result;
    renderReport(result);
    renderWorkflowTrace(result.workflow_attempts);
    renderEvaluationPanel(result.evaluation, result.elapsed_seconds);

  } catch (err) {
    dom.reportArea.innerHTML = `<p class="empty-state">Research failed: ${escHtml(err.message)}</p>`;
    showToast(`Research failed: ${err.message}`, 6000);
  } finally {
    state.isResearching = false;
    dom.researchBtn.disabled = false;
  }
}

/* ── Report Rendering ─────────────────────────────────────────── */
function renderReport(result) {
  const container = document.createElement('div');
  container.className = 'report-content';

  // Convert Markdown → HTML with citation markers injected
  const html = markdownToHtml(result.answer_text, result.citations);
  container.innerHTML = html;

  // Inject citation marker buttons
  injectCitationMarkers(container, result.citations);

  // Build footnote list
  if (result.citations.length > 0) {
    const footnoteList = buildFootnoteList(result.citations, result);
    container.appendChild(footnoteList);
  }

  dom.reportArea.innerHTML = '';
  dom.reportArea.appendChild(container);
}

/**
 * Very minimal Markdown → HTML converter.
 * Handles: h1/h2/h3 (# prefix), paragraphs, **bold**, hr (---).
 * [^N] markers are replaced with citation button placeholders.
 */
function markdownToHtml(markdown, citations) {
  const citedIds = new Set((citations || []).map(c => c.footnote_id));

  let html = escHtml(markdown)
    // h1/h2/h3
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // horizontal rule
    .replace(/^---$/gm, '<hr>')
    // citation markers → data attribute placeholders
    .replace(/\[\^(\d+)\]/g, (_, n) => `<cite-marker data-fn="${n}"></cite-marker>`)
    // paragraphs (double newline)
    .replace(/\n\n+/g, '</p><p>')
    // wrap in p
    .replace(/^(.+)$/, '<p>$1</p>');

  return html;
}

/**
 * Replace <cite-marker> placeholders with real <button> elements.
 */
function injectCitationMarkers(container, citations) {
  const markers = container.querySelectorAll('cite-marker');
  markers.forEach(marker => {
    const fn = parseInt(marker.dataset.fn, 10);
    const citation = citations.find(c => c.footnote_id === fn);

    const btn = document.createElement('button');
    btn.className = 'citation-marker';
    btn.setAttribute('aria-label', `Citation ${fn} — click to view source`);
    btn.setAttribute('data-footnote-id', fn);
    btn.textContent = fn;

    if (citation) {
      btn.addEventListener('click', () => activateCitation(citation, btn));
    } else {
      btn.disabled = true;
      btn.setAttribute('aria-label', `Citation ${fn} — source not found`);
    }

    marker.replaceWith(btn);
  });
}

function buildFootnoteList(citations, result) {
  const ol = document.createElement('ol');
  ol.className = 'footnote-list';

  // Deduplicate by footnote_id
  const seen = new Set();
  citations.forEach(cit => {
    if (seen.has(cit.footnote_id)) return;
    seen.add(cit.footnote_id);

    // Find document info
    const doc = result._docRegistry?.[cit.chunk_id] || null;

    const li = document.createElement('li');
    li.className = 'footnote-item';
    li.innerHTML = `
      <span class="footnote-num">${cit.footnote_id}</span>
      <span>${escHtml(cit.excerpt || '(source excerpt)')}</span>
    `;
    ol.appendChild(li);
  });

  return ol;
}

/* ── Citation Activation — the signature interaction ─────────── */
function activateCitation(citation, buttonEl) {
  // Deactivate previous
  document.querySelectorAll('button.citation-marker.is-active').forEach(b => {
    b.classList.remove('is-active');
  });

  // Activate this marker
  buttonEl.classList.add('is-active');
  state.activeCitation = citation.footnote_id;

  // Show source with highlighted span
  showSourceHighlight(citation);
}

function showSourceHighlight(citation) {
  // Find the chunk text from the last research result
  const chunks = getChunksFromReport();
  const chunk = chunks.find(c => c.chunk_id === citation.chunk_id);
  const chunkText = chunk?.text || '[Source text not available]';

  const relStart = citation.char_start - (chunk?.char_start ?? 0);
  const relEnd   = citation.char_end   - (chunk?.char_start ?? 0);

  const before = escHtml(chunkText.slice(0, relStart));
  const highlighted = escHtml(chunkText.slice(relStart, relEnd));
  const after = escHtml(chunkText.slice(relEnd));

  const metaHtml = `chunk_id: ${escHtml(citation.chunk_id)}<br>char_start: ${citation.char_start} · char_end: ${citation.char_end}`;
  const sourceHtml = `${before}<mark class="source-highlight">${highlighted}</mark>${after}`;

  const isMobile = window.innerWidth < 768;

  if (isMobile) {
    dom.sourceMetaMobile.innerHTML = metaHtml;
    dom.sourceTextMobile.innerHTML = sourceHtml;
    openSourceSheet();
  } else {
    dom.sourceMeta.innerHTML = metaHtml;
    dom.sourceText.innerHTML = sourceHtml;
    dom.sourcePrompt.hidden = true;
    dom.sourceViewer.hidden = false;

    // Scroll the highlighted span into view within the source panel
    requestAnimationFrame(() => {
      const mark = dom.sourceText.querySelector('.source-highlight');
      if (mark) {
        mark.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    });
  }
}

function getChunksFromReport() {
  // Extract chunk data surfaced in the research response
  // The API sends contexts via citations; we reconstruct what we can
  if (!state.report) return [];
  // We have citation.chunk_id, char_start, char_end, excerpt — enough for highlighting
  return (state.report.citations || []).map(cit => ({
    chunk_id: cit.chunk_id,
    text: cit.excerpt || '',
    char_start: cit.char_start,
    char_end: cit.char_end,
  }));
}

function resetSourceViewer() {
  dom.sourcePrompt.hidden = false;
  dom.sourceViewer.hidden = true;
  dom.sourceMeta.innerHTML = '';
  dom.sourceText.innerHTML = '';
}

/* ── Mobile bottom sheet ──────────────────────────────────────── */
function openSourceSheet() {
  dom.sourceSheet.hidden = false;
  dom.sourceSheet.removeAttribute('hidden');
  dom.sourceBackdrop.hidden = false;
  // Allow focus management
  dom.sourceSheetClose.focus();
}

function closeSourceSheet() {
  dom.sourceSheet.hidden = true;
  dom.sourceBackdrop.hidden = true;
}

dom.sourceSheetClose.addEventListener('click', closeSourceSheet);
dom.sourceBackdrop.addEventListener('click', closeSourceSheet);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !dom.sourceSheet.hidden) closeSourceSheet();
});

/* ── Workflow Trace Rendering ─────────────────────────────────── */
function renderWorkflowTrace(attempts) {
  if (!attempts || attempts.length === 0) {
    dom.workflowTrace.hidden = true;
    return;
  }

  if (attempts.length === 1 && !attempts[0].triggered_retry) {
    // Single passing attempt — no need to show trace
    dom.workflowTrace.hidden = true;
    return;
  }

  let html = `<h3 class="workflow-trace-heading">Workflow Trace</h3>`;

  attempts.forEach((attempt, i) => {
    const ev = attempt.evaluation;
    const isRetry = attempt.triggered_retry;
    const attemptClass = isRetry ? 'workflow-attempt is-retry' : 'workflow-attempt';
    const arrow = isRetry ? `<span class="attempt-arrow">→ reformulated queries</span>` : '';

    html += `
      <div class="${attemptClass}" aria-label="Attempt ${i + 1}${isRetry ? ', triggered retry' : ', final'}">
        <span class="attempt-scores">
          attempt ${attempt.attempt_number + 1}&nbsp;&nbsp;
          citation_coverage: ${ev.citation_coverage.toFixed(2)}&nbsp;&nbsp;
          citation_utilization: ${ev.citation_utilization.toFixed(2)}&nbsp;&nbsp;
          relevance: ${ev.answer_relevance.toFixed(2)}
        </span>
        ${arrow}
      </div>
    `;
  });

  dom.workflowTrace.innerHTML = html;
  dom.workflowTrace.hidden = false;
}

/* ── Evaluation Panel Rendering ───────────────────────────────── */
function renderEvaluationPanel(ev, elapsed) {
  if (!ev) {
    dom.evaluationPanel.hidden = true;
    return;
  }

  dom.evaluationPanel.innerHTML = `
    <h3 class="eval-heading">Quality Scores</h3>
    <div class="eval-grid">
      <div class="eval-item">
        <span class="eval-label">citation_coverage</span>
        <span class="eval-value">${ev.citation_coverage.toFixed(2)}</span>
      </div>
      <div class="eval-item">
        <span class="eval-label">citation_utilization</span>
        <span class="eval-value">${ev.citation_utilization.toFixed(2)}</span>
      </div>
      <div class="eval-item">
        <span class="eval-label">answer_relevance</span>
        <span class="eval-value">${ev.answer_relevance.toFixed(2)}</span>
      </div>
      <div class="eval-item">
        <span class="eval-label">hallucination_risk</span>
        <span class="eval-value">${ev.hallucination_risk.toFixed(2)}</span>
      </div>
    </div>
    <div class="eval-elapsed">elapsed: ${elapsed.toFixed(1)}s</div>
  `;
  dom.evaluationPanel.hidden = false;
}

/* ── Utilities ────────────────────────────────────────────────── */
function escHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ── Initialise ───────────────────────────────────────────────── */
(async function init() {
  await loadDocuments();
})();
