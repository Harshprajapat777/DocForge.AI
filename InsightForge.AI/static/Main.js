/* ================================================================
   InsightForge.AI — Main.js
   Handles chat UI: send query, render answer, citations, steps
   ================================================================ */

const API_URL   = '/chat';
const LOGO_URL  = '/logo/InsightAI.jpg';

const welcomeEl  = document.getElementById('welcome');
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('queryInput');
const sendBtn    = document.getElementById('sendBtn');

let isLoading = false;

// ── Sidebar quick-query ─────────────────────────────────────────
function setQuery(text) {
  inputEl.value = text;
  autoResize(inputEl);
  inputEl.focus();
}

// ── Auto-resize textarea ────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Enter = send, Shift+Enter = new line ────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

// ── Show messages panel, hide welcome ──────────────────────────
function showMessages() {
  if (welcomeEl) welcomeEl.style.display = 'none';
  messagesEl.style.display = 'flex';
}

// ── Scroll chat to bottom ───────────────────────────────────────
function scrollBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

// ── Format timestamp ────────────────────────────────────────────
function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

// ── Append user message bubble ──────────────────────────────────
function appendUserMsg(text) {
  showMessages();
  const el = document.createElement('div');
  el.className = 'msg-user';
  el.innerHTML = `<div class="msg-user-bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(el);
  scrollBottom();
}

// ── Append typing indicator ─────────────────────────────────────
function appendTyping() {
  const el = document.createElement('div');
  el.className = 'msg-agent typing-indicator';
  el.id = 'typing';
  el.innerHTML = `
    <div class="agent-avatar"><img src="${LOGO_URL}" alt="Agent" /></div>
    <div class="typing-bubble">
      <span class="typing-text">InsightForge.AI is thinking</span>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  messagesEl.appendChild(el);
  scrollBottom();
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

// ── Append agent response ───────────────────────────────────────
function appendAgentMsg(data) {
  const el = document.createElement('div');
  el.className = 'msg-agent';

  // Citations HTML
  let citationsHtml = '';
  if (data.citations && data.citations.length) {
    const badges = data.citations
      .map(c => `<span class="citation-badge">${escapeHtml(c)}</span>`)
      .join('');
    citationsHtml = `
      <div class="citations-row">
        <span class="citations-label">Citations</span>
        ${badges}
      </div>`;
  }

  // Tools used HTML
  let toolsHtml = '';
  if (data.tools_used && data.tools_used.length) {
    const tags = data.tools_used
      .map(t => `<span class="tool-tag">${escapeHtml(t)}</span>`)
      .join('');
    toolsHtml = `<div class="tools-row">${tags}</div>`;
  }

  // Agent steps accordion
  let stepsHtml = '';
  if (data.agent_steps && data.agent_steps.length) {
    const stepItems = data.agent_steps.map((s, i) => `
      <div class="step-item">
        ${s.thought ? `<div class="step-thought"><strong>Thought ${i+1}:</strong> ${escapeHtml(s.thought)}</div>` : ''}
        ${s.action  ? `<div class="step-action">&#9881; Tool: ${escapeHtml(s.action)}</div>` : ''}
        ${s.observation ? `<div class="step-obs">${escapeHtml(s.observation)}</div>` : ''}
      </div>`).join('');

    stepsHtml = `
      <div class="steps-accordion">
        <button class="steps-toggle" onclick="toggleSteps(this)">
          &#128270; Agent Reasoning &amp; Tool Calls (${data.agent_steps.length} steps)
          <span class="arrow">&#9660;</span>
        </button>
        <div class="steps-body hidden">${stepItems}</div>
      </div>`;
  }

  el.innerHTML = `
    <div class="agent-avatar"><img src="${LOGO_URL}" alt="Agent" /></div>
    <div class="msg-agent-body">
      <div class="agent-name">InsightForge.AI</div>
      <div class="agent-answer-card">
        ${escapeHtml(data.answer)}
        ${citationsHtml}
        ${toolsHtml}
        ${stepsHtml}
      </div>
      <div class="msg-time">${fmtTime(data.timestamp)}</div>
    </div>`;

  messagesEl.appendChild(el);
  scrollBottom();
}

// ── Append error card ───────────────────────────────────────────
function appendError(msg) {
  const el = document.createElement('div');
  el.className = 'msg-agent';
  el.innerHTML = `
    <div class="agent-avatar"><img src="${LOGO_URL}" alt="Agent" /></div>
    <div class="msg-agent-body">
      <div class="agent-name">InsightForge.AI</div>
      <div class="error-card">&#9888; ${escapeHtml(msg)}</div>
    </div>`;
  messagesEl.appendChild(el);
  scrollBottom();
}

// ── Toggle steps accordion ──────────────────────────────────────
function toggleSteps(btn) {
  btn.classList.toggle('open');
  const body = btn.nextElementSibling;
  body.classList.toggle('hidden');
}

// ── Main send function ──────────────────────────────────────────
async function sendQuery() {
  if (isLoading) return;
  const query = inputEl.value.trim();
  if (!query) return;

  isLoading = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  appendUserMsg(query);
  appendTyping();

  try {
    const res = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query }),
    });

    removeTyping();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      appendError(err.detail || `Server error ${res.status}`);
      return;
    }

    const data = await res.json();
    appendAgentMsg(data);

  } catch (err) {
    removeTyping();
    appendError('Could not reach the server. Is it running on port 8001?');
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ── XSS-safe HTML escape ────────────────────────────────────────
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
