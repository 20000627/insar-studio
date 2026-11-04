/* InSAR Studio · local web client
   Vanilla JS: guided wizard, live job console, SVG network and product viewer. */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const TECH_FALLBACK = {
    dinsar: {
      name: 'D-InSAR', cn: '差分干涉', badge: '快速',
      tagline: '单对 / 多对差分，快速提取同震、滑坡、沉降形变',
      desc: '两景 SAR 直接差分，几分钟得到视线向位移。适合事件响应和快速踏勘。'
    },
    ps: {
      name: 'PS-InSAR', cn: '永久散射体', badge: '毫米级',
      tagline: '相干点时序，适合城市建筑、岩石裸露区毫米级监测',
      desc: '挑选长期稳定的强散射点，保留点目标相位，适合城区与裸岩。'
    },
    sbas: {
      name: 'SBAS-InSAR', cn: '小基线集', badge: '推荐',
      tagline: '多主影像小基线网络，兼顾精度与空间覆盖',
      desc: '短基线组合 + 联合时序反演，是沉降、滑坡监测的稳健默认路线。'
    },
    ds: {
      name: 'DS-InSAR', cn: '分布式散射体', badge: '前沿',
      tagline: '相位链接 + 自适应多视，适合植被、裸土等低相干区',
      desc: '先把相邻像元做相位链接，再对分布式目标做时序反演。'
    }
  };

  const FIELD_LABELS = {
    wrapped: '缠绕相位',
    coherence: '相干性',
    unwrapped: '解缠相位',
    displacement: '位移',
    atmcorr: '大气校正相位',
    atmosphere: '大气延迟'
  };

  const GROUP_LABELS = {
    arrays: '数值产品 · NumPy',
    geotiff: '地理栅格 · GeoTIFF',
    figures: '图件 · PNG',
    report: '报告 · Markdown / DOCX',
    exports: '成果包 · ZIP',
    root: '其他'
  };

  const state = {
    cap: null,
    projects: [],
    projectId: null,
    project: null,
    overview: null,
    step: 'project',
    mode: localStorage.getItem('insar.mode') || 'guided',
    selectedTech: null,
    selectedPair: null,
    pairField: 'unwrapped',
    atmSource: null,
    results: null,
    resultsLoading: false,
    resultsError: '',
    peaks: null,
    series: null,
    seriesPoint: null,
    jobId: null,
    job: null,
    jobProjectId: null,
    jobRunning: false,
    jobTimer: null,
    dlResults: [],
    searchBusy: false,
    newProject: null
  };

  /* --------------------------------------------------------------- utils */
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function fmt(value, digits = 2, fallback = '—') {
    if (value === null || value === undefined || value === '') return fallback;
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return n.toFixed(digits);
  }

  function fmtSigned(value, digits = 2, unit = '') {
    if (value === null || value === undefined) return '—';
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}${unit}`;
  }

  function fmtPct(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function fmtBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
  }

  function todayISO(offsetDays = 0) {
    const d = new Date();
    d.setDate(d.getDate() + offsetDays);
    return d.toISOString().slice(0, 10);
  }

  function shortDate(value) {
    if (!value) return '—';
    return String(value).slice(5) || String(value);
  }

  function slug(value) {
    return String(value || '').replace(/[^a-z0-9_]/gi, '_');
  }

  function techMeta(key) {
    const capTech = state.cap && state.cap.techniques ? state.cap.techniques[key] : null;
    return { ...(TECH_FALLBACK[key] || TECH_FALLBACK.sbas), ...(capTech || {}) };
  }

  function techDefaults(key) {
    if (key === 'ps') return { multilook: 1, max_temporal_baseline: 96, max_pairs: 140, ps_star_all: true };
    if (key === 'ds') return { multilook: 2, max_temporal_baseline: 72, ds_patch: 9, ds_overlap: 0.5 };
    if (key === 'dinsar') return { multilook: 2 };
    return { multilook: 2 };
  }

  function applyTechDefaults(key) {
    if (!state.project) return;
    state.project.params = { ...(state.project.params || {}), ...techDefaults(key) };
  }

  function currentTech() {
    return state.selectedTech || (state.project && state.project.technique) || 'sbas';
  }

  function pval(key, fallback = undefined) {
    const p = state.project && state.project.params ? state.project.params : {};
    if (Object.prototype.hasOwnProperty.call(p, key) && p[key] !== undefined) return p[key];
    if (fallback !== undefined) return fallback;
    if (state.cap && state.cap.defaults && key in state.cap.defaults) return state.cap.defaults[key];
    return undefined;
  }

  function statusFor(key) {
    if (!state.overview || !state.overview.status) return { done: false, detail: '' };
    return state.overview.status.find(s => s.key === key) || { done: false, detail: '' };
  }

  function isDone(key) {
    return !!statusFor(key).done;
  }

  function assetUrl(rel) {
    if (!state.projectId || !rel) return '';
    return `/api/projects/${encodeURIComponent(state.projectId)}/asset/${rel.split('/').map(encodeURIComponent).join('/')}`;
  }

  function downloadUrl(rel) {
    if (!state.projectId || !rel) return '#';
    return `/api/projects/${encodeURIComponent(state.projectId)}/download/${rel.split('/').map(encodeURIComponent).join('/')}`;
  }

  async function api(path, options = {}) {
    const opts = { ...options };
    opts.headers = { ...(options.headers || {}) };
    if (opts.body && typeof opts.body !== 'string') {
      opts.body = JSON.stringify(opts.body);
      opts.headers['Content-Type'] = 'application/json';
    }
    const response = await fetch(path, opts);
    const type = response.headers.get('content-type') || '';
    let data = null;
    if (type.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = text ? { message: text } : {};
    }
    if (!response.ok) {
      const msg = (data && (data.error || data.message)) || `${response.status} ${response.statusText}`;
      const error = new Error(msg);
      error.status = response.status;
      error.payload = data;
      throw error;
    }
    return data;
  }

  function toast(message, type = 'info') {
    const host = $('#toast-host');
    if (!host) return;
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    const glyph = type === 'error' ? '!' : (type === 'success' ? '✓' : '·');
    el.innerHTML = `<span class="toast-glyph">${glyph}</span><span>${esc(message)}</span>`;
    host.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity .3s';
      setTimeout(() => el.remove(), 320);
    }, type === 'error' ? 6500 : 4200);
  }

  /* --------------------------------------------------------------- modal */
  function showModal(inner, wide = false) {
    const root = $('#modal-root');
    root.innerHTML = `<div class="modal-backdrop"><div class="modal ${wide ? 'wide' : ''}" role="dialog" aria-modal="true">${inner}</div></div>`;
    document.body.style.overflow = 'hidden';
    const backdrop = $('.modal-backdrop', root);
    backdrop.addEventListener('mousedown', ev => { if (ev.target === backdrop) closeModal(); });
    $$('.modal-close', root).forEach(btn => btn.addEventListener('click', closeModal));
  }

  function closeModal() {
    $('#modal-root').innerHTML = '';
    document.body.style.overflow = '';
  }

  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape' && $('#modal-root').innerHTML) closeModal();
  });

  /* --------------------------------------------------------------- init */
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    bindGlobal();
    applyMode(state.mode, false);
    try {
      state.cap = await api('/api/capabilities');
      $('#version-chip').textContent = `v${state.cap.version || '1.0.0'}`;
      const ws = $('#workspace-path');
      if (ws && state.cap.workspace) {
        ws.textContent = state.cap.workspace;
        ws.title = state.cap.workspace;
      }
    } catch (err) {
      toast(`无法读取系统能力：${err.message}`, 'error');
    }
    await loadProjects();
    const last = localStorage.getItem('insar.lastProject');
    if (last && state.projects.some(p => p.id === last)) {
      await openProject(last);
    } else {
      renderNoProject();
    }
  }

  function bindGlobal() {
    $('#btn-new').addEventListener('click', () => openNewProject());
    $('#btn-help').addEventListener('click', openHelp);
    $('#btn-doctor').addEventListener('click', openDoctor);
    $('#btn-cancel-job').addEventListener('click', cancelJob);
    $('#btn-clear-log').addEventListener('click', () => {
      const log = $('#console-log');
      if (log) log.innerHTML = '<div class="console-empty"><span class="console-glyph">⌁</span><p>日志区已清空，任务记录仍保存在工作区。</p></div>';
      $('#btn-clear-log').disabled = true;
    });
    $('#project-search').addEventListener('input', renderSidebar);
    const consolePanel = $('#console');
    if (consolePanel) {
      const head = $('.console-head', consolePanel);
      const toggleConsole = () => {
        if (!window.matchMedia('(max-width: 1180px)').matches) return;
        const open = consolePanel.classList.toggle('open');
        head.setAttribute('aria-expanded', open ? 'true' : 'false');
      };
      head.setAttribute('role', 'button');
      head.setAttribute('tabindex', '0');
      head.setAttribute('aria-expanded', 'false');
      head.setAttribute('title', '点击展开 / 收起任务控制台');
      head.addEventListener('click', toggleConsole);
      head.addEventListener('keydown', ev => {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggleConsole(); }
      });
    }
    $$('.mode-switch button').forEach(btn => {
      btn.addEventListener('click', () => applyMode(btn.dataset.mode, true));
    });
    document.addEventListener('click', ev => {
      const fig = ev.target.closest('.figure-card');
      if (fig && fig.dataset.src) {
        const box = document.createElement('div');
        box.className = 'lightbox';
        box.innerHTML = `<img src="${esc(fig.dataset.src)}" alt="">`;
        box.addEventListener('click', () => box.remove());
        document.body.appendChild(box);
      }
    });
    window.addEventListener('error', ev => {
      if (ev && ev.message) console.error(ev.message);
    });
  }

  function applyMode(mode, rerender = true) {
    state.mode = mode === 'expert' ? 'expert' : 'guided';
    localStorage.setItem('insar.mode', state.mode);
    document.body.classList.toggle('expert', state.mode === 'expert');
    $$('.mode-switch button').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === state.mode));
    if (rerender && state.overview) {
      try {
        const p = collectParams($('#step-content'));
        state.project.params = { ...state.project.params, ...p };
      } catch (err) { /* panel has no form */ }
      renderStepContent();
    }
  }

  /* --------------------------------------------------------------- projects */
  async function loadProjects() {
    try {
      const data = await api('/api/projects');
      state.projects = data.projects || [];
      renderSidebar();
    } catch (err) {
      toast(`项目列表读取失败：${err.message}`, 'error');
    }
  }

  function renderSidebar() {
    const host = $('#project-list');
    if (!host) return;
    const query = ($('#project-search') && $('#project-search').value || '').trim().toLowerCase();
    const rows = state.projects.filter(p => !query || `${p.name} ${p.id}`.toLowerCase().includes(query));
    if (!rows.length) {
      host.innerHTML = `<div class="rail-empty">${state.projects.length ? '没有匹配的项目。' : '还没有项目。<br>点击「新建」创建演示项目或导入数据。'}</div>`;
      return;
    }
    host.innerHTML = rows.map(p => {
      const tech = techMeta(p.technique);
      const active = p.id === state.projectId ? ' active' : '';
      const job = p.active_job;
      const dates = Array.isArray(p.dates) && p.dates.length === 2
        ? `${p.dates[0] || '—'} → ${p.dates[1] || '—'}` : '';
      const pct = job ? Math.round((Number(job.progress) || 0) * 100)
        : (p.steps_total ? Math.round(100 * p.steps_done / p.steps_total) : 0);
      const badge = job ? `<span class="badge amber"><i class="pulse-dot"></i>${job.status === 'cancelling' ? '取消中' : '运行中'}</span>`
        : (p.demo ? '<span class="badge amber">演示</span>' : '<span class="badge grey">导入</span>');
      const staleText = p.stale ? ' · 参数待重跑' : '';
      return `<button class="project-item${active}${job ? ' running' : ''}" data-project="${esc(p.id)}">
        <div class="pi-top">
          <span class="pi-name" title="${esc(p.name)}">${esc(p.name)}</span>
          ${badge}
        </div>
        <div class="pi-meta">
          <span>${esc(tech.name)}</span>
          <span>·</span>
          <span>${p.n_dates || 0} 景</span>
          <span style="margin-left:auto">${job ? `${pct}%` : `${p.steps_done}/${p.steps_total}`}</span>
        </div>
        <div class="pi-dates">${esc(dates || p.id)}${staleText}</div>
        <div class="progress-inline"><i style="width:${pct}%"></i></div>
      </button>`;
    }).join('');
    $$('.project-item', host).forEach(btn => {
      btn.addEventListener('click', () => openProject(btn.dataset.project));
    });
  }

  function renderNoProject() {
    if (state.jobTimer) { clearTimeout(state.jobTimer); state.jobTimer = null; }
    state.projectId = null;
    state.project = null;
    state.overview = null;
    state.jobId = null;
    state.job = null;
    state.jobProjectId = null;
    state.jobRunning = false;
    state.step = 'project';
    const head = $('#stage-head');
    if (head) {
      head.innerHTML = `<div class="stage-title-row"><div class="stage-title"><span class="eyebrow">INSAR STUDIO</span><h2>从一叠 SAR 影像，到一张可信的形变图</h2><div class="sub">配准、干涉、解缠、大气校正、时序反演与成果导出，全部在本地完成。</div></div></div>`;
    }
    const stepper = $('#stepper');
    if (stepper) stepper.innerHTML = '';
    const content = $('#step-content');
    if (content) {
      content.innerHTML = welcomeHtml();
      $('#welcome-new').addEventListener('click', openNewProject);
      $('#welcome-demo').addEventListener('click', runQuickDemo);
    }
  }

  function welcomeHtml() {
    return `<section class="welcome">
      <div class="welcome-inner">
        <span class="eyebrow">LOCAL · OFFLINE-FIRST</span>
        <h2>把形变看见</h2>
        <p>InSAR Studio 把复杂的时序干涉流程收进七步向导：创建项目、载入影像、设计网络、生成干涉图、校正大气、时序反演，最后导出速率图、时间序列与处理报告。演示数据自带真值，可先跑通全链路，再换成自己的数据。</p>
        <div class="welcome-actions">
          <button class="btn btn-primary" id="welcome-new">＋ 新建项目</button>
          <button class="btn btn-amber" id="welcome-demo">用演示数据跑一遍</button>
          <button class="btn btn-ghost" onclick="document.getElementById('btn-help').click()">查看使用说明</button>
        </div>
        <div class="welcome-features">
          <div class="welcome-feature"><b>四条技术路线</b><span>D-InSAR · PS · SBAS · DS</span></div>
          <div class="welcome-feature"><b>大气校正</b><span>GACOS / 经验高程 / 关闭</span></div>
          <div class="welcome-feature"><b>真值校验</b><span>合成数据逐步对账</span></div>
          <div class="welcome-feature"><b>可交付成果</b><span>GeoTIFF · CSV · DOCX · ZIP</span></div>
        </div>
      </div>
    </section>`;
  }

  async function openProject(id) {
    if (!id) return;
    if (state.jobTimer) { clearTimeout(state.jobTimer); state.jobTimer = null; }
    state.jobId = null;
    state.job = null;
    state.jobRunning = false;
    document.body.classList.remove('job-active');
    state.projectId = id;
    localStorage.setItem('insar.lastProject', id);
    try {
      const ov = await api(`/api/projects/${encodeURIComponent(id)}`);
      state.overview = ov;
      state.project = ov.project;
      state.selectedTech = ov.project.technique || 'sbas';
      state.results = null;
      state.peaks = null;
      state.series = null;
      state.seriesPoint = null;
      state.selectedPair = ov.pairs && ov.pairs.length ? ov.pairs[0].id : null;
      state.atmSource = (ov.atmosphere && ov.atmosphere.source) || (ov.project.demo ? 'demo' : 'local');
      state.step = firstPendingStep(ov);
      renderSidebar();
      renderAll();
      if (ov.active_job) {
        attachJob(ov.active_job, { notify: false });
        toast(`检测到「${ov.active_job.label || ov.active_job.kind}」正在运行，已重新连接。`, 'info');
      }
    } catch (err) {
      toast(`项目打开失败：${err.message}`, 'error');
    }
  }

  function firstPendingStep(ov) {
    if (!ov.stack || !ov.stack.n_dates) return 'data';
    const stale = (ov.status || []).find(s => s.stale);
    if (stale && stale.key !== 'project' && stale.key !== 'data') return stale.key;
    if (!ov.pairs || !ov.pairs.length) return 'network';
    if (!ov.interferometry || !ov.interferometry.n_pairs) return 'interferometry';
    if (!ov.atmosphere || !ov.atmosphere.method) return 'atmosphere';
    if (!ov.inversion || !ov.inversion.technique) return 'inversion';
    return 'results';
  }

  async function refreshOverview({ keepStep = true } = {}) {
    if (!state.projectId) return;
    const ov = await api(`/api/projects/${encodeURIComponent(state.projectId)}`);
    state.overview = ov;
    state.project = ov.project;
    if (!state.selectedTech) state.selectedTech = ov.project.technique;
    if (!state.selectedPair && ov.pairs && ov.pairs.length) state.selectedPair = ov.pairs[0].id;
    if (!keepStep) state.step = firstPendingStep(ov);
    await loadProjects();
  }

  function renderAll() {
    if (!state.overview) {
      renderNoProject();
      return;
    }
    renderHeader();
    renderStepper();
    renderStepContent();
    renderSidebar();
  }

  function renderHeader() {
    const ov = state.overview;
    if (!ov) return;
    const p = ov.project;
    const tech = techMeta(currentTech());
    const dates = ov.stack && ov.stack.dates ? ov.stack.dates : [];
    const dateText = dates.length ? `${dates[0]} → ${dates[dates.length - 1]}` : '暂无影像';
    const demo = p.demo ? '<span class="demo-pill technique-pill">合成演示 · 有真值</span>' : '';
    const head = $('#stage-head');
    head.innerHTML = `<div class="stage-title-row">
      <div class="stage-title">
        <span class="eyebrow">PROJECT · ${esc(p.id)}</span>
        <h2>${esc(p.name)}</h2>
        <div class="sub">
          <span class="technique-pill"><i class="dot"></i>${esc(tech.name)} · ${esc(tech.cn)}</span>
          ${demo}
          <span>${esc(dateText)}</span>
          <span class="meta-sep">/</span>
          <span>${ov.n_dates || 0} 景</span>
          <span class="meta-sep">/</span>
          <span>${ov.stack && ov.stack.rows ? `${ov.stack.rows} × ${ov.stack.cols} px` : '尺寸未知'}</span>
        </div>
      </div>
      <div class="stage-actions">
        <button class="btn btn-ghost" id="btn-health">项目体检</button>
        <button class="btn btn-ghost" id="btn-refresh">↻ 刷新</button>
        ${isDone('inversion') ? '<button class="btn btn-ghost" id="btn-validate">✓ 重新校验</button>' : ''}
        <button class="btn btn-amber" id="btn-run-all" data-run>▶ 一键跑完全流程</button>
      </div>
    </div>`;
    $('#btn-health').addEventListener('click', openHealth);
    $('#btn-refresh').addEventListener('click', async () => {
      await refreshOverview({ keepStep: true });
      state.results = null;
      renderAll();
      toast('项目状态已刷新');
    });
    const validateBtn = $('#btn-validate');
    if (validateBtn) validateBtn.addEventListener('click', validateNow);
    $('#btn-run-all').addEventListener('click', () => runAll());
    if (state.jobRunning) $$('[data-run]').forEach(b => { b.disabled = true; });
  }

  function renderStepper() {
    if (!state.overview) return;
    const labels = { project: '项目', data: '数据', network: '网络', interferometry: '干涉', atmosphere: '大气', inversion: '反演', results: '成果' };
    const keys = ['project', 'data', 'network', 'interferometry', 'atmosphere', 'inversion', 'results'];
    const stepper = $('#stepper');
    stepper.innerHTML = keys.map((key, idx) => {
      const st = statusFor(key);
      const cls = [st.done ? 'done' : '', st.stale ? 'stale' : '', key === state.step ? 'active' : ''].filter(Boolean).join(' ');
      const mark = st.stale ? '!' : (st.done && key !== state.step ? '✓' : idx + 1);
      return `<button class="step-btn ${cls}" data-step="${key}" title="${esc(st.detail || labels[key])}" aria-label="${esc(labels[key])}: ${esc(st.detail || '')}">
        <span class="step-num">${mark}</span>
        <span class="label">${labels[key]}</span>
      </button>`;
    }).join('');
    $$('.step-btn', stepper).forEach(btn => btn.addEventListener('click', () => gotoStep(btn.dataset.step)));
  }

  function gotoStep(step) {
    if (!state.overview) return;
    try {
      const p = collectParams($('#step-content'));
      state.project.params = { ...state.project.params, ...p };
    } catch (err) { /* no form */ }
    state.step = step;
    renderStepper();
    renderStepContent();
    $('#stage-head').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* --------------------------------------------------------------- step router */
  function renderStepContent() {
    const host = $('#step-content');
    if (!host || !state.overview) return;
    let html = '';
    switch (state.step) {
      case 'project': html = projectStep(); break;
      case 'data': html = dataStep(); break;
      case 'network': html = networkStep(); break;
      case 'interferometry': html = interferometryStep(); break;
      case 'atmosphere': html = atmosphereStep(); break;
      case 'inversion': html = inversionStep(); break;
      case 'results': html = resultsStep(); break;
      default: html = projectStep();
    }
    host.innerHTML = html;
    markWideFigures();
    bindStep(state.step);
  }

  function markWideFigures() {
    document.querySelectorAll('.figure-card img').forEach(img => {
      const apply = () => {
        const card = img.closest('.figure-card');
        if (!card || !img.naturalWidth || !img.naturalHeight) return;
        card.classList.toggle('wide', img.naturalWidth / img.naturalHeight >= 2);
      };
      if (img.complete) apply();
      else img.addEventListener('load', apply, { once: true });
    });
  }

  function bindStep(step) {
    switch (step) {
      case 'project': bindProjectStep(); break;
      case 'data': bindDataStep(); break;
      case 'network': bindNetworkStep(); break;
      case 'interferometry': bindInterferometryStep(); break;
      case 'atmosphere': bindAtmosphereStep(); break;
      case 'inversion': bindInversionStep(); break;
      case 'results': bindResultsStep(); break;
    }
  }

  function guide(icon, title, text, tag = 'GUIDE') {
    return `<div class="guide-banner">
      <div class="guide-icon">${icon}</div>
      <div><strong>${esc(title)}</strong><p>${text}</p></div>
      <span class="guide-tag">${esc(tag)}</span>
    </div>`;
  }

  function actionButton(id, label, kind = 'primary', extra = '') {
    const cls = kind === 'primary' ? 'btn btn-primary' : (kind === 'amber' ? 'btn btn-amber' : 'btn');
    return `<button class="${cls}" id="${id}" data-run ${extra}>${label}</button>`;
  }

  function statCard(k, v, d = '', cls = '') {
    return `<div class="stat ${cls}"><div class="k">${esc(k)}</div><div class="v">${v}</div>${d ? `<div class="d">${esc(d)}</div>` : ''}</div>`;
  }

  /* --------------------------------------------------------------- step 1 · project */
  function projectStep() {
    const ov = state.overview;
    const p = ov.project;
    const tech = currentTech();
    const techDef = techMeta(tech);
    const done = ov.status ? ov.status.filter(s => s.done).length : 0;
    const total = ov.status ? ov.status.length : 7;
    const dates = ov.stack && ov.stack.dates ? ov.stack.dates : [];
    return `
      ${guide('⌖', '先确认项目与技术路线', '技术路线决定网络设计与反演方式。新手建议先用 SBAS 跑通全流程；演示项目自带已知形变、大气与 DEM 误差，适合对照学习。', 'STEP 1 / 7')}
      <div class="quick-run">
        <div>
          <h3>${esc(p.name)}</h3>
          <p>${done}/${total} 个阶段已完成 · ${esc(techDef.name)} · ${dates.length ? `${dates[0]} 至 ${dates[dates.length - 1]}` : '等待载入影像'}</p>
        </div>
        <div class="btn-row">
          <button class="btn" id="quick-project-data">查看数据</button>
          <button class="btn" id="quick-project-health">项目体检</button>
          <button class="btn btn-amber" id="quick-project-run" data-run>▶ 运行全流程</button>
        </div>
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>项目设置</h3><div class="hint">名称与备注会写入处理报告。</div></div></div>
          <div class="field"><label>项目名称</label><input type="text" id="proj-name" value="${esc(p.name)}"></div>
          <div class="field" style="margin-top:10px"><label>备注</label><textarea id="proj-notes" placeholder="例如：四川某滑坡隐患点，2023 年雨季重点复查。">${esc(p.notes || '')}</textarea></div>
          <div class="btn-row" style="margin-top:12px"><button class="btn btn-primary" id="btn-save-project">保存设置</button></div>
          <div class="divider"></div>
          <div class="kv-row"><span class="k">项目编号</span><span class="v">${esc(p.id)}</span></div>
          <div class="kv-row"><span class="k">创建时间</span><span class="v">${esc(p.created_at)}</span></div>
          <div class="kv-row"><span class="k">数据来源</span><span class="v">${p.demo ? '合成演示数据' : '本地导入 / 下载'}</span></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>七步进度</h3><div class="hint">点击任意阶段可直接跳转，已完成的阶段可以单独重跑。</div></div></div>
          <div>
            ${(ov.status || []).map(s => `<div class="kv-row"><span class="k">${esc(s.label)} ${s.stale ? '<span class="badge amber">参数待重跑</span>' : (s.done ? '<span class="badge green">完成</span>' : '<span class="badge grey">待处理</span>')}</span><span class="v">${esc(s.detail || (s.done ? '完成' : '待处理'))}</span></div>`).join('')}
          </div>
          <div class="inline-note" style="margin-top:12px">引导模式只暴露最关键的参数；需要调网络阈值、滤波窗口、PS/DS 阈值时，切换到右上角「专家」。</div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>处理路线</h3><div class="hint">四条路线共用同一套数据与干涉产品，可随时切换做对比。</div></div><span class="badge grey">${esc(techDef.name)} 已选择</span></div>
        ${techGrid(tech, 'tech-card', 'data-tech')}
        <div class="inline-note" style="margin-top:12px">${esc(techDef.desc)}</div>
      </div>
      <div class="card danger-zone">
        <div class="card-head"><div><h3>项目维护</h3><div class="hint">重置只清空派生结果，保留原始影像；删除会移除整个项目目录。</div></div>
          <div class="btn-row"><button class="btn btn-small" id="btn-reset-project">重置处理结果</button><button class="btn btn-danger btn-small" id="btn-delete-project">删除此项目</button></div>
        </div>
      </div>`;
  }

  function techGrid(selected, cls = 'tech-card', attr = 'data-tech') {
    const keys = ['sbas', 'ps', 'ds', 'dinsar'];
    return `<div class="tech-grid">${keys.map(key => {
      const t = techMeta(key);
      return `<button class="${cls} ${selected === key ? 'selected' : ''}" ${attr}="${key}">
        <div class="tc-top"><span class="tc-name">${esc(t.name)}</span><span class="tc-badge">${esc(t.badge || '')}</span></div>
        <div class="tc-cn">${esc(t.cn)}</div>
        <div class="tc-desc">${esc(t.tagline || t.desc || '')}</div>
        <span class="tc-check">✓</span>
      </button>`;
    }).join('')}</div>`;
  }

  function bindProjectStep() {
    const p = state.project;
    $('#quick-project-data').addEventListener('click', () => gotoStep('data'));
    $('#quick-project-health').addEventListener('click', openHealth);
    $('#quick-project-run').addEventListener('click', () => runAll());
    $('#btn-save-project').addEventListener('click', async () => {
      const name = $('#proj-name').value.trim() || '未命名项目';
      const notes = $('#proj-notes').value;
      try {
        await api(`/api/projects/${encodeURIComponent(state.projectId)}`, { method: 'PATCH', body: { name, notes } });
        state.project.name = name;
        state.project.notes = notes;
        await loadProjects();
        renderHeader();
        toast('项目设置已保存', 'success');
      } catch (err) { toast(err.message, 'error'); }
    });
    $$('[data-tech]').forEach(btn => btn.addEventListener('click', async () => {
      const key = btn.dataset.tech;
      if (key === currentTech()) return;
      const hasProducts = (state.overview.status || []).some(s => s.done && !['project', 'data'].includes(s.key));
      if (hasProducts && !confirm(`切换到 ${techMeta(key).name} 会清空现有网络、干涉、大气与反演产品（保留原始数据与配准结果）。继续？`)) return;
      state.selectedTech = key;
      applyTechDefaults(key);
      try {
        await api(`/api/projects/${encodeURIComponent(state.projectId)}`, { method: 'PATCH', body: { technique: key, params: techDefaults(key) } });
        state.project.technique = key;
        state.overview.project.technique = key;
        await refreshOverview({ keepStep: true });
        state.step = hasProducts ? 'network' : 'project';
        renderAll();
        toast(hasProducts ? `已切换到 ${techMeta(key).name}，旧产品已清理，请重建网络。` : `已切换到 ${techMeta(key).name}。`);
      } catch (err) { toast(err.message, 'error'); }
    }));
    const resetBtn = $('#btn-reset-project');
    if (resetBtn) resetBtn.addEventListener('click', resetProjectProducts);
    $('#btn-delete-project').addEventListener('click', async () => {
      if (!confirm(`确定删除项目「${p.name}」？所有中间产品与成果将被移除。`)) return;
      try {
        await api(`/api/projects/${encodeURIComponent(state.projectId)}`, { method: 'DELETE' });
        localStorage.removeItem('insar.lastProject');
        state.projectId = null;
        state.project = null;
        state.overview = null;
        await loadProjects();
        renderNoProject();
        toast('项目已删除');
      } catch (err) { toast(err.message, 'error'); }
    });
  }

  /* --------------------------------------------------------------- step 2 · data */
  function dataStep() {
    const ov = state.overview;
    const s = ov.stack || {};
    const dates = s.dates || ov.dates || [];
    const refIdx = Number(pval('ref_index', s.ref_index || 0));
    const amplitude = ov.assets && ov.assets.amplitude;
    const source = s.source || (ov.project.demo ? 'demo' : 'import');
    return `
      ${guide('▦', '数据是一切的底图', '确认影像数量、时间跨度、尺寸与地理范围。演示数据包含 24 景 Sentinel-1 纹理、已知形变和大气；真实数据建议先用 SNAP / ISCE 完成配准与 burst 拼接，再导入这里。', 'STEP 2 / 7')}
      <div class="grid-4">
        ${statCard('影像数量', `${s.n_dates || dates.length || 0}<small> 景</small>`, s.platform || '')}
        ${statCard('影像尺寸', s.rows ? `${s.rows}<small> × ${s.cols} px</small>` : '—', s.crs || '')}
        ${statCard('时间范围', dates.length ? `${shortDate(dates[0])}<small> → ${shortDate(dates[dates.length - 1])}</small>` : '—', dates.length ? `${dates.length} 个时相` : '')}
        ${statCard('数据来源', ov.project.demo ? '演示' : '导入', source === 'demo' ? '带真值校验' : '本地文件', 'accent')}
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>影像栈</h3><div class="hint">${esc(s.name || '本地 SLC 栈')}</div></div><span class="badge ${ov.project.demo ? 'amber' : 'grey'}">${ov.project.demo ? 'SYNTHETIC' : 'LOCAL'}</span></div>
          <dl class="meta-list">
            <dt>平台 / 轨道</dt><dd>${esc(s.platform || '—')} · ${esc(s.orbit_direction || '—')} ${s.relative_orbit ? '#' + s.relative_orbit : ''}</dd>
            <dt>波长</dt><dd>${s.wavelength_m ? `${fmt(s.wavelength_m * 1000, 2)} mm` : '—'}</dd>
            <dt>入射角</dt><dd>${s.incidence_deg ? `${fmt(s.incidence_deg, 1)}°` : '—'}</dd>
            <dt>像元间距</dt><dd>${s.range_spacing_m ? `${fmt(s.range_spacing_m, 2)} m × ${fmt(s.azimuth_spacing_m, 2)} m` : '—'}</dd>
            <dt>参考影像</dt><dd>${dates[refIdx] || '—'}（第 ${refIdx + 1} 景）</dd>
            <dt>真值</dt><dd>${ov.truth_available ? '可用 · 自动对账' : '无（真实数据）'}</dd>
          </dl>
          <div class="stack-dates">${dates.map((d, i) => `<span class="date-chip ${i === refIdx ? 'ref' : ''}" title="第 ${i + 1} 景">${esc(d)}</span>`).join('')}</div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>幅度预览</h3><div class="hint">平均幅度越高，通常代表建筑、岩石或强散射体。</div></div></div>
          ${amplitude ? `<div class="thumb-card"><img src="${assetUrl(amplitude)}" alt="幅度预览"><div class="thumb-cap">多时相平均幅度 · 白色为强散射 / 稳定像元</div></div>` : `<div class="empty-mini">尚无预览图。导入数据后可运行配准生成幅度预览。</div>`}
          <div class="inline-note" style="margin-top:10px">${ov.project.demo ? '演示数据由确定性场景生成，形变、大气与 DEM 误差都有真值，可反复回归。' : '真实数据请检查轨道方向、极化方式与坐标系；速率结果默认是 LOS 向，未做升/降轨分解。'}</div>
        </div>
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>导入本地数据</h3><div class="hint">支持原生 .npy 栈、ENVI 栈、复数 GeoTIFF 目录与 Sentinel-1 SAFE（实验）。</div></div></div>
          <div class="field"><label>数据目录或文件路径</label><input type="text" id="import-path" placeholder="D:\\data\\sentinel1\\stack"></div>
          <div class="field" style="margin-top:9px"><label>DEM 路径 <span class="unit">可选</span></label><input type="text" id="import-dem" placeholder="D:\\data\\dem.tif"></div>
          <div class="btn-row" style="margin-top:11px"><button class="btn btn-primary" id="btn-import">导入到当前项目</button><span class="small muted">导入会覆盖当前影像栈。</span></div>
          <div class="inline-note" style="margin-top:10px">目录中放 <span class="mono">slc.npy</span>（形状：时相 × 行 × 列）与可选的 <span class="mono">dem.npy</span>、<span class="mono">mask.npy</span>、<span class="mono">meta.json</span> 即可。</div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>在线检索 Sentinel-1</h3><div class="hint">实验性适配器；下载后仍需 SNAP / ISCE 做 burst 与配准。</div></div><span class="badge grey">EXPERIMENTAL</span></div>
          <div class="form-grid">
            <div class="field"><label>数据源</label><select id="dl-provider"><option value="copernicus">Copernicus Data Space</option><option value="asf">ASF DAAC</option></select></div>
            <div class="field"><label>最大结果数</label><input type="number" id="dl-max" value="8" min="1" max="30"></div>
          </div>
          <div class="form-grid three" style="margin-top:8px">
            <div class="field"><label>西经 / 西边界</label><input type="number" id="dl-west" value="106.52" step="0.001"></div>
            <div class="field"><label>南纬 / 南边界</label><input type="number" id="dl-south" value="29.55" step="0.001"></div>
            <div class="field"><label>东经 / 东边界</label><input type="number" id="dl-east" value="106.63" step="0.001"></div>
          </div>
          <div class="form-grid three" style="margin-top:8px">
            <div class="field"><label>北纬 / 北边界</label><input type="number" id="dl-north" value="29.63" step="0.001"></div>
            <div class="field"><label>开始日期</label><input type="date" id="dl-start" value="2023-01-01"></div>
            <div class="field"><label>结束日期</label><input type="date" id="dl-end" value="2023-12-31"></div>
          </div>
          <details style="margin-top:8px"><summary class="small muted" style="cursor:pointer">账号（仅本次请求使用，不写入项目）</summary>
            <div class="form-grid" style="margin-top:8px">
              <div class="field"><label>用户名 / 邮箱</label><input type="text" id="dl-user" autocomplete="off"></div>
              <div class="field"><label>密码</label><input type="password" id="dl-pass" autocomplete="new-password"></div>
            </div>
          </details>
          <div class="btn-row" style="margin-top:10px"><button class="btn" id="btn-dl-search">检索影像</button><span class="small muted">Copernicus 需要账号；ASF 检索公开。</span></div>
          <div class="result-list" id="dl-results">${downloadResultsHtml()}</div>
        </div>
      </div>`;
  }

  function bindDataStep() {
    $('#btn-import').addEventListener('click', async () => {
      const path = $('#import-path').value.trim();
      if (!path) { toast('请填写数据路径', 'error'); return; }
      if (!confirm('导入会覆盖当前影像栈，并清空网络、干涉、大气与反演成果。继续？')) return;
      const btn = $('#btn-import');
      btn.disabled = true;
      btn.textContent = '导入中…';
      try {
        const res = await api(`/api/projects/${encodeURIComponent(state.projectId)}/import`, {
          method: 'POST', body: { path, dem_path: $('#import-dem').value.trim() }
        });
        state.results = null;
        await refreshOverview({ keepStep: true });
        renderAll();
        toast(`导入完成：${res.stack.n_dates} 景，${res.stack.rows} × ${res.stack.cols} px`, 'success');
      } catch (err) {
        toast(err.message, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = '导入到当前项目';
      }
    });
    $('#btn-dl-search').addEventListener('click', searchDownloads);
    bindDownloadResults();
  }

  function downloadResultsHtml() {
    if (state.searchBusy) return '<div class="empty-mini">正在检索远端目录…</div>';
    if (!state.dlResults.length) return '<div class="empty-mini">填入区域与时间范围后点击「检索影像」。结果会显示在下方，可直接加入下载队列。</div>';
    return state.dlResults.map((item, idx) => `<div class="dl-result">
      <div><div class="dr-name">${esc(item.name || item.id || 'Sentinel-1 SLC')}</div>
      <div class="dr-meta">${esc(item.date || '')} · ${esc(item.size_human || fmtBytes(item.size))} · ${esc(item.provider || '')}</div></div>
      <button class="btn btn-small" data-dl-index="${idx}">下载</button>
    </div>`).join('');
  }

  function bindDownloadResults() {
    const host = $('#dl-results');
    if (!host) return;
    $$('[data-dl-index]', host).forEach(btn => btn.addEventListener('click', () => startDownload(Number(btn.dataset.dlIndex))));
  }

  async function searchDownloads() {
    const provider = $('#dl-provider').value;
    const bbox = ['dl-west', 'dl-south', 'dl-east', 'dl-north'].map(id => Number($(`#${id}`).value));
    if (bbox.some(v => !Number.isFinite(v))) { toast('请填写完整的经纬度范围', 'error'); return; }
    state.searchBusy = true;
    const host = $('#dl-results');
    if (host) host.innerHTML = downloadResultsHtml();
    try {
      const res = await api('/api/download/search', {
        method: 'POST',
        body: {
          provider, bbox,
          start: $('#dl-start').value, end: $('#dl-end').value,
          max_results: Number($('#dl-max').value) || 8,
          username: $('#dl-user').value, password: $('#dl-pass').value
        }
      });
      state.dlResults = res.results || [];
      if (!state.dlResults.length) toast('检索完成，但没有找到符合条件的 SLC。');
    } catch (err) {
      state.dlResults = [];
      toast(`检索失败：${err.message}`, 'error');
    } finally {
      state.searchBusy = false;
      if ($('#dl-results')) {
        $('#dl-results').innerHTML = downloadResultsHtml();
        bindDownloadResults();
      }
    }
  }

  async function startDownload(index) {
    const item = state.dlResults[index];
    if (!item) return;
    try {
      const res = await api('/api/download/fetch', {
        method: 'POST',
        body: {
          project_id: state.projectId, url: item.url, provider: item.provider,
          name: item.name, username: $('#dl-user') ? $('#dl-user').value : '',
          password: $('#dl-pass') ? $('#dl-pass').value : ''
        }
      });
      trackJob(res.job);
      toast('下载任务已进入后台队列');
    } catch (err) { toast(err.message, 'error'); }
  }

  /* --------------------------------------------------------------- step 3 · network */
  function networkStep() {
    const ov = state.overview;
    const net = ov.network || {};
    const stats = net.stats || {};
    const pairs = ov.pairs || [];
    const tech = currentTech();
    const dates = (ov.stack && ov.stack.dates) || ov.dates || [];
    const refIdx = Number(pval('ref_index', stats.ref_index || (ov.stack && ov.stack.ref_index) || 0));
    const pair = state.selectedPair || (pairs[0] && pairs[0].id);
    return `
      ${guide('⌁', '网络决定时序能不能解', '每个干涉对就是一条观测方程。短基线组合能抑制失相干，同时保留足够冗余；网络断开会让部分日期无法求解。绿色线是干涉对，橙色点是参考影像。', 'STEP 3 / 7')}
      <div class="grid-4">
        ${statCard('干涉对', `${stats.n_pairs || pairs.length}<small> 对</small>`, '每条线是一个观测')}
        ${statCard('冗余度', `${fmt(stats.redundancy, 2)}<small> ×</small>`, '越高越稳健')}
        ${statCard('最大时间基线', `${fmt(stats.max_temporal_days, 0)}<small> 天</small>`, `平均 ${fmt(stats.mean_temporal_days, 1)} 天`)}
        ${statCard('连通分量', `${stats.connected_components || 1}`, (stats.connected_components || 1) === 1 ? '全图连通' : '存在断网风险', (stats.connected_components || 1) === 1 ? 'good' : 'bad')}
      </div>
      <div class="split-main" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>时空基线网络</h3><div class="hint">横轴为时间，纵轴为垂直基线。点击任意连线可直接查看该干涉对。</div></div><span class="badge grey">${pairs.length} PAIRS</span></div>
          ${pairs.length ? networkSvg(ov) : '<div class="empty-mini">尚未构建网络。</div>'}
          <div class="legend-row"><span><i></i>干涉对</span><span><i class="amber"></i>选中 / 点击查看</span><span><i class="dot"></i>影像时相</span></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>网络参数</h3><div class="hint">引导模式给出稳健默认值，通常无需修改。</div></div></div>
          ${tech === 'dinsar' ? dinsarPairFields(pairs) : `
            <div class="form-grid">
              <div class="field"><label>最大时间基线 <span class="unit">天</span></label><input type="number" data-param="max_temporal_baseline" min="6" max="365" value="${esc(pval('max_temporal_baseline', 72))}"></div>
              <div class="field"><label>最大垂直基线 <span class="unit">m</span></label><input type="number" data-param="max_perpendicular_baseline" min="10" max="2000" step="10" value="${esc(pval('max_perpendicular_baseline', 180))}"></div>
            </div>
            <div class="form-grid expert-only" style="margin-top:8px">
              <div class="field"><label>参考影像序号 <span class="unit">0 起</span></label><input type="number" data-param="ref_index" min="0" max="${Math.max(0, dates.length - 1)}" value="${esc(refIdx)}"></div>
              <div class="field"><label>最大对数</label><input type="number" data-param="max_pairs" min="10" max="600" value="${esc(pval('max_pairs', 80))}"></div>
            </div>
            ${tech === 'ps' ? `<label class="checkline expert-only"><input type="checkbox" data-param="ps_star_all" ${pval('ps_star_all', true) ? 'checked' : ''}><span>PS 使用全星形网络（所有影像连到参考影像）<span class="help">关闭后只保留短时间基线星形对；相邻对仍会自动补充用于闭合检查。</span></span></label>` : ''}
            <div class="inline-note" style="margin-top:10px">${tech === 'ps' ? 'PS 默认 45 对星形 + 相邻网络，多视 1，保留点目标。' : (tech === 'ds' ? 'DS 默认 72 天 / 180 m 基线，相位链接在干涉步骤执行。' : 'SBAS 默认 72 天 / 180 m 基线，得到约 80 对；冗余度 3 以上较稳健。')}</div>
          `}
          <div class="btn-row" style="margin-top:13px">${actionButton('btn-run-network', pairs.length ? '重建网络' : '构建网络')}</div>
          <div class="small muted" style="margin-top:7px">若尚未配准，构建网络会自动先完成 SLC 配准。重建网络后，需要重新运行干涉、大气与反演。</div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>干涉对清单</h3><div class="hint">点击一行，跳到干涉步骤查看该对相位。</div></div><span class="badge grey">${pairs.length} 对</span></div>
        ${pairs.length ? `<div class="table-wrap medium"><table class="data"><thead><tr><th>编号</th><th>主影像</th><th>从影像</th><th>时间基线</th><th>垂直基线</th><th>状态</th></tr></thead><tbody>
          ${pairs.map(p => `<tr class="selectable ${p.id === pair ? 'selected' : ''}" data-pair-select="${esc(p.id)}">
            <td><span class="pair-id">${esc(p.id)}</span></td>
            <td class="num">${esc(p.ref_date)}</td>
            <td class="num">${esc(p.sec_date)}</td>
            <td class="num">${fmt(p.dt_days, 0)} d</td>
            <td class="num">${fmtSigned(p.bperp_diff, 1, ' m')}</td>
            <td>${isDone('interferometry') ? '<span class="badge green">已处理</span>' : '<span class="badge grey">待处理</span>'}</td>
          </tr>`).join('')}
        </tbody></table></div>` : '<div class="empty-mini">暂无干涉对。</div>'}
      </div>`;
  }

  function dinsarPairFields(pairs) {
    const dates = (state.overview.stack && state.overview.stack.dates) || state.overview.dates || [];
    const existing = pairs[0] || null;
    const dparams = pval('dinsar_pair', null);
    let ref = existing ? existing.ref : Math.max(0, Math.floor(dates.length / 3));
    let sec = existing ? existing.sec : Math.max(1, dates.length - 1);
    if (Array.isArray(dparams) && dparams.length === 2) { ref = Number(dparams[0]); sec = Number(dparams[1]); }
    return `<div class="inline-note">D-InSAR 只处理一个干涉对。选择主、从影像后重建网络，再运行干涉与反演。</div>
      <div class="form-grid" style="margin-top:10px">
        <div class="field"><label>主影像</label><select data-dinsar-ref>${dates.map((d, i) => `<option value="${i}" ${i === ref ? 'selected' : ''}>${i + 1}. ${esc(d)}</option>`).join('')}</select></div>
        <div class="field"><label>从影像</label><select data-dinsar-sec>${dates.map((d, i) => `<option value="${i}" ${i === sec ? 'selected' : ''}>${i + 1}. ${esc(d)}</option>`).join('')}</select></div>
      </div>`;
  }

  function networkSvg(ov) {
    const pairs = ov.pairs || [];
    const meta = ov.stack || {};
    const dates = meta.dates || ov.dates || [];
    const bperp = meta.bperp_m || [];
    const W = 760, H = 300;
    const m = { l: 48, r: 18, t: 26, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const day = d => {
      const t = Date.parse(`${d}T00:00:00`);
      return Number.isFinite(t) ? t / 86400000 : 0;
    };
    const d0 = dates.length ? day(dates[0]) : 0;
    const xs = dates.map(d => day(d) - d0);
    const xmax = Math.max(1, ...xs);
    const bp = dates.map((_, i) => Number(bperp[i] || 0));
    let ymin = Math.min(0, ...bp), ymax = Math.max(0, ...bp);
    if (ymin === ymax) { ymin -= 10; ymax += 10; }
    const pad = (ymax - ymin) * 0.12;
    ymin -= pad; ymax += pad;
    const X = v => m.l + (v / xmax) * iw;
    const Y = v => m.t + ih - ((v - ymin) / (ymax - ymin)) * ih;
    let grid = '';
    for (let i = 0; i <= 4; i++) {
      const yy = m.t + (ih * i) / 4;
      const val = ymax - ((ymax - ymin) * i) / 4;
      grid += `<line class="net-grid" x1="${m.l}" y1="${yy.toFixed(1)}" x2="${W - m.r}" y2="${yy.toFixed(1)}"/>`;
      grid += `<text class="net-label" x="${m.l - 6}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${val.toFixed(0)}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const xx = m.l + (iw * i) / 4;
      const idx = Math.min(dates.length - 1, Math.round((dates.length - 1) * i / 4));
      grid += `<line class="net-grid" x1="${xx.toFixed(1)}" y1="${m.t}" x2="${xx.toFixed(1)}" y2="${m.t + ih}"/>`;
      if (dates[idx]) grid += `<text class="net-label" x="${xx.toFixed(1)}" y="${H - 14}" text-anchor="middle">${esc(shortDate(dates[idx]))}</text>`;
    }
    const lines = pairs.map(p => {
      const x1 = X(xs[p.ref] || 0), x2 = X(xs[p.sec] || 0);
      const y1 = Y(Number(bperp[p.ref] || 0)), y2 = Y(Number(bperp[p.sec] || 0));
      const cls = p.id === state.selectedPair ? 'net-line selected' : 'net-line';
      return `<line class="${cls}" data-pair-line="${esc(p.id)}" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"><title>${esc(p.id)} · ${esc(p.ref_date)} → ${esc(p.sec_date)} · Δt ${fmt(p.dt_days, 0)} d</title></line>`;
    }).join('');
    const refIdx = Number(pval('ref_index', (ov.network && ov.network.stats && ov.network.stats.ref_index) || 0));
    const points = dates.map((d, i) => {
      const isRef = i === refIdx;
      return `<circle class="${isRef ? 'net-point ref' : 'net-point'}" cx="${X(xs[i]).toFixed(1)}" cy="${Y(bp[i]).toFixed(1)}" r="${isRef ? 4 : 2.6}"><title>${esc(d)} · B⊥ ${fmt(bp[i], 1)} m${isRef ? ' · 参考影像' : ''}</title></circle>`;
    }).join('');
    return `<div class="network-wrap"><svg class="network-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="时空基线网络">
      <text class="net-title" x="${m.l}" y="15">垂直基线 B⊥ (m) / 时间</text>
      ${grid}
      <line class="net-axis" x1="${m.l}" y1="${m.t + ih}" x2="${W - m.r}" y2="${m.t + ih}"/>
      <line class="net-axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + ih}"/>
      ${lines}${points}
    </svg></div>`;
  }

  function bindNetworkStep() {
    $$('[data-pair-line]').forEach(line => line.addEventListener('click', () => {
      state.selectedPair = line.dataset.pairLine;
      state.pairField = 'unwrapped';
      gotoStep('interferometry');
    }));
    $$('[data-pair-select]').forEach(row => row.addEventListener('click', () => {
      state.selectedPair = row.dataset.pairSelect;
      state.pairField = 'unwrapped';
      gotoStep('interferometry');
    }));
    $('#btn-run-network').addEventListener('click', () => runStep('network'));
  }

  /* --------------------------------------------------------------- step 4 · interferometry */
  function interferometryStep() {
    const ov = state.overview;
    const ifg = ov.interferometry || {};
    const pairs = ov.pairs || [];
    const pair = pairs.find(p => p.id === state.selectedPair) || pairs[0] || null;
    const field = state.pairField || 'unwrapped';
    const img = pair ? `/api/projects/${encodeURIComponent(state.projectId)}/pairs/${encodeURIComponent(pair.id)}/image?field=${encodeURIComponent(field)}&v=${Date.now()}` : '';
    const closure = ifg.closure || {};
    const tech = currentTech();
    return `
      ${guide('◫', '把两景相位变成一条可解的观测', '干涉图包含形变、地形、大气与噪声。滤波抑制斑点，解缠把缠绕相位展开成连续相位；相干性低的像元会被掩膜。先看缠绕相位与相干性，再决定是否调整参数。', 'STEP 4 / 7')}
      <div class="grid-4">
        ${statCard('已处理干涉对', `${ifg.n_pairs || 0}<small> 对</small>`, pairs.length ? `共 ${pairs.length} 对` : '')}
        ${statCard('平均相干性', fmt(ifg.mean_coherence, 2), '0 – 1，越高越可靠', 'accent')}
        ${statCard('闭合差中位数', `${fmt(closure.median_closure_rad, 3)}<small> rad</small>`, closure.warning || '三角闭合检查')}
        ${statCard('多视 / 有效像元', `${ifg.multilook || pval('multilook', 2)}<small> ×</small>`, ifg.mean_valid_fraction ? `有效 ${fmtPct(ifg.mean_valid_fraction)}` : '')}
      </div>
      <div class="split-main" style="margin-top:14px">
        <div class="card">
          <div class="card-head">
            <div><h3>干涉对浏览</h3><div class="hint">切换干涉对与显示场。相位图为弧度，位移图为毫米。</div></div>
            <select class="select" id="pair-select">${pairs.map(p => `<option value="${esc(p.id)}" ${pair && p.id === pair.id ? 'selected' : ''}>${esc(p.id)} · ${esc(shortDate(p.ref_date))} → ${esc(shortDate(p.sec_date))}</option>`).join('')}</select>
          </div>
          <div class="field-tabs" style="margin-bottom:9px">
            ${Object.keys(FIELD_LABELS).map(k => `<button class="field-tab ${field === k ? 'active' : ''}" data-field="${k}">${FIELD_LABELS[k]}</button>`).join('')}
          </div>
          <div class="viewer">
            ${pair && ifg.n_pairs ? `<img src="${img}" alt="${FIELD_LABELS[field]}" onerror="this.style.display='none';this.nextElementSibling.style.display='block'"><div class="viewer-empty" style="display:none">该场尚未生成。请先运行干涉处理，或换一个显示场。</div>` : '<div class="viewer-empty">尚未生成干涉图。设置参数后点击右侧「运行干涉处理」。</div>'}
            ${pair ? `<div class="viewer-cap"><span>${esc(pair.id)} · ${esc(pair.ref_date)} → ${esc(pair.sec_date)} · Δt ${fmt(pair.dt_days, 0)} 天 · B⊥ ${fmtSigned(pair.bperp_diff, 1, ' m')}</span><span>${FIELD_LABELS[field]}</span></div>` : ''}
          </div>
          <div class="inline-note" style="margin-top:10px">${tech === 'ps' ? 'PS 路线使用单视（multilook = 1）保留点目标；强滤波会削弱永久散射体，建议只用轻量 boxcar。' : (tech === 'ds' ? 'DS 路线会先做相位链接，再重建干涉对相位；本页可预览普通干涉结果，最终 DS 产品在反演步骤生成。' : '相位图出现大面积条纹跳变或噪声，优先提高相干性阈值或减小多视；不要用滤波掩盖解缠错误。')}</div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>处理参数</h3><div class="hint">默认 boxcar 3×3 + 质量引导解缠。</div></div></div>
          <div class="form-grid">
            <div class="field"><label>多视因子</label><select data-param="multilook">${[1, 2, 4].map(v => `<option value="${v}" ${Number(pval('multilook', 2)) === v ? 'selected' : ''}>${v} × ${v}</option>`).join('')}</select></div>
            <div class="field"><label>相位滤波</label><select data-param="filter">${[['boxcar', 'Boxcar 均值'], ['goldstein', 'Goldstein 自适应'], ['gaussian', '高斯低通'], ['nlmeans', 'NL-Means'], ['none', '不滤波']].map(([v, l]) => `<option value="${v}" ${pval('filter', 'boxcar') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
            <div class="field"><label>解缠算法</label><select data-param="unwrap_method">${[['quality', '质量引导（内置）'], ['snaphu', 'SNAPHU（需安装）'], ['skimage', '最小二乘（skimage）']].map(([v, l]) => `<option value="${v}" ${pval('unwrap_method', 'quality') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
            <div class="field"><label>相干性阈值</label><input type="number" data-param="unwrap_coherence_min" min="0" max="0.9" step="0.01" value="${esc(pval('unwrap_coherence_min', 0.18))}"></div>
          </div>
          <label class="checkline expert-only"><input type="checkbox" data-param="remove_ramp" ${pval('remove_ramp', true) ? 'checked' : ''}><span>解缠后去除线性/二次趋势面<span class="help">会同时移除真实的长波形变，仅在确认存在轨道或长波误差时使用。</span></span></label>
          <div class="form-grid expert-only" style="margin-top:6px">
            <div class="field"><label>相干性窗口 <span class="unit">px</span></label><input type="number" data-param="coherence_window" min="3" max="15" value="${esc(pval('coherence_window', 5))}"></div>
            <div class="field"><label>有效相干性下限</label><input type="number" data-param="min_coherence" min="0" max="0.9" step="0.01" value="${esc(pval('min_coherence', 0.25))}"></div>
            <div class="field"><label>Boxcar 窗口 <span class="unit">px</span></label><input type="number" data-param="boxcar_size" min="1" max="15" value="${esc(pval('boxcar_size', 3))}"></div>
            <div class="field"><label>Goldstein α</label><input type="number" data-param="goldstein_alpha" min="0" max="1" step="0.05" value="${esc(pval('goldstein_alpha', 0.65))}"></div>
            <div class="field"><label>Goldstein 块大小</label><input type="number" data-param="goldstein_patch" min="8" max="128" value="${esc(pval('goldstein_patch', 32))}"></div>
            <div class="field"><label>高斯 σ</label><input type="number" data-param="gaussian_sigma" min="0.5" max="8" step="0.5" value="${esc(pval('gaussian_sigma', 2))}"></div>
          </div>
          <div class="btn-row" style="margin-top:13px">${actionButton('btn-run-ifg', ifg.n_pairs ? '重新处理干涉' : '运行干涉处理')}</div>
          <div class="small muted" style="margin-top:7px">处理 ${pairs.length} 对约需 ${pairs.length > 60 ? '10–20' : '5–10'} 秒（演示数据）。</div>
        </div>
      </div>`;
  }

  function bindInterferometryStep() {
    const sel = $('#pair-select');
    if (sel) sel.addEventListener('change', () => {
      state.selectedPair = sel.value;
      renderStepContent();
    });
    $$('[data-field]').forEach(btn => btn.addEventListener('click', () => {
      state.pairField = btn.dataset.field;
      renderStepContent();
    }));
    const run = $('#btn-run-ifg');
    if (run) run.addEventListener('click', () => runStep('interferometry'));
  }

  /* --------------------------------------------------------------- step 5 · atmosphere */
  function atmosphereStep() {
    const ov = state.overview;
    const atm = ov.atmosphere || {};
    const method = pval('atmosphere_method', atm.method || (ov.project.demo ? 'gacos' : 'empirical'));
    const source = pval('gacos_source', state.atmSource || atm.source || (ov.project.demo ? 'demo' : 'local'));
    const val = atm.validation || {};
    const hasAtm = !!atm.method;
    const methodLabel = { gacos: 'GACOS 逐日延迟', empirical: '经验高程校正', off: '不校正' }[method] || method;
    return `
      ${guide('☁', '大气是毫米级形变最大的“假信号”', '对流层延迟在空间上平滑、在时间上变化，和形变混在一起。GACOS 用外部天气模型逐日估计延迟；经验法则用相位—高程关系扣除分层大气。没有外部数据时，反演中的时间高通也能压掉一部分。', 'STEP 5 / 7')}
      <div class="grid-4">
        ${statCard('当前方法', methodLabel, atm.source ? `数据源 ${atm.source}` : '尚未校正', 'accent')}
        ${statCard('校正干涉对', `${atm.n_pairs || 0}<small> 对</small>`, hasAtm ? '已写入 unwrapped_atmcorr' : '')}
        ${statCard('相位离散度降低', atm.phase_std_reduction_pct !== undefined ? `${fmt(atm.phase_std_reduction_pct, 1)}<small> %</small>` : '—', '校正前后稳健标准差', 'good')}
        ${statCard('成对残差', val.pair_residual_after_mm !== undefined ? `${fmt(val.pair_residual_after_mm, 2)}<small> mm</small>` : '—', val.pair_residual_before_mm !== undefined ? `校正前 ${fmt(val.pair_residual_before_mm, 2)} mm` : '无真值时不显示')}
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>校正方法</h3><div class="hint">演示项目可用“模拟 GACOS”，它与合成大气完全一致，便于观察上限。</div></div></div>
          <div class="tech-grid" style="grid-template-columns:repeat(3,minmax(0,1fr))">
            ${[['gacos', 'GACOS', '外部逐日天顶延迟，转 LOS 后逐对差分'], ['empirical', '经验高程', '相位—高程稳健回归，无需外部数据'], ['off', '不校正', '保留原始相位，反演中做时间高通']].map(([k, n, d]) => `
              <button class="tech-card ${method === k ? 'selected' : ''}" data-atm-method="${k}">
                <div class="tc-cn" style="margin-top:0">${n}</div><div class="tc-desc">${d}</div><span class="tc-check">✓</span>
              </button>`).join('')}
          </div>
          ${method === 'gacos' ? `
            <div class="form-grid" style="margin-top:12px">
              <div class="field"><label>GACOS 数据源</label><select data-param="gacos_source" id="gacos-source">
                <option value="demo" ${source === 'demo' ? 'selected' : ''}>演示 · 合成逐日延迟</option>
                <option value="local" ${source === 'local' ? 'selected' : ''}>本地 · .ztd / .tif / .npy</option>
                <option value="online" ${source === 'online' ? 'selected' : ''}>在线 · 自定义服务</option>
              </select></div>
              <div class="field"><label>多视因子 <span class="unit">与干涉一致</span></label><input type="number" data-param="multilook" min="1" max="8" value="${esc(pval('multilook', 2))}"></div>
            </div>
            ${source === 'local' ? `<div class="field" style="margin-top:9px"><label>GACOS 目录</label><input type="text" data-param="gacos_dir" value="${esc(pval('gacos_dir', ''))}" placeholder="D:\\gacos\\2023"><span class="help">目录内文件名需包含日期（YYYYMMDD 或 YYYY-MM-DD），支持 .ztd/.rsc、.tif、.npy。</span></div>` : ''}
            ${source === 'online' ? `<div class="form-grid expert-only" style="margin-top:9px">
              <div class="field"><label>服务地址</label><input type="text" data-param="gacos_url" value="${esc(pval('gacos_url', ''))}" placeholder="https://…"></div>
              <div class="field"><label>邮箱 / 令牌</label><input type="text" data-param="gacos_email" value="" placeholder="仅本次任务使用"></div>
            </div><div class="field expert-only" style="margin-top:8px"><label>访问令牌</label><input type="password" data-param="gacos_token" value="" placeholder="不会写入项目"></div>` : ''}
            ${source === 'demo' && !ov.project.demo ? '<div class="inline-note" style="margin-top:9px">当前项目不是演示数据，没有内置模拟大气；请改用本地或在线 GACOS，或选择经验高程校正。</div>' : ''}
          ` : ''}
          <div class="btn-row" style="margin-top:13px">${actionButton('btn-run-atm', hasAtm ? '重新校正大气' : '运行大气校正')}</div>
          <div class="small muted" style="margin-top:7px">校正结果写入每个干涉对的 unwrapped_atmcorr.npy，反演会自动优先读取。</div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>诊断</h3><div class="hint">左侧为逐日 APS 屏，右侧为校正前后相位分布与逐对离散度。</div></div></div>
          ${ov.assets && ov.assets.atmosphere ? `<div class="thumb-card"><img src="${assetUrl(ov.assets.atmosphere)}" alt="大气诊断"></div>` : '<div class="empty-mini">运行大气校正后生成诊断图。</div>'}
          <div style="margin-top:10px">
            <div class="kv-row"><span class="k">方法</span><span class="v">${esc(atm.method || '—')}</span></div>
            <div class="kv-row"><span class="k">数据源</span><span class="v">${esc(atm.source || '—')}</span></div>
            <div class="kv-row"><span class="k">逐日屏数量</span><span class="v">${esc(atm.n_screens || '—')}</span></div>
            <div class="kv-row"><span class="k">平均 |延迟|</span><span class="v">${atm.mean_abs_screen_mm !== undefined ? `${fmt(atm.mean_abs_screen_mm, 2)} mm` : '—'}</span></div>
            ${val.pair_residual_before_mm !== undefined ? `<div class="kv-row"><span class="k">真值成对残差（前 → 后）</span><span class="v">${fmt(val.pair_residual_before_mm, 2)} → ${fmt(val.pair_residual_after_mm, 2)} mm</span></div>` : ''}
            ${val.pair_residual_noise_floor_mm !== undefined ? `<div class="kv-row"><span class="k">噪声底</span><span class="v">${fmt(val.pair_residual_noise_floor_mm, 2)} mm</span></div>` : ''}
          </div>
          <div class="inline-note" style="margin-top:10px">校正过度会吃掉真实长波形变；若速率图出现大面积负相关纹理，回退到“不校正”或只保留时间高通。</div>
        </div>
      </div>`;
  }

  function bindAtmosphereStep() {
    $$('[data-atm-method]').forEach(btn => btn.addEventListener('click', () => {
      state.project.params = { ...state.project.params, atmosphere_method: btn.dataset.atmMethod };
      renderStepContent();
    }));
    const sel = $('#gacos-source');
    if (sel) sel.addEventListener('change', () => {
      state.atmSource = sel.value;
      state.project.params = { ...state.project.params, gacos_source: sel.value, atmosphere_method: 'gacos' };
      renderStepContent();
    });
    const run = $('#btn-run-atm');
    if (run) run.addEventListener('click', () => runStep('atmosphere'));
  }

  /* --------------------------------------------------------------- step 6 · inversion */
  function inversionStep() {
    const ov = state.overview;
    const inv = ov.inversion || {};
    const tech = currentTech();
    const dates = (ov.stack && ov.stack.dates) || ov.dates || [];
    const existing = inv.technique === tech;
    const demWarn = pval('estimate_dem_error', false);
    return `
      ${guide('∑', '把几十对相位，反演成一条时间序列', '反演同时估计每个日期的形变、DEM 误差与大气残差，并用正则化抑制病态解。短时序、低冗余网络不建议估计 DEM 误差；本演示默认关闭，避免与形变历史共线。', 'STEP 6 / 7')}
      <div class="grid-4">
        ${statCard('当前路线', techMeta(tech).name, existing ? '已有同路线结果' : '尚未按此路线反演', 'accent')}
        ${statCard('位移中位数', inv.displacement_median_mm !== undefined && String(inv.technique).toLowerCase() === 'dinsar' ? `${fmt(inv.displacement_median_mm, 2)}<small> mm</small>` : (inv.velocity_median_mm_yr !== undefined ? `${fmt(inv.velocity_median_mm_yr, 2)}<small> mm/yr</small>` : '—'), 'LOS 向，未做升/降轨分解')}
        ${statCard('有效像元', inv.valid_pixels ? `${inv.valid_pixels}` : '—', inv.valid_fraction ? `占 ${fmtPct(inv.valid_fraction)}` : '')}
        ${statCard('时序相干性', inv.temporal_coherence_median !== undefined ? fmt(inv.temporal_coherence_median, 2) : '—', inv.residual_median_mm !== undefined ? `残差 ${fmt(inv.residual_median_mm, 2)} mm` : '')}
      </div>
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>反演路线</h3><div class="hint">切换路线会按新路线的采样与网络参数重建下游产品，避免旧干涉图与新路线混用。</div></div><span class="badge ${existing ? 'green' : 'amber'}">${esc(techMeta(tech).name)}</span></div>
        ${techGrid(tech, 'tech-card', 'data-inv-tech')}
        <div class="btn-row" style="margin-top:12px">${actionButton('btn-run-inv', existing ? '重新反演' : '运行时序反演', 'primary')}<span class="small muted">${tech === 'ps' ? 'PS 约 40–60 秒' : (tech === 'ds' ? 'DS 约 20–30 秒' : (tech === 'dinsar' ? 'D-InSAR 约 5 秒' : 'SBAS 约 20–30 秒'))}（演示数据）</span></div>
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>${tech === 'dinsar' ? '干涉对' : '时序参数'}</h3><div class="hint">${tech === 'dinsar' ? 'D-InSAR 只反演一个干涉对，不生成速率。' : '正则化越大越平滑；过大将压低真实形变。'}</div></div></div>
          ${tech === 'dinsar' ? dinsarPairFields(ov.pairs || []) : `
            <div class="form-grid">
              <div class="field"><label>多视因子</label><select data-param="multilook">${[1, 2, 4].map(v => `<option value="${v}" ${Number(pval('multilook', 2)) === v ? 'selected' : ''}>${v} × ${v}</option>`).join('')}</select></div>
              <div class="field"><label>正则化权重</label>
                <div class="range-row"><input type="range" id="reg-range" data-param="regularization" min="0" max="1" step="0.01" value="${esc(pval('regularization', 0.3))}"><output id="reg-out">${fmt(pval('regularization', 0.3), 2)}</output></div>
              </div>
            </div>
            <label class="checkline"><input type="checkbox" data-param="remove_ramp" ${pval('remove_ramp', true) ? 'checked' : ''}><span>去除残余趋势面<span class="help">扣除轨道/长波误差，同时会移除同尺度的真实形变。</span></span></label>
            <label class="checkline expert-only"><input type="checkbox" data-param="estimate_dem_error" ${demWarn ? 'checked' : ''}><span>估计 DEM 误差<span class="help">短时序、低冗余时与形变历史共线，容易产生错误符号；默认关闭。</span></span></label>
            <div class="form-grid expert-only" style="margin-top:6px">
              <div class="field"><label>稳健迭代次数</label><input type="number" data-param="robust_iterations" min="0" max="10" value="${esc(pval('robust_iterations', 3))}"></div>
              <div class="field"><label>时间高通迭代</label><input type="number" data-param="aps_iterations" min="0" max="5" value="${esc(pval('aps_iterations', 1))}"></div>
              <div class="field"><label>APS 截止波长 <span class="unit">px</span></label><input type="number" data-param="aps_sigma_px" min="2" max="50" value="${esc(pval('aps_sigma_px', 10))}"></div>
              <div class="field"><label>APS 方法</label><select data-param="aps_method">${[['spatiotemporal', '时空高通'], ['off', '关闭']].map(([v, l]) => `<option value="${v}" ${pval('aps_method', 'spatiotemporal') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
            </div>
          `}
          <div class="divider expert-only"></div>
          <div class="form-grid expert-only">
            <div class="field"><label>空间参考</label><select data-param="reference_mode">${[['auto', '自动稳定点'], ['manual', '手动指定']].map(([v, l]) => `<option value="${v}" ${pval('reference_mode', 'auto') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
            <div class="field"><label>参考行 row</label><input type="number" data-param="reference_row" min="0" value="${esc(pval('reference_row', '') ?? '')}" placeholder="自动"></div>
            <div class="field"><label>参考列 col</label><input type="number" data-param="reference_col" min="0" value="${esc(pval('reference_col', '') ?? '')}" placeholder="自动"></div>
          </div>
          <div class="small muted expert-only">手动参考点必须在有效像元内，否则自动回退到稳定点；修改后重跑反演即可重新定标。</div>
          ${tech === 'ps' ? `<div class="divider"></div><div class="form-grid">
            <div class="field"><label>PS 幅度离散阈值 ADI</label><input type="number" data-param="ps_adi_max" min="0.05" max="0.6" step="0.01" value="${esc(pval('ps_adi_max', 0.28))}"></div>
            <div class="field"><label>PS 相干性下限</label><input type="number" data-param="ps_coherence_min" min="0.2" max="0.95" step="0.01" value="${esc(pval('ps_coherence_min', 0.62))}"></div>
          </div>` : ''}
          ${tech === 'ds' ? `<div class="divider"></div><div class="form-grid">
            <div class="field"><label>DS 块大小 <span class="unit">px</span></label><input type="number" data-param="ds_patch" min="4" max="32" value="${esc(pval('ds_patch', 10))}"></div>
            <div class="field"><label>块重叠</label><input type="number" data-param="ds_overlap" min="0" max="0.9" step="0.1" value="${esc(pval('ds_overlap', 0.5))}"></div>
            <div class="field"><label>相位链接</label><select data-param="ds_phase_linking">${[['emi', '特征值分解（EMI）'], ['eigenvalue', '特征值分解（兼容）']].map(([v, l]) => `<option value="${v}" ${pval('ds_phase_linking', 'emi') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
          </div>` : ''}
        </div>
        <div class="card">
          <div class="card-head"><div><h3>结果摘要</h3><div class="hint">${existing ? '当前项目已有这一路线的产品。' : '运行后在此显示摘要，并自动进入成果页。'}</div></div></div>
          ${inv.technique ? `<dl class="meta-list">
            <dt>路线</dt><dd>${esc(String(inv.technique).toUpperCase())}</dd>
            <dt>反演时间</dt><dd>${esc(inv.created_at || '—')}</dd>
            <dt>有效像元</dt><dd>${inv.valid_pixels || '—'} / ${fmtPct(inv.valid_fraction)}</dd>
            ${inv.reference_pixel ? `<dt>参考像元</dt><dd>row ${inv.reference_pixel[0]} · col ${inv.reference_pixel[1]}</dd>` : ''}
            ${inv.pair ? `<dt>干涉对</dt><dd>${esc(inv.pair.id)} · ${esc(inv.pair.ref_date)} → ${esc(inv.pair.sec_date)}</dd>` : ''}
          </dl>` : '<div class="empty-mini">尚未运行反演。</div>'}
          <div class="divider"></div>
          <div class="inline-note">短时序演示中，<b>不要同时开启 DEM 误差估计与强正则化</b>：两者都会吸收长波信号，导致速率被系统性低估。默认值已针对本数据集调过。</div>
          <div class="btn-row" style="margin-top:12px"><button class="btn" id="btn-save-params">保存参数</button></div>
        </div>
      </div>`;
  }

  function bindInversionStep() {
    $$('[data-inv-tech]').forEach(btn => btn.addEventListener('click', () => {
      state.selectedTech = btn.dataset.invTech;
      try {
        const p = collectParams($('#step-content'));
        state.project.params = { ...state.project.params, ...p, ...techDefaults(state.selectedTech) };
      } catch (err) { /* ignore */ }
      renderStepContent();
      renderHeader();
    }));
    const reg = $('#reg-range');
    const out = $('#reg-out');
    if (reg && out) reg.addEventListener('input', () => { out.textContent = Number(reg.value).toFixed(2); });
    $('#btn-run-inv').addEventListener('click', () => {
      if (currentTech() !== state.project.technique) {
        if (!confirm(`切换至 ${techMeta(currentTech()).name} 会重建网络、干涉图、大气与反演产品，耗时更长。继续？`)) return;
      }
      runStep('inversion');
    });
    $('#btn-save-params').addEventListener('click', async () => {
      const p = collectParams($('#step-content'));
      try {
        await saveParams(p, false);
        toast('参数已保存到项目', 'success');
      } catch (err) { toast(err.message, 'error'); }
    });
  }

  /* --------------------------------------------------------------- step 7 · results */
  function resultsStep() {
    if (!state.results && !state.resultsLoading) {
      setTimeout(loadResults, 30);
    }
    if (state.resultsLoading) {
      return `${guide('◎', '成果整理与交付', '正在读取反演产品与真值校验…', 'STEP 7 / 7')}<div class="card"><div class="empty-mini">读取成果中…</div></div>`;
    }
    if (state.resultsError) {
      return `${guide('◎', '成果整理与交付', '成果读取失败，请先完成反演。', 'STEP 7 / 7')}<div class="card"><div class="empty-mini">${esc(state.resultsError)}</div></div>`;
    }
    const ov = state.results;
    if (!ov || !ov.inversion || !ov.inversion.technique) {
      return `${guide('◎', '成果整理与交付', '先完成一次时序反演，才能生成速率图、时间序列与报告。', 'STEP 7 / 7')}<div class="card"><div class="empty-mini">尚无成果。点击上方「一键跑完全流程」或在反演步骤运行。</div></div>`;
    }
    const inv = ov.inversion || {};
    const val = ov.validation || {};
    const tech = String(inv.technique || '').toLowerCase();
    const isDinsar = tech === 'dinsar';
    const grade = val.grade || '—';
    const rmse = val.displacement_rmse_mm;
    const figs = [
      ['velocity', 'LOS 形变速率', isDinsar ? '' : 'mm/yr'],
      ['ps', 'PS 候选点', ''],
      ['ds', 'DS 时序相干性', ''],
      ['dinsar', 'D-InSAR 位移', 'mm'],
      ['temporal_coherence', '时序相干性', ''],
      ['residual', '反演残差', 'mm'],
      ['dem_error', 'DEM 误差', 'm'],
      ['atmosphere', '大气诊断', ''],
      ['network', '干涉网络', ''],
      ['coreg_gain', '配准相干性增益', '']
    ].filter(([key]) => ov.assets && ov.assets[key]);
    return `
      ${guide('◎', '把结果变成可交付的成果', grade !== '—' ? `真值校验：位移 RMSE <b>${fmt(rmse, 2)} mm</b>，评价「${esc(grade)}」。合成数据只验证算法链路，不代表真实场景精度。` : '本项目没有合成真值；请结合相干性、闭合差与外部 GNSS/水准判断可靠性。', 'STEP 7 / 7')}
      <div class="grade-card">
        <div class="grade-seal">${esc(grade)}</div>
        <div><h3>${isDinsar ? '差分干涉成果' : '时序形变成果'} · ${esc(techMeta(tech).name)}</h3><p>${esc(ov.project.name)} · 参考像元 ${inv.reference_pixel ? `row ${inv.reference_pixel[0]} / col ${inv.reference_pixel[1]}` : '自动'} · LOS 向</p></div>
        <div class="grade-metrics">
          <div class="grade-metric"><div class="gm-v">${isDinsar ? fmt(inv.displacement_median_mm, 2) : fmt(inv.velocity_median_mm_yr, 2)}</div><div class="gm-k">${isDinsar ? '位移中位数 / mm' : '速率中位数 / mm/yr'}</div></div>
          <div class="grade-metric"><div class="gm-v">${val.displacement_rmse_mm !== undefined ? fmt(val.displacement_rmse_mm, 2) : '—'}</div><div class="gm-k">位移 RMSE / mm</div></div>
          <div class="grade-metric"><div class="gm-v">${val.velocity_correlation !== undefined ? fmt(val.velocity_correlation, 3) : '—'}</div><div class="gm-k">速率空间相关</div></div>
        </div>
      </div>
      <div class="grid-4" style="margin-top:14px">
        ${statCard('有效像元', inv.valid_pixels || '—', inv.valid_fraction ? `占 ${fmtPct(inv.valid_fraction)}` : '')}
        ${statCard('时序相干性', fmt(inv.temporal_coherence_median, 2), '中位数')}
        ${statCard(isDinsar ? '相干性中位数' : '反演残差', isDinsar ? fmt(inv.coherence_median, 2) : `${fmt(inv.residual_median_mm, 2)}<small> mm</small>`, isDinsar ? '' : '越小越贴合观测')}
        ${statCard('成果文件', `${(ov.files || []).length}<small> 个</small>`, 'GeoTIFF · CSV · PNG · DOCX')}
      </div>
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>主图</h3><div class="hint">点击任意图件可放大查看。</div></div><div class="btn-row"><button class="btn btn-small" id="btn-make-report">生成报告</button><a class="btn btn-small btn-primary" href="/api/projects/${encodeURIComponent(state.projectId)}/export">下载成果包 ZIP</a></div></div>
        ${figs.length ? `<div class="figure-grid">${figs.map(([key, label, unit]) => `
          <div class="figure-card" data-src="${assetUrl(ov.assets[key])}">
            <img src="${assetUrl(ov.assets[key])}" alt="${esc(label)}">
            <div class="fig-cap"><b>${esc(label)}</b><span>${esc(unit || key)} · 点击放大</span></div>
          </div>`).join('')}</div>` : '<div class="empty-mini">尚无图件。</div>'}
      </div>
      <div class="grid-2" style="margin-top:14px">
        <div class="card">
          <div class="card-head"><div><h3>像元时间序列</h3><div class="hint">输入行列，或直接跳到形变峰值 / 参考点。</div></div><span class="badge grey" id="series-badge">${state.series ? `${state.series.row}, ${state.series.col}` : '未选择'}</span></div>
          <div class="point-bar">
            <div class="field"><label>行 row</label><input type="number" id="series-row" min="0" value="${state.series ? state.series.row : 0}"></div>
            <div class="field"><label>列 col</label><input type="number" id="series-col" min="0" value="${state.series ? state.series.col : 0}"></div>
            <button class="btn btn-small" id="btn-series-query">查询</button>
            <button class="btn btn-small btn-ghost" id="btn-series-peak">形变峰值</button>
            <button class="btn btn-small btn-ghost" id="btn-series-ref">参考点</button>
            <a class="btn btn-small btn-ghost" id="btn-series-csv" href="#" download>导出该点 CSV</a>
            <button class="btn btn-small btn-ghost" id="btn-set-reference">设为空间参考</button>
          </div>
          <div class="series-wrap" style="margin-top:10px"><canvas id="series-canvas" class="series-canvas"></canvas></div>
          <div class="small muted" id="series-meta" style="margin-top:6px">${state.series ? seriesMetaText(state.series) : ''}</div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>真值校验</h3><div class="hint">只对合成演示数据有效。</div></div><button class="btn btn-small btn-ghost" id="btn-validate-2">重新校验</button></div>
          ${val && val.technique ? validationTable(val) : '<div class="empty-mini">暂无真值校验。真实数据不提供此项。</div>'}
        </div>
      </div>
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>成果文件</h3><div class="hint">按类型分组；压缩包包含产品、报告与项目配置。</div></div><div class="btn-row"><a class="btn btn-small" href="/api/projects/${encodeURIComponent(state.projectId)}/points.csv">点位表 CSV</a><span class="badge grey">${(ov.files || []).length} FILES</span></div></div>
        ${filesHtml(ov.files || [])}
      </div>`;
  }

  function seriesMetaText(series) {
    if (!series) return '';
    const parts = [`row ${series.row} · col ${series.col}`];
    if (series.velocity_mm_yr !== null && series.velocity_mm_yr !== undefined) parts.push(`速率 ${fmtSigned(series.velocity_mm_yr, 2, ' mm/yr')}`);
    if (series.temporal_coherence !== null && series.temporal_coherence !== undefined) parts.push(`时序相干性 ${fmt(series.temporal_coherence, 2)}`);
    parts.push(series.valid ? '有效像元' : '掩膜像元');
    return parts.join(' · ');
  }

  function validationTable(val) {
    const rows = [
      ['位移 RMSE', 'displacement_rmse_mm', ' mm', 2],
      ['位移偏差', 'displacement_bias_mm', ' mm', 2],
      ['位移 P95', 'displacement_p95_mm', ' mm', 2],
      ['速率 RMSE', 'velocity_rmse_mm_yr', ' mm/yr', 2],
      ['速率偏差', 'velocity_bias_mm_yr', ' mm/yr', 2],
      ['速率空间相关', 'velocity_correlation', '', 3],
      ['DEM 误差相关', 'dem_correlation', '', 3]
    ].filter(([, key]) => val[key] !== undefined && val[key] !== null);
    return `<div class="table-wrap short"><table class="data"><thead><tr><th>指标</th><th>数值</th></tr></thead><tbody>
      ${rows.map(([label, key, unit, digits]) => `<tr><td>${label}</td><td class="num">${fmt(val[key], digits)}${unit}</td></tr>`).join('')}
      ${val.atmosphere ? `<tr><td>大气校正前成对残差</td><td class="num">${fmt(val.atmosphere.pair_residual_before_mm, 2)} mm</td></tr>
        <tr><td>大气校正后成对残差</td><td class="num">${fmt(val.atmosphere.pair_residual_after_mm, 2)} mm</td></tr>` : ''}
      <tr><td>综合评价</td><td><span class="badge ${val.grade === '优秀' ? 'green' : (val.grade === '需检查' ? 'red' : 'amber')}">${esc(val.grade || '—')}</span></td></tr>
    </tbody></table></div><div class="inline-note" style="margin-top:8px">${esc(val.note || '')}</div>`;
  }

  function filesHtml(files) {
    if (!files.length) return '<div class="empty-mini">暂无成果文件。</div>';
    const groups = {};
    files.forEach(f => {
      const g = f.group || 'root';
      (groups[g] = groups[g] || []).push(f);
    });
    return Object.keys(groups).sort().map(g => `<div class="file-group"><h4>${esc(GROUP_LABELS[g] || g)}</h4>
      ${groups[g].map(f => {
        const isImage = /\.(png|jpg|jpeg)$/i.test(f.name);
        return `<div class="file-row"><span class="fr-name" title="${esc(f.path)}">${esc(f.name)}</span><span class="fr-size">${fmtBytes(f.size)}</span>
          ${isImage ? `<a href="${assetUrl(f.path)}" target="_blank" rel="noopener">查看</a>` : ''}
          <a href="${downloadUrl(f.path)}">下载</a></div>`;
      }).join('')}
    </div>`).join('');
  }

  async function loadResults() {
    if (!state.projectId) return;
    state.resultsLoading = true;
    state.resultsError = '';
    renderStepContent();
    try {
      const results = await api(`/api/projects/${encodeURIComponent(state.projectId)}/results`);
      state.results = results;
      let peaks = null;
      try { peaks = await api(`/api/projects/${encodeURIComponent(state.projectId)}/peaks`); } catch (err) { peaks = null; }
      state.peaks = peaks;
      if (peaks && peaks.peak) {
        let row = Number(state.seriesPoint ? state.seriesPoint.row : peaks.peak.row);
        let col = Number(state.seriesPoint ? state.seriesPoint.col : peaks.peak.col);
        await loadSeries(row, col, false);
      }
    } catch (err) {
      state.resultsError = err.message;
    } finally {
      state.resultsLoading = false;
      renderStepContent();
    }
  }

  async function loadSeries(row, col, redraw = true) {
    if (!state.projectId) return;
    try {
      const series = await api(`/api/projects/${encodeURIComponent(state.projectId)}/series?row=${encodeURIComponent(row)}&col=${encodeURIComponent(col)}`);
      state.series = series;
      state.seriesPoint = { row: Number(row), col: Number(col) };
      if (redraw) {
        renderStepContent();
        bindResultsStep();
      }
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  function bindResultsStep() {
    const report = $('#btn-make-report');
    if (report) report.addEventListener('click', makeReport);
    const valBtn = $('#btn-validate-2');
    if (valBtn) valBtn.addEventListener('click', validateNow);
    const query = $('#btn-series-query');
    if (query) query.addEventListener('click', () => {
      const row = Number($('#series-row').value);
      const col = Number($('#series-col').value);
      loadSeries(row, col);
    });
    const peakBtn = $('#btn-series-peak');
    if (peakBtn) peakBtn.addEventListener('click', () => {
      if (!state.peaks || !state.peaks.peak) { toast('尚未读取峰值点'); return; }
      loadSeries(state.peaks.peak.row, state.peaks.peak.col);
    });
    const refBtn = $('#btn-series-ref');
    if (refBtn) refBtn.addEventListener('click', () => {
      if (!state.peaks || !state.peaks.reference_pixel) { toast('尚未读取参考点'); return; }
      loadSeries(state.peaks.reference_pixel[0], state.peaks.reference_pixel[1]);
    });
    const csvBtn = $('#btn-series-csv');
    if (csvBtn && state.seriesPoint) {
      csvBtn.href = `/api/projects/${encodeURIComponent(state.projectId)}/series.csv?row=${encodeURIComponent(state.seriesPoint.row)}&col=${encodeURIComponent(state.seriesPoint.col)}`;
    }
    const setRef = $('#btn-set-reference');
    if (setRef) setRef.addEventListener('click', setCurrentPointAsReference);
    if (state.seriesPoint && $('#series-row')) {
      $('#series-row').value = state.seriesPoint.row;
      $('#series-col').value = state.seriesPoint.col;
    }
    drawSeries();
  }

  async function makeReport() {
    const btn = $('#btn-make-report');
    if (btn) { btn.disabled = true; btn.textContent = '生成中…'; }
    try {
      const out = await api(`/api/projects/${encodeURIComponent(state.projectId)}/report`, { method: 'POST' });
      toast(`报告已生成：${out.docx ? 'Markdown + DOCX' : 'Markdown'}`, 'success');
      await refreshOverview({ keepStep: true });
      state.results = await api(`/api/projects/${encodeURIComponent(state.projectId)}/results`);
      renderStepContent();
      bindResultsStep();
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '生成报告'; }
    }
  }

  async function validateNow() {
    if (!state.projectId) return;
    try {
      const val = await api(`/api/projects/${encodeURIComponent(state.projectId)}/validate`, { method: 'POST' });
      state.results = null;
      if (state.step === 'results') {
        await loadResults();
      } else {
        await refreshOverview({ keepStep: true });
        renderAll();
      }
      toast(`校验完成：位移 RMSE ${fmt(val.displacement_rmse_mm, 2)} mm · ${val.grade}`, 'success');
    } catch (err) { toast(err.message, 'error'); }
  }

  function drawSeries() {
    const canvas = $('#series-canvas');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(280, rect.width);
    const height = 260;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const series = state.series;
    if (!series || !Array.isArray(series.disp_mm) || !series.disp_mm.length) {
      ctx.fillStyle = '#73786d';
      ctx.font = '12px "Microsoft YaHei", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('查询一个像元后显示 LOS 位移时间序列', width / 2, height / 2);
      return;
    }
    const vals = series.disp_mm.map(v => (v === null || v === undefined ? NaN : Number(v)));
    const dates = series.dates || [];
    const finite = vals.filter(Number.isFinite);
    if (!finite.length) {
      ctx.fillStyle = '#73786d';
      ctx.font = '12px "Microsoft YaHei", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('该像元被掩膜或无有效观测', width / 2, height / 2);
      return;
    }
    let lo = Math.min(0, ...finite);
    let hi = Math.max(0, ...finite);
    if (lo === hi) { lo -= 1; hi += 1; }
    const span = hi - lo;
    lo -= span * 0.12;
    hi += span * 0.12;
    const margin = { l: 48, r: 14, t: 14, b: 30 };
    const iw = width - margin.l - margin.r;
    const ih = height - margin.t - margin.b;
    const n = Math.max(vals.length, 2);
    const X = i => margin.l + (iw * i) / (n - 1);
    const Y = v => margin.t + ih - ((v - lo) / (hi - lo)) * ih;

    ctx.font = '10px "Cascadia Mono", Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = hi - ((hi - lo) * i) / 4;
      const y = Y(v);
      ctx.strokeStyle = '#ece6d8';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(margin.l, y); ctx.lineTo(width - margin.r, y); ctx.stroke();
      ctx.fillStyle = '#8a9085';
      ctx.fillText(v.toFixed(1), margin.l - 7, y);
    }
    const zeroY = Y(0);
    ctx.strokeStyle = '#cfc6b2';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(margin.l, zeroY); ctx.lineTo(width - margin.r, zeroY); ctx.stroke();
    ctx.setLineDash([]);

    const xTickCount = Math.min(6, dates.length || 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < xTickCount; i++) {
      const idx = Math.round(((dates.length - 1 || 1) * i) / Math.max(1, xTickCount - 1));
      const x = X(idx);
      ctx.strokeStyle = '#ece6d8';
      ctx.beginPath(); ctx.moveTo(x, margin.t); ctx.lineTo(x, margin.t + ih); ctx.stroke();
      ctx.fillStyle = '#8a9085';
      ctx.fillText(shortDate(dates[idx] || ''), x, margin.t + ih + 6);
    }

    const validPoints = [];
    vals.forEach((v, i) => { if (Number.isFinite(v)) validPoints.push({ x: X(i), y: Y(v), v, i }); });
    if (validPoints.length > 1) {
      const gradient = ctx.createLinearGradient(0, margin.t, 0, margin.t + ih);
      gradient.addColorStop(0, 'rgba(47, 127, 110, .22)');
      gradient.addColorStop(1, 'rgba(47, 127, 110, .015)');
      ctx.beginPath();
      ctx.moveTo(validPoints[0].x, zeroY);
      validPoints.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.lineTo(validPoints[validPoints.length - 1].x, zeroY);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();
    }
    ctx.strokeStyle = '#2f7f6e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    vals.forEach((v, i) => {
      if (!Number.isFinite(v)) { started = false; return; }
      if (!started) { ctx.moveTo(X(i), Y(v)); started = true; }
      else ctx.lineTo(X(i), Y(v));
    });
    ctx.stroke();
    validPoints.forEach(p => {
      ctx.beginPath();
      ctx.fillStyle = p.i === 0 ? '#c27a2d' : '#2f7f6e';
      ctx.arc(p.x, p.y, 2.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fffdf6';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
    ctx.fillStyle = '#73786d';
    ctx.font = '9.5px "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('LOS 位移 / mm', margin.l, 2);
  }

  window.addEventListener('resize', () => {
    if (state.step === 'results') drawSeries();
  });

  /* --------------------------------------------------------------- job console */
  async function runAll() {
    if (!state.projectId || state.jobRunning) return;
    if (!confirm('将依次运行配准、网络、干涉、大气与反演，已有产品会被覆盖。继续？')) return;
    const params = { ...(state.project.params || {}), ...collectParams($('#step-content')) };
    try {
      await saveParams(params, true);
    } catch (err) {
      toast(err.message, 'error');
      return;
    }
    const res = await startJob(`/api/projects/${encodeURIComponent(state.projectId)}/run`, {
      technique: currentTech(), params
    });
    if (res) state.jobMeta = { kind: 'pipeline' };
  }

  async function runQuickDemo() {
    const btn = $('#welcome-demo');
    if (btn) { btn.disabled = true; btn.textContent = '正在创建演示项目…'; }
    try {
      const data = await api('/api/projects', {
        method: 'POST',
        body: { name: '快速体验 · 阿尔卑斯山谷', technique: 'sbas', source: 'demo' }
      });
      await loadProjects();
      await openProject(data.project.id);
      const res = await startJob(`/api/projects/${encodeURIComponent(data.project.id)}/run`, {
        technique: 'sbas', params: state.project.params || {}
      });
      if (res) {
        state.jobMeta = { kind: 'pipeline' };
        toast('演示项目已创建，正在自动跑完整条链路。', 'success');
      }
    } catch (err) {
      toast(`快速体验创建失败：${err.message}`, 'error');
      if (btn) { btn.disabled = false; btn.textContent = '用演示数据跑一遍'; }
    }
  }

  async function runStep(kind) {
    if (!state.projectId || state.jobRunning) return;
    const params = { ...(state.project.params || {}), ...collectParams($('#step-content')) };
    if (kind === 'network' && state.overview.pairs && state.overview.pairs.length) {
      if (!confirm('重建网络会清空下游产品（干涉、大气、反演需要重跑）。继续？')) return;
    }
    try {
      await saveParams(params, true);
    } catch (err) {
      toast(err.message, 'error');
      return;
    }
    const res = await startJob(`/api/projects/${encodeURIComponent(state.projectId)}/${kind}`, {
      technique: currentTech(), params
    });
    if (res) state.jobMeta = { kind };
  }

  async function saveParams(params, patch = true) {
    if (!state.projectId) return;
    const clean = {};
    Object.keys(params || {}).forEach(key => {
      const k = key.toLowerCase();
      if (k.includes('password') || k.includes('secret') || k.includes('token') || k === 'gacos_email' || k === 'username') return;
      clean[key] = params[key];
    });
    state.project.params = { ...(state.project.params || {}), ...clean };
    state.overview.project.params = state.project.params;
    if (!patch) return;
    try {
      await api(`/api/projects/${encodeURIComponent(state.projectId)}`, {
        method: 'PATCH',
        body: { params: clean }
      });
    } catch (err) {
      throw new Error(`参数保存失败：${err.message}`);
    }
  }

  async function setCurrentPointAsReference() {
    if (!state.seriesPoint) { toast('请先查询一个像元', 'error'); return; }
    const row = Number(state.seriesPoint.row);
    const col = Number(state.seriesPoint.col);
    if (!confirm(`把 row ${row} / col ${col} 设为空间参考点，并重新运行反演？该点位移将归零。`)) return;
    try {
      await saveParams({ reference_mode: 'manual', reference_row: row, reference_col: col }, true);
      const res = await startJob(`/api/projects/${encodeURIComponent(state.projectId)}/inversion`, {
        technique: currentTech(), params: state.project.params
      });
      if (res) state.jobMeta = { kind: 'inversion' };
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function resetProjectProducts() {
    if (!state.projectId || state.jobRunning) return;
    if (!confirm('重置会清空干涉、大气、反演与导出成果，保留原始影像和配准结果。继续？')) return;
    try {
      await api(`/api/projects/${encodeURIComponent(state.projectId)}/reset`, {
        method: 'POST', body: { keep_coreg: true }
      });
      state.results = null;
      state.series = null;
      state.peaks = null;
      await refreshOverview({ keepStep: true });
      state.step = 'network';
      renderAll();
      toast('派生结果已清理，可从网络步骤重新开始。', 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  function collectParams(root) {
    if (!root) return {};
    const out = {};
    $$('[data-param]', root).forEach(el => {
      const key = el.dataset.param;
      if (!key) return;
      if (el.type === 'checkbox') out[key] = !!el.checked;
      else if (el.type === 'number' || el.type === 'range') out[key] = el.value === '' ? null : Number(el.value);
      else out[key] = el.value;
    });
    const ref = $('[data-dinsar-ref]', root);
    const sec = $('[data-dinsar-sec]', root);
    if (ref && sec) {
      const a = Number(ref.value), b = Number(sec.value);
      if (Number.isFinite(a) && Number.isFinite(b) && a !== b) out.dinsar_pair = [Math.min(a, b), Math.max(a, b)];
    }
    return out;
  }

  async function startJob(url, body) {
    if (state.jobRunning) { toast('已有任务在运行，请等待完成或取消。'); return null; }
    try {
      const data = await api(url, { method: 'POST', body });
      if (!data || !data.job) throw new Error('后台任务创建失败');
      trackJob(data.job);
      return data.job;
    } catch (err) {
      if (err.status === 409 && err.payload && err.payload.job) {
        attachJob(err.payload.job, { notify: false });
        toast('该项目已有任务在运行，已重新连接进度。', 'info');
        return err.payload.job;
      }
      toast(err.message, 'error');
      return null;
    }
  }

  function attachJob(job, { notify = false } = {}) {
    if (!job || !job.id) return;
    if (state.jobTimer) { clearTimeout(state.jobTimer); state.jobTimer = null; }
    state.jobId = job.id;
    state.job = job;
    state.jobProjectId = job.project_id || state.projectId;
    state.jobRunning = !['done', 'failed', 'cancelled', 'interrupted'].includes(job.status);
    const stamp = job.started_at ? Date.parse(String(job.started_at).replace(' ', 'T')) : NaN;
    state.jobStarted = Number.isFinite(stamp) ? stamp : Date.now();
    renderConsole(job);
    document.body.classList.add('job-active');
    const panel = $('#console');
    if (panel) panel.classList.add('open');
    $$('[data-run]').forEach(btn => { btn.disabled = true; });
    if (notify) toast(`任务已启动：${job.label || job.kind}`);
    if (state.jobRunning) {
      pollJob();
    }
  }

  function trackJob(job) {
    attachJob(job, { notify: true });
  }

  async function pollJob() {
    if (!state.jobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(state.jobId)}`);
      state.job = job;
      renderConsole(job);
      if (['done', 'failed', 'cancelled', 'interrupted'].includes(job.status)) {
        await finishJob(job);
        return;
      }
    } catch (err) {
      console.error(err);
      if (err.status === 404) {
        state.jobRunning = false;
        state.jobId = null;
        toast('后台任务记录已不存在，请刷新项目状态。', 'error');
        return;
      }
    }
    state.jobTimer = setTimeout(pollJob, 1000);
  }

  async function finishJob(job) {
    state.jobRunning = false;
    state.jobId = null;
    document.body.classList.remove('job-active');
    $$('[data-run]').forEach(btn => { btn.disabled = false; });
    const belongs = !state.jobProjectId || state.jobProjectId === state.projectId;
    const kind = (job.meta && job.meta.kind === 'pipeline') || job.kind === 'pipeline'
      ? 'pipeline' : job.kind;
    if (job.status === 'done') {
      const elapsed = job.started_at && job.finished_at
        ? Math.max(0, (Date.parse(job.finished_at.replace(' ', 'T')) - Date.parse(job.started_at.replace(' ', 'T'))) / 1000)
        : (Date.now() - state.jobStarted) / 1000;
      toast(`${job.label || '任务'}完成${elapsed ? ` · ${elapsed.toFixed(1)} 秒` : ''}`, 'success');
      if (belongs) {
        state.results = null;
        await refreshOverview({ keepStep: true }).catch(() => {});
        const target = nextStepFor(kind);
        if (target) state.step = target;
      } else {
        await loadProjects().catch(() => {});
      }
      if (window.matchMedia('(max-width: 1180px)').matches) {
        setTimeout(() => { const panel = $('#console'); if (panel) panel.classList.remove('open'); }, 1400);
      }
    } else if (job.status === 'cancelled') {
      toast('任务已取消');
      if (belongs) await refreshOverview({ keepStep: true }).catch(() => {});
    } else {
      toast(`任务失败：${job.error || job.stage || '未知错误'}`, 'error');
      if (belongs) await refreshOverview({ keepStep: true }).catch(() => {});
    }
    if (belongs) {
      renderAll();
    } else {
      renderSidebar();
    }
  }

  function nextStepFor(kind) {
    if (!state.overview) return null;
    if (kind === 'pipeline') return state.overview.inversion && state.overview.inversion.technique ? 'results' : 'inversion';
    if (kind === 'network') return 'interferometry';
    if (kind === 'interferometry') return 'atmosphere';
    if (kind === 'atmosphere') return 'inversion';
    if (kind === 'inversion') return 'results';
    return null;
  }

  function renderConsole(job) {
    const badge = $('#console-state');
    const bar = $('#job-progress');
    const stage = $('#job-stage');
    const pct = $('#job-percent');
    const log = $('#console-log');
    const cancel = $('#btn-cancel-job');
    const clear = $('#btn-clear-log');
    if (!badge) return;
    if (!job) {
      badge.className = 'console-state idle';
      badge.innerHTML = '<i></i><span>空闲</span>';
      if (bar) bar.style.width = '0%';
      if (stage) stage.textContent = '等待任务';
      if (pct) pct.textContent = '0%';
      if (cancel) cancel.disabled = true;
      return;
    }
    const statusMap = {
      queued: ['queued', '排队中'],
      running: ['running', '运行中'],
      cancelling: ['running', '取消中'],
      done: ['done', '已完成'],
      failed: ['failed', '失败'],
      cancelled: ['done', '已取消'],
      interrupted: ['failed', '已中断']
    };
    const [cls, label] = statusMap[job.status] || ['idle', job.status || '未知'];
    badge.className = `console-state ${cls}`;
    badge.innerHTML = `<i></i><span>${label}</span>`;
    const progress = Number(job.progress) || 0;
    if (bar) bar.style.width = `${Math.round(progress * 100)}%`;
    if (pct) pct.textContent = `${Math.round(progress * 100)}%`;
    if (stage) stage.textContent = job.stage || job.label || '';
    if (cancel) cancel.disabled = !['queued', 'running'].includes(job.status);
    if (clear) clear.disabled = !(job.logs && job.logs.length);
    const logs = job.logs || [];
    if (logs.length && log) {
      log.innerHTML = logs.map(line => `<div class="log-line ${line.level === 'error' ? 'error' : (line.level === 'warn' ? 'warn' : '')}"><span class="t">${esc(line.time)}</span><span class="m">${esc(line.message)}</span></div>`).join('');
      log.scrollTop = log.scrollHeight;
    } else if (log) {
      log.innerHTML = `<div class="console-empty"><span class="console-glyph">⌁</span><p>${esc(job.stage || '任务已排队…')}</p></div>`;
    }
  }

  async function cancelJob() {
    if (!state.jobId) return;
    try {
      await api(`/api/jobs/${encodeURIComponent(state.jobId)}/cancel`, { method: 'POST' });
      toast('已发送取消请求，当前步骤结束后停止');
    } catch (err) { toast(err.message, 'error'); }
  }

  /* --------------------------------------------------------------- new project modal */
  function openNewProject(technique = null) {
    if (typeof technique !== 'string') technique = null;
    state.newProject = {
      source: 'demo',
      technique: technique || 'sbas',
      name: technique ? `${techMeta(technique).cn}演示项目` : '阿尔卑斯山谷 · 演示',
      path: '',
      dem: ''
    };
    renderNewProjectModal();
  }

  function renderNewProjectModal() {
    const np = state.newProject;
    const isImport = np.source === 'import';
    showModal(`<div class="modal-head">
      <div><span class="eyebrow">NEW PROJECT</span><h2>新建项目</h2><p>先用合成数据跑通全流程，或把本地 SLC 栈接进来。</p></div>
      <button class="modal-close" aria-label="关闭">×</button>
    </div>
    <div class="modal-body">
      <div class="source-switch">
        <button id="np-source-demo" class="${!isImport ? 'active' : ''}">合成演示数据</button>
        <button id="np-source-import" class="${isImport ? 'active' : ''}">导入本地数据</button>
      </div>
      <div class="field"><label>项目名称</label><input type="text" id="np-name" value="${esc(np.name)}" placeholder="例如：四川某滑坡隐患点"></div>
      <div class="field"><label>技术路线</label></div>
      ${techGrid(np.technique, 'tech-card', 'data-np-tech')}
      ${isImport ? `
        <div class="field" style="margin-top:12px"><label>数据目录或文件路径</label><input type="text" id="np-path" value="${esc(np.path)}" placeholder="D:\\data\\sentinel1\\stack"></div>
        <div class="field"><label>DEM 路径 <span class="unit">可选</span></label><input type="text" id="np-dem" value="${esc(np.dem)}" placeholder="D:\\data\\dem.tif"></div>
        <div class="inline-note">支持原生 NumPy 栈、ENVI 栈、复数 GeoTIFF 目录与 Sentinel-1 SAFE（实验）。导入时会尽量读取同目录的 meta.json。</div>
      ` : `
        <div class="inline-note">将生成 24 景 Sentinel-1 风格 SLC，包含已知滑坡形变、季节性波动、大气延迟、DEM 误差与亚像元配准偏移。<b>完全离线</b>，创建后即可一键运行。</div>
      `}
    </div>
    <div class="modal-foot">
      <button class="btn btn-ghost" id="np-cancel">取消</button>
      <button class="btn btn-primary" id="np-create">${isImport ? '创建并导入' : '创建并生成数据'}</button>
    </div>`, true);
    $('#np-source-demo').addEventListener('click', () => { np.source = 'demo'; captureNewProjectFields(); renderNewProjectModal(); });
    $('#np-source-import').addEventListener('click', () => { np.source = 'import'; captureNewProjectFields(); renderNewProjectModal(); });
    $$('[data-np-tech]').forEach(btn => btn.addEventListener('click', () => {
      np.technique = btn.dataset.npTech;
      captureNewProjectFields();
      renderNewProjectModal();
    }));
    $('#np-name').addEventListener('input', ev => { np.name = ev.target.value; });
    const pathInput = $('#np-path');
    if (pathInput) pathInput.addEventListener('input', ev => { np.path = ev.target.value; });
    const demInput = $('#np-dem');
    if (demInput) demInput.addEventListener('input', ev => { np.dem = ev.target.value; });
    $('#np-cancel').addEventListener('click', closeModal);
    $('#np-create').addEventListener('click', createProject);
  }

  function captureNewProjectFields() {
    const np = state.newProject;
    if ($('#np-name')) np.name = $('#np-name').value;
    if ($('#np-path')) np.path = $('#np-path').value;
    if ($('#np-dem')) np.dem = $('#np-dem').value;
  }

  async function createProject() {
    captureNewProjectFields();
    const np = state.newProject;
    if (!np.name.trim()) { toast('请填写项目名称', 'error'); return; }
    if (np.source === 'import' && !np.path.trim()) { toast('请填写数据路径', 'error'); return; }
    const btn = $('#np-create');
    btn.disabled = true;
    btn.textContent = np.source === 'demo' ? '正在生成演示数据…' : '正在导入…';
    try {
      const body = { name: np.name.trim(), technique: np.technique, source: np.source };
      if (np.source === 'import') { body.path = np.path.trim(); body.dem_path = np.dem.trim(); }
      const data = await api('/api/projects', { method: 'POST', body });
      closeModal();
      await loadProjects();
      await openProject(data.project.id);
      toast(np.source === 'demo' ? '演示项目已生成，可以开始跑流程了。' : '数据导入完成。', 'success');
    } catch (err) {
      toast(err.message, 'error');
      btn.disabled = false;
      btn.textContent = np.source === 'demo' ? '创建并生成数据' : '创建并导入';
    }
  }

  /* --------------------------------------------------------------- help & doctor */
  function openHelp() {
    showModal(`<div class="modal-head"><div><span class="eyebrow">WORKFLOW</span><h2>七步跑完一条 InSAR 链路</h2><p>面向零基础，也给专家留了全部参数。</p></div><button class="modal-close">×</button></div>
      <div class="modal-body">
        <div class="help-section"><h3>推荐路线</h3><p>第一次使用，直接选 <b>SBAS-InSAR</b> + <b>合成演示数据</b>，点「一键跑完全流程」。约 20–30 秒后就能在成果页看到速率图、时间序列与真值校验。确认链路正常后，再导入自己的数据。</p></div>
        <div class="help-section"><h3>每一步在做什么</h3><ul>
          <li><b>数据</b> 确认影像数量、时间跨度、尺寸与参考影像。</li>
          <li><b>网络</b> 用短时间/垂直基线把影像连成图，红线是观测方程。</li>
          <li><b>干涉</b> 配准后的两景共轭相乘、滤波、解缠，得到连续相位。</li>
          <li><b>大气</b> 用 GACOS 或相位—高程回归扣掉对流层延迟。</li>
          <li><b>反演</b> 联合所有干涉对，估计每个日期的 LOS 形变。</li>
          <li><b>成果</b> 导出 GeoTIFF、CSV、PNG、DOCX 报告与 ZIP。</li>
        </ul></div>
        <div class="help-section"><h3>四条路线怎么选</h3><ul>
          <li><b>D-InSAR</b>：只有两景、应急事件、快速看位移。没有速率。</li>
          <li><b>SBAS</b>：通用首选，兼顾覆盖与精度，适合沉降/滑坡。</li>
          <li><b>PS</b>：城区、裸岩、桥梁等强散射点，毫米级但密度低。</li>
          <li><b>DS</b>：植被、裸土等低相干区，相位链接后做分布式目标。</li>
        </ul></div>
        <div class="help-section"><h3>参数与可靠性</h3><p>引导模式的默认值偏稳健。专家模式里最值得关注的是：多视因子、相干性阈值、解缠算法、正则化权重与 DEM 误差估计。短时序不要轻易开启 DEM 误差估计，它和形变历史容易共线。所有结果都是 LOS 向，未做升/降轨分解或 GNSS 基准校正。</p></div>
      </div>
      <div class="modal-foot"><button class="btn btn-primary" id="help-create">新建演示项目</button><button class="btn btn-ghost modal-close">关闭</button></div>`, true);
    $('#help-create').addEventListener('click', () => { closeModal(); openNewProject('sbas'); });
  }

  async function openDoctor() {
    showModal(`<div class="modal-head"><div><span class="eyebrow">ENVIRONMENT</span><h2>运行环境检查</h2><p>依赖、工作区写入、磁盘空间。</p></div><button class="modal-close">×</button></div>
      <div class="modal-body"><div class="empty-mini">检查中…</div></div>`, true);
    try {
      const data = await api('/api/doctor');
      const list = (data.checks || []).map(c => `<li><span class="${c.ok ? 'ok' : 'no'}">${c.ok ? '✓' : '!'}</span><span>${esc(c.name)}</span><span class="cv">${esc(c.value)}</span></li>`).join('');
      $('.modal-body').innerHTML = `<div class="inline-note" style="margin-bottom:10px">${data.ok ? '核心依赖齐全，可以开始处理。' : '有检查项未通过，部分功能可能不可用。'} · ${esc(data.platform)}</div><ul class="check-list">${list}</ul>`;
    } catch (err) {
      $('.modal-body').innerHTML = `<div class="empty-mini">检查失败：${esc(err.message)}</div>`;
    }
  }

  async function openHealth() {
    if (!state.projectId) return;
    showModal(`<div class="modal-head"><div><span class="eyebrow">PROJECT HEALTH</span><h2>项目体检</h2><p>只读检查数据、网络、干涉、大气与成果的一致性。</p></div><button class="modal-close">×</button></div>
      <div class="modal-body"><div class="empty-mini">检查中…</div></div>`, true);
    try {
      const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/health`);
      const icon = { ok: '✓', warn: '!', error: '×', info: '·' };
      const checks = (data.checks || []).map(c => `<li class="health-${esc(c.status)}"><span class="health-icon">${icon[c.status] || '·'}</span><span><b>${esc(c.label)}</b><small>${esc(c.detail || '')}</small></span></li>`).join('');
      const issueList = (data.issues || []).filter(i => i.status !== 'ok').map(i => `<div class="inline-note ${i.status === 'error' ? 'danger' : ''}"><b>${esc(i.label)}</b> · ${esc(i.detail)}</div>`).join('');
      const nextLabel = { data: '前往数据', network: '前往网络', interferometry: '前往干涉', atmosphere: '前往大气', inversion: '前往反演', results: '前往成果' }[data.next_step] || '前往下一步';
      $('.modal-body').innerHTML = `
        <div class="health-summary ${data.grade === '健康' ? 'ok' : (data.grade === '需要处理' ? 'bad' : 'warn')}">
          <div><span class="health-grade">${esc(data.grade)}</span><b>${esc(data.technique || '').toUpperCase()} 项目</b></div>
          <div class="health-stats"><span>${data.summary ? data.summary.checks : 0} 项检查</span><span>${data.summary ? data.summary.warnings : 0} 条提醒</span><span>${data.summary ? data.summary.errors : 0} 个问题</span><span>占用 ${data.summary ? data.summary.size_human : '—'}</span><span>磁盘可用 ${data.summary ? data.summary.free_human : '—'}</span></div>
        </div>
        ${issueList ? `<div class="health-issues">${issueList}</div>` : '<div class="inline-note" style="margin:10px 0">核心文件与数组一致，没有发现阻塞问题。</div>'}
        <ul class="check-list health-list">${checks}</ul>
        <div class="inline-note">体检只读取元信息和数组头，不会修改项目；修复方式通常是重跑对应阶段或重置派生结果。</div>`;
      $('.modal-foot') && $('.modal-foot').remove();
      const foot = document.createElement('div');
      foot.className = 'modal-foot';
      foot.innerHTML = `<button class="btn btn-ghost modal-close">关闭</button><button class="btn btn-primary" id="health-next">${nextLabel}</button>`;
      $('.modal', $('#modal-root')).appendChild(foot);
      $$('.modal-close', $('#modal-root')).forEach(btn => btn.addEventListener('click', closeModal));
      $('#health-next').addEventListener('click', () => {
        closeModal();
        gotoStep(data.next_step || 'project');
      });
    } catch (err) {
      $('.modal-body').innerHTML = `<div class="empty-mini">项目体检失败：${esc(err.message)}</div>`;
    }
  }

})();
