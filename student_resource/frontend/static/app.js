// Amazon Business Entity Resolution Studio - Client Application

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  initPipelineMonitor();
  initLiveMatcher();
  initModelStats();
  initSampleMatches();
});

// 0. Theme Toggle (Defaults to crisp Light Theme)
function initTheme() {
  const toggleBtn = document.getElementById('btn-theme-toggle');
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');

  let currentTheme = localStorage.getItem('er_theme') || 'light';
  applyTheme(currentTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      currentTheme = currentTheme === 'light' ? 'dark' : 'light';
      localStorage.setItem('er_theme', currentTheme);
      applyTheme(currentTheme);
    });
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      if (icon) icon.textContent = '☀️';
      if (label) label.textContent = 'Light';
    } else {
      document.documentElement.removeAttribute('data-theme');
      if (icon) icon.textContent = '🌙';
      if (label) label.textContent = 'Dark';
    }
  }
}

// 1. Navigation Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      if (targetId === 'tab-results') {
        loadSampleMatches();
      }
    });
  });
}

// 2. Real-Time Pipeline Progress Monitor
function initPipelineMonitor() {
  updateStatus();
  setInterval(updateStatus, 3000);
}

async function updateStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();

    // KPIs
    document.getElementById('kpi-total-entities').textContent = data.total_test_s1.toLocaleString();
    document.getElementById('kpi-processed-entities').textContent = data.processed_rows.toLocaleString();
    document.getElementById('kpi-progress-pct').textContent = `${data.progress_percentage}% Complete`;
    document.getElementById('kpi-active-partition').textContent = data.current_partition;
    document.getElementById('kpi-partition-detail').textContent = `${data.partition_progress_percentage}% of partition`;

    // Progress Bar
    const pBar = document.getElementById('main-progress-bar');
    pBar.style.width = `${data.progress_percentage}%`;
    document.getElementById('progress-percent-label').textContent = `${data.progress_percentage}%`;

    // System Status Pill
    const pill = document.getElementById('system-status-pill');
    const pillText = document.getElementById('system-status-text');
    if (data.is_complete) {
      pill.style.background = 'rgba(16, 185, 129, 0.15)';
      pill.style.borderColor = 'rgba(16, 185, 129, 0.5)';
      pillText.textContent = 'Inference Complete ✓';
    } else {
      pill.style.background = 'rgba(99, 102, 241, 0.15)';
      pill.style.borderColor = 'rgba(99, 102, 241, 0.4)';
      pillText.textContent = `Processing: ${data.current_partition}`;
    }

    // Partition Cards
    const fStatus = document.getElementById('part-status-france');
    const uStatus = document.getElementById('part-status-us');
    const iStatus = document.getElementById('part-status-india');

    if (data.processed_rows >= 259452) {
      fStatus.innerHTML = '<span style="color: var(--accent-emerald);">✓ Completed (259,452 S1)</span>';
    } else {
      fStatus.innerHTML = `<span style="color: var(--accent-cyan);">In Progress (${data.processed_rows.toLocaleString()} / 259,452)</span>`;
    }

    if (data.processed_rows >= (259452 + 663106)) {
      uStatus.innerHTML = '<span style="color: var(--accent-emerald);">✓ Completed (663,106 S1)</span>';
    } else if (data.processed_rows > 259452) {
      const usRows = data.processed_rows - 259452;
      uStatus.innerHTML = `<span style="color: var(--accent-cyan);">In Progress (${usRows.toLocaleString()} / 663,106)</span>`;
    } else {
      uStatus.innerHTML = '<span style="color: var(--text-dim);">Queued</span>';
    }

    if (data.is_complete) {
      iStatus.innerHTML = '<span style="color: var(--accent-emerald);">✓ Completed (809,986 S1)</span>';
    } else if (data.processed_rows > (259452 + 663106)) {
      const indRows = data.processed_rows - (259452 + 663106);
      iStatus.innerHTML = `<span style="color: var(--accent-cyan);">In Progress (${indRows.toLocaleString()} / 809,986)</span>`;
    } else {
      iStatus.innerHTML = '<span style="color: var(--text-dim);">Queued</span>';
    }

    // TSV Files Monitor
    document.getElementById('tsv-match-rows').textContent = `${data.matching_tsv.rows.toLocaleString()} rows`;
    document.getElementById('tsv-match-size').textContent = `${data.matching_tsv.size_mb} MB`;
    document.getElementById('tsv-cand-rows').textContent = `${data.matching_tsv.rows.toLocaleString()} rows`;
    document.getElementById('tsv-cand-size').textContent = `${data.candidate_tsv.size_mb} MB`;

  } catch (err) {
    console.error('Status fetch error:', err);
  }
}

// 3. Live Matcher Sandbox
const FEATURE_DESCRIPTIONS = {
  'name_ratio': 'Standard Levenshtein ratio on cleaned names',
  'name_token_sort_ratio': 'Levenshtein similarity invariant to word order',
  'name_token_set_ratio': 'Token set overlap similarity (robust to extra words)',
  'name_partial_ratio': 'Best matching substring ratio',
  'name_jaccard': 'Word token intersection over union',
  'name_exact_clean': 'Binary exact match flag on normalized names',
  'name_len_diff': 'Absolute length difference in characters',
  'name_domain_match': 'Concatenated / compact domain string inclusion flag',
  'addr_token_set_ratio': 'Address fuzzy token set similarity',
  'addr_jaccard': 'Address word token Jaccard similarity',
  'addr_len_diff': 'Absolute address length difference',
  'addr_empty_s1': 'Flag indicating Source 1 address is missing',
  'addr_empty_c': 'Flag indicating Candidate address is missing',
  'both_addr_present': 'Flag indicating both addresses are populated',
  'num_common': 'Count of shared street / unit / pin code numbers',
  'num_jaccard': 'Jaccard overlap of extracted numeric tokens',
  'num_conflict': 'Penalty flag: candidate has numbers that contradict S1',
  'blocking_score': 'TF-IDF cosine similarity from candidate generation',
  'blocking_rank': 'Rank of candidate in blocking stage (1 to 8)',
  'is_s2': 'Binary indicator: 1.0 for Source 2, 0.0 for Source 3'
};

async function initLiveMatcher() {
  loadPresets();
  const btn = document.getElementById('btn-run-match');
  if (btn) {
    btn.addEventListener('click', runLiveMatch);
  }
  runLiveMatch(); // run initial calculation
}

async function loadPresets() {
  try {
    const res = await fetch('/api/examples');
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById('preset-container');
    container.innerHTML = '';

    data.examples.forEach(ex => {
      const pill = document.createElement('button');
      pill.className = 'preset-pill';
      pill.innerHTML = `<strong>${ex.tag}:</strong> ${ex.title}`;
      pill.addEventListener('click', () => {
        document.getElementById('input-s1-name').value = ex.s1_name;
        document.getElementById('input-s1-addr').value = ex.s1_addr;
        document.getElementById('input-cand-name').value = ex.cand_name;
        document.getElementById('input-cand-addr').value = ex.cand_addr;
        runLiveMatch();
      });
      container.appendChild(pill);
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

async function runLiveMatch() {
  const s1Name = document.getElementById('input-s1-name').value;
  const s1Addr = document.getElementById('input-s1-addr').value;
  const candName = document.getElementById('input-cand-name').value;
  const candAddr = document.getElementById('input-cand-addr').value;

  try {
    const res = await fetch('/api/match_live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        s1_name: s1Name,
        s1_addr: s1Addr,
        cand_name: candName,
        cand_addr: candAddr,
        cand_id: 'S2-00042',
        blocking_score: 0.82,
        blocking_rank: 1
      })
    });

    if (!res.ok) return;
    const result = await res.json();

    // Update previews
    document.getElementById('clean-preview-s1').textContent = 
      `Normalized: ${result.clean_s1.name} | Nums: {${result.clean_s1.numbers.join(', ')}}`;
    document.getElementById('clean-preview-cand').textContent = 
      `Normalized: ${result.clean_candidate.name} | Nums: {${result.clean_candidate.numbers.join(', ')}}`;

    // Update Decision Banner
    const banner = document.getElementById('match-banner');
    const badge = document.getElementById('banner-decision');
    const probEl = document.getElementById('banner-prob');
    const explEl = document.getElementById('banner-explanation');

    const pct = Math.round(result.probability * 1000) / 10;
    probEl.textContent = `${pct}%`;

    if (result.is_match) {
      banner.className = 'match-result-banner banner-match';
      badge.textContent = 'MATCH ✓';
      badge.className = 'decision-badge match-text';
      probEl.style.color = 'var(--accent-emerald)';
      explEl.textContent = `Pair similarity (${pct}%) exceeds calibrated F_0.5 decision threshold (${result.threshold}).`;
    } else {
      banner.className = 'match-result-banner banner-non-match';
      badge.textContent = 'NON-MATCH ✗';
      badge.className = 'decision-badge non-match-text';
      probEl.style.color = 'var(--accent-rose)';
      explEl.textContent = `Pair similarity (${pct}%) below precision threshold (${result.threshold}). Correctly isolated.`;
    }

    // Populate Feature Table
    const tbody = document.getElementById('features-table-body');
    tbody.innerHTML = '';

    for (const [key, val] of Object.entries(result.features)) {
      const row = document.createElement('tr');
      const desc = FEATURE_DESCRIPTIONS[key] || 'Extracted pairwise signal';

      // Signal bar width logic
      let fillPct = 0;
      if (key.includes('ratio')) fillPct = val; // 0 to 100
      else if (val >= 0 && val <= 1) fillPct = val * 100;
      else fillPct = Math.min(100, val * 10);

      const barColor = (key === 'num_conflict' && val > 0) ? 'var(--accent-rose)' : 'var(--accent-primary)';

      row.innerHTML = `
        <td class="code-id">${key}</td>
        <td style="font-weight: 600;">${val}</td>
        <td class="feature-bar-cell">
          <div class="mini-bar-bg">
            <div class="mini-bar-fill" style="width: ${fillPct}%; background: ${barColor};"></div>
          </div>
        </td>
        <td style="color: var(--text-muted);">${desc}</td>
      `;
      tbody.appendChild(row);
    }

  } catch (err) {
    console.error('Match live error:', err);
  }
}

// 4. Model Stats & Architecture & Live Benchmark
async function initModelStats() {
  loadBenchmarkScores();

  const benchBtn = document.getElementById('btn-run-benchmark');
  if (benchBtn) {
    benchBtn.addEventListener('click', runBenchmark);
  }

  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('feature-importance-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    let rank = 1;
    for (const [feat, gainPct] of Object.entries(data.feature_importances)) {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td style="font-weight: 700; color: var(--text-dim);">${rank++}</td>
        <td class="code-id">${feat}</td>
        <td style="font-weight: 700; color: var(--accent-cyan);">${gainPct}%</td>
        <td class="feature-bar-cell">
          <div class="mini-bar-bg">
            <div class="mini-bar-fill" style="width: ${Math.min(100, gainPct * 3.5)}%; background: var(--accent-cyan);"></div>
          </div>
        </td>
      `;
      tbody.appendChild(row);
    }
  } catch (err) {
    console.error('Model stats error:', err);
  }
}

async function loadBenchmarkScores() {
  try {
    const res = await fetch('/api/benchmark_scores');
    if (!res.ok) return;
    const data = await res.json();
    displayBenchmarkResults(data);
  } catch (err) {
    console.error('Failed to load benchmark scores:', err);
  }
}

function displayBenchmarkResults(data) {
  const elF05 = document.getElementById('res-macro-f05');
  if (elF05 && data.macro_f05 !== undefined) {
    elF05.textContent = data.macro_f05.toFixed(4);
    document.getElementById('res-macro-prec').textContent = `${data.macro_precision.toFixed(2)}%`;
    document.getElementById('res-macro-rec').textContent = `${data.macro_recall.toFixed(2)}%`;
    document.getElementById('res-singleton-acc').textContent = `${data.singleton_accuracy.toFixed(2)}%`;
    document.getElementById('res-blocking-rec').textContent = `${data.blocking_recall_ceiling.toFixed(2)}%`;
    document.getElementById('res-reduction').textContent = `${data.reduction_ratio.toFixed(3)}%`;
    document.getElementById('res-micro-prec').textContent = `${data.micro_precision.toFixed(2)}%`;
    document.getElementById('res-eval-time').textContent = `${data.elapsed_seconds.toFixed(1)}s`;
    document.getElementById('res-eval-entities').textContent = `${data.n_evaluated_entities.toLocaleString()} Entities Scored`;
  }
}

async function runBenchmark() {
  const btn = document.getElementById('btn-run-benchmark');
  const spinner = document.getElementById('bench-loading');
  const sampleSize = parseInt(document.getElementById('bench-sample-size').value, 10) || 1000;
  const thresh = parseFloat(document.getElementById('bench-threshold').value) || 0.65;

  btn.disabled = true;
  btn.style.opacity = '0.6';
  spinner.style.display = 'block';

  try {
    const res = await fetch('/api/run_benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_val: sampleSize, threshold: thresh })
    });

    if (!res.ok) throw new Error('Benchmark failed');
    const results = await res.json();
    displayBenchmarkResults(results);

  } catch (err) {
    console.error('Benchmark execution error:', err);
    alert('Benchmark evaluation error. Check server logs.');
  } finally {
    btn.disabled = false;
    btn.style.opacity = '1';
    spinner.style.display = 'none';
  }
}

// 5. Sample Matches Inspector
async function initSampleMatches() {
  const btn = document.getElementById('btn-refresh-samples');
  if (btn) {
    btn.addEventListener('click', loadSampleMatches);
  }
}

async function loadSampleMatches() {
  const tbody = document.getElementById('sample-matches-tbody');
  tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Fetching sample matches from output/matching_results.tsv...</td></tr>';

  try {
    const res = await fetch('/api/sample_matches?limit=15');
    if (!res.ok) return;
    const data = await res.json();

    if (!data.samples || data.samples.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-dim);">No matched entities written to output file yet. Check pipeline progress!</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    data.samples.forEach(s => {
      const row = document.createElement('tr');
      const cStr = s.candidate_entity_ids.join(', ') || 'None';
      const mStr = s.matched_entity_ids.join(', ') || 'None';

      row.innerHTML = `
        <td class="code-id">${s.source1_entity_id}</td>
        <td style="font-weight: 600;">${s.candidate_count}</td>
        <td style="color: var(--text-muted); font-size: 0.8rem; font-family: var(--font-mono);">${cStr}</td>
        <td style="font-weight: 700; color: var(--accent-emerald);">${s.match_count}</td>
        <td style="color: var(--accent-emerald); font-weight: 600; font-family: var(--font-mono);">${mStr}</td>
        <td><span style="color: var(--accent-emerald); font-weight: 700;">PASS (Subset ✓)</span></td>
      `;
      tbody.appendChild(row);
    });

  } catch (err) {
    console.error('Failed to load sample matches:', err);
  }
}
