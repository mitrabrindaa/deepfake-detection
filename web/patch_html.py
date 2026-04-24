"""Patch index.html: replace static scan cards with dynamic ones + add modal."""
from pathlib import Path

html_path = Path(__file__).parent / "templates" / "index.html"
content = html_path.read_text(encoding="utf-8")

# ── 1. Replace static scan-cards block with empty dynamic grid ──────────────
start_marker = '<div class="scan-cards">'
end_marker_after = "</div>\n      </div><!-- /container -->"

start_idx = content.find(start_marker)
# find the closing </div> of scan-cards
depth = 0
i = start_idx
while i < len(content):
    if content[i:i+4] == "<div":
        depth += 1
    elif content[i:i+6] == "</div>":
        depth -= 1
        if depth == 0:
            end_idx = i + 6
            break
    i += 1

new_grid = '<div class="scan-cards" id="scan-cards"></div>'
content = content[:start_idx] + new_grid + content[end_idx:]

# ── 2. Add modal HTML before </body> ────────────────────────────────────────
modal_html = """
  <!-- Report Modal -->
  <div id="report-modal" style="display:none;position:fixed;inset:0;z-index:100;align-items:center;justify-content:center;background:rgba(0,0,0,0.7);backdrop-filter:blur(6px);">
    <div style="background:#0d1120;border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:2rem;width:90%;max-width:380px;position:relative;">
      <button onclick="closeModal()" style="position:absolute;top:1rem;right:1rem;background:none;border:none;color:#94a3b8;font-size:1.3rem;cursor:pointer;">&#x2715;</button>
      <div id="modal-thumb-wrap" style="text-align:center;margin-bottom:1.25rem;">
        <img id="modal-thumb" src="" alt="" style="max-height:180px;border-radius:10px;object-fit:cover;border:1px solid rgba(255,255,255,0.1);" />
      </div>
      <div id="modal-label" style="font-size:1.3rem;font-weight:700;text-align:center;padding:0.5rem 1rem;border-radius:8px;margin-bottom:1.25rem;"></div>
      <div style="display:flex;flex-direction:column;gap:0.6rem;">
        <div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#94a3b8;margin-bottom:4px;">
            <span>Fake</span><span id="modal-fake-pct"></span>
          </div>
          <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:9px;overflow:hidden;">
            <div id="modal-fake-bar" style="height:100%;border-radius:99px;background:linear-gradient(90deg,#ef4444,#f87171);transition:width 0.7s ease;width:0%"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#94a3b8;margin-bottom:4px;">
            <span>Real</span><span id="modal-real-pct"></span>
          </div>
          <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:9px;overflow:hidden;">
            <div id="modal-real-bar" style="height:100%;border-radius:99px;background:linear-gradient(90deg,#22c55e,#4ade80);transition:width 0.7s ease;width:0%"></div>
          </div>
        </div>
      </div>
      <p style="font-size:0.75rem;color:#475569;text-align:center;margin-top:1rem;" id="modal-time"></p>
    </div>
  </div>
"""
content = content.replace("</body>", modal_html + "</body>")

# ── 3. Replace the JS block with updated version ────────────────────────────
old_script_start = "<script>"
old_script_end = "</script>"
s = content.rfind(old_script_start)
e = content.rfind(old_script_end) + len(old_script_end)

new_script = """<script>
    const dropWrapper = document.getElementById('drop-wrapper');
    const dropZone    = document.getElementById('drop-zone');
    const fileInput   = document.getElementById('file-input');
    const previewWrap = document.getElementById('preview-wrap');
    const preview     = document.getElementById('preview');
    const analyzeBtn  = document.getElementById('analyze-btn');
    const spinner     = document.getElementById('spinner');
    const errorMsg    = document.getElementById('error-msg');
    const result      = document.getElementById('result');
    const resultLabel = document.getElementById('result-label');
    const fakeBar     = document.getElementById('fake-bar');
    const realBar     = document.getElementById('real-bar');
    const fakePct     = document.getElementById('fake-pct');
    const realPct     = document.getElementById('real-pct');
    const resetBtn    = document.getElementById('reset-btn');
    const scanCards   = document.getElementById('scan-cards');

    let selectedFile = null;
    const scans = []; // { imgSrc, label, prob_fake, prob_real, time }

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropWrapper.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropWrapper.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropWrapper.classList.remove('dragover');
      const f = e.dataTransfer.files[0]; if (f) handleFile(f);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

    function handleFile(file) {
      selectedFile = file;
      preview.src = URL.createObjectURL(file);
      previewWrap.style.display = 'block';
      analyzeBtn.style.display = 'block';
      result.style.display = 'none';
      resetBtn.style.display = 'none';
      errorMsg.style.display = 'none';
      dropWrapper.style.display = 'none';
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      analyzeBtn.disabled = true;
      spinner.style.display = 'block';
      errorMsg.style.display = 'none';
      result.style.display = 'none';

      const form = new FormData();
      form.append('image', selectedFile);

      try {
        const res  = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        if (data.detail) throw new Error(data.detail);

        resultLabel.textContent = data.label.toUpperCase();
        resultLabel.className   = 'result-label ' + data.label;
        fakeBar.style.width = data.prob_fake + '%';
        realBar.style.width = data.prob_real + '%';
        fakePct.textContent = data.prob_fake + '%';
        realPct.textContent = data.prob_real + '%';
        result.style.display   = 'block';
        resetBtn.style.display = 'block';
        analyzeBtn.style.display = 'none';

        // add to recent scans
        const imgSrc = preview.src;
        const scan = { imgSrc, label: data.label, prob_fake: data.prob_fake, prob_real: data.prob_real, time: new Date() };
        scans.unshift(scan);
        renderScans();
      } catch (err) {
        errorMsg.textContent   = err.message || 'Something went wrong.';
        errorMsg.style.display = 'block';
      } finally {
        spinner.style.display  = 'none';
        analyzeBtn.disabled    = false;
      }
    });

    resetBtn.addEventListener('click', () => {
      selectedFile = null;
      fileInput.value = '';
      preview.src = '';
      previewWrap.style.display  = 'none';
      analyzeBtn.style.display   = 'none';
      result.style.display       = 'none';
      resetBtn.style.display     = 'none';
      errorMsg.style.display     = 'none';
      dropWrapper.style.display  = 'block';
    });

    function timeAgo(date) {
      const s = Math.floor((Date.now() - date) / 1000);
      if (s < 60) return 'Just now';
      if (s < 3600) return Math.floor(s/60) + ' min ago';
      return Math.floor(s/3600) + ' hr ago';
    }

    function renderScans() {
      scanCards.innerHTML = '';
      scans.slice(0, 8).forEach((scan, idx) => {
        const isReal = scan.label === 'real';
        const isUncertain = scan.label === 'uncertain';
        const badgeClass = isReal ? 'real' : isUncertain ? 'uncertain' : 'fake';
        const badgeIcon  = isReal ? '&#10003;' : isUncertain ? '?' : '&#9888;';
        const badgeText  = isReal ? 'Real' : isUncertain ? 'Uncertain' : 'AI';
        const card = document.createElement('div');
        card.className = 'scan-card';
        card.innerHTML = `
          <div class="scan-thumb">
            <img src="${scan.imgSrc}" alt="Scan ${idx+1}" />
            <span class="scan-badge ${badgeClass}"><span class="badge-dot">${badgeIcon}</span> ${badgeText}</span>
          </div>
          <div class="scan-meta">
            <p class="scan-time">${timeAgo(scan.time)}</p>
            <a href="#" class="scan-link" data-idx="${idx}">View Report</a>
          </div>`;
        card.querySelector('.scan-link').addEventListener('click', e => {
          e.preventDefault(); openModal(idx);
        });
        scanCards.appendChild(card);
      });
    }

    function openModal(idx) {
      const scan = scans[idx];
      const modal = document.getElementById('report-modal');
      document.getElementById('modal-thumb').src = scan.imgSrc;
      const lbl = document.getElementById('modal-label');
      lbl.textContent = scan.label.toUpperCase();
      lbl.style.cssText = scan.label === 'real'
        ? 'background:rgba(74,222,128,0.12);color:#4ade80;border:1px solid rgba(74,222,128,0.25);font-size:1.3rem;font-weight:700;text-align:center;padding:0.5rem 1rem;border-radius:8px;margin-bottom:1.25rem;'
        : scan.label === 'uncertain'
        ? 'background:#2a2510;color:#facc15;font-size:1.3rem;font-weight:700;text-align:center;padding:0.5rem 1rem;border-radius:8px;margin-bottom:1.25rem;'
        : 'background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);font-size:1.3rem;font-weight:700;text-align:center;padding:0.5rem 1rem;border-radius:8px;margin-bottom:1.25rem;';
      document.getElementById('modal-fake-pct').textContent = scan.prob_fake + '%';
      document.getElementById('modal-real-pct').textContent = scan.prob_real + '%';
      document.getElementById('modal-time').textContent = scan.time.toLocaleTimeString();
      modal.style.display = 'flex';
      // animate bars after paint
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          document.getElementById('modal-fake-bar').style.width = scan.prob_fake + '%';
          document.getElementById('modal-real-bar').style.width = scan.prob_real + '%';
        });
      });
    }

    function closeModal() {
      const modal = document.getElementById('report-modal');
      modal.style.display = 'none';
      document.getElementById('modal-fake-bar').style.width = '0%';
      document.getElementById('modal-real-bar').style.width = '0%';
    }

    document.getElementById('report-modal').addEventListener('click', function(e) {
      if (e.target === this) closeModal();
    });
  </script>"""

content = content[:s] + new_script + content[e:]
html_path.write_text(content, encoding="utf-8")
print("Done. Size:", html_path.stat().st_size)
