function copyBibtex() {
  const el = document.getElementById('bibtex');
  const text = el.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copybtn');
    const old = btn.innerText;
    btn.innerText = 'Copied';
    setTimeout(()=>btn.innerText=old, 1200);
  }).catch(()=> alert('Copy failed. Please copy manually.'));
}
document.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  const href = a.getAttribute('href');
  if (!href || !href.startsWith('#')) return;
  const target = document.querySelector(href);
  if (!target) return;
  e.preventDefault();
  target.scrollIntoView({behavior:'smooth', block:'start'});
  history.replaceState(null, '', href);
});
