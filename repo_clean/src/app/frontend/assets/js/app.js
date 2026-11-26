/* MT2KD Audio Intelligence — page logic (split from the single-file index.html).
   Demo mode works without the backend; Live mode talks to the FastAPI backend (same origin). */
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const css = getComputedStyle(document.documentElement);
  const C = { sage: css.getPropertyValue('--sage').trim(), line: '#D9DCD1', ink: '#141613', faint: '#C3C7BB' };

  // ---------- waveform helper ----------
  function mixc(a,b,k){ const p = s => [1,3,5].map(i=>parseInt(s.slice(i,i+2),16)); const A=p(a), B=p(b); return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*k)).join(',')})`; }
  function rng(seed){ return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }
  function envelope(n, seed){
    const r = rng(seed), bursts = []; let x = 0.03;
    while (x < 0.97){ const w = 0.025 + r()*0.06; bursts.push([x + w/2, w, 0.35 + r()*0.65]); x += w*1.3 + r()*0.05; }
    const out = [];
    for (let i=0;i<n;i++){
      const t = i/n; let v = 0.04;
      for (const [c,w,a] of bursts) v += a * Math.exp(-((t-c)**2) / (2*(w/3)**2));
      out.push(Math.min(1, v * (0.7 + 0.3*r())));
    }
    return out;
  }
  const waves = [];
  class Wave{
    constructor(canvas, o){ this.c = canvas; this.ctx = canvas.getContext('2d'); this.o = Object.assign({bars:120, seed:7, gap:2, progress:null, playhead:true, amp:0.92, drift:0}, o); this.env = envelope(this.o.drift ? this.o.bars*3 : this.o.bars, this.o.seed); this.p = 0; this.visible = true; this.resize(); waves.push(this);
      new ResizeObserver(()=>this.resize()).observe(canvas);
      new IntersectionObserver(([e])=>{ this.visible = e.isIntersecting; }).observe(canvas); }
    reseed(s){ this.env = envelope(this.o.bars, s); }
    resize(){ const d = Math.min(2, devicePixelRatio||1); const r = this.c.getBoundingClientRect(); this.w = r.width; this.h = r.height; this.c.width = Math.max(1, r.width*d); this.c.height = Math.max(1, r.height*d); this.ctx.setTransform(d,0,0,d,0,0); this.draw(0); }
    draw(t){
      const {ctx,w,h,env} = this, L = env.length, n = this.o.bars, bw = w/n, mid = h/2, D = this.o.drift;
      ctx.clearRect(0,0,w,h);
      const shift = D && !reduce ? t/D : 0, base = Math.floor(shift), frac = shift - base;
      for (let i=0;i<=n;i++){
        if (!D && i===n) break;
        const v = env[(base + i) % L];
        const live = reduce ? 1 : 0.96 + 0.04*Math.sin(t/2600 + i*0.12);
        const bh = Math.max(2, v*h*this.o.amp*live);
        const x = (i - frac)*bw + this.o.gap/2, ww = Math.max(1, bw - this.o.gap);
        if (D){ // soft fade at both edges, one even colour
          const e = Math.max(0, Math.min(1, (x/w)/0.14, (1 - (x+ww)/w)/0.14)); if (!e) continue;
          ctx.globalAlpha = e; ctx.fillStyle = this.o.color || C.sage;
        } else if (this.o.playhead){ // played bars ease in over a few bars instead of snapping
          const k = Math.max(0, Math.min(1, (this.p*n - i)/4)); ctx.fillStyle = mixc(this.o.color || '#C9CEC0', '#6E9E4C', k);
        } else ctx.fillStyle = this.o.color || C.faint;
        ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, mid - bh/2, ww, bh, ww/2) : ctx.rect(x, mid-bh/2, ww, bh); ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (this.o.playhead && !D){ const px = this.p*w; ctx.fillStyle = 'rgba(20,22,19,.28)'; ctx.fillRect(px, 6, 1, h-12); }
    }
  }

  // ---------- hero ----------
  const hero = new Wave(document.getElementById('heroWave'), {bars:96, seed:11, gap:3, amp:.95, playhead:false, drift:420, color:'#7FAE5C'});
  const cta = new Wave(document.getElementById('ctaWave'), {bars:70, seed:5, gap:3, playhead:false, amp:.8, drift:520, color:'#B9CFA4'});

  // ---------- features ----------
  const fw = new Wave(document.getElementById('featWave'), {bars:110, seed:23, gap:2});
  const svA = new Wave(document.getElementById('svA'), {bars:44, seed:3, gap:2, playhead:false, color:'#B8CDA3'});
  const svB = new Wave(document.getElementById('svB'), {bars:44, seed:9, gap:2, playhead:false, color:'#B8CDA3'});

  const words = "the last train left the platform at nine and the station slowly went quiet".split(' ');
  const tx = document.getElementById('tx');
  tx.innerHTML = words.map(w=>`<span class="w">${w}</span>`).join(' ') + '<span class="caret"></span>';
  const wEls = [...tx.querySelectorAll('.w')], wc = document.getElementById('wc'), tNow = document.getElementById('tNow');
  const DUR = 8;

  const tagData = [["Speech",.93],["Train",.84],["Rail transport",.77],["Vehicle",.61],["Crowd",.34],["Wind",.12]];
  const tagsEl = document.getElementById('tags');
  tagsEl.innerHTML = tagData.map(([n])=>`<div class="trow"><span>${n}</span><span class="bar"><i></i></span><span class="v">0.00</span></div>`).join('');
  const tRows = [...tagsEl.children];
  function fillTags(on){
    tRows.forEach((row,i)=>{ const v = on ? tagData[i][1] : 0;
      setTimeout(()=>{ row.querySelector('i').style.width = (v*100)+'%'; row.classList.toggle('low', v < .3 && on); }, on ? i*140 : 0);
      countTo(row.querySelector('.v'), v, on ? 900 + i*140 : 0); });
  }
  function countTo(el, target, ms){
    const start = parseFloat(el.textContent)||0, t0 = performance.now();
    if (reduce || !ms){ el.textContent = target.toFixed(2); return; }
    (function step(now){ const k = Math.min(1,(now-t0)/ms), e = 1-Math.pow(1-k,3); el.textContent = (start + (target-start)*e).toFixed(2); if (k<1) requestAnimationFrame(step); })(t0);
  }

  const pairs = [
    {mode:'Mode / Reference + test', pair:'Pair A', ref:'Reference', sim:.78, a:3, b:4, chip:'m1'},
    {mode:'Mode / Registered speaker', pair:'Pair B', ref:'Registered · SPK-014', sim:.21, a:17, b:29, chip:'m2'}
  ];
  let pi = 0, svTimer = null;
  const simV = document.getElementById('simV'), simBar = document.getElementById('simBar'), verdict = document.getElementById('verdict');
  function showPair(){
    const p = pairs[pi];
    document.getElementById('svMode').textContent = p.mode;
    document.getElementById('svPair').textContent = p.pair;
    document.getElementById('refLbl').textContent = p.ref;
    svA.reseed(p.a); svB.reseed(p.b);
    document.getElementById('m1').classList.toggle('on', p.chip==='m1');
    document.getElementById('m2').classList.toggle('on', p.chip==='m2');
    simBar.style.width = '0%'; simV.textContent = '0.00'; verdict.style.opacity = 0;
    setTimeout(()=>{
      simBar.style.width = (p.sim*100)+'%'; simBar.style.background = p.sim >= .5 ? C.sage : '#C98A6B';
      countTo(simV, p.sim, 1100);
      setTimeout(()=>{ const yes = p.sim >= .5; verdict.className = 'verdict ' + (yes?'yes':'no'); verdict.querySelector('span').textContent = yes ? 'Same speaker' : 'Different speaker'; verdict.style.opacity = 1; }, 1150);
    }, 250);
  }

  let step = -1;
  const outs = [...document.querySelectorAll('.out')], heads = [...document.querySelectorAll('#fheads span')];
  const steps = [...document.querySelectorAll('.step')];
  function setStep(i){
    if (i === step) return; step = i;
    steps.forEach((s,k)=>s.classList.toggle('on', k===i));
    outs.forEach((o,k)=>o.classList.toggle('on', k===i));
    heads.forEach((h,k)=>h.classList.toggle('on', k===i));
    fillTags(i===1);
    clearInterval(svTimer);
    if (i===2){ pi = 0; showPair(); svTimer = setInterval(()=>{ pi = (pi+1)%pairs.length; showPair(); }, 5200); }
    setLabel(steps[i].dataset.label);
  }

  // ---------- section counter ----------
  const cnt = document.getElementById('cnt'); let lastLabel = cnt ? cnt.textContent : '';
  function setLabel(l){
    if (!cnt || !l || l === lastLabel) return; lastLabel = l;
    if (reduce){ cnt.textContent = l; return; }
    cnt.style.transform = 'translateY(-110%)'; cnt.style.opacity = 0;
    setTimeout(()=>{ cnt.style.transition = 'none'; cnt.textContent = l; cnt.style.transform = 'translateY(110%)';
      requestAnimationFrame(()=>{ requestAnimationFrame(()=>{ cnt.style.transition = ''; cnt.style.transform = ''; cnt.style.opacity = 1; }); }); }, 220);
  }
  const mid = {rootMargin:'-48% 0px -48% 0px'};
  const secIO = new IntersectionObserver(es=>es.forEach(e=>{ if (e.isIntersecting && !e.target.classList.contains('feat')) setLabel(e.target.dataset.label); }), mid);
  document.querySelectorAll('section[data-label]').forEach(s=>secIO.observe(s));
  const stepIO = new IntersectionObserver(es=>es.forEach(e=>{ if (e.isIntersecting) setStep(+e.target.dataset.step); }), mid);
  steps.forEach(s=>stepIO.observe(s));
  setStep(0); setLabel('01 / Overview');

  // ---------- architecture stages ----------
  const arch = document.getElementById('arch'), b1 = document.getElementById('st1'), b2 = document.getElementById('st2');
  let manual = false;
  function stage(n){ arch.dataset.stage = n; b1.setAttribute('aria-pressed', n==1); b2.setAttribute('aria-pressed', n==2);
    document.getElementById('n1').classList.toggle('on', n==1); document.getElementById('n2').classList.toggle('on', n==2); }
  b1.onclick = ()=>{ manual = true; stage(1); }; b2.onclick = ()=>{ manual = true; stage(2); };
  setInterval(()=>{ if (!manual && !reduce) stage(arch.dataset.stage === '1' ? 2 : 1); }, 4800);

  // ---------- reveals (only for things starting below the fold) ----------
  const rIO = new IntersectionObserver(es=>es.forEach(e=>{ if (e.isIntersecting){ e.target.classList.remove('pre'); rIO.unobserve(e.target); } }), {rootMargin:'0px 0px -8% 0px'});
  if (!reduce) document.querySelectorAll('.reveal').forEach(el=>{ if (el.getBoundingClientRect().top > innerHeight){ el.classList.add('pre'); rIO.observe(el); } });


  // ================= TEACHERS =================
  const mix = (a, b, k) => { const p = s => [1,3,5].map(i=>parseInt(s.slice(i,i+2),16)); const A=p(a), B=p(b); return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*k)).join(',')})`; };
  const PALE = '#E6E9DF', DEEP = '#4E7A36';
  function rr(ctx,x,y,w,h,r){ ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x,y,w,h,Math.min(r,w/2,h/2)); else ctx.rect(x,y,w,h); ctx.fill(); }
  const hash = (n) => { const x = Math.sin(n*127.1)*43758.5453; return x - Math.floor(x); };

  const drawers = {
    wave(ctx,w,h,t,hot,o){
      const n = 46, bw = w/n, env = o.env || (o.env = envelope(200, +o.seed||7)), sh = Math.floor(t/240);
      for (let i=0;i<n;i++){ const v = env[(i+sh)%env.length]*(0.95+0.05*Math.sin(t/1100+i*.3)); const bh = Math.max(2, v*h*.9);
        ctx.fillStyle = hot ? mix('#B9D39E', DEEP, v) : '#CDD2C4'; rr(ctx, i*bw+1, (h-bh)/2, Math.max(1,bw-2), bh, 2); }
    },
    mel(ctx,w,h,t,hot,o){
      const rows = +o.rows||16, cols = 36, cw = w/cols, ch = h/rows, sh = t/(hot?90:180);
      for (let c=0;c<cols;c++) for (let r=0;r<rows;r++){
        const x = c + sh, low = 1 - r/rows;
        let v = 0.25*low + 0.45*Math.max(0,Math.sin(x*0.35 + r*0.15))*low + 0.3*Math.max(0, Math.sin(x*0.9 - r*0.6)*Math.cos(x*0.21));
        v = Math.min(1, v*(0.75+0.5*hash(Math.floor(x)*31+r)));
        ctx.fillStyle = mix(PALE, hot?DEEP:'#9AA88E', v); ctx.fillRect(c*cw, (rows-1-r)*ch, cw-1, ch-1);
      }
    },
    layers(ctx,w,h,t,hot,o){
      const n = +o.n||12, gap = n>20?2:4, bw = (w - gap*(n-1))/n, s = ((t/(hot?1100:2200))%1)*(n+8)-4;
      for (let i=0;i<n;i++){ const k = Math.exp(-((i-s)**2)/(n>20?6:2)); const bh = h*(0.55+0.35*k);
        ctx.fillStyle = mix('#D9DDD0', DEEP, hot?k:k*.35); rr(ctx, i*(bw+gap), (h-bh)/2, bw, bh, Math.min(3,bw/2)); }
    },
    patches(ctx,w,h,t,hot,o){
      const cols = 14, rows = 4, cw = w/cols, ch = h/rows, cyc = Math.floor(t/2600), ph = (t%2600)/2600;
      for (let r=0;r<rows;r++) for (let c=0;c<cols;c++){
        const id = r*cols+c, masked = hash(id*7.3+cyc*13.1) < .75, val = hash(id*3.7+1.3)*0.8 + (1-r/rows)*.2;
        const x = c*cw+1, y = r*ch+1, ww = cw-2, hh = ch-2;
        if (!masked){ ctx.fillStyle = mix(PALE, hot?DEEP:'#9AA88E', val); rr(ctx,x,y,ww,hh,2); }
        else { const fill = hot ? Math.max(0, Math.min(1,(ph*1.6 - hash(id)*.8))) : 0;
          ctx.fillStyle = '#F1F2EC'; rr(ctx,x,y,ww,hh,2); ctx.strokeStyle = '#D6DACE'; ctx.lineWidth = 1; ctx.strokeRect(x+.5,y+.5,ww-1,hh-1);
          if (fill>0){ ctx.globalAlpha = fill; ctx.fillStyle = mix(PALE, '#7FAE5C', val); rr(ctx,x,y,ww,hh,2); ctx.globalAlpha = 1; } }
      }
    },
    codes(ctx,w,h,t,hot,o){
      const n = w < 200 ? 3 : w < 260 ? 4 : 5, gap = 6, cw = (w-gap*(n-1))/n, step = Math.floor(t/(hot?420:900)), off = ((t%(hot?420:900))/(hot?420:900));
      ctx.font = '500 11px "DM Mono", ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let i=0;i<n;i++){ const code = Math.floor(hash(step*5+i)*1024); const k = i===n-1 ? Math.min(1, off*3) : 1;
        ctx.globalAlpha = k; ctx.fillStyle = hot ? (i===n-1 ? '#CFE3B9' : '#E2EED5') : '#ECEEE6'; rr(ctx, i*(cw+gap), h*.22, cw, h*.56, 7);
        ctx.fillStyle = hot ? '#3F6527' : '#8A8F83'; ctx.fillText('#'+String(code).padStart(4,'0'), i*(cw+gap)+cw/2, h/2+1); ctx.globalAlpha = 1; }
    },
    tags(ctx,w,h,t,hot,o){
      const L = [['Speech',.86],['Vehicle',.58],['Music',.21]], rh = h/3, lw = 62;
      ctx.font = '400 11px "DM Mono", ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      L.forEach(([n,v],i)=>{ const live = v*(hot ? 0.92+0.08*Math.sin(t/300+i*2) : 0.6); const y = i*rh + rh/2;
        ctx.fillStyle = '#6C7067'; ctx.fillText(n, 0, y); ctx.fillStyle = '#ECEEE6'; rr(ctx, lw, y-3, w-lw, 6, 3);
        ctx.fillStyle = hot ? '#7FAE5C' : '#C8CEBE'; rr(ctx, lw, y-3, (w-lw)*live, 6, 3); });
    },
    res2(ctx,w,h,t,hot,o){
      const n = 3, gap = 10, bw = (w-gap*(n-1))/n, act = Math.floor(t/(hot?650:1300))%n;
      ctx.font = '500 10px "DM Mono", ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      for (let b=0;b<n;b++){ const on = b===act && hot, x = b*(bw+gap);
        ctx.fillStyle = on ? '#E2EED5' : '#F1F2EC'; rr(ctx, x, 0, bw, h, 9);
        for (let s=0;s<8;s++){ const k = on ? Math.max(0, Math.sin(t/120 - s*.7)) : .15; const sw = (bw-20)/8;
          ctx.fillStyle = mix('#D9DDD0', DEEP, k); rr(ctx, x+10+s*sw+1, 8, sw-2, h-28, 2); }
        ctx.fillStyle = on ? '#3F6527' : '#8A8F83'; ctx.fillText(o.labels ? o.labels.split(',')[b] : 'd=' + (b+2), x+bw/2, h-7); }
    },
    mix(ctx,w,h,t,hot,o){
      const n = 60, sh = t/(hot?700:1400), mid = h/2;
      const line = (col, f, ph, amp) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
        for (let i=0;i<=n;i++){ const x = i/n*w, e = 0.35 + 0.65*Math.abs(Math.sin(i*0.21 + ph + sh*0.6)); const y = mid + Math.sin(i*f + sh*2 + ph)*e*amp*h*0.42; i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); } ctx.stroke(); };
      line(hot ? '#B9D39E' : '#D6DACE', 0.55, 0, .8); line(hot ? '#C9B98A' : '#DEDBD0', 0.95, 2, .6); line(hot ? DEEP : '#9AA88E', 0.75, 1, 1);
    },
    pool(ctx,w,h,t,hot,o){
      const n = 40, bw = w/n, c = (Math.sin(t/(hot?900:1800))*.35+.5)*n, c2 = (Math.cos(t/1300)*.3+.35)*n;
      for (let i=0;i<n;i++){ const a = 0.15 + .85*Math.max(Math.exp(-((i-c)**2)/18), .7*Math.exp(-((i-c2)**2)/10)) * (0.85+.15*hash(i));
        const bh = Math.max(2, a*h*.92); ctx.fillStyle = mix('#DCE0D3', hot?DEEP:'#9AA88E', a); rr(ctx, i*bw+1, h-bh, Math.max(1,bw-2), bh, 2); }
    },
    embed(ctx,w,h,t,hot,o){
      const cols = 48, rows = 4, cw = w/cols, ch = h/rows, sp = hot ? 1 : .35;
      for (let i=0;i<cols*rows;i++){ const v = .5 + .5*Math.sin(hash(i)*40 + t/600*sp*(1+(i%5)*.2));
        ctx.fillStyle = mix(PALE, hot?DEEP:'#9AA88E', v); ctx.fillRect((i%cols)*cw, Math.floor(i/cols)*ch, cw-1, ch-1); }
    }
  };

  const widgets = [];
  document.querySelectorAll('.st-w canvas[data-kind]').forEach(cv=>{
    const o = Object.assign({}, cv.dataset), st = cv.closest('.st'), ctx = cv.getContext('2d');
    const wd = {cv, ctx, o, st, vis:false, w:0, h:0};
    const fit = ()=>{ const d = Math.min(2, devicePixelRatio||1), r = cv.getBoundingClientRect(); wd.w = r.width; wd.h = r.height; cv.width = Math.max(1,r.width*d); cv.height = Math.max(1,r.height*d); ctx.setTransform(d,0,0,d,0,0); };
    fit(); new ResizeObserver(fit).observe(cv);
    new IntersectionObserver(([e])=>{ wd.vis = e.isIntersecting; }).observe(cv);
    widgets.push(wd);
  });
  function drawWidgets(t){
    for (const wd of widgets){ if (!wd.vis || !wd.w) continue; if (tScroll.classList.contains('js-pin') && !wd.st.closest('.model').classList.contains('on')) continue;
      wd.ctx.clearRect(0,0,wd.w,wd.h); drawers[wd.o.kind](wd.ctx, wd.w, wd.h, t, wd.st.classList.contains('hot') || reduce, wd.o); }
  }

  // stage cycling per model + decoder tokens
  const models = [...document.querySelectorAll('.model')];
  models.forEach(m=>{
    const sts = [...m.querySelectorAll('.st')], cons = [...m.querySelectorAll('.conn')];
    let k = 0; sts[0].classList.add('hot');
    if (reduce){ sts.forEach(s=>s.classList.add('hot')); return; }
    setInterval(()=>{
      sts[k].classList.remove('hot');
      if (k < cons.length){ const c = cons[k]; c.classList.remove('go'); void c.offsetWidth; c.classList.add('go'); }
      k = (k+1) % sts.length;
      setTimeout(()=>sts[k].classList.add('hot'), k===0 ? 0 : 380);
    }, 1500);
  });
  document.querySelectorAll('.toks').forEach(el=>{
    const toks = el.dataset.toks.split(','); let i = 0;
    const add = ()=>{ if (i >= toks.length){ el.innerHTML = ''; i = 0; }
      const s = document.createElement('span'); s.textContent = toks[i]; if (toks[i].startsWith('<|')) s.className = 'sp'; el.appendChild(s); i++;
      while (el.scrollWidth > el.clientWidth + 1 && el.children.length > 1) el.removeChild(el.firstChild); };
    if (reduce){ toks.slice(2,7).forEach(tk=>{ const s=document.createElement('span'); s.textContent=tk; el.appendChild(s); }); return; }
    add(); setInterval(add, 520);
  });

  // pinned scroll scene
  const tScroll = document.getElementById('tScroll');
  const railBtns = [...document.querySelectorAll('.t-rail button')], trk = [...document.querySelectorAll('.t-rail .trk i')];
  const pinMQ = matchMedia('(min-width: 961px) and (min-height: 620px)');
  let mIdx = -1;
  function setModel(i){ if (i === mIdx) return; mIdx = i; models.forEach((m,k)=>m.classList.toggle('on', k===i)); railBtns.forEach((b,k)=>{ b.classList.toggle('on', k===i); b.setAttribute('aria-selected', k===i); }); }
  function tScrollTick(){
    const r = tScroll.getBoundingClientRect(), mid = innerHeight/2, inView = r.top < mid && r.bottom > mid;
    if (!tScroll.classList.contains('js-pin')){ if (inView) setLabel('03 / Teachers'); return; }
    const span = r.height - innerHeight, p = Math.min(1, Math.max(0, -r.top/span));
    trk.forEach((el,k)=>el.style.width = (Math.min(1, Math.max(0, p*3-k))*100)+'%');
    setModel(Math.min(2, Math.floor(p*3)));
    if (inView) setLabel(models[mIdx].dataset.label);
  }
  function pinMode(){
    const on = pinMQ.matches && !reduce; tScroll.classList.toggle('js-pin', on); mIdx = -1;
    if (on){ models.forEach(m=>m.classList.remove('pre')); tScrollTick(); } else models.forEach(m=>m.classList.add('on'));
  }
  addEventListener('scroll', tScrollTick, {passive:true});
  pinMQ.addEventListener ? pinMQ.addEventListener('change', pinMode) : pinMQ.addListener(pinMode);
  railBtns.forEach((b,k)=>b.addEventListener('click', ()=>{ const r = tScroll.getBoundingClientRect(), span = r.height - innerHeight; scrollTo({top: scrollY + r.top + span*(k+.5)/3, behavior: reduce?'auto':'smooth'}); }));

  // ================= UPLOAD: Demo mode (simulated, labelled) + Live mode (FastAPI backend) =================
  const fileBox = document.getElementById('fileBox'), fileIn = document.getElementById('fileIn'), dz = document.getElementById('dz'), errEl = document.getElementById('err');
  const fWave = document.getElementById('fWave'), runBtn = document.getElementById('run'), statusEl = document.getElementById('status'), doneEl = document.getElementById('done');
  const resultEl = document.getElementById('result'), sampleBtn = document.getElementById('useSample'), modeNote = document.getElementById('modeNote');
  const modeBtns = [...document.querySelectorAll('#runMode button')], spkSel = document.getElementById('spk'), spkHint = document.getElementById('spkHint');
  const CFG = window.MT2KD_CONFIG || {};
  const API = location.protocol.startsWith('http') ? (CFG.apiBase || '') : null;            // the backend serves this page -> same origin
  let task = 0, hasFile = false, peaks = null, running = false, live = false, liveOK = false, liveMsg = '';
  let audioId = null, refAudioId = null, speakers = [], lastText = '', fileName = '';
  const fmt = s => { s = Math.round(s); return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0'); };
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const svRegistered = () => document.querySelector('#svModeSw button:last-child').getAttribute('aria-pressed') === 'true';
  const topK = () => +((document.querySelector('#topK [aria-pressed="true"]') || {}).textContent || 5);

  async function api(path, opts){
    const r = await fetch(API + path, opts); let j = null;
    try { j = await r.json(); } catch(e){}
    if (!r.ok) throw new Error(j && j.detail ? (typeof j.detail === 'string' ? j.detail : 'The request was not accepted: ' + JSON.stringify(j.detail)) : `The server answered with error ${r.status}.`);
    return j;
  }
  async function upload(f){ const fd = new FormData(); fd.append('file', f); return api('/api/audio', {method:'POST', body: fd}); }

  function drawPeaks(){
    if (!peaks) return; const d = Math.min(2, devicePixelRatio||1), r = fWave.getBoundingClientRect(); if (!r.width) return;
    fWave.width = r.width*d; fWave.height = r.height*d; const ctx = fWave.getContext('2d'); ctx.setTransform(d,0,0,d,0,0);
    const n = peaks.length, bw = r.width/n, max = Math.max(...peaks) || 1;
    const pl = document.getElementById('player'), frac = (pl && pl.getAttribute('src') && pl.duration) ? pl.currentTime/pl.duration : 0;
    peaks.forEach((v,i)=>{ const bh = Math.max(2, (v/max)*(r.height-24)); const done = frac > 0 && (i+0.5)/n <= frac;
      ctx.fillStyle = done ? '#2F5220' : mix('#B9D39E', DEEP, v/max); rr(ctx, i*bw+1, (r.height-bh)/2, Math.max(1,bw-2), bh, 2); });
    if (frac > 0 && frac < 1){ ctx.fillStyle = '#2F5220'; ctx.fillRect(Math.round(frac*r.width)-1, 6, 2, r.height-12); }
  }
  new ResizeObserver(drawPeaks).observe(fWave);
  function showFile(name, meta, dur, ch, sample){
    hasFile = true; fileName = name; fileBox.classList.add('has-file'); errEl.hidden = true;
    document.getElementById('fName').textContent = name; document.getElementById('fMeta').textContent = meta;
    document.getElementById('fDur').textContent = dur; document.getElementById('fCh').textContent = ch;
    document.getElementById('sampleBadge').hidden = !sample; document.getElementById('fileState').textContent = live ? 'On the server' : 'Loaded';
    requestAnimationFrame(drawPeaks); resetRun(true);
  }
  const player = document.getElementById('player'); let playerURL = null;
  function setPlayer(f){
    player.pause(); if (playerURL){ URL.revokeObjectURL(playerURL); playerURL = null; }
    if (f){ playerURL = URL.createObjectURL(f); player.src = playerURL; player.hidden = false; }
    else { player.removeAttribute('src'); player.load(); player.hidden = true; }
  }
  player.hidden = true;
  let playRAF = 0; const playLoop = ()=>{ drawPeaks(); if (!player.paused && !player.ended) playRAF = requestAnimationFrame(playLoop); };
  player.addEventListener('play', ()=>{ cancelAnimationFrame(playRAF); playLoop(); });
  ['pause','ended','seeked','loadedmetadata'].forEach(ev=>player.addEventListener(ev, ()=>drawPeaks()));
  fWave.addEventListener('click', e=>{
    if (!player.getAttribute('src') || !player.duration) return; const r = fWave.getBoundingClientRect();
    player.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left)/r.width)) * player.duration; if (player.paused) player.play();
  });
  function clearFile(){
    setPlayer(null);
    hasFile = false; audioId = null; peaks = null; fileIn.value = ''; fileBox.classList.remove('has-file');
    document.getElementById('fileState').textContent = 'No file yet'; resetRun(true);
  }
  function showErr(msg){ errEl.textContent = msg; errEl.hidden = false; }
  const sizeOf = f => f.size > 1048576 ? (f.size/1048576).toFixed(1)+' MB' : Math.ceil(f.size/1024)+' KB';
  const isAudio = f => (f.type && f.type.startsWith('audio/')) || /\.(wav|mp3|flac|ogg|m4a)$/i.test(f.name);

  async function takeFile(f){
    if (!f || running) return;
    if (!isAudio(f)){ showErr(`${f.name} isn't an audio file. Choose a WAV, MP3, FLAC or OGG file.`); return; }
    errEl.hidden = true;
    if (live){
      document.getElementById('fileState').textContent = 'Uploading…'; runBtn.disabled = true; runBtn.textContent = 'Uploading…';
      try {
        const j = await upload(f); audioId = j.audio_id; peaks = j.peaks;
        showFile(f.name, `${sizeOf(f)} · ${(j.sample_rate_in/1000).toFixed(1).replace('.0','')} kHz in`, fmt(j.duration_s), j.channels > 1 ? 'Stereo → mono' : 'Mono', false); setPlayer(f);
      } catch(e){ clearFile(); showErr(e.message); }
      return;
    }
    try {
      const buf = await f.arrayBuffer(); const AC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      const audio = await new AC(1, 1, 44100).decodeAudioData(buf);
      const data = audio.getChannelData(0), n = 140, blk = Math.max(1, Math.floor(data.length/n)); peaks = [];
      for (let i=0;i<n;i++){ let m = 0; for (let j=i*blk;j<(i+1)*blk && j<data.length;j+=8) m = Math.max(m, Math.abs(data[j])); peaks.push(m); }
      showFile(f.name, sizeOf(f) + ' · ' + (f.type || 'audio'), fmt(audio.duration), audio.numberOfChannels > 1 ? 'Stereo → mono' : 'Mono', false); setPlayer(f);
    } catch(e){ showErr(`The browser couldn't read the audio in ${f.name}. Try exporting it as WAV.`); }
  }
  fileIn.addEventListener('change', e=>takeFile(e.target.files[0]));
  ['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev, e=>{ e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(ev=>dz.addEventListener(ev, e=>{ e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e=>takeFile(e.dataTransfer.files[0]));
  document.getElementById('replace').addEventListener('click', ()=>fileIn.click());
  sampleBtn.addEventListener('click', ()=>{ setPlayer(null); peaks = envelope(140, 23); showFile('platform_evening.wav', 'Sample clip · demo only', '00:08', 'Mono', true); });

  // ---- live recording from the microphone -> 16-bit PCM WAV in the browser -> same path as an uploaded file
  const MAX_REC_S = CFG.maxRecordSeconds || 120, recTime = document.getElementById('recTime'), recLevel = document.getElementById('recLevel');
  let rec = null;
  function encodeWav(x, sr){
    const n = x.length, buf = new ArrayBuffer(44 + n*2), v = new DataView(buf), w = (o, t)=>{ for (let i=0;i<t.length;i++) v.setUint8(o+i, t.charCodeAt(i)); };
    w(0,'RIFF'); v.setUint32(4, 36 + n*2, true); w(8,'WAVE'); w(12,'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr*2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36,'data'); v.setUint32(40, n*2, true);
    for (let i=0;i<n;i++){ const s = Math.max(-1, Math.min(1, x[i])); v.setInt16(44 + i*2, s < 0 ? s*0x8000 : s*0x7FFF, true); }
    return new Blob([buf], {type: 'audio/wav'});
  }
  async function startRec(){
    if (running || rec || refRec) return;
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
      showErr('This page cannot use the microphone. Open it as http://localhost:8000 in Chrome, Edge or Firefox (the browser only allows recording on localhost or https).'); return; }
    errEl.hidden = true; player.pause();
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}}); }
    catch(e){ showErr(e.name === 'NotAllowedError' ? 'Microphone access was blocked. Allow the microphone for this page (icon in the address bar) and try again.'
                     : e.name === 'NotFoundError' ? 'No microphone was found on this computer.' : 'The microphone could not be started: ' + e.message); return; }
    const AC = window.AudioContext || window.webkitAudioContext, ctx = new AC();
    if (ctx.state === 'suspended') await ctx.resume();
    const src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1), mute = ctx.createGain(); mute.gain.value = 0;
    const r = {stream, ctx, src, proc, mute, chunks: [], peak: 0, t0: performance.now(), raf: 0, prevState: document.getElementById('fileState').textContent};
    proc.onaudioprocess = e => { if (rec !== r) return; const d = e.inputBuffer.getChannelData(0); r.chunks.push(new Float32Array(d));
      let m = 0; for (let i=0;i<d.length;i+=4) m = Math.max(m, Math.abs(d[i])); r.peak = m; };
    src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);       // silent output: nothing is played back while recording
    rec = r; fileBox.classList.add('recording'); runBtn.disabled = true; document.getElementById('fileState').textContent = 'Recording…';
    recTime.textContent = '00:00'; recLevel.style.width = '0%';
    const tick = ()=>{ if (rec !== r) return; const sec = (performance.now() - r.t0)/1000;
      recTime.textContent = fmt(Math.floor(sec)); recLevel.style.width = Math.min(100, Math.round(Math.sqrt(r.peak)*130)) + '%';
      if (sec >= MAX_REC_S){ stopRec(true); return; } r.raf = requestAnimationFrame(tick); };
    tick();
  }
  function stopRec(use){
    if (!rec) return; const r = rec; rec = null; cancelAnimationFrame(r.raf);
    try { r.src.disconnect(); r.proc.disconnect(); r.mute.disconnect(); } catch(e){}
    r.stream.getTracks().forEach(t=>t.stop()); r.ctx.close();
    fileBox.classList.remove('recording'); document.getElementById('fileState').textContent = r.prevState; resetRun(false);
    if (!use) return;
    const len = r.chunks.reduce((a, c)=>a + c.length, 0);
    if (len < r.ctx.sampleRate * 1.0){ showErr('The recording is too short. Record at least 1 second (5–15 seconds works best).'); return; }
    const pcm = new Float32Array(len); let o = 0; for (const c of r.chunks){ pcm.set(c, o); o += c.length; }
    const d = new Date(), p2 = x=>String(x).padStart(2,'0');
    const name = `recording_${d.getFullYear()}${p2(d.getMonth()+1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}.wav`;
    takeFile(new File([encodeWav(pcm, r.ctx.sampleRate)], name, {type: 'audio/wav'}));
  }
  document.getElementById('recStart').addEventListener('click', startRec);
  document.getElementById('recAgain').addEventListener('click', startRec);
  document.getElementById('recStop').addEventListener('click', ()=>stopRec(true));
  document.getElementById('recCancel').addEventListener('click', ()=>stopRec(false));

  // reference recording (SV mode 1): a file OR a live recording from the microphone
  const refName = document.getElementById('refName'), refPlayer = document.getElementById('refPlayer');
  const refRecRow = document.getElementById('refRecRow'), refRecLive = document.getElementById('refRecLive');
  const refRecTime = document.getElementById('refRecTime'), refRecLevel = document.getElementById('refRecLevel');
  let refPlayerURL = null, refRec = null;
  function setRefPlayer(f){
    refPlayer.pause(); if (refPlayerURL){ URL.revokeObjectURL(refPlayerURL); refPlayerURL = null; }
    if (f){ refPlayerURL = URL.createObjectURL(f); refPlayer.src = refPlayerURL; refPlayer.hidden = false; }
    else { refPlayer.removeAttribute('src'); refPlayer.load(); refPlayer.hidden = true; }
  }
  async function takeRef(f){
    if (!f) return; refAudioId = null; setRefPlayer(null);
    if (!isAudio(f)){ refName.textContent = `${f.name} isn't an audio file. Choose another.`; return; }
    if (!live){ refName.textContent = 'Reference: ' + f.name; setRefPlayer(f); resetRun(); return; }
    refName.textContent = 'Uploading ' + f.name + '…';
    try { const j = await upload(f); refAudioId = j.audio_id; refName.textContent = `Reference: ${f.name} (${fmt(j.duration_s)})`; setRefPlayer(f); }
    catch(err){ refName.textContent = err.message; }
    resetRun();
  }
  document.getElementById('refIn').addEventListener('change', e=>{ takeRef(e.target.files[0]); e.target.value = ''; });

  async function openMic(){
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw new Error('This page cannot use the microphone. Open it as http://localhost:8000 in Chrome, Edge or Firefox.');
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}}); }
    catch(e){ throw new Error(e.name === 'NotAllowedError' ? 'Microphone access was blocked. Allow the microphone for this page (icon in the address bar) and try again.'
                              : e.name === 'NotFoundError' ? 'No microphone was found on this computer.' : 'The microphone could not be started: ' + e.message); }
    const AC = window.AudioContext || window.webkitAudioContext, ctx = new AC();
    if (ctx.state === 'suspended') await ctx.resume();
    const src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1), mute = ctx.createGain(); mute.gain.value = 0;
    const m = {stream, ctx, src, proc, mute, chunks: [], peak: 0, t0: performance.now(), raf: 0, open: true};
    proc.onaudioprocess = e => { if (!m.open) return; const d = e.inputBuffer.getChannelData(0); m.chunks.push(new Float32Array(d));
      let x = 0; for (let i=0;i<d.length;i+=4) x = Math.max(x, Math.abs(d[i])); m.peak = x; };
    src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
    return m;
  }
  function closeMic(m){
    m.open = false; cancelAnimationFrame(m.raf);
    try { m.src.disconnect(); m.proc.disconnect(); m.mute.disconnect(); } catch(e){}
    m.stream.getTracks().forEach(t=>t.stop()); m.ctx.close();
    const len = m.chunks.reduce((a, c)=>a + c.length, 0), pcm = new Float32Array(len); let o = 0;
    for (const c of m.chunks){ pcm.set(c, o); o += c.length; }
    return {pcm, sr: m.ctx.sampleRate};
  }
  async function startRefRec(){
    if (running || rec || refRec) return;
    player.pause(); refPlayer.pause();
    let m; try { m = await openMic(); } catch(e){ refName.textContent = e.message; return; }
    refRec = m; refRecRow.hidden = true; refRecLive.hidden = false; runBtn.disabled = true; refRecTime.textContent = '00:00'; refRecLevel.style.width = '0%';
    const tick = ()=>{ if (refRec !== m) return; const sec = (performance.now() - m.t0)/1000;
      refRecTime.textContent = fmt(Math.floor(sec)); refRecLevel.style.width = Math.min(100, Math.round(Math.sqrt(m.peak)*130)) + '%';
      if (sec >= MAX_REC_S){ stopRefRec(true); return; } m.raf = requestAnimationFrame(tick); };
    tick();
  }
  function stopRefRec(use){
    if (!refRec) return; const m = refRec; refRec = null; const {pcm, sr} = closeMic(m);
    refRecLive.hidden = true; refRecRow.hidden = false; resetRun(false);
    if (!use) return;
    if (pcm.length < sr * 1.0){ refName.textContent = 'The reference recording is too short. Record at least 1 second (5–15 seconds works best).'; return; }
    const d = new Date(), p2 = x=>String(x).padStart(2,'0');
    const name = `reference_${d.getFullYear()}${p2(d.getMonth()+1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}.wav`;
    takeRef(new File([encodeWav(pcm, sr)], name, {type: 'audio/wav'}));
  }
  document.getElementById('refRecStart').addEventListener('click', startRefRec);
  document.getElementById('refRecStop').addEventListener('click', ()=>stopRefRec(true));
  document.getElementById('refRecCancel').addEventListener('click', ()=>stopRefRec(false));

  // registered speakers (SV mode 2)
  function fillSpeakers(){
    if (!live){ spkSel.innerHTML = '<option>SPK-014 (demo)</option><option>SPK-022 (demo)</option>'; spkSel.disabled = false; spkHint.textContent = 'Demo speakers. Live mode lists the real registered speakers.'; return; }
    if (!speakers.length){ spkSel.innerHTML = '<option value="">No registered speakers yet</option>'; spkSel.disabled = true;
      spkHint.textContent = 'To register a speaker, put 3 to 5 clean clips (5 to 15 s each) in speakers/<Name>/ next to the app and restart it.'; return; }
    spkSel.disabled = false; spkSel.innerHTML = speakers.map(s=>`<option value="${esc(s.id)}">${esc(s.name)} · ${s.clips} clips</option>`).join('');
    spkHint.textContent = 'Each registered voice is the average of its enrolment clips.';
  }

  // generic segmented controls
  document.querySelectorAll('.toggle.small').forEach(g=>g.addEventListener('click', e=>{ const b = e.target.closest('button'); if (!b || b.disabled || g.id === 'runMode') return;
    g.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', x===b));
    if (g.id === 'svModeSw'){ const reg = b === g.lastElementChild; document.getElementById('svRef').hidden = reg; document.getElementById('svReg').hidden = !reg; }
    resetRun(); }));

  // demo / live switch
  function setMode(wantLive){
    live = !!wantLive && liveOK; modeBtns.forEach((b,k)=>b.setAttribute('aria-pressed', (k===1) === live));
    sampleBtn.hidden = live; refAudioId = null; refName.textContent = '+ Add a reference recording of the claimed speaker'; setRefPlayer(null); stopRefRec(false);
    modeNote.innerHTML = live ? 'Live: results come from the trained MT2KD model running on this computer.'
      : 'Demo: sample results, clearly labelled. No model is used.' + (liveMsg ? ` <span class="warn">${esc(liveMsg)}</span>` : '');
    fillSpeakers(); clearFile(); errEl.hidden = true;
  }
  modeBtns[0].addEventListener('click', ()=>setMode(false));
  modeBtns[1].addEventListener('click', ()=>setMode(true));

  const tps = [...document.querySelectorAll('.tp')], cfgs = [...document.querySelectorAll('.cfg-p')];
  function pickTask(i){ task = i; tps.forEach((t,k)=>{ t.setAttribute('aria-checked', k===i); t.tabIndex = k===i ? 0 : -1; }); cfgs.forEach((c,k)=>c.classList.toggle('on', k===i)); resetRun(true); }
  tps.forEach((t,k)=>{ t.addEventListener('click', ()=>pickTask(k)); t.addEventListener('keydown', e=>{ if (e.key==='ArrowRight'||e.key==='ArrowDown'){ e.preventDefault(); pickTask((k+1)%3); tps[(k+1)%3].focus(); } if (e.key==='ArrowLeft'||e.key==='ArrowUp'){ e.preventDefault(); pickTask((k+2)%3); tps[(k+2)%3].focus(); } }); });
  function resetRun(clearResult){
    if (running) return;
    const labels = live ? ['Transcribe','Tag sounds','Verify speaker'] : ['Transcribe in demo mode','Tag sounds in demo mode','Verify speaker in demo mode'];
    runBtn.disabled = !hasFile; runBtn.textContent = hasFile ? labels[task] : 'Choose a file to start';
    if (clearResult){ statusEl.hidden = true; doneEl.hidden = true; resultEl.hidden = true; }
  }

  // ---- demo run (simulated, labelled as such)
  const pipeSteps = [
    ['Checking the file', '—'], ['Resampling to 16 kHz mono', '—'], ['Extracting features', '—'], ['Running the shared encoder', (window.MT2KD_ENC_NAME || 'Zipformer-S')],
    [['Decoding the transcript','Scoring sound events','Comparing speaker embeddings'], '—']
  ];
  function runDemo(){
    running = true; runBtn.disabled = true; runBtn.textContent = 'Running (simulated)…'; doneEl.hidden = true; resultEl.hidden = true;
    statusEl.innerHTML = pipeSteps.map(([l,m])=>`<li><i></i><span>${Array.isArray(l)?l[task]:l}</span>${m!=='—'?`<span class="mono">${m}</span>`:''}</li>`).join(''); statusEl.hidden = false;
    const lis = [...statusEl.children]; let k = 0;
    const next = ()=>{ if (k>0) lis[k-1].className = 'ok'; if (k === lis.length){ running = false; runBtn.disabled = false; runBtn.textContent = 'Run again'; doneEl.hidden = false; return; }
      lis[k].className = 'act'; k++; setTimeout(next, reduce ? 50 : 650 + Math.random()*400); };
    next();
  }

  // ---- live run (real model)
  async function runLive(){
    let path, body;
    if (task === 0){ path = '/api/asr'; body = {audio_id: audioId}; }
    else if (task === 1){ path = '/api/at'; body = {audio_id: audioId, top_k: topK()}; }
    else if (svRegistered()){
      if (!spkSel.value){ showResultErr('There are no registered speakers yet. Add clips to the speakers folder, or use "Reference + test".'); return; }
      path = '/api/sv'; body = {audio_id: audioId, speaker_id: spkSel.value};
    } else {
      if (!refAudioId){ showResultErr('Add a reference recording of the claimed speaker first.'); return; }
      path = '/api/sv'; body = {audio_id: audioId, reference_audio_id: refAudioId};
    }
    running = true; runBtn.disabled = true; runBtn.textContent = 'Running on the model…'; resultEl.hidden = true; doneEl.hidden = true;
    statusEl.innerHTML = `<li class="ok"><i></i><span>Uploaded and resampled to 16 kHz mono</span></li><li class="act"><i></i><span>${['Transcribing','Tagging sounds','Comparing the two voices'][task]}</span><span class="mono">${window.MT2KD_ENC_NAME || 'Zipformer-S'}</span></li>`;
    statusEl.hidden = false;
    try {
      const r = await api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      statusEl.lastElementChild.className = 'ok'; showResult(r);
    } catch(e){ statusEl.lastElementChild.className = ''; showResultErr(e.message); }
    finally { running = false; runBtn.disabled = false; runBtn.textContent = 'Run again'; }
  }
  function showResultErr(msg){ resultEl.innerHTML = `<p class="res-err">${esc(msg)}</p>`; resultEl.hidden = false; }
  function showResult(r){
    const head = (title, meta) => `<div class="res-h"><span>${title}</span><span><span class="live-tag">Live</span> ${meta}</span></div>`;
    if (r.task === 'asr'){
      lastText = r.text;
      resultEl.innerHTML = head(r.decoder === 'ctc+lm' ? 'Transcript · with language model' : 'Transcript · greedy', `${r.words} words · ${r.compute_s} s for ${r.duration_s} s of audio`) +
        `<p class="res-text">${r.text ? esc(r.text) : '<span class="muted">No speech was recognised.</span>'}</p>` +
        (r.decoder === 'ctc+lm' ? `<p class="res-alt"><span>Without the language model</span>${r.text_greedy ? esc(r.text_greedy) : '—'}</p>` : '') +
        `<div class="res-actions"><button class="chip" type="button" id="copyTx">Copy</button><button class="chip" type="button" id="dlTx">Download .txt</button></div>` +
        `<p class="res-note">${esc(r.note)}${r.segments > 1 ? ` Long audio was split into ${r.segments} parts at quiet points.` : ''}</p>`;
      document.getElementById('copyTx').onclick = async e=>{ try { await navigator.clipboard.writeText(lastText); e.target.textContent = 'Copied'; } catch(_){ e.target.textContent = 'Copy failed'; } };
      document.getElementById('dlTx').onclick = ()=>{ const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lastText + '\n'], {type:'text/plain'}));
        a.download = (fileName.replace(/\.[^.]+$/, '') || 'transcript') + '_mt2kd.txt'; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 2000); };
    } else if (r.task === 'at'){
      resultEl.innerHTML = head('Sound events', `top ${r.labels.length} · ${r.compute_s} s for ${r.duration_s} s of audio`) +
        `<div class="tags">${r.labels.map(l=>`<div class="trow${l.score < .3 ? ' low' : ''}"><span>${esc(l.name)}</span><span class="bar"><i style="width:${(l.score*100).toFixed(1)}%"></i></span><span class="v">${l.score.toFixed(2)}</span></div>`).join('')}</div>` +
        `<p class="res-note">${esc(r.note)}${r.windows > 1 ? ` Long audio: ${r.windows} windows of 10 s, highest score per class.` : ''}</p>`;
    } else {
      const thr = r.threshold, pos = v => Math.max(0, Math.min(1, v)) * 100;
      const verdict = r.same_speaker == null ? '' : `<div class="verdict ${r.same_speaker ? 'yes' : 'no'}"><i></i><span>${r.same_speaker ? 'Same speaker' : 'Different speaker'}</span></div>`;
      resultEl.innerHTML = head('Speaker match', `${r.compute_s} s`) +
        `<div class="res-sim"><strong>${r.similarity.toFixed(2)}</strong><span class="mono muted">cosine similarity vs ${esc(r.against.name)}</span></div>` +
        (thr != null ? `<div class="gauge"><i style="width:${pos(r.similarity)}%;background:${r.same_speaker ? 'var(--sage)' : '#C98A6B'}"></i><span class="th" style="left:${pos(thr)}%"><span class="mono">Threshold ${thr.toFixed(2)}</span></span></div>` : '') +
        verdict + `<p class="res-note">${esc(r.note)}</p>`;
    }
    resultEl.hidden = false;
  }
  runBtn.addEventListener('click', ()=>{ if (!hasFile || running) return; live ? runLive() : runDemo(); });
  document.getElementById('seeOut').addEventListener('click', e=>{ e.preventDefault(); steps[task].scrollIntoView({behavior: reduce?'auto':'smooth', block:'center'}); });

  // ---- backend detection, speakers, measured numbers
  function pct(v){ return v == null ? null : (v*100).toFixed(2); }
  function setEval(i, val, badge, desc){ if (val == null) return; document.getElementById('evV'+i).textContent = val;
    const b = document.getElementById('evB'+i); b.textContent = badge; b.classList.toggle('official', badge === 'Official test'); if (desc) document.getElementById('evD'+i).textContent = desc; }
  async function loadModelInfo(){
    try {
      const m = await api('/api/model-info'), v = m.validation || {};
      if (m.model_name) { window.MT2KD_ENC_NAME = m.model_name; document.querySelectorAll('.enc-name').forEach(e => { e.textContent = m.model_name; }); }
      setEval(0, pct(v.asr_dev_clean_wer), 'Validation'); setEval(1, pct(v.at_fsd50k_val_map), 'Validation'); setEval(2, pct(v.sv_val_eer_nonofficial), 'Validation');
      if (m.lm && m.lm.dev_clean_half_B) document.getElementById('evS0').innerHTML = `With the LibriSpeech word LM: <b>${pct(m.lm.dev_clean_half_B.lm_wer)}%</b> vs ${pct(m.lm.dev_clean_half_B.greedy_wer)}% greedy (held-out dev-clean speakers)`;
      const o = m.official_test;
      if (o){
        setEval(0, pct(o.asr_test_clean_wer), 'Official test', 'LibriSpeech test-clean, greedy CTC, no language model. Lower is better.');
        if (o.sv_vox1o_eer != null && o.sv_status === 'official') setEval(2, pct(o.sv_vox1o_eer), 'Official test', 'VoxCeleb1-O trial list, cosine scoring. Lower is better.');
        if (o.at_eval_map != null) setEval(1, pct(o.at_eval_map), 'Official test', 'FSD50K evaluation split, 200 classes. Higher is better.');
        document.getElementById('evFoot').textContent = 'Numbers marked "Official test" come from the final evaluation of this model.';
      }
      // same-protocol numbers of the earlier 64M model (Zipformer-M, MT2KD_E2E_main_E0 final evaluation), shown only when the app runs Zipformer-S
      if (m.model_name === 'Zipformer-S') {
        [['evC0', 'Zipformer-M (64M): 45.08% on the same test set'], ['evC1', 'Zipformer-M (64M): 39.36% on the same split'],
         ['evC2', 'Zipformer-M (64M): 8.44% on the same trial list']].forEach(([id, t]) => { const e = document.getElementById(id); if (e) { e.textContent = t; e.hidden = false; } });
      }
      document.getElementById('evLede').textContent = `Numbers for the model this app runs (Stage-2 epoch ${m.stage2_epoch}). "Validation" numbers were measured during training; "Official test" numbers come from the final evaluation.`;
    } catch(e){}
  }
  async function detectBackend(){
    if (API === null){ liveMsg = 'Live mode needs the backend: start it and open http://localhost:8000.'; setMode(false); return; }
    try {
      const h = await api('/api/health');
      if (!h.model_loaded){ liveMsg = 'The backend is running but the model is not loaded. ' + (h.error || ''); setMode(false); return; }
      liveOK = true; modeBtns[1].disabled = false;
      try { speakers = await api('/api/speakers'); } catch(_){ speakers = []; }
      setMode(true); loadModelInfo();
    } catch(e){ liveMsg = 'The backend is not reachable, so only demo mode is available.'; setMode(false); }
  }
  detectBackend();

  pinMode();

  // ---------- main loop ----------
  const t0 = performance.now();
  function loop(now){
    const t = now - t0;
    const fp = (t/(DUR*1000)) % 1; fw.p = fp;
    const sec = fp*DUR; tNow.textContent = '00:0' + Math.floor(sec);
    const said = Math.floor(fp * (words.length + 2));
    wEls.forEach((w,k)=>{ w.classList.toggle('said', k < said); w.classList.toggle('now', k === said-1); });
    wc.textContent = Math.min(said, words.length) + ' words';
    for (const w of waves) if (w.visible) w.draw(t);
    drawWidgets(t);
    requestAnimationFrame(loop);
  }
  if (reduce){ fw.p = 1; wEls.forEach(w=>w.classList.add('said')); wc.textContent = words.length+' words'; waves.forEach(w=>w.draw(0)); setTimeout(()=>drawWidgets(0), 300); }
  else requestAnimationFrame(loop);
})();
