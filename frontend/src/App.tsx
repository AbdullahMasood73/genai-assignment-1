import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Camera, Check, ChevronRight, Download, FlaskConical, ImagePlus, Layers3, LoaderCircle, PanelLeftClose, RefreshCw, Route, Sparkles, Upload, X } from 'lucide-react';

type Key = 'universal' | 'hard' | 'soft' | 'gan';
type Result = { input: string; output: string; target?: string; error_map?: string; inference_ms?: number; probabilities?: number[]; weights?: number[]; predicted?: string; expert?: string; settings?: Record<string, unknown> };
type Health = { status: string; workspaces: Record<Key, boolean>; models: Record<string, string> };
type Experiments = { status: string; message?: string; split?: string; pet_input_count?: number; face_pair_count?: number;
  classifier?: { accuracy: number; 'macro avg': { precision: number; recall: number; 'f1-score': number } };
  restoration?: { system: string; corruption: string; severity: string; n: number; l1: number; psnr: number; ssim: number }[];
  face?: { style: number; n: number; l1: number; psnr: number; ssim: number }[];
  tracking?: { source: string; runs: { id: string; name: string; status: string; epoch: number | null; validation_objective: number | null }[] } };
const workspaces = [
  { key: 'universal' as Key, name: 'Universal Restoration', short: 'Universal', icon: Sparkles, tag: '01', description: 'One compact autoencoder. Three kinds of corruption. Restore an image through a shared learned representation.', endpoint: 'universal-restoration' },
  { key: 'hard' as Key, name: 'Hard-Routed Restoration', short: 'Hard routing', icon: Route, tag: '02', description: 'Let the classifier identify the corruption, then send your image to the corresponding specialist.', endpoint: 'hard-routing' },
  { key: 'soft' as Key, name: 'Soft Mixture-of-Experts Restoration', short: 'Soft mixture', icon: Layers3, tag: '03', description: 'Combine the identity branch and three restoration experts with weights learned by a jointly trained gate.', endpoint: 'soft-mixture' },
  { key: 'gan' as Key, name: 'Face-to-Sketch Generator', short: 'Face to sketch', icon: FlaskConical, tag: '04', description: 'Translate a facial photograph into a sketch using one of three learned FS2K style conditions.', endpoint: 'face-to-sketch' },
];
const labels = ['Clean identity', 'Salt & pepper', 'Gaussian blur', 'Occlusion'];
// Tailwind utility strings for the table, weight-bar and footer components.
const tableWrap = 'overflow-auto mt-4 mb-6';
const table = 'w-full border-collapse text-[11px] text-left';
const cell = 'px-2.5 py-[9px] border-b border-[#dde5df] whitespace-nowrap';
const head = cell + ' bg-[#edf3e8] sticky top-0';
const step = 'bg-mist border border-[#dce4d0] rounded-[5px] text-[9px] px-3 py-[11px] text-[#6e835d] whitespace-nowrap max-[640px]:whitespace-normal max-[640px]:text-center max-[640px]:p-[9px] max-[640px]:text-[8px] max-[640px]:flex-1';

function ImagePanel({ title, url, caption }: { title: string; url?: string; caption: string }) {
  return <div className="image-panel"><div className="panel-title"><span>{title}</span><span className="mono">128 × 128</span></div>
    <div className={'image-stage ' + (url ? 'has-image' : '')}>{url ? <img src={url} alt={title} /> : <div className="stage-placeholder"><ImagePlus size={38} strokeWidth={1} /><span>{caption}</span></div>}</div>
  </div>;
}

export default function App() {
  const [key, setKey] = useState<Key>('universal');
  const [health, setHealth] = useState<Health | null>(null);
  const [connected, setConnected] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [corruption, setCorruption] = useState('none');
  const [severity, setSeverity] = useState('medium');
  const [seed, setSeed] = useState('42');
  const [style, setStyle] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [samples, setSamples] = useState<{ id: string; url: string }[]>([]);
  const [showExperiments, setShowExperiments] = useState(false);
  const [experiments, setExperiments] = useState<Experiments | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const operation = useRef(0);
  const current = workspaces.find(w => w.key === key)!;

  async function refreshHealth() {
    try {
      const response = await fetch('/api/health');
      if (!response.ok) throw new Error('API unavailable');
      setHealth(await response.json()); setConnected(true);
    } catch { setConnected(false); }
  }
  useEffect(() => { void refreshHealth(); fetch('/api/samples').then(r => r.ok ? r.json() : []).then(setSamples).catch(() => {}); }, []);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => { stream.current?.getTracks().forEach(track => track.stop()); }, []);

  function chooseFile(selected?: File) {
    if (!selected) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(selected.type)) { setError('Choose a PNG, JPEG, or WebP image.'); return; }
    if (selected.size > 10 * 1024 * 1024) { setError('Choose an image smaller than 10 MB.'); return; }
    operation.current += 1;
    setFile(selected); setResult(null); setError(''); setBusy(false);
  }
  function switchWorkspace(next: Key) {
    operation.current += 1;
    setKey(next); setResult(null); setError(''); setBusy(false); setNavigationOpen(false);
  }
  function changeSetting(action: () => void) {
    operation.current += 1; action(); setResult(null); setBusy(false); setError('');
  }
  async function request(previewOnly = false) {
    if (!file) return;
    const id = ++operation.current;
    setBusy(true); setError('');
    try {
      const payload = new FormData(); payload.append('file', file);
      payload.append('corruption', corruption === 'none' ? 'clean' : corruption);
      payload.append('severity', severity); payload.append('seed', seed); payload.append('style', String(style));
      if (!previewOnly) payload.set('corruption', corruption);
      const response = await fetch('/api/' + (previewOnly ? 'corrupt' : current.endpoint), { method: 'POST', body: payload });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the image and selected settings.');
      if (id !== operation.current) return;
      if (previewOnly) setResult({ input: data.output, output: '', target: data.input, settings: data.settings });
      else setResult(data);
    } catch (reason) { if (id === operation.current) setError(reason instanceof Error ? reason.message : 'Unable to complete the request.'); }
    finally { if (id === operation.current) setBusy(false); }
  }
  async function chooseSample(sample: { id: string; url: string }) {
    try {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error('Sample unavailable.');
      chooseFile(new File([await response.blob()], sample.id, { type: 'image/png' }));
      setCorruption('salt');
    } catch { setError('This sample could not be loaded.'); }
  }
  async function openCamera() {
    setError('');
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      setCameraOpen(true);
    } catch { setError('Camera access is unavailable. Upload a photograph instead.'); }
  }
  useEffect(() => { if (cameraOpen && video.current && stream.current) video.current.srcObject = stream.current; }, [cameraOpen]);
  function closeCamera() { stream.current?.getTracks().forEach(t => t.stop()); stream.current = null; setCameraOpen(false); }
  function capture() {
    if (!video.current?.videoWidth) return;
    const canvas = document.createElement('canvas'); canvas.width = video.current.videoWidth; canvas.height = video.current.videoHeight;
    canvas.getContext('2d')!.drawImage(video.current, 0, 0);
    canvas.toBlob(blob => { if (blob) chooseFile(new File([blob], 'camera-photo.png', { type: 'image/png' })); }, 'image/png');
    closeCamera();
  }
  async function inspectExperiments() {
    setShowExperiments(true);
    try { const response = await fetch('/api/experiments'); if (!response.ok) throw new Error(); setExperiments(await response.json()); }
    catch { setExperiments({ status: 'unavailable', message: 'The experiment service is unavailable.' }); }
  }
  const weights = result?.weights || result?.probabilities;
  const ready = Boolean(health?.workspaces[key]);

  return <div className="app-shell">
    <aside className={'sidebar ' + (navigationOpen ? 'mobile-open' : '')}>
      <a className="brand" href="#" onClick={event => { event.preventDefault(); switchWorkspace('universal'); }}><div className="brand-mark"><Layers3 size={22}/></div><span>Restoration<span className="brand-lab">Lab</span></span></a>
      <div className="sidebar-label">GENERATIVE IMAGING</div>
      <nav aria-label="Workspaces">{workspaces.map(w => <button key={w.key} className={'nav-item ' + (key === w.key ? 'active' : '')} onClick={() => switchWorkspace(w.key)}><w.icon size={19}/><span>{w.short}</span><span className="nav-number">{w.tag}</span></button>)}</nav>
      <div className="sidebar-divider"/><button className="nav-item" onClick={inspectExperiments}><FlaskConical size={19}/><span>Experiments</span><ChevronRight size={15}/></button>
      <div className="sidebar-bottom"><div className="lab-note"><span className="note-dot"/>A small laboratory<br/><span className="muted">for learning to restore.</span></div><div className="sidebar-foot">GENERATIVE AI · ASSIGNMENT 01</div></div>
    </aside>
    {navigationOpen && <button className="mobile-overlay" aria-label="Close navigation" onClick={() => setNavigationOpen(false)}/>}
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setNavigationOpen(true)}><PanelLeftClose size={20}/></button><span>Workspaces</span><ChevronRight size={14}/><strong>{current.short}</strong></div><div className="connection"><span className={'status-dot ' + (connected ? 'online' : '')}/>{connected ? 'Service connected' : 'Service offline'}<button aria-label="Refresh service status" title="Refresh service status" onClick={refreshHealth}><RefreshCw size={14}/></button></div></header>
      <main>
        <div className="intro"><div><div className="eyebrow"><span className="tiny-line"/>WORKSPACE {current.tag}</div><h1>{current.name}</h1><p>{current.description}</p></div><div className="workspace-symbol"><current.icon size={40} strokeWidth={1.2}/></div></div>
        <div className="workspace-layout">
          <section className="controls-card" aria-label="Image and model controls"><div className="section-heading"><span className="step-number">1</span><h2>Set up your image</h2></div>
            <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" className="hidden" onChange={e => { chooseFile(e.target.files?.[0]); e.target.value = ''; }}/>
            <button className={'upload-zone ' + (dragging ? 'dragging' : '')} onClick={() => fileInput.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files[0]); }}>
              <div className="upload-icon"><Upload size={23} strokeWidth={1.5}/></div><strong>{file ? 'Change your image' : 'Drop an image here'}</strong><span>{file ? file.name : 'or browse your files'}</span><small>PNG, JPG, WEBP · up to 10 MB</small>
            </button>
            {key === 'gan' && <button className="secondary camera-button" onClick={openCamera}><Camera size={16}/>Take a photograph</button>}
            {key !== 'gan' && samples.length > 0 && <div className="sample-section"><label>Or try a clean sample</label><div className="sample-list">{samples.map((sample, i) => <button key={sample.id} onClick={() => chooseSample(sample)} aria-label={'Select clean pet sample ' + (i + 1)}><img src={sample.url} alt={'Pet sample ' + (i + 1)}/></button>)}</div></div>}
            <div className="control-divider"/>
            {key === 'gan' ? <><label className="field-label">Sketch style</label><div className="style-picker">{[0, 1, 2].map(s => <button key={s} className={style === s ? 'selected' : ''} onClick={() => changeSetting(() => setStyle(s))}>Style {s + 1}{style === s && <Check size={13}/>}</button>)}</div><p className="field-help">Styles correspond to the three categories in FS2K.</p></> : <>
              <label className="field-label" htmlFor="corruption">Input condition</label><select id="corruption" value={corruption} onChange={e => changeSetting(() => setCorruption(e.target.value))}><option value="none">Already corrupted / use as uploaded</option><option value="clean">Clean image</option><option value="salt">Apply salt & pepper noise</option><option value="blur">Apply Gaussian blur</option><option value="occlusion">Apply rectangular occlusion</option></select>
              {['salt', 'blur', 'occlusion'].includes(corruption) && <><label className="field-label severity-label">Corruption severity</label><div className="severity-picker">{['low', 'medium', 'high'].map(level => <button key={level} className={severity === level ? 'selected' : ''} onClick={() => changeSetting(() => setSeverity(level))}>{level}</button>)}</div><label className="field-label seed-label" htmlFor="seed">Random seed</label><input id="seed" type="number" min="0" max="2147483647" value={seed} onChange={e => changeSetting(() => setSeed(e.target.value))}/><button className="text-button" onClick={() => request(true)} disabled={!file || busy}>Preview corruption<ArrowRight size={14}/></button></>}
              <p className="field-help">Images are resized to 128 × 128 for inference.</p>
            </>}
            <button className="run-button" onClick={() => request()} disabled={!file || busy || !ready}>{busy ? <LoaderCircle className="spin" size={17}/> : <current.icon size={17}/>}<span>{busy ? 'Processing image…' : key === 'gan' ? 'Generate sketch' : 'Restore image'}</span><ArrowRight size={17}/></button>
            {!ready && <div className="model-notice">{connected ? 'Awaiting trained model files. Image uploads and corruption preview are available.' : 'Start the application service to process images.'}</div>}
          </section>
          <section className="results-card" aria-label="Image results"><div className="section-heading"><span className="step-number">2</span><h2>{key === 'gan' ? 'Inspect the sketch' : 'Inspect the restoration'}</h2><span className="result-badge">{result?.output ? 'COMPLETE' : 'READY WHEN YOU ARE'}</span></div>
            {error && <div className="flex justify-between items-center gap-2.5 bg-[#fcf0e8] border border-[#ebd0bd] text-[#996340] p-3 rounded-md text-[11px] mb-[18px]" role="alert">{error}<button className="flex" aria-label="Dismiss error" onClick={() => setError('')}><X size={15}/></button></div>}
            <div className="image-comparison"><ImagePanel title={key === 'gan' ? 'Original photograph' : 'Input image'} url={result?.input || preview} caption="Your uploaded image appears here"/><ImagePanel title={key === 'gan' ? 'Generated sketch' : 'Restored image'} url={result?.output} caption={key === 'gan' ? 'A new sketch, generated from your photo' : 'Your restoration will appear here'}/></div>
            <div className="flex items-center justify-between border-t border-line mt-[21px] pt-[18px] gap-2.5 max-[1200px]:flex-wrap"><div className="flex items-center gap-[7px] text-[9px] text-[#9ca78f] max-[640px]:text-[8px]"><span className="status-dot online"/><span>{result?.inference_ms !== undefined ? `${result.inference_ms.toFixed(2)} ms inference` : 'Inference time appears after processing'}</span></div>{result?.output ? <a className="download-button" href={result.output} download={key === 'gan' ? 'generated-sketch.png' : 'restored-image.png'}><Download size={16}/>Download result</a> : <span className="empty-download"><Download size={16}/>Download result</span>}</div>
            {weights && <div className="routing-panel"><div className="routing-heading"><h3>{key === 'hard' ? 'Classifier probabilities' : 'Expert contributions'}</h3>{key === 'hard' && <span>Selected: {result?.expert}</span>}</div><div className="grid grid-cols-2 gap-[17px] max-[640px]:gap-[13px] mt-[19px]">{weights.map((weight, i) => <div key={labels[i]}><div className="flex justify-between text-[10px] text-[#819173]"><span>{labels[i]}</span><strong className="font-semibold text-moss">{(weight * 100).toFixed(1)}%</strong></div><div className="h-[5px] bg-track rounded-[5px] mt-2 overflow-hidden"><div className="h-full bg-sage rounded-[5px] transition-[width] duration-300" style={{ width: `${Math.max(0, Math.min(100, weight * 100))}%` }}/></div></div>)}</div></div>}
            {result?.error_map && <details className="analysis-details"><summary>Compare with the clean target</summary><div className="image-comparison"><ImagePanel title="Clean target" url={result.target} caption=""/><ImagePanel title="Absolute error map" url={result.error_map} caption=""/></div></details>}
            {result?.settings && <details className="settings-details"><summary>Applied corruption settings</summary><pre>{JSON.stringify(result.settings, null, 2)}</pre></details>}
          </section>
        </div>
        <section className="flex items-center justify-between gap-6 bg-[#edf1e5] border border-[#e0e6d6] rounded-[10px] mt-[26px] p-6 max-[1200px]:flex-col max-[1200px]:items-start max-[640px]:p-[18px] max-[640px]:gap-[18px]"><div><span className="flex items-center gap-[9px] text-[8px] tracking-[1.5px] font-semibold text-[#7f8e78]">UNDER THE SURFACE</span><h2 className="text-sm font-medium tracking-[-.2px] mt-[7px] mb-0 text-[#506543]">{key === 'universal' ? 'A shared path back to a clean image.' : key === 'hard' ? 'Recognize. Route. Restore.' : key === 'soft' ? 'Let every expert contribute.' : 'A photograph, interpreted in a learned style.'}</h2></div><div className="flex items-center gap-3 text-[#a5b396] max-[640px]:gap-[7px] max-[640px]:w-full"><span className={step}>{key === 'gan' ? 'Photograph + style' : 'Input image'}</span><ArrowRight className="shrink-0 max-[640px]:w-[13px]" size={18}/><span className={step}>{key === 'universal' ? 'Compressed latent' : key === 'hard' ? 'Classifier → specialist' : key === 'soft' ? 'Gate + weighted experts' : 'Conditional U-Net'}</span><ArrowRight className="shrink-0 max-[640px]:w-[13px]" size={18}/><span className={step}>{key === 'gan' ? 'Generated sketch' : 'Reconstruction'}</span></div></section>
        <footer className="flex justify-between gap-2.5 pt-[25px] pb-2 text-[9px] text-[#a5af99] max-[640px]:text-[8px]"><span className="font-semibold tracking-[.6px]">Restoration Lab</span><span>Four approaches. One imaging workspace.</span></footer>
      </main>
    </div>
    {showExperiments && <div className="modal-backdrop" onClick={() => setShowExperiments(false)}><section className="modal" role="dialog" aria-modal="true" aria-label="Experiment results" onClick={e => e.stopPropagation()}><div className="modal-heading"><h2>Experiment records</h2><button aria-label="Close experiment results" onClick={() => setShowExperiments(false)}><X/></button></div><p>Recorded evaluation results from trained checkpoints. Training histories and Optuna studies are stored in the repository (artifacts/ and experiments/).</p>{!experiments ? <p>Loading records…</p> : experiments.status !== 'evaluated' ? <p>{experiments.message || 'Evaluation pending.'}</p> : <>
      <p><strong>{experiments.split} split</strong> · {experiments.pet_input_count?.toLocaleString()} restoration inputs · {experiments.face_pair_count?.toLocaleString()} paired sketches</p>
      <details className="analysis-details"><summary>Training records from MLflow</summary>{experiments.tracking?.runs.length ? <><p>Latest recorded epoch and validation objective for each run. Full histories are stored in artifacts/mlflow_artifacts.</p><div className={tableWrap}><table className={table}><thead><tr><th className={head}>Run</th><th className={head}>Status</th><th className={head}>Epoch</th><th className={head}>Validation objective ↓</th></tr></thead><tbody>{experiments.tracking.runs.map(run => <tr key={run.id} className="even:bg-[#f7f9f5]"><td className={cell}>{run.name}</td><td className={cell}>{run.status}</td><td className={cell}>{run.epoch ?? '—'}</td><td className={cell}>{run.validation_objective?.toFixed(5) ?? '—'}</td></tr>)}</tbody></table></div></> : <p>Training database unavailable.</p>}</details>
      {experiments.classifier && <section aria-label="Classifier evaluation"><h3>Corruption classifier</h3><p>Accuracy {(experiments.classifier.accuracy * 100).toFixed(1)}% · Macro precision {(experiments.classifier['macro avg'].precision * 100).toFixed(1)}% · Macro recall {(experiments.classifier['macro avg'].recall * 100).toFixed(1)}% · Macro F1 {(experiments.classifier['macro avg']['f1-score'] * 100).toFixed(1)}%</p></section>}
      <h3>Restoration by condition</h3><div className={tableWrap}><table className={table}><thead><tr><th className={head}>Method</th><th className={head}>Condition</th><th className={head}>Severity</th><th className={head}>L1 ↓</th><th className={head}>PSNR ↑</th><th className={head}>SSIM ↑</th></tr></thead><tbody>{experiments.restoration?.map(row => <tr key={`${row.system}-${row.corruption}-${row.severity}`} className="even:bg-[#f7f9f5]"><td className={cell}>{row.system.replaceAll('_', ' ')}</td><td className={cell}>{row.corruption}</td><td className={cell}>{row.severity}</td><td className={cell}>{row.l1.toFixed(4)}</td><td className={cell}>{row.psnr.toFixed(2)} dB</td><td className={cell}>{row.ssim.toFixed(4)}</td></tr>)}</tbody></table></div>
      <h3>Paired sketch results</h3><div className={tableWrap}><table className={table}><thead><tr><th className={head}>Style</th><th className={head}>Images</th><th className={head}>L1 ↓</th><th className={head}>PSNR ↑</th><th className={head}>SSIM ↑</th></tr></thead><tbody>{experiments.face?.map(row => <tr key={row.style} className="even:bg-[#f7f9f5]"><td className={cell}>Style {row.style + 1}</td><td className={cell}>{row.n}</td><td className={cell}>{row.l1.toFixed(4)}</td><td className={cell}>{row.psnr.toFixed(2)} dB</td><td className={cell}>{row.ssim.toFixed(4)}</td></tr>)}</tbody></table></div>
      <p>Lower L1 and higher PSNR/SSIM indicate better paired reconstruction. Read the condition-specific results and report failure examples before drawing conclusions.</p></>}</section></div>}
    {cameraOpen && <div className="modal-backdrop"><section className="modal camera-modal" role="dialog" aria-modal="true" aria-label="Capture photograph"><div className="modal-heading"><h2>Capture a photograph</h2><button aria-label="Close camera" onClick={closeCamera}><X/></button></div><video ref={video} autoPlay playsInline muted/><button className="run-button" onClick={capture}><Camera size={18}/>Capture image</button></section></div>}
  </div>;
}
