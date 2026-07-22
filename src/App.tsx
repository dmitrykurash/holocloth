import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useDialKitController, DialRoot, DialStore } from 'dialkit';
import { HoloApp, type HoloParams, type ImagesState } from './scene.ts';

/** Shared renderer settings — identical across every preset. */
const RENDER_DEFAULTS = {
  background: '#0b0c12',
  exposure: 0.5,
  environment: 0.73,
  bloom: 0.05,
  bloomThreshold: 1.41,
  noise: 0.345,
  toneMapping: 'Neutral',
  occlusion: true,
  occlusionStrength: 1,
  dof: false,
  dofAperture: 40,
  dofBlur: 0.04,
  dofRange: 0.3,
};

/** finish → surface response; presets reference these by name */
const FINISH_VALUES: Record<string, { roughness: number; clearcoat: number; coatRoughness: number }> = {
  Glossy: { roughness: 0.1, clearcoat: 1.0, coatRoughness: 0.08 },
  Satin: { roughness: 0.3, clearcoat: 0.45, coatRoughness: 0.3 },
  Matte: { roughness: 0.62, clearcoat: 0.06, coatRoughness: 0.7 },
};

interface PresetBundle {
  material: Record<string, unknown>;
  render: Record<string, unknown>;
}

const PRESET_VALUES: Record<string, PresetBundle> = {
  Holo: {
    material: {
      finish: 'Matte',
      baseColor: '#20242d',
      holoIntensity: 3.78,
      holoScale: 400,
      bandFreq: 1.1,
      saturation: 1.0,
      hueShift: 0.37,
      sparkle: 0.73,
      specTint: 0.33,
      iridescence: 0.81,
      metalness: 1.0,
      sheen: 0,
      bump: 3.0,
      bumpTiling: 3,
      ...FINISH_VALUES.Matte,
    },
    render: { ...RENDER_DEFAULTS },
  },
  Chrome: {
    material: {
      finish: 'Glossy',
      baseColor: '#dfe3e8',
      holoIntensity: 0,
      sparkle: 0.2,
      specTint: 0,
      iridescence: 0,
      metalness: 1,
      sheen: 0,
      bump: 0.05,
      ...FINISH_VALUES.Glossy,
      roughness: 0.04,
      coatRoughness: 0.04,
    },
    render: { ...RENDER_DEFAULTS },
  },
  'Black Cloth': {
    material: {
      finish: 'Satin',
      baseColor: '#101114',
      holoIntensity: 0.1,
      holoScale: 8,
      bandFreq: 0.2,
      saturation: 0,
      hueShift: 0,
      sparkle: 0,
      specTint: 0.82,
      iridescence: 0,
      metalness: 0.43,
      sheen: 0.08,
      bump: 0,
      ...FINISH_VALUES.Satin,
      roughness: 0.83,
      clearcoat: 0.22,
      coatRoughness: 0.32,
    },
    render: { ...RENDER_DEFAULTS },
  },
};

/**
 * Anchor a container div directly AFTER the panel button with the given
 * label. DialKit's panel is config-driven (no image control), so chips are
 * portaled into containers we insert ourselves; a MutationObserver re-inserts
 * them whenever DialKit re-renders or a folder collapses/expands.
 */
function usePanelAnchor(label: string): HTMLElement | null {
  const [container, setContainer] = useState<HTMLElement | null>(null);
  const currentRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const ensure = () => {
      let btn: HTMLElement | null = null;
      document.querySelectorAll('button').forEach((b) => {
        if (b.textContent?.trim() === label) btn = b;
      });
      if (!btn) {
        if (currentRef.current) {
          currentRef.current = null;
          setContainer(null);
        }
        return;
      }
      const next = (btn as HTMLElement).nextElementSibling as HTMLElement | null;
      if (next && next.dataset.holochip === label) {
        if (currentRef.current !== next) {
          currentRef.current = next;
          setContainer(next);
        }
        return;
      }
      const c = document.createElement('div');
      c.dataset.holochip = label;
      (btn as HTMLElement).insertAdjacentElement('afterend', c);
      currentRef.current = c;
      setContainer(c);
    };
    ensure();
    const observer = new MutationObserver(ensure);
    observer.observe(document.body, { childList: true, subtree: true });
    const interval = window.setInterval(ensure, 1000); // safety net
    return () => {
      observer.disconnect();
      window.clearInterval(interval);
      currentRef.current?.remove();
      currentRef.current = null;
    };
  }, [label]);

  return container;
}

/** One uploaded-image row, styled to sit among DialKit's controls. */
function ImageChipRow({ thumb, label, onRemove }: { thumb: string; label: string; onRemove: () => void }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '5px 8px',
        margin: '6px 0',
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 8,
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        fontSize: 11,
        color: '#c9cbd2',
        userSelect: 'none',
      }}
    >
      <img
        src={thumb}
        alt={label}
        style={{ width: 28, height: 28, objectFit: 'cover', borderRadius: 4, display: 'block' }}
      />
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </span>
      <button
        onClick={onRemove}
        title={`Remove ${label}`}
        style={{
          background: 'rgba(255,255,255,0.08)',
          border: 'none',
          borderRadius: 5,
          color: '#e8e9ee',
          width: 20,
          height: 20,
          lineHeight: '18px',
          fontSize: 12,
          cursor: 'pointer',
          padding: 0,
          flexShrink: 0,
        }}
      >
        ✕
      </button>
    </div>
  );
}

/** Fetch an image with byte-level progress, then decode it. */
function loadImageWithProgress(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'blob';
    xhr.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`${url}: HTTP ${xhr.status}`));
        return;
      }
      const blobUrl = URL.createObjectURL(xhr.response as Blob);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(blobUrl);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(blobUrl);
        reject(new Error(`${url}: decode failed`));
      };
      img.src = blobUrl;
    };
    xhr.onerror = () => reject(new Error(`${url}: network error`));
    xhr.send();
  });
}

/** Circular determinate loader shown while the startup assets download. */
function Loader({ percent, hiding }: { percent: number; hiding: boolean }) {
  const r = 26;
  const circumference = 2 * Math.PI * r;
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        display: 'grid',
        placeItems: 'center',
        background: '#0b0c12',
        zIndex: 2147483600,
        opacity: hiding ? 0 : 1,
        transition: 'opacity 0.5s ease',
        pointerEvents: hiding ? 'none' : 'auto',
      }}
    >
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r={r} fill="none" stroke="rgba(255,255,255,0.09)" strokeWidth="2" />
        <circle
          cx="36"
          cy="36"
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.85)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - percent / 100)}
          transform="rotate(-90 36 36)"
          style={{ transition: 'stroke-dashoffset 0.15s linear' }}
        />
        <text
          x="36"
          y="36"
          textAnchor="middle"
          dominantBaseline="central"
          fill="#c9cbd2"
          fontSize="12"
          fontFamily="ui-sans-serif, system-ui, sans-serif"
        >
          {Math.round(percent)}%
        </text>
      </svg>
    </div>
  );
}

function loadImageFile(file: File, onLoad: (img: HTMLImageElement) => void) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // decoded pixels stay usable for canvas drawing after revocation
    URL.revokeObjectURL(url);
    onLoad(img);
  };
  img.src = url;
}

export default function App() {
  const hostRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<HoloApp | null>(null);
  const decalInputRef = useRef<HTMLInputElement>(null);
  const clothInputRef = useRef<HTMLInputElement>(null);
  const bumpInputRef = useRef<HTMLInputElement>(null);

  const dial = useDialKitController(
    'Holocloth',
    {
      performance: {
        type: 'select' as const,
        options: ['High', 'Medium', 'Low'],
        default: 'High',
      },
      material: {
        preset: {
          type: 'select' as const,
          options: ['Holo', 'Chrome', 'Black Cloth'],
          default: 'Holo',
        },
        finish: {
          type: 'select' as const,
          options: ['Glossy', 'Satin', 'Matte'],
          default: 'Matte',
        },
        baseColor: '#20242d',
        holoIntensity: [3.78, 0, 4, 0.01],
        holoScale: [400, 8, 400, 1],
        bandFreq: [1.1, 0.2, 10, 0.05],
        saturation: [1, 0, 1, 0.01],
        hueShift: [0.37, 0, 1, 0.01],
        sparkle: [0.73, 0, 2, 0.01],
        specTint: [0.33, 0, 1, 0.01],
        iridescence: [0.81, 0, 1, 0.01],
        roughness: [0.62, 0, 1, 0.01],
        metalness: [1, 0, 1, 0.01],
        clearcoat: [0.06, 0, 1, 0.01],
        coatRoughness: [0.7, 0, 1, 0.01],
        sheen: [0, 0, 1, 0.01],
        bump: [3, 0, 3, 0.01],
        bumpTiling: [3, 1, 12, 0.5],
        uploadBump: { type: 'action' as const, label: 'Upload bump map' },
      },
      physics: {
        _collapsed: true,
        viscosity: [0.6, 0.0, 0.6, 0.005],
        stiffness: [1, 0.2, 1, 0.01],
        iterations: [14, 1, 14, 1],
        smoothing: [0.045, 0, 0.3, 0.005],
        grabRadius: [0.27, 0.05, 1.2, 0.01],
      },
      images: {
        _collapsed: true,
        useImage: false,
        edit: false,
        scale: [0.35, 0.02, 2.5, 0.01],
        rotation: [0, -180, 180, 1],
        opacity: [1, 0, 1, 0.01],
        cornerRadius: [0, 0, 1, 0.01],
        addImage: { type: 'action' as const, label: 'Add image / SVG' },
        makeCloth: { type: 'action' as const, label: 'Image as cloth…' },
        clearImages: { type: 'action' as const, label: 'Clear images' },
      },
      render: {
        _collapsed: true,
        background: { type: 'color' as const, default: RENDER_DEFAULTS.background },
        exposure: [RENDER_DEFAULTS.exposure, 0.2, 2.5, 0.01],
        environment: [RENDER_DEFAULTS.environment, 0, 3, 0.01],
        bloom: [RENDER_DEFAULTS.bloom, 0, 1.2, 0.01],
        bloomThreshold: [RENDER_DEFAULTS.bloomThreshold, 0, 2, 0.01],
        noise: [RENDER_DEFAULTS.noise, 0, 0.6, 0.005],
        toneMapping: {
          type: 'select' as const,
          options: ['AgX', 'ACES', 'Neutral'],
          default: RENDER_DEFAULTS.toneMapping,
        },
        occlusion: RENDER_DEFAULTS.occlusion,
        occlusionStrength: [RENDER_DEFAULTS.occlusionStrength, 0, 1, 0.01],
        dof: RENDER_DEFAULTS.dof,
        dofAperture: [RENDER_DEFAULTS.dofAperture, 1, 150, 1],
        dofBlur: [RENDER_DEFAULTS.dofBlur, 0, 0.15, 0.001],
        dofRange: [RENDER_DEFAULTS.dofRange, 0, 3, 0.01],
        pickFocus: { type: 'action' as const, label: 'Pick focus point' },
        autoFocus: { type: 'action' as const, label: 'Auto focus' },
      },
      exportPNG: { type: 'action' as const, label: 'Export PNG' },
      exportPNGClear: { type: 'action' as const, label: 'Export PNG (no background)' },
      resetCloth: { type: 'action' as const, label: 'Reset cloth' },
      poke: { type: 'action' as const, label: 'Poke' },
    },
    {
      id: 'holocloth',
      onAction: (path: string) => {
        const name = path.split('.').pop();
        const app = appRef.current;
        if (!app) return;
        if (name === 'resetCloth') app.resetCloth();
        else if (name === 'poke') app.poke();
        else if (name === 'exportPNG') app.exportPNG(false);
        else if (name === 'exportPNGClear') app.exportPNG(true);
        else if (name === 'addImage') decalInputRef.current?.click();
        else if (name === 'makeCloth') clothInputRef.current?.click();
        else if (name === 'uploadBump') bumpInputRef.current?.click();
        else if (name === 'pickFocus') app.startPickFocus();
        else if (name === 'autoFocus') app.clearPickFocus();
        else if (name === 'clearImages') {
          app.clearImages();
          dial.setValues({ images: { useImage: false } } as never);
        }
      },
    },
  );

  const params = dial.values;

  const [, setImagesRev] = useState(0);
  const [loadPercent, setLoadPercent] = useState(0);
  const [assetsReady, setAssetsReady] = useState(false);
  const [loaderGone, setLoaderGone] = useState(false);

  useEffect(() => {
    if (!hostRef.current) return;
    const app = new HoloApp(hostRef.current);
    appRef.current = app;
    app.onDecalSelect = (scale, rotation) => {
      dial.setValues({ images: { scale, rotation } } as never);
    };
    app.onImagesChanged = () => setImagesRev((v) => v + 1);
    // startup assets: cloth poster + bump map, with combined download
    // progress driving the loader; the cloth stays hidden until both land
    const assets = [
      { url: '/holo-bg-2.jpg', loaded: 0, total: 0, done: false },
      { url: '/bump-scratches.jpg', loaded: 0, total: 0, done: false },
    ];
    const report = () => {
      const total = assets.reduce((s, a) => s + a.total, 0);
      const loaded = assets.reduce((s, a) => s + a.loaded, 0);
      // before any Content-Length arrives, fall back to a coarse item count
      const pct = total > 0
        ? (loaded / total) * 100
        : (assets.filter((a) => a.done).length / assets.length) * 100;
      setLoadPercent((prev) => Math.max(prev, Math.min(100, pct)));
    };
    Promise.all(
      assets.map((a) =>
        loadImageWithProgress(a.url, (loaded, total) => {
          a.loaded = loaded;
          a.total = total;
          report();
        }).then((img) => {
          a.done = true;
          a.loaded = a.total || a.loaded;
          report();
          return img;
        }),
      ),
    )
      .then(([clothImg, bumpImg]) => {
        if (appRef.current !== app) return; // unmounted meanwhile
        app.setBumpMap(bumpImg);
        app.setClothImage(clothImg);
        dial.setValues({ images: { useImage: true } } as never);
        setLoadPercent(100);
        app.reveal();
        setAssetsReady(true);
        window.setTimeout(() => setLoaderGone(true), 550);
      })
      .catch((err) => {
        console.error('[holocloth] asset load failed', err);
        if (appRef.current !== app) return;
        // never trap the user behind the loader
        app.reveal();
        setAssetsReady(true);
        window.setTimeout(() => setLoaderGone(true), 550);
      });

    // Version manager: DialKit already versions the dial values in-memory.
    // Images live in the engine, so keep a side-table keyed by the active
    // preset id (null = "Version 1") and swap it on version switches.
    const imageStates = new Map<string | null, ImagesState>();
    let prevVersion = DialStore.getActivePresetId('holocloth');
    const unsubVersions = DialStore.subscribe('holocloth', () => {
      const cur = DialStore.getActivePresetId('holocloth');
      if (cur === prevVersion) return;
      imageStates.set(prevVersion, app.snapshotImages());
      const saved = imageStates.get(cur);
      // a freshly created version ("+") has no entry yet — it inherits the
      // current images, which become its own on the next switch-away
      if (saved) app.restoreImages(saved);
      prevVersion = cur;
    });

    return () => {
      unsubVersions();
      app.dispose();
      appRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // finish drives the gloss/satin/matte response (ref declared before the
  // preset effect so presets can arm it)
  const skipFinish = useRef(true);

  // preset picks a whole material bundle
  const preset = params.material.preset;
  const skipPreset = useRef(true);
  useEffect(() => {
    if (skipPreset.current) {
      skipPreset.current = false;
      return;
    }
    const bundle = PRESET_VALUES[preset];
    if (!bundle) return;
    // Preset bundles already include their finish response (possibly
    // overridden, e.g. Chrome's tighter roughness). If this write changes
    // `finish`, the finish effect would fire next render and clobber those
    // overrides — skip that one run.
    if (bundle.material.finish !== params.material.finish) skipFinish.current = true;
    dial.setValues({ material: bundle.material, render: bundle.render } as never);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  const finish = params.material.finish;
  useEffect(() => {
    if (skipFinish.current) {
      skipFinish.current = false;
      return;
    }
    const bundle = FINISH_VALUES[finish];
    if (bundle) dial.setValues({ material: bundle } as never);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finish]);

  // push live dial values into the engine on every change
  useEffect(() => {
    appRef.current?.applyParams(params as unknown as HoloParams);
  });

  const app = appRef.current;
  const clothThumb = app?.getClothThumbnail() ?? null;
  const decalThumbs = app?.getDecalThumbnails() ?? [];
  const bumpThumb = app?.getBumpThumbnail() ?? null;
  const clothAnchor = usePanelAnchor('Image as cloth…');
  const decalAnchor = usePanelAnchor('Add image / SVG');
  const bumpAnchor = usePanelAnchor('Upload bump map');

  return (
    <>
      <div
        id="canvas-host"
        ref={hostRef}
        style={{ opacity: assetsReady ? 1 : 0, transition: 'opacity 0.6s ease' }}
      />
      <input
        ref={decalInputRef}
        type="file"
        accept="image/*,.svg"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) loadImageFile(file, (img) => appRef.current?.addDecal(img));
          e.target.value = '';
        }}
      />
      <input
        ref={clothInputRef}
        type="file"
        accept="image/*,.svg"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            loadImageFile(file, (img) => {
              appRef.current?.setClothImage(img);
              dial.setValues({ images: { useImage: true } } as never);
            });
          }
          e.target.value = '';
        }}
      />
      <input
        ref={bumpInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) loadImageFile(file, (img) => appRef.current?.setBumpMap(img));
          e.target.value = '';
        }}
      />
      {clothAnchor && clothThumb &&
        createPortal(
          <ImageChipRow
            thumb={clothThumb}
            label="Cloth image"
            onRemove={() => dial.setValues({ images: { useImage: false } } as never)}
          />,
          clothAnchor,
        )}
      {decalAnchor && decalThumbs.length > 0 &&
        createPortal(
          <div>
            {decalThumbs.map((t, i) => (
              <ImageChipRow
                key={i}
                thumb={t}
                label={`Image ${i + 1}`}
                onRemove={() => appRef.current?.removeDecal(i)}
              />
            ))}
          </div>,
          decalAnchor,
        )}
      {bumpAnchor && bumpThumb &&
        createPortal(
          <ImageChipRow
            thumb={bumpThumb}
            label="Bump map"
            onRemove={() => appRef.current?.setBumpMap(null)}
          />,
          bumpAnchor,
        )}
      <DialRoot position="top-right" defaultOpen productionEnabled />
      {!loaderGone && <Loader percent={loadPercent} hiding={assetsReady} />}
    </>
  );
}
