import { StrictMode, useState, useEffect } from "react"
import { createRoot } from "react-dom/client"
import { 
  BarChart3, Boxes, BrainCircuit, CircleHelp, Database, 
  FileImage, LayoutDashboard, Menu, Microscope, Network, 
  Settings2, Sparkles, UploadCloud, AlertCircle, Loader2, ArrowRight
} from "lucide-react"
import { classNames, classNamesEs, type PageKey, type Model } from "./types"
import { api, type PredictionResult } from "./api"
import { ModelTuningPanel } from "./components/ModelTuningPanel"
import "./styles.css"

const navigation: { label: PageKey; titleEs: string; icon: typeof LayoutDashboard; group: string }[] = [
  { label: "Dashboard", titleEs: "Panel General", icon: LayoutDashboard, group: "Espacio de Trabajo" },
  { label: "Classification", titleEs: "Clasificación en Vivo", icon: FileImage, group: "Espacio de Trabajo" },
  { label: "Models", titleEs: "Modelos y MLOps", icon: Boxes, group: "Espacio de Trabajo" },
  { label: "Analytics", titleEs: "Analítica y Métricas", icon: BarChart3, group: "Espacio de Trabajo" },
  { label: "Dataset", titleEs: "Explorador EuroSAT", icon: Database, group: "Espacio de Trabajo" },
  { label: "Error Analysis", titleEs: "Análisis de Errores", icon: Microscope, group: "Espacio de Trabajo" },
  { label: "VAE Lab", titleEs: "Laboratorio VAE", icon: Network, group: "Investigación" },
  { label: "Methodology", titleEs: "Metodología Cátedra", icon: CircleHelp, group: "Investigación" },
]

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) { 
  return <section className={`card ${className}`}>{children}</section> 
}

function SectionHead({ title, action, onAction }: { title: string; action?: string; onAction?: () => void }) { 
  return (
    <div className="section-head">
      <h2>{title}</h2>
      {action && (
        <button className="link" onClick={onAction} style={{ display: "inline-flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
          {action} <ArrowRight size={12} />
        </button>
      )}
    </div>
  ) 
}

function Dashboard({ models, onNavigate }: { models: Model[]; onNavigate: (page: PageKey) => void }) {
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([])
  const bestModel = models[0] || { accuracy: 35.2, f1: 0.3306, latency: 30.37, name: "Baseline (Regresión Logística)" }

  useEffect(() => {
    api.metrics().then((data) => {
      if (data.confusion_matrix && data.confusion_matrix.length > 0) {
        setConfusionMatrix(data.confusion_matrix)
      }
    })
  }, [])

  return (
    <>
      <p className="eyebrow">Visión General / Ecosistema Productivo</p>
      <h1>EuroSAT AI Lab</h1>
      <p className="subtitle">Laboratorio de Inteligencia Artificial sobre Sentinel-2 (UNaB 2026 - Lic. Pablo Moreira)</p>
      
      <div className="grid kpis">
        <Card>
          <p className="card-title">Métrica Principal (Macro F1)</p>
          <div className="metric">{(bestModel.f1 * 100).toFixed(1)}%<span>Conjunto Dev</span></div>
          <p className="metric-note">Evaluación armónica multiclase (Clase 1 y 2)</p>
        </Card>
        <Card>
          <p className="card-title">Exactitud Global (Accuracy)</p>
          <div className="metric">{bestModel.accuracy}%<span>Conjunto Dev</span></div>
          <p className="metric-note">Diagnóstico: Sesgo Alto (Underfitting)</p>
        </Card>
        <Card>
          <p className="card-title">Muestras del Dataset</p>
          <div className="metric">27,000<span>Imágenes RGB</span></div>
          <p className="metric-note">Partición 80% Train, 10% Dev, 10% Test</p>
        </Card>
        <Card>
          <p className="card-title">Latencia de Inferencia</p>
          <div className="metric">{bestModel.latency}ms<span>CPU</span></div>
          <p className="metric-note">Métrica de satisfacción ≤ 50ms</p>
        </Card>
      </div>

      <div className="grid two">
        <Card>
          <SectionHead 
            title="Diagnóstico Clínico: Sesgo Alto en el Baseline" 
            action="Ver Metodología" 
            onAction={() => onNavigate("Methodology")} 
          />
          <div style={{ padding: "8px 0", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
            <p style={{ margin: "0 0 8px" }}>
              El <strong>Baseline de Regresión Logística</strong> alcanza un <strong>{bestModel.accuracy}%</strong> de exactitud sobre píxeles aplanados (12.288 entradas).
            </p>
            <p style={{ margin: "0 0 8px" }}>
              Al ser un modelo lineal, no logra capturar fronteras de decisión curvas ni patrones espaciales de los parches satelitales.
            </p>
            <div style={{ background: "#112233", padding: 10, borderRadius: 6, border: "1px solid #1e3a55", color: "#38bdf8" }}>
              <strong>Acción Pedagógica (Etapa 2):</strong> Pasar a un Perceptrón Multicapa (MLP) con funciones no lineales (GELU/ReLU) para reducir el sesgo estructural.
            </div>
          </div>
        </Card>
        
        <Card>
          <SectionHead 
            title="Distribución de Clases de Uso de Suelo" 
            action="Explorar Dataset" 
            onAction={() => onNavigate("Dataset")} 
          />
          <div className="class-bars">
            {classNames.slice(0, 6).map((name, i) => (
              <div className="class-row" key={name}>
                <span style={{ fontSize: 11 }}>{classNamesEs[name] || name}</span>
                <div className="track"><div className="fill" style={{ width: `${[100, 86, 77, 66, 56, 49][i]}%` }} /></div>
                <span>{[2714, 2340, 2083, 1792, 1517, 1328][i]}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid two">
        <Card>
          <SectionHead 
            title="Registro de Modelos Evaluados" 
            action="Ajustar Hiperparámetros" 
            onAction={() => onNavigate("Models")} 
          />
          <table className="table">
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Exactitud</th>
                <th>Macro F1</th>
                <th>Latencia</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td><strong>{m.name}</strong><br /><span style={{ color:"#5f7892", fontSize:10 }}>{m.version}</span></td>
                  <td>{m.accuracy}%</td>
                  <td>{m.f1}</td>
                  <td>{m.latency}ms</td>
                  <td><span className="pill">{m.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card>
          <SectionHead title="Matriz de Confusión en Español (Conjunto Dev)" />
          {confusionMatrix.length > 0 ? (
            <div style={{ overflowX: "auto" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(10, 1fr)", gap: 2, minWidth: 260 }}>
                {confusionMatrix.flatMap((row, rIdx) => 
                  row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const maxVal = Math.max(...row, 1);
                    const intensity = Math.min(val / maxVal, 1);
                    return (
                      <div 
                        key={`${rIdx}-${cIdx}`} 
                        title={`Real: ${classNamesEs[classNames[rIdx]]} | Predicción: ${classNamesEs[classNames[cIdx]]} | Cantidad: ${val}`}
                        style={{
                          aspectRatio: "1",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 9,
                          fontWeight: isDiagonal ? "bold" : "normal",
                          background: isDiagonal 
                            ? `rgba(37, 212, 197, ${0.25 + intensity * 0.75})` 
                            : `rgba(99, 102, 241, ${intensity * 0.6})`,
                          color: intensity > 0.4 ? "#fff" : "#94a3b8",
                          borderRadius: 2
                        }}
                      >
                        {val > 0 ? val : ""}
                      </div>
                    )
                  })
                )}
              </div>
              <div className="axis" style={{ marginTop: 8 }}>
                <span>Eje Vertical: Clase Real</span>
                <span>Eje Horizontal: Predicción</span>
              </div>
            </div>
          ) : (
            <p style={{ color: "#64748b", fontSize: 12 }}>Cargando matriz del modelo...</p>
          )}
        </Card>
      </div>
    </>
  )
}

function Classification() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewDataUrl, setPreviewDataUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setResult(null)
      setErrorMsg(null)

      const reader = new FileReader()
      reader.onload = (event) => {
        setPreviewDataUrl(event.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleClassify = async () => {
    if (!selectedFile) return
    setLoading(true)
    setErrorMsg(null)
    try {
      const data = await api.predict(selectedFile)
      setResult(data)
    } catch (err: any) {
      setErrorMsg(err.message || "Error al conectar con la API de FastAPI")
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <p className="eyebrow">Inferencia en Tiempo Real / Sentinel-2</p>
      <h1>Clasificador de Cobertura Terrestre</h1>
      <p className="subtitle">Sube un parche satelital de 64x64 píxeles para clasificarlo con el modelo en FastAPI.</p>

      <div className="grid two">
        <Card>
          <SectionHead title="1. Subir Imagen Satelital" />
          <div style={{ border:"1px dashed #2b5b72", borderRadius:8, minHeight:240, display:"grid", placeItems:"center", textAlign:"center", background:"#0b2033", padding:20 }}>
            {previewDataUrl ? (
              <div>
                <img 
                  src={previewDataUrl} 
                  alt="Vista previa" 
                  style={{ width: 96, height: 96, borderRadius: 8, objectFit: "cover", margin: "0 auto 10px", border: "2px solid #25d4c5" }} 
                />
                <p style={{ fontSize:12, color:"#dbe9f7", margin:0 }}>{selectedFile?.name}</p>
                <label className="link" style={{ cursor:"pointer", display:"block", marginTop:8 }}>
                  Cambiar imagen
                  <input type="file" hidden accept="image/*" onChange={handleFileChange} />
                </label>
              </div>
            ) : (
              <div>
                <UploadCloud size={32} color="#25d4c5" style={{ margin:"0 auto 10px" }} />
                <p style={{ fontSize:13, color:"#dbe9f7", margin:"0 0 4px" }}>Selecciona una imagen de EuroSAT</p>
                <p style={{ color:"#68829a", fontSize:11, margin:"0 0 12px" }}>Formatos: JPG, PNG, TIF (64×64 px)</p>
                <label className="link" style={{ cursor:"pointer", padding:"6px 12px", background:"#17364b", borderRadius:6 }}>
                  Examinar archivos
                  <input type="file" hidden accept="image/*" onChange={handleFileChange} />
                </label>
              </div>
            )}
          </div>

          <button 
            className="primary" 
            onClick={handleClassify} 
            disabled={!selectedFile || loading}
            style={{ 
              marginTop:16, width:"100%", padding:12, borderRadius:6, border:0, 
              background: selectedFile && !loading ? "#24c8bb" : "#244052", 
              color: selectedFile && !loading ? "#062019" : "#7890a0", 
              fontWeight:700, display:"flex", justifyContent:"center", alignItems:"center", gap:8 
            }}
          >
            {loading && <Loader2 size={16} className="animate-spin" />}
            {loading ? "Procesando en FastAPI..." : "Clasificar Cobertura"}
          </button>

          {errorMsg && (
            <div style={{ marginTop:12, padding:10, background:"#38161a", border:"1px solid #73222a", borderRadius:6, color:"#fca5a5", fontSize:11, display:"flex", gap:8, alignItems:"center" }}>
              <AlertCircle size={16} />
              <span>{errorMsg}</span>
            </div>
          )}
        </Card>

        <Card>
          <SectionHead title="2. Resultado de la Inferencia" />
          <div style={{ minHeight:240, display:"grid", placeItems:"center", textAlign:"center" }}>
            {result ? (
              <div style={{ width:"100%", textAlign:"left" }}>
                <div style={{ color:"#25d4c5", fontFamily:"DM Mono", fontSize:11, textTransform:"uppercase" }}>
                  CLASE PREDICHA
                </div>
                <div style={{ fontSize:28, fontWeight:800, margin:"4px 0", color:"#f0fdf4" }}>
                  {result.predicted_class_es}
                </div>
                <div style={{ color:"#64748b", fontSize:11, marginBottom:6 }}>
                  Etiqueta técnica: {result.predicted_class}
                </div>
                <div style={{ color:"#54d4ab", fontSize:13, marginBottom:12 }}>
                  {result.confidence}% de certeza · {result.inference_time_ms} ms de latencia
                </div>

                <p style={{ fontSize:11, color:"#7890a0", marginBottom:8 }}>Probabilidad por Categoría (Top 5):</p>
                <div className="class-bars">
                  {result.ranked_probabilities.slice(0, 5).map((item) => (
                    <div className="class-row" key={item.className}>
                      <span style={{ fontSize:11 }}>{item.classNameEs}</span>
                      <div className="track">
                        <div className="fill" style={{ width: `${Math.max(item.probability, 2)}%` }} />
                      </div>
                      <span style={{ fontSize:11 }}>{item.probability}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <BrainCircuit size={36} color="#55738e" style={{ margin:"0 auto 10px" }} />
                <p style={{ color:"#7088a0", fontSize:12, margin:0 }}>
                  Sube una foto satelital y haz clic en "Clasificar" para consultar a la red.
                </p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </>
  )
}

function DatasetPage() {
  return (
    <>
      <p className="eyebrow">Academic Research / Dataset Explorer</p>
      <h1>Explorador de EuroSAT</h1>
      <p className="subtitle">Catálogo de 27.000 imágenes Sentinel-2 divididas en 10 clases balanceadas.</p>
      
      <div className="grid two">
        <Card>
          <SectionHead title="Resumen de Metadatos" />
          <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.8 }}>
            <p><strong>Resolución Espacial:</strong> 64 × 64 píxeles (RGB)</p>
            <p><strong>Satélite:</strong> ESA Sentinel-2 (Órbita polar terrestre)</p>
            <p><strong>Total de Muestras:</strong> 27.000 parches etiquetados</p>
            <p><strong>Partición Estratificada:</strong></p>
            <ul>
              <li>Entrenamiento (Train): 21.600 imágenes (80%)</li>
              <li>Validación (Dev): 2.700 imágenes (10%)</li>
              <li>Evaluación Final (Test): 2.700 imágenes (10%)</li>
            </ul>
          </div>
        </Card>

        <Card>
          <SectionHead title="Las 10 Clases de Cobertura Terrestre (Traducción Oficial)" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {classNames.map((c) => (
              <div key={c} style={{ background: "#1e3a55", padding: "6px 10px", borderRadius: 6, fontSize: 11 }}>
                <strong style={{ color: "#38bdf8" }}>{classNamesEs[c] || c}</strong>
                <span style={{ color: "#64748b", display: "block", fontSize: 10 }}>({c})</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </>
  )
}

function GenericPage({ page, models, onRefresh }: { page: PageKey; models: Model[]; onRefresh: () => void }) {
  if (page === "Dataset") return <DatasetPage />

  if (page === "Models") {
    return (
      <>
        <p className="eyebrow">Investigación Académica / Registro de Modelos y MLOps</p>
        <h1>Modelos y Experimentación</h1>
        <p className="subtitle">Ajuste de hiperparámetros y registro de bitácora MLOps (UNaB 2026).</p>

        <ModelTuningPanel onRetrained={onRefresh} />

        <Card>
          <SectionHead title="Modelos Registrados en artifacts/metrics/models_summary.json" />
          <table className="table">
            <thead>
              <tr>
                <th>Arquitectura</th>
                <th>Versión / Parámetros</th>
                <th>Exactitud Dev</th>
                <th>Macro F1</th>
                <th>Latencia</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td><strong>{m.name}</strong></td>
                  <td>{m.version}</td>
                  <td>{m.accuracy}%</td>
                  <td>{m.f1}</td>
                  <td>{m.latency}ms</td>
                  <td><span className="pill">{m.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </>
    )
  }

  return (
    <>
      <p className="eyebrow">Investigación Académica / {page}</p>
      <h1>{page}</h1>
      <p className="subtitle">Sección académica sincronizada con los Notebooks de la cátedra.</p>

      {page === "Methodology" && (
        <Card>
          <SectionHead title="Las 5 Etapas del Trabajo Práctico (UNaB 2026)" />
          <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.8 }}>
            <p><strong>Etapa 1:</strong> EDA exhaustivo + Partición 80/10/10 + Baseline Lineal (Regresión Logística).</p>
            <p><strong>Etapa 2:</strong> Análisis cualitativo de 50 errores + Red Neuronal Densa (MLP de 3+ capas con GELU, BatchNorm y Dropout).</p>
            <p><strong>Etapa 3:</strong> Redes Convolucionales (CNN de 3 bloques) + Data Augmentation + Reducción de varianza.</p>
            <p><strong>Etapa 4:</strong> Autoencoder Variacional (VAE) y exploración del espacio latente.</p>
            <p><strong>Etapa 5:</strong> Tabla maestra comparativa y evaluación final en el conjunto Test (una única vez).</p>
          </div>
        </Card>
      )}

      {page !== "Methodology" && (
        <Card>
          <div style={{ display:"flex", alignItems:"center", gap:14, padding:"16px 0" }}>
            <Sparkles color="#25d4c5" />
            <div>
              <h2 style={{ margin:0, fontSize:15 }}>Módulo de {page} Activo</h2>
              <p style={{ color:"#7189a2", fontSize:12, margin:"4px 0 0" }}>
                Este módulo se alimenta de los artefactos generados al ejecutar los Notebooks en JupyterLab.
              </p>
            </div>
          </div>
        </Card>
      )}
    </>
  )
}

export default function App() {
  const [page, setPage] = useState<PageKey>("Dashboard")
  const [mobileOpen, setMobileOpen] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [apiOnline, setApiOnline] = useState<boolean>(false)

  const reloadData = () => {
    api.health().then((h) => setApiOnline(h.status === "healthy"))
    api.models().then((m) => setModels(m))
  }

  useEffect(() => {
    reloadData()
  }, [])

  return (
    <div className="app">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">ES</div>
          <div>EUROSAT <small>AI RESEARCH LAB</small></div>
        </div>

        {["Espacio de Trabajo", "Investigación"].map((group) => (
          <div key={group}>
            <div className="nav-label">{group}</div>
            {navigation
              .filter((n) => n.group === group)
              .map(({ label, titleEs, icon: Icon }) => (
                <button
                  className={`nav-item ${page === label ? "active" : ""}`}
                  key={label}
                  onClick={() => { setPage(label); setMobileOpen(false) }}
                >
                  <Icon />
                  {titleEs}
                </button>
              ))}
          </div>
        ))}

        <div className="sidebar-footer">
          <span className="dot" style={{ background: apiOnline ? "#33d69a" : "#f87171" }} />
          <span>{apiOnline ? "API Conectada" : "API Desconectada"}</span>
          <Settings2 size={14} style={{ marginLeft:"auto" }} />
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="crumb">
            <button className="icon-btn mobile-menu" onClick={() => setMobileOpen(!mobileOpen)}>
              <Menu size={18} />
            </button>
            <span style={{ marginLeft: 10 }}>UNaB 2026 /</span>
            <strong>{navigation.find(n => n.label === page)?.titleEs || page}</strong>
          </div>
          <div className="top-actions">
            <span className="status" style={{ color: apiOnline ? "#55d4ae" : "#f87171", borderColor: apiOnline ? "#1e604f" : "#7f1d1d" }}>
              ● {apiOnline ? "API FASTAPI ONLINE" : "API OFFLINE"}
            </span>
            <div className="avatar">IA</div>
          </div>
        </header>

        <div className="content">
          {page === "Dashboard" && <Dashboard models={models} onNavigate={(p) => setPage(p)} />}
          {page === "Classification" && <Classification />}
          {page !== "Dashboard" && page !== "Classification" && (
            <GenericPage page={page} models={models} onRefresh={reloadData} />
          )}
        </div>
      </main>
    </div>
  )
}

const root = document.getElementById("root")
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>
  )
}
