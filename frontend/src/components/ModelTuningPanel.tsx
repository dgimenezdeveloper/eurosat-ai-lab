import { useState, useEffect } from "react"
import { api, type ExperimentRun } from "../api"
import { Sliders, Play, Loader2, History } from "lucide-react"

export function ModelTuningPanel({ onRetrained }: { onRetrained: () => void }) {
  const [cParam, setCParam] = useState<number>(1.0)
  const [maxIter, setMaxIter] = useState<number>(200)
  const [penalty, setPenalty] = useState<string>("l2")
  const [solver, setSolver] = useState<string>("lbfgs")
  const [scalerType, setScalerType] = useState<string>("standard")
  const [samples, setSamples] = useState<number>(3000)
  const [notes, setNotes] = useState<string>("Prueba de hiperparámetros")
  const [loading, setLoading] = useState<boolean>(false)
  const [resultMsg, setResultMsg] = useState<string | null>(null)
  const [history, setHistory] = useState<ExperimentRun[]>([])

  const loadHistory = () => {
    api.experiments().then((data) => setHistory(data))
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const handleRetrain = async () => {
    setLoading(true)
    setResultMsg(null)
    try {
      const res = await api.retrainBaseline({
        C: cParam,
        max_iter: maxIter,
        penalty: penalty,
        solver: solver,
        scaler_type: scalerType,
        max_samples: samples,
        notes: notes
      })
      setResultMsg(`✓ ${res.run_id} registrado: Exactitud Dev = ${res.accuracy}% | Macro F1 = ${res.f1} | ${res.diagnosis}`)
      loadHistory()
      onRetrained()
    } catch (e: any) {
      setResultMsg(`Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: "grid", gap: 16, marginBottom: 20 }}>
      <div style={{ background: "rgba(13,31,51,0.85)", border: "1px solid #1b3a55", borderRadius: 10, padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <Sliders size={18} color="#25d4c5" />
          <h3 style={{ margin: 0, fontSize: 15, color: "#f0fdf4" }}>Laboratorio de Hiperparámetros de la Cátedra (Baseline)</h3>
        </div>
        <p style={{ fontSize: 12, color: "#94a3b8", margin: "0 0 16px" }}>
          Ajusta regularización (L1/L2), escalado y algoritmos de optimización para diagnosticar sesgo vs. varianza.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14, marginBottom: 16 }}>
          {/* Slider C */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Parámetro C (Inverso L2/L1): <strong>{cParam}</strong>
            </label>
            <input 
              type="range" min="0.01" max="5" step="0.05" value={cParam}
              onChange={(e) => setCParam(parseFloat(e.target.value))}
              style={{ width: "100%" }}
            />
            <small style={{ color: "#64748b", fontSize: 10 }}>C bajo = más penalización (L2/L1)</small>
          </div>

          {/* Slider Iteraciones */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Máx. Iteraciones: <strong>{maxIter}</strong>
            </label>
            <input 
              type="range" min="50" max="500" step="25" value={maxIter}
              onChange={(e) => setMaxIter(parseInt(e.target.value))}
              style={{ width: "100%" }}
            />
            <small style={{ color: "#64748b", fontSize: 10 }}>Evita el ConvergenceWarning</small>
          </div>

          {/* Selector Penalización (L1 vs L2) */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Tipo de Regularización:
            </label>
            <select 
              value={penalty} 
              onChange={(e) => setPenalty(e.target.value)}
              style={{ width: "100%", background: "#0b2033", border: "1px solid #1b3a55", color: "#e2e8f0", padding: "6px", borderRadius: 6, fontSize: 11 }}
            >
              <option value="l2">L2 - Ridge (Pesos pequeños)</option>
              <option value="l1">L1 - Lasso (Sparsidad)</option>
            </select>
          </div>

          {/* Selector Escalador */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Escalado de Datos:
            </label>
            <select 
              value={scalerType} 
              onChange={(e) => setScalerType(e.target.value)}
              style={{ width: "100%", background: "#0b2033", border: "1px solid #1b3a55", color: "#e2e8f0", padding: "6px", borderRadius: 6, fontSize: 11 }}
            >
              <option value="standard">StandardScaler (Media 0, Var 1)</option>
              <option value="minmax">MinMaxScaler (Rango [0, 1])</option>
            </select>
          </div>

          {/* Selector Solver */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Algoritmo Solver:
            </label>
            <select 
              value={solver} 
              onChange={(e) => setSolver(e.target.value)}
              style={{ width: "100%", background: "#0b2033", border: "1px solid #1b3a55", color: "#e2e8f0", padding: "6px", borderRadius: 6, fontSize: 11 }}
            >
              <option value="lbfgs">lbfgs (Rápido, L2)</option>
              <option value="saga">saga (Escalable, L1/L2)</option>
              <option value="liblinear">liblinear (Clásico)</option>
            </select>
          </div>

          {/* Muestras */}
          <div>
            <label style={{ fontSize: 11, color: "#38bdf8", display: "block", marginBottom: 4 }}>
              Muestras: <strong>{samples}</strong>
            </label>
            <input 
              type="range" min="1000" max="6000" step="500" value={samples}
              onChange={(e) => setSamples(parseInt(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>
        </div>

        <div style={{ marginBottom: 16 }}>
          <input 
            type="text" value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Hipótesis: Ej. Probando regularización L1 con SAGA para forzar coeficientes nulos..."
            style={{ width: "100%", background: "#0b2033", border: "1px solid #1b3a55", color: "#e2e8f0", padding: "8px 12px", borderRadius: 6, fontSize: 11 }}
          />
        </div>

        <button
          onClick={handleRetrain}
          disabled={loading}
          style={{
            background: loading ? "#1e3a55" : "#25d4c5",
            color: "#07111f",
            fontWeight: 700,
            border: 0,
            borderRadius: 6,
            padding: "10px 20px",
            cursor: loading ? "not-allowed" : "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12
          }}
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          {loading ? "Reentrenando en CPU..." : "Ejecutar y Registrar Experimento"}
        </button>

        {resultMsg && (
          <div style={{ marginTop: 12, padding: 10, background: "#0f2922", border: "1px solid #166534", borderRadius: 6, color: "#86efac", fontSize: 11 }}>
            {resultMsg}
          </div>
        )}
      </div>

      {/* Bitácora Histórica */}
      <div style={{ background: "rgba(13,31,51,0.85)", border: "1px solid #1b3a55", borderRadius: 10, padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <History size={18} color="#38bdf8" />
          <h3 style={{ margin: 0, fontSize: 15, color: "#f0fdf4" }}>Bitácora de Experimentos (MLOps Ledger)</h3>
        </div>

        {history.length > 0 ? (
          <table className="table">
            <thead>
              <tr>
                <th>ID / Fecha</th>
                <th>Hiperparámetros (C, Norma, Solver, Escala)</th>
                <th>Train Acc</th>
                <th>Dev Acc</th>
                <th>Gap</th>
                <th>Macro F1</th>
                <th>Diagnóstico Clínico</th>
              </tr>
            </thead>
            <tbody>
              {history.map((run) => (
                <tr key={run.run_id}>
                  <td><strong>{run.run_id}</strong><br /><span style={{ color: "#64748b", fontSize: 10 }}>{run.timestamp}</span></td>
                  <td>
                    C={run.hyperparameters.C} · {run.hyperparameters.penalty?.toUpperCase()} · {run.hyperparameters.solver} · {run.hyperparameters.scaler}
                  </td>
                  <td>{run.metrics.train_accuracy}%</td>
                  <td style={{ color: "#38bdf8", fontWeight: 600 }}>{run.metrics.dev_accuracy}%</td>
                  <td>{run.metrics.gap}%</td>
                  <td style={{ color: "#25d4c5", fontWeight: 600 }}>{run.metrics.macro_f1}</td>
                  <td>
                    <span style={{ 
                      fontSize: 10, 
                      padding: "3px 8px", 
                      borderRadius: 4, 
                      background: run.metrics.dev_accuracy < 40 ? "rgba(239, 68, 68, 0.15)" : "rgba(34, 197, 94, 0.15)",
                      color: run.metrics.dev_accuracy < 40 ? "#fca5a5" : "#86efac",
                      border: run.metrics.dev_accuracy < 40 ? "1px solid rgba(239, 68, 68, 0.3)" : "1px solid rgba(34, 197, 94, 0.3)"
                    }}>
                      {run.diagnosis}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: "#64748b", fontSize: 12 }}>No hay experimentos previos registrados.</p>
        )}
      </div>
    </div>
  )
}
