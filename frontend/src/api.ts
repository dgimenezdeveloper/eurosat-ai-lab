import { classNames, classNamesEs, type Model } from "./types"

const baseUrl = "/api"

export interface PredictionResult {
  predicted_class: string
  predicted_class_es: string
  confidence: number
  probabilities: Record<string, number>
  ranked_probabilities: { className: string; classNameEs: string; probability: number }[]
  inference_time_ms: number
  model_version: string
}

export interface RetrainParams {
  C: number
  max_iter: number
  penalty: string
  solver: string
  scaler_type: string
  max_samples: number
  notes?: string
}

export interface ExperimentRun {
  run_id: string
  timestamp: string
  model_name: string
  hyperparameters: { 
    C: number
    penalty: string
    solver: string
    scaler: string
    max_iter: number 
  }
  metrics: {
    train_accuracy: number
    dev_accuracy: number
    macro_f1: number
    latency_ms: number
    gap: number
  }
  diagnosis: string
  notes: string
}

export const api = {
  async health() {
    try {
      const res = await fetch(`${baseUrl}/health`)
      if (!res.ok) throw new Error("API offline")
      return await res.json()
    } catch {
      return { status: "offline", device: "cpu", model_loaded: false }
    }
  },

  async models(): Promise<Model[]> {
    try {
      const res = await fetch(`${baseUrl}/models`)
      if (!res.ok) throw new Error("Error fetching models")
      const data = await res.json()
      return data.length > 0 ? data : mockModels
    } catch {
      return mockModels
    }
  },

  async experiments(): Promise<ExperimentRun[]> {
    try {
      const res = await fetch(`${baseUrl}/experiments`)
      if (!res.ok) throw new Error("Error fetching experiments")
      return await res.json()
    } catch {
      return []
    }
  },

  async metrics() {
    try {
      const res = await fetch(`${baseUrl}/metrics`)
      if (!res.ok) throw new Error("Error fetching metrics")
      return await res.json()
    } catch {
      return { confusion_matrix: [] }
    }
  },

  async classes(): Promise<string[]> {
    try {
      const res = await fetch(`${baseUrl}/classes`)
      if (!res.ok) throw new Error("Error fetching classes")
      return await res.json()
    } catch {
      return classNames
    }
  },

  async predict(file: File): Promise<PredictionResult> {
    const formData = new FormData()
    formData.append("file", file)

    const res = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      body: formData,
    })

    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || "Error en la inferencia")
    }

    return await res.json()
  },

  async retrainBaseline(params: RetrainParams) {
    const res = await fetch(`${baseUrl}/train/baseline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
    if (!res.ok) throw new Error("Fallo al reentrenar modelo")
    return await res.json()
  }
}

export const mockModels: Model[] = [
  { name: "Baseline (Regresión Logística)", version: "v1.0.0", accuracy: 35.2, f1: 0.3306, latency: 30.37, status: "Evaluated" }
]
