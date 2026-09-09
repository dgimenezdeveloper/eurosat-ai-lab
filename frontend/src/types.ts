export type PageKey = "Dashboard" | "Classification" | "Models" | "Analytics" | "Dataset" | "Error Analysis" | "VAE Lab" | "Methodology"

export const classNames = [
  "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
  "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

export const classNamesEs: Record<string, string> = {
  "AnnualCrop": "Cultivo Anual",
  "Forest": "Bosque",
  "HerbaceousVegetation": "Vegetación Herbácea",
  "Highway": "Autopista",
  "Industrial": "Zona Industrial",
  "Pasture": "Pastizal",
  "PermanentCrop": "Cultivo Permanente",
  "Residential": "Zona Residencial",
  "River": "Río",
  "SeaLake": "Mar o Lago"
}

export type Model = { name: string; version: string; accuracy: number; f1: number; latency: number; status: string }
export type Experiment = { name: string; model: string; dataset: string; accuracy: number; date: string; status: "Completed" | "Running" }
