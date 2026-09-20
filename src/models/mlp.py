import torch
import torch.nn as nn

class EurosatMLP(nn.Module):
    """
    Perceptrón Multicapa (MLP Profundo) para EuroSAT (64x64x3 = 12.288 features).
    Diseñado para contrastar el impacto directo de la regularización (Etapa 2).
    """
    def __init__(self, input_dim: int = 12288, num_classes: int = 10, regularized: bool = True, dropout_rate: float = 0.35):
        super().__init__()
        self.regularized = regularized

        if regularized:
            self.network = nn.Sequential(
                # Capa Oculta 1: 12288 -> 512
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(dropout_rate),

                # Capa Oculta 2: 512 -> 256
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout_rate),

                # Capa Oculta 3: 256 -> 128
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(dropout_rate),

                # Capa de Salida: 128 -> 10
                nn.Linear(128, num_classes)
            )
        else:
            self.network = nn.Sequential(
                # Capa Oculta 1 (Sin regularizar)
                nn.Linear(input_dim, 512),
                nn.ReLU(),

                # Capa Oculta 2
                nn.Linear(512, 256),
                nn.ReLU(),

                # Capa Oculta 3
                nn.Linear(256, 128),
                nn.ReLU(),

                # Capa de Salida
                nn.Linear(128, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)