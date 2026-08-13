import torch
import torch.nn as nn

# Estructura básica de una Red Neuronal Simple
class RedNeuronalBase(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super(RedNeuronalBase, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    print("Iniciando entorno de Redes Neuronales...")
    modelo = RedNeuronalBase()
    datos_prueba = torch.randn(5, 10)
    predicciones = modelo(datos_prueba)
    print("Prueba de Forward Pass exitosa. Salida shape:", predicciones.shape)
