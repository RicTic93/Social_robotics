import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Behavioral Cloning Model
class BehavioralCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BehavioralCloningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    # Passage dans le réseau
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Entraînement du modèle avec les démonstrations expertes
    def train(self, states, actions, epochs, batch_size):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        #loss_fn = nn.CrossEntropyLoss() --> discrete actions
        loss_fn = nn.MSELoss() # --> continuous actions

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_values = []  
        # Training loop per epoch
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                predicted_actions = self(batch_states)
                # Calcule la perte entre les actions prédites et les actions expertes
                loss = loss_fn(predicted_actions, batch_actions)
                # Rétropropagation et mise à jour des poids
                loss.backward()
                # Mise à jour des poids du modèle
                optimizer.step()
                total_loss += loss.item()
            loss_values.append(total_loss / len(dataloader))

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_values[-1]:.4f}')
        # Calcule deviation standard des pertes
        loss_std = np.std(loss_values)

        return loss_values, loss_std
