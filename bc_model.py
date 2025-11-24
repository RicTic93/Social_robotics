import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from env.fauteuil_env import FauteuilEnv
from config import config
from simple_policy_bc import generate_expert_demonstrations
from simple_policy_bc import run_policy

# Define the Behavioral Cloning Model
class BehavioralCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BehavioralCloningModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, states, actions, epochs=50, batch_size=32):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        #loss_fn = nn.CrossEntropyLoss() --> discrete actions
        loss_fn = nn.MSELoss() # --> continuous actions

        states = torch.tensor(states, dtype=torch.float32)
        #print('states', states)
        print("States size", states.size())

        actions = torch.tensor(actions, dtype=torch.float32)
        print("Actions size", actions.size())

        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # In BC continuo, l'accuracy non ha senso (non esistono classi) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Puoi tenerla per debug, ma NON è significativa.
        accuracy_values = []
        loss_values = []  # Initialize loss_values to track loss values

        for epoch in range(epochs):
            total_loss = 0.0
            #correct_predictions = 0.0

            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                predicted_actions = self(batch_states)
                loss = loss_fn(predicted_actions, batch_actions)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                #accuracy_values.append(0)

                # Calculate the number of correct predictions in the current batch
                #correct_predictions += (predicted_actions.argmax(dim=1) == batch_actions).sum().item()

            # Calculate accuracy for the current epoch and store it
            #accuracy = correct_predictions / len(dataset)
            #accuracy_values.append(accuracy)

            # Append the average loss for the current epoch to loss_values
            loss_values.append(total_loss / len(dataloader))

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_values[-1]:.4f}')#, Accuracy: {accuracy * 100:.2f}%')

        # Calculate standard deviation for loss and accuracy
        loss_std = np.std(loss_values)
        #accuracy_std = np.std(accuracy_values)

        return loss_values, accuracy_values, loss_std, #accuracy_std # Return loss and accuracy values
    

# Extract states and actions from expert demonstrations
states = []
actions = []

env = FauteuilEnv(config)

# Generate expert demonstrations
num_demos = 250
expert_demonstrations, episode_rewards = generate_expert_demonstrations(env, num_demos)
#for i, reward in enumerate(episode_rewards):
#    print(f"Episode {i+1}: Total Reward = {reward}")

for demonstration in expert_demonstrations:
    for state, action in demonstration:
        states.append(state)
        actions.append(action)
states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

# Define state_dim and action_dim based on your data
state_dim = states.shape[1]#env.observation_space.shape[0]  
action_dim = actions.shape[1]#env.action_space.shape[0]

# Instantiate the model
model = BehavioralCloningModel(state_dim, action_dim)

# Train the model with the expert demonstration data
loss_values, accuracy_values, loss_std = model.train(states, actions, epochs=50, batch_size=32)

print("Num demo generated:", len(expert_demonstrations))
for i, traj in enumerate(expert_demonstrations):
    print(f"  Demo {i} length = {len(traj)}")

"""# Controlliamo anche stato iniziale e prima action
if len(expert_demonstrations) > 0 and len(expert_demonstrations[0]) > 0:
    print("Esempio primo stato:", expert_demonstrations[0][0][0])
    print("Esempio prima action:", expert_demonstrations[0][0][1])"""

run_policy(env, model, render=True)
time.sleep(0.05)