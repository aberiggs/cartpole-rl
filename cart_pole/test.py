# Visualization to show the results of the model

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Load the model from "model.pt"

model = DQN(4, 2).to(device)
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

def select_action(state):
    # t.max(1) will return the largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
    return model(state).max(1).indices.view(1, 1)

env = gym.make('CartPole-v1', render_mode="human")
state, info = env.reset()

count = 0
total_count = 0
episode = 0
while episode < 5:
    count += 1
    total_count += 1

    env.render()
    input = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(input).item()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Lasted for {count} steps" + (" (truncated)" if truncated else ""))
        env.reset()
        count = 0
        episode += 1
    else:
        state = observation
    
        
print(f"-----\nAveraged {total_count/10} steps\n-----")
env.close()
