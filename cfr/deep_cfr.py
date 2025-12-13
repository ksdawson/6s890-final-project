import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mccfr import MonteCarloCFR
import numpy as np
from utils import state_space_size
from game import StochasticGame

class RegretNetwork(nn.Module):
    def __init__(self, num_actions, input_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, input_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

def flatten(item):
    if isinstance(item, (tuple, list)):
        flat_list = []
        for x in item:
            flat_list.extend(flatten(x))
        return flat_list
    else:
        return [item]

def infoset_to_tensor(infoset, num_states, max_history_length, state_len, num_players=2, device='cpu'):
    player, player_s, player_history = infoset
    
    # Flatten all inputs just to be safe
    flat_player_s = flatten(player_s)
    flat_history = [flatten(h) for h in player_history]

    # Calculate vector size
    state_block_size = num_states * state_len
    total_size = num_players + state_block_size + (max_history_length * state_block_size)
    
    # Setup features
    features = np.zeros(total_size, dtype=np.float32)
    current_idx = 0
    
    # Encode player
    p_idx = player - 1 
    if 0 <= p_idx < num_players:
        features[current_idx + p_idx] = 1.0
    current_idx += num_players
    
    # Encode current local state
    for i, val in enumerate(flat_player_s):
        if i >= state_len: break 
        if isinstance(val, int) and 0 <= val < num_states:
            features[current_idx + (i * num_states) + val] = 1.0       
    current_idx += state_block_size
    
    # Encode history sequence
    for t, state_list in enumerate(flat_history):
        if t >= max_history_length: break
        
        time_offset = t * state_block_size
        
        for i, val in enumerate(state_list):
            if i >= state_len: break
            
            if isinstance(val, int) and 0 <= val < num_states:
                features[current_idx + time_offset + (i * num_states) + val] = 1.0
            
    # Convert to tensor
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

class DeepCFR:
    def __init__(self, game):
        # Get game info
        self.game = game
        self.num_states = max(state_space_size(*game.p1), state_space_size(*game.p2))
        self.state_len = max(self.game.p1[1], self.game.p2[1]) + 1
        self.input_size = 2 + self.num_states * self.state_len + (self.game.max_depth * self.num_states * self.state_len)

        # Initialize the solver
        self.mccfr = MonteCarloCFR(game, deep_cfr=True, model=self.model)

        # Value Network predicts advantages/regrets (Unbounded)
        self.value_net = RegretNetwork(self.num_states, self.input_size) # states is an upper bound on actions
        # Policy Network predicts the average strategy (Probability distribution)
        self.policy_net = PolicyNetwork(self.num_states, self.input_size)
        
        # Define Optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Define Loss Function (MSE is standard for Deep CFR)
        self.loss_fn = nn.MSELoss()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_net.to(self.device)
        self.policy_net.to(self.device)

    def model(self, infoset):
        infoset_tensor = infoset_to_tensor(infoset, self.num_states, self.game.max_depth, self.state_len, device=self.device)
        output = self.value_net(infoset_tensor)
        return output.detach().cpu().numpy()[0]

    def train(self, iters, traversals_per_iter=1000, batch_size=1024):
        for i in range(iters):
            # Collect data
            self.mccfr.train(traversals_per_iter) 

            # Get data
            regret_data = self.mccfr.regret_samples
            policy_data = self.mccfr.policy_samples

            # Train regret network
            self.train_network(
                self.value_net, 
                self.value_optimizer, 
                regret_data, 
                batch_size
            )
            
            # Clear regret samples to prevent mixing old (bad) data with new data
            self.mccfr.regret_samples = []

            # Train policy network
            self.train_network(
                self.policy_net, 
                self.policy_optimizer, 
                policy_data, 
                batch_size,
                is_policy=True
            )
            
            # Optional: Clear policy samples if memory is an issue
            # self.mccfr.policy_samples = []

    def train_network(self, model, optimizer, data, batch_size, is_policy=False):
        if not data:
            return

        # Unzip the raw data
        raw_inputs, raw_targets = zip(*data)

        # Convert faw infosets to feature tensors
        processed_inputs = []
        for raw_info in raw_inputs:
            tensor = infoset_to_tensor(
                raw_info, 
                self.num_states, 
                self.game.max_depth, 
                self.state_len, 
                device='cpu'
            )
            processed_inputs.append(tensor)
        
        # Stack inputs: Shape [Batch_Size, Input_Size]
        inputs = torch.cat(processed_inputs).to(self.device)

        # Process Targets: PAD to fixed length
        # The network always outputs 'self.num_states' values. 
        # We must make sure our target matches that shape.
        padded_targets = []
        for target in raw_targets:
            # target is a list like [10.5, -2.0]
            current_len = len(target)
            
            # Calculate how many zeros we need
            pad_len = self.num_states - current_len
            
            if pad_len > 0:
                # Add zeros to the end
                padded_target = list(target) + [0.0] * pad_len
            else:
                # Truncate if somehow larger (safety check)
                padded_target = target[:self.num_states]
                
            padded_targets.append(padded_target)

        # Now all rows are the same length -> Safe to convert
        targets = torch.tensor(np.array(padded_targets), dtype=torch.float32).to(self.device)
        
        # Create DataLoader & Train
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        epochs = 2 
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                if is_policy:
                    outputs = F.softmax(outputs, dim=1)
                
                # Calculate Loss
                loss = self.loss_fn(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

if __name__ == '__main__':
    p1, p2 = (1, 2), (1, 2)
    game = StochasticGame(p1, p2, depth=3, t=None, transition=None)
    deepcfr = DeepCFR(game)
    deepcfr.train(5, 100)