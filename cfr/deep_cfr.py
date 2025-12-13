import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mccfr import MonteCarloCFR
import numpy as np
from utils import state_space_size, time_func
from game import StochasticGame
from transition_func import load_demand, load_graph, load_zones, Transition

class RegretNetwork(nn.Module):
    def __init__(self, num_actions, num_input_slots, num_embeddings, embedding_dim=16, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.flat_input_size = num_input_slots * embedding_dim
        
        self.fc1 = nn.Linear(self.flat_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_input_slots, num_embeddings, embedding_dim=16, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.flat_input_size = num_input_slots * embedding_dim

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)

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
    
    # Flatten inputs
    flat_player_s = flatten(player_s)
    flat_history = [flatten(h) for h in player_history]

    # Calculate number of slots
    total_slots = 1 + state_len + (max_history_length * state_len)
    
    # Create indices array
    indices = np.zeros(total_slots, dtype=np.int64)
    current_idx = 0
    
    # We define the maximum allowed index
    # (Must match num_embeddings in the network initialization)
    MAX_VALID_INDEX = num_states 
    
    # Encode player
    indices[current_idx] = max(0, min(player - 1, MAX_VALID_INDEX)) 
    current_idx += 1
    
    # Encode local state
    for i, val in enumerate(flat_player_s):
        if i >= state_len: break
        if isinstance(val, int):
            safe_val = max(0, min(val, MAX_VALID_INDEX))
            indices[current_idx + i] = safe_val
    current_idx += state_len
    
    # Encode history
    for t, state_list in enumerate(flat_history):
        if t >= max_history_length: break
        
        time_offset = t * state_len
        for i, val in enumerate(state_list):
            if i >= state_len: break
            if isinstance(val, int):
                safe_val = max(0, min(val, MAX_VALID_INDEX))
                indices[current_idx + time_offset + i] = safe_val

    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
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
        self.num_slots = 1 + self.state_len + (self.game.max_depth * self.state_len)
        # Pick an embedding dimension (Tune this! 8 to 64 is usually good)
        EMBED_DIM = 16 
        # Pass the total distinct values (num_states) so the embedding layer knows how big the dictionary is
        self.value_net = RegretNetwork(
            num_actions=self.num_states, 
            num_input_slots=self.num_slots, 
            num_embeddings=self.num_states + 1, # +1 for safety/padding
            embedding_dim=EMBED_DIM
        )

        # Policy Network predicts the average strategy (Probability distribution)
        self.policy_net = PolicyNetwork(
            self.num_states,
            self.num_slots,
            num_embeddings=self.num_states + 1,
            embedding_dim=EMBED_DIM
        )
        
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
            MAX_POLICY_SAMPLES = 20000
            if len(self.mccfr.policy_samples) > MAX_POLICY_SAMPLES:
                # Randomly sample to keep size manageable
                indices = np.random.choice(len(self.mccfr.policy_samples), MAX_POLICY_SAMPLES, replace=False)
                self.mccfr.policy_samples = [self.mccfr.policy_samples[i] for i in indices]

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
    # Reduced depth for debugging
    p1, p2 = (10, 5), (10, 5)
    t = 60 * 60 # 60 mins
    depth = (24 * 60 * 60) // t # 60 min steps -> 24 layers

    # Setup transition info
    zones = load_zones('../data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('../data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('../data/networks/example_network/base/nodes.csv', '../data/networks/example_network/base/edges.csv')
    transition = Transition(p1, p2, graph, zones, demands)

    # Create game
    game = StochasticGame(p1, p2, depth, t, transition=transition)
    
    deepcfr = DeepCFR(game)
    time_func(deepcfr.train, {'iters': 5, 'traversals_per_iter': 100})