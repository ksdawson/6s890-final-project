import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cfr.mccfr import MonteCarloCFR
import numpy as np
from cfr.utils import time_func, max_actions
from cfr.game import StochasticGame
from cfr.transition import load_demand, load_graph, load_zones, Transition

class RegretNetwork(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

def flatten(x):
    for i in x:
        if isinstance(i, (tuple, list)):
            yield from flatten(i)
        else:
            yield i

def infoset_to_tensor(infoset, max_infoset_size, device='cpu'):
    # TODO: Consider truncation or seperator?
    # Discard player
    p, s, h = infoset
    # Flatten and concatenate everything
    arr = np.fromiter(flatten((s, h)), dtype=int)
    # Pad to full length w/ -1 as 0 is significant
    padded = np.pad(arr, (0, max(0, max_infoset_size - len(arr))), constant_values=-1)
    # Convert to a tensor
    tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

def save_policy(net, path='policy.pth'):
    # It is good practice to save the state_dict rather than the entire object
    torch.save(net.state_dict(), path)
    print(f'Policy network saved to {path}')

def load_policy(net, device, path='policy.pth'):
    # map_location ensures we can load a GPU model onto a CPU if needed
    checkpoint = torch.load(path, map_location=device)
    net.load_state_dict(checkpoint)
    
    # Set to eval mode (important for inference)
    net.eval() 
    print(f'Policy network loaded from {path}')

class DeepCFR:
    def __init__(self, game):
        # Get game info
        self.game = game
        self.num_actions = max(max_actions(*self.game.p1), max_actions(*self.game.p1))
        self.state_len = max(self.game.p1[1], self.game.p2[1])

        # Initialize the solver
        self.mccfr = MonteCarloCFR(game, deep_cfr=True, model=self.model)

        # Setup networks 
        self.max_in_size = self.state_len + (self.game.max_depth * self.state_len * 2) # could add +1 for player
        # Value Network predicts advantages/regrets (Unbounded)
        self.value_net = RegretNetwork(in_size=self.max_in_size, out_size=self.num_actions)
        # Policy Network predicts the average strategy (Probability distribution)
        self.policy_net = PolicyNetwork(in_size=self.max_in_size, out_size=self.num_actions)
        
        # Define Optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Define Loss Function (MSE is standard for Deep CFR)
        self.loss_fn = nn.MSELoss()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_net.to(self.device)
        self.policy_net.to(self.device)

    def model(self, infoset, model_type):
        infoset_tensor = infoset_to_tensor(infoset, self.max_in_size, device=self.device)
        if model_type == 'regret':
            output = self.value_net(infoset_tensor)
        else:
            output = self.policy_net(infoset_tensor)
        return output.detach().cpu().numpy()[0]

    def train(self, iters, traversals_per_iter=1000, batch_size=1024):
        print(f'Starting deep CFR training with {iters} train runs of {traversals_per_iter} hands...')
        for i in range(iters):
            # Collect data
            self.mccfr.train(traversals_per_iter)
            regret_data = self.mccfr.regret_buffer
            policy_data = self.mccfr.policy_buffer

            # Train networks
            self.train_network(
                self.value_net, 
                self.value_optimizer, 
                regret_data, 
                batch_size
            )
            self.train_network(
                self.policy_net, 
                self.policy_optimizer, 
                policy_data, 
                batch_size,
                is_policy=True
            )

            # Clear regret samples to prevent mixing old (bad) data with new data
            self.mccfr.regret_buffer = []

            # Optional: Clear policy samples if memory is an issue
            MAX_POLICY_SAMPLES = 20000
            if len(policy_data) > MAX_POLICY_SAMPLES:
                # Randomly sample to keep size manageable
                indices = np.random.choice(len(policy_data), MAX_POLICY_SAMPLES, replace=False)
                self.mccfr.policy_buffer = [policy_data[i] for i in indices]

            print(f'Progress: {round((i+1)/iters * 100, 2)}% done')

    def train_network(self, model, optimizer, data, batch_size, is_policy=False):
        if not data:
            return
        
        # Convert infosets to feature tensors
        inputs, targets = zip(*data)
        input_tensors = tuple(infoset_to_tensor(infoset, self.max_in_size, device=self.device) for infoset in inputs)
        inputs = torch.cat(input_tensors)

        # Pad targets to full
        targets = np.stack([
            np.pad(
                np.asarray(t, dtype=np.float32),
                (0, self.num_actions - len(t)),
                constant_values=0.0
            )
            for t in targets
        ])
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Create dataLoader
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
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

def train_example_network():
    p1, p2 = (5, 6), (5, 6)
    t = 10 * 60 # 10 mins
    depth = (24 * 60 * 60) // t # 10 min steps -> 144 layers

    # Setup transition info
    zones = load_zones('./data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('./data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('./data/networks/example_network/base/nodes.csv', './data/networks/example_network/base/edges.csv')
    transition = Transition(p1, p2, graph, zones, demands)

    # Create game
    game = StochasticGame(p1, p2, depth, t, transition=transition)
    
    # Train
    deepcfr = DeepCFR(game)
    time_func(deepcfr.train, {'iters': 5, 'traversals_per_iter': 100})

    # Save the policy
    save_policy(deepcfr.policy_net, './cfr/example_network_policy_5c6z10m.pth')

def train_nyc_network():
    p1, p2 = (5, 8), (5, 8)
    t = 10 * 60 # 10 mins
    depth = (24 * 60 * 60) // t # 10 min steps -> 144 layers

    # Setup transition info
    zones = load_zones('./data/zones/manhattan_zones/Manhattan_corrected_12min_max/Manhattan_2019_corrected/node_zone_info.csv')
    demand_files = [
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-11.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-12.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-13.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-14.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-15.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-16.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-17.csv',
        './data/demand/manhattan_demand/matched/Manhattan_2019_corrected/2018-11-18.csv'
    ]
    demands = [load_demand(demand_file) for demand_file in demand_files]
    graph = load_graph('./data/networks/manhattan_network/base/nodes.csv', './data/networks/manhattan_network/base/edges.csv')
    transition = Transition(p1, p2, graph, zones, demands)

    # Create game
    game = StochasticGame(p1, p2, depth, t, transition=transition)
    
    # Train
    deepcfr = DeepCFR(game)
    time_func(deepcfr.train, {'iters': 5, 'traversals_per_iter': 100})

    # Save the policy
    save_policy(deepcfr.policy_net, './cfr/manhattan_network_policy_5c8z10m.pth')

if __name__ == '__main__':
    # Example network
    train_example_network()

    # NYC network
    # train_nyc_network()