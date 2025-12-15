import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from cfr.mccfr import MonteCarloCFR
import numpy as np
from cfr.utils import time_func, max_actions
from cfr.game import StochasticGame
from cfr.transition import load_demand, load_graph, load_zones, Transition

def predict(infoset, model, max_in_size, device):
    # Switch to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        # Prepare input: Move to device and add batch dimension (1, N)
        tensor_in = infoset_to_tensor(infoset, max_in_size, device=device)
        
        # Forward pass
        output_tensor = model(tensor_in)
        
        # Post-process: Detach from graph -> Move to CPU -> Convert to Numpy
        # [0] is used to unwrap the batch dimension
        result = output_tensor.detach().cpu().numpy()[0]

    # (Optional) Switch back to train mode if we're inside a training loop
    model.train()

    return result

class GameEval:
    def __init__(self):
        pass

    def sample_action(self, actions, sigma):
        return random.choices(actions, weights=[sigma[a] for a in actions], k=1)[0]

    def get_model_strategy(self, infoset, actions, model, model_in_size, device):
        strategy = predict(infoset, model, model_in_size, device)
        sigma = {a: strategy[i] for i, a in enumerate(actions)}
        return sigma

    def get_random_strategy(self, infoset, actions):
        num_actions = len(actions)
        uniform_prob = 1/num_actions
        return {a: uniform_prob for a in actions}

    def play_game(self, game, num_hands, model, model_in_size, device):
        r1s, r2s = [], []
        for hand in range(num_hands):
            # Get root of game tree
            s = game.initial_state()

            # Keep track of player rewards
            total_r1, total_r2 = 0.0, 0.0

            # Play until we reach a leaf node
            k = 0
            while not game.is_terminal(s):
                # Get player info
                infoset_1, actions_1 = s.infoset_key(1), game.actions(s)[0]
                infoset_2, actions_2 = s.infoset_key(2), game.actions(s)[1]

                # Get player strategies
                sigma_1 = self.get_model_strategy(infoset_1, actions_1, model, model_in_size, device)
                sigma_2 = self.get_random_strategy(infoset_2, actions_2)

                # Sample player actions
                a1_sampled = self.sample_action(actions_1, sigma_1)
                a2_sampled = self.sample_action(actions_2, sigma_2)

                # Sample from chance node to transition to next layer
                next_s, (r1, r2), chance_prob = game.step(s, a1_sampled, a2_sampled)

                # Update total reward with discounted reward
                total_r1 += game.gamma**k * r1
                total_r2 += game.gamma**k * r2

                # Proceed to next state
                s = next_s
                k += 1
            
            r1s.append(total_r1)
            r2s.append(total_r2)

        # Avg rewards
        avg_r1, avg_r2 = sum(r1s)/len(r1s), sum(r2s)/len(r2s)

        return avg_r1, avg_r2

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
        self.num_actions = max(max_actions(*self.game.p1), max_actions(*self.game.p2))
        self.state_len = max(self.game.p1[1], self.game.p2[1])

        # Game eval framework
        self.game_eval = GameEval()
        self.num_hands = 100

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
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_net.to(self.device)
        self.policy_net.to(self.device)

    def model(self, infoset, model_type):
        if model_type == 'regret':
            return predict(infoset, self.value_net, self.max_in_size, self.device)
        else:
            return predict(infoset, self.policy_net, self.max_in_size, self.device)

    def train(self, iters, traversals_per_iter=1000, batch_size=1024):
        print(f'Starting deep CFR training with {iters} train runs of {traversals_per_iter} hands...')
        for i in range(iters):
            print(f'Starting train run {i}...')

            # Collect data
            self.mccfr.train(traversals_per_iter, traversals_per_iter*i)
            regret_data = self.mccfr.regret_buffer
            policy_data = self.mccfr.policy_buffer

            # Separate into train and test sets to see if (1) it's learning (2) it generalizes
            regret_train_data, regret_val_data = train_test_split(regret_data.data, test_size=0.2, shuffle=True)
            policy_train_data, policy_val_data = train_test_split(policy_data.data, test_size=0.2, shuffle=True)

            # Train networks
            self.train_network(
                self.value_net, 
                self.value_optimizer, 
                regret_train_data, regret_val_data,
                batch_size
            )
            self.train_network(
                self.policy_net, 
                self.policy_optimizer, 
                policy_train_data, policy_val_data,
                batch_size,
                is_policy=True
            )

            # Evaluate the learned strategy
            avg_r1, avg_r2 = self.game_eval.play_game(self.game, self.num_hands, self.policy_net, self.max_in_size, self.device)
            print(f'Learned strategy vs random strategy got average rewards of {avg_r1}, {avg_r2} over {self.num_hands} on train run {i}')

            print(f'Progress: {round((i+1)/iters * 100, 2)}% done')

    def prepare_dataset(self, data):
        # Extract data
        if len(data[0]) == 3:
            inputs, targets, weights = zip(*data)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            inputs, targets = zip(*data)
            weights = torch.ones(len(targets), dtype=torch.float32, device=self.device)

        # Convert infosets to feature tensors
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

        return inputs, targets, weights

    def train_network(self, model, optimizer, train_data, val_data, batch_size, is_policy=False):
        if not train_data or not val_data:
            return
        mode_name = 'policy' if is_policy else 'regret'
        print(f'Training {mode_name} network...')
        
        # Convert infosets to feature tensors
        train_x, train_y, train_w = self.prepare_dataset(train_data)
        if is_policy:
            # Normalize targets so they sum to 1 (like Softmax)
            # Add epsilon (1e-12) to prevent division by zero
            train_y = train_y / (train_y.sum(dim=1, keepdim=True) + 1e-12)
            # Use KL Div Loss for Policy
            loss_fn = nn.KLDivLoss(reduction='none')
        else:
            # Use MSE for Regret Net
            loss_fn = nn.MSELoss(reduction='none')
        
        # Create dataLoader
        dataset = TensorDataset(train_x, train_y, train_w)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        model.train()
        epochs = 2 
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y, batch_w in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)

                if is_policy:
                    log_outputs = F.log_softmax(outputs, dim=1)
                    loss_per_sample = loss_fn(log_outputs, batch_y).sum(dim=1)
                else:
                    loss_per_sample = loss_fn(outputs, batch_y).mean(dim=1)

                # Apply Linear Weights (t)
                # We normalize weights by the mean weight of the batch to keep gradient scale consistent
                normalized_weights = batch_w / (batch_w.mean() + 1e-12)
                weighted_loss = (loss_per_sample * normalized_weights).mean()

                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item()
            
            print(f'Epoch {epoch} loss: {total_loss:.4f}')

        # Test
        model.eval() # Switch to evaluation mode (disables dropout/batchnorm updates)
        with torch.no_grad(): # Disable gradient calculation (saves memory)
            # Preprocess Validation Data
            val_x, val_y, val_w = self.prepare_dataset(val_data)
            
            # Forward pass on full validation set
            # (If val set is huge, wrap this in a DataLoader loop too)
            val_outputs = model(val_x)
            
            if is_policy:
                val_y = val_y / (val_y.sum(dim=1, keepdim=True) + 1e-12)
                val_outputs = F.log_softmax(val_outputs, dim=1)
                raw_loss = loss_fn(val_outputs, val_y)
                val_loss = raw_loss.sum(dim=1).mean().item()
            else:
                raw_loss = loss_fn(val_outputs, val_y)
                val_loss = raw_loss.mean().item()

            print(f'>> Validation Loss ({mode_name}): {val_loss:.4f}')
        
        model.train() # Switch back to train mode for safety

def train_example_network():
    p1, p2 = (5, 6), (5, 6)
    t = 10 * 60 # 10 mins
    depth = (24 * 60 * 60) // t # 10 min steps -> 144 layers

    # Setup transition info
    zones = load_zones('./data/zones/example_zones/example_network/node_zone_info.csv')
    demand_files = [
        './data/demand/example_demand/matched/example_network/example_100.csv',
        './data/demand/example_demand/matched/example_network/example_200.csv',
        './data/demand/example_demand/matched/example_network/example_400.csv'
    ]
    demands = [load_demand(demand_file) for demand_file in demand_files]
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