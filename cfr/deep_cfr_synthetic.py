import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mccfr import MonteCarloCFR
import numpy as np
from utils import time_func, max_actions
from game import StochasticGame
from transition import load_demand, load_graph, load_zones, Transition
from itertools import product

def get_combos(target, num_slots, current_combo=None):
    if current_combo is None:
        current_combo = []
    if num_slots == 1:
        if target >= 0:
            yield tuple(current_combo + [target])
        return
    for i in range(target + 1):
        yield from get_combos(target - i, num_slots - 1, current_combo + [i])

def generate_states(p, q):
    return tuple({combo for tot in range(p+1) for combo in get_combos(tot, q)})

def get_actions_for(s):
    p, q = sum(s), len(s)
    new_states = get_combos(p, q)
    actions = {tuple(new_s[i] - s[i] for i in range(q)) for new_s in new_states}
    
    # Filter to only keep actions that sum to 0
    valid_actions = {action for action in actions if sum(action) == 0}
    
    return tuple(sorted(valid_actions))  # deterministic

def action_space_size(p, q):
    states = StochasticGame.generate_states(p, q)

    actions = set()




    for state in states:
        actions.update(StochasticGame.get_actions_for(state))


def generate_action_space():
    """
    Generate all valid actions based on game parameters.
    Action is a redistribution of cars across zones (must sum to 0).
    
    Args:
        num_cars: Total number of cars a player has
        num_zones: Total number of zones
    
    Returns:
        List of valid action tuples
    """
    # actions = []
    
    # # Maximum change in any zone is bounded by total cars
    # # A zone can lose all cars (-num_cars) or gain all cars (+num_cars)
    # max_change = num_cars
    
    # # Generate all combinations for first (num_zones - 1) positions
    # # The last position is determined by the constraint that sum = 0
    # ranges = [range(-max_change, max_change + 1) for _ in range(num_zones - 1)]
    
    # for combination in product(*ranges):
    #     # Calculate last position to ensure sum = 0
    #     last_value = -sum(combination)
        
    #     # Check if last value is in valid range
    #     if -max_change <= last_value <= max_change:
    #         action = combination + (last_value,)
    #         actions.append(action)
    
    # return actions

    actions = []

    all_states = generate_states(10, 5)
    for state in all_states:
        actions.extend(list(get_actions_for(state)))



    return list(set(actions))



def compute_variance(state):
    """Compute variance of state values."""
    return np.var(state)

def compute_optimal_action(state):
    """
    Directly compute the action that minimizes variance.
    The optimal state has all values equal to the mean.
    """
    state = np.array(state)
    total = state.sum()
    num_zones = len(state)
    
    # Target: distribute total evenly
    target_per_zone = total / num_zones
    
    # If total divides evenly, we can reach perfect equality
    if total % num_zones == 0:
        target_state = np.full(num_zones, total // num_zones)
    else:
        # Otherwise, distribute as evenly as possible
        # Some zones get floor(mean), others get ceil(mean)
        base = total // num_zones
        remainder = total % num_zones
        target_state = np.full(num_zones, base)
        target_state[:remainder] += 1  # Give extra to first 'remainder' zones
    
    # Action is the difference
    action = tuple(target_state - state)
    return action

def generate_equalize_strategy_data(num_samples=1000, action_space=None, total_cars=10, num_zones=5):
    """
    Generate synthetic data where optimal action minimizes state variance.
    Target is a probability distribution over actions.
    """
    if action_space is None:
        action_space = generate_action_space()

    num_actions = len(action_space)
    data = []
    
    for i in range(num_samples):
        # Convert to Python int explicitly
        state = tuple(int(x) for x in np.random.multinomial(total_cars, [1/num_zones]*num_zones))
        history = tuple(int(x) for x in np.random.randint(0, total_cars+1, 3*num_zones))
       
        infoset = (state, history)
        
        # Directly compute optimal action (no search needed!)
        best_action = compute_optimal_action(state)
        
        # Create one-hot probability distribution
        action_probs = np.zeros(num_actions, dtype=np.float32)
        
        # Find index of this action in action_space
        try:
            best_action_idx = action_space.index(best_action)
            action_probs[best_action_idx] = 1.0
        except ValueError:
            # If the optimal action isn't in action_space, skip or use closest
            print(f"Warning: optimal action {best_action} not in action space")
            continue
        
        data.append((infoset, action_probs))
    
    return data, action_space

def generate_soft_equalize_strategy_data(num_samples=1000, action_space=None, 
                                         max_change=2, temperature=0.1):
    """
    Generate data with soft probabilities based on how good each action is.
    Actions that reduce variance more get higher probability.
    """
    if action_space is None:
        action_space = generate_action_space()
        print('length of action space', action_space)
    
    num_actions = len(action_space)
    data = []
    
    for _ in range(num_samples):
        state = tuple(np.random.randint(0, 20, 5))
        history = tuple(np.random.randint(0, 20, 15))
        
        infoset = (state, history)
        
        # Compute variance for each action
        current_state = np.array(state)
        variances = []
        
        for action in action_space:
            next_state = current_state + np.array(action)
            variance = compute_variance(next_state)
            variances.append(variance)
        
        variances = np.array(variances)
        
        # Convert to probabilities (lower variance = higher probability)
        # Negate because we want to minimize variance
        scores = -variances
        
        # Softmax with temperature
        exp_scores = np.exp(scores / temperature)
        action_probs = exp_scores / exp_scores.sum()
        action_probs = action_probs.astype(np.float32)
        
        data.append((infoset, action_probs))
    
    return data, action_space

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
    if len(infoset) == 3:
        player_id, state, history = infoset
    else:
        state, history = infoset
    
    # Convert state and history to numpy arrays and concatenate
    state_arr = np.asarray(state, dtype=np.float32)
    history_arr = np.asarray(history, dtype=np.float32)
    arr = np.concatenate([state_arr, history_arr])
    
    # Pad to full length w/ -1 as 0 is significant
    padded = np.pad(arr, (0, max(0, max_infoset_size - len(arr))), constant_values=-1)
    
    # Convert to tensor
    tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

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
        print('max infoset size', self.max_in_size)

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

    def model(self, infoset):
        infoset_tensor = infoset_to_tensor(infoset, self.max_in_size, device=self.device)
        output = self.value_net(infoset_tensor)
        return output.detach().cpu().numpy()[0]

    def train(self, iters, traversals_per_iter=1000, batch_size=1024):
        for i in range(iters):
            # Collect data
            # self.mccfr.train(traversals_per_iter)
            # regret_data = self.mccfr.regret_samples
            # policy_data = self.mccfr.policy_samples


            synthetic_data = generate_equalize_strategy_data(num_samples=1000)

            print(len(synthetic_data))

            # Train networks
            # self.train_network(
            #     self.value_net, 
            #     self.value_optimizer, 
            #     regret_data, 
            #     batch_size
            # )

            self.train_network(
                self.policy_net, 
                self.policy_optimizer, 
                synthetic_data, 
                batch_size,
                is_policy=True
            )

            # Clear regret samples to prevent mixing old (bad) data with new data
            self.mccfr.regret_samples = []

            # Optional: Clear policy samples if memory is an issue
            MAX_POLICY_SAMPLES = 20000
            if len(self.mccfr.policy_samples) > MAX_POLICY_SAMPLES:
                # Randomly sample to keep size manageable
                indices = np.random.choice(len(self.mccfr.policy_samples), MAX_POLICY_SAMPLES, replace=False)
                self.mccfr.policy_samples = [self.mccfr.policy_samples[i] for i in indices]

    def train_network(self, model, optimizer, data, batch_size, is_policy=False):
        if not data:
            return
        
        # Convert infosets to feature tensors
        inputs, targets = data
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
        epochs = 10
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                if is_policy:
                    # Cross-entropy loss (don't apply softmax - it's built into the loss)
                    log_probs = F.log_softmax(outputs, dim=1)
                    loss = -(batch_y * log_probs).sum(dim=1).mean()
                else:
                    # MSE loss for regret network
                    loss = self.loss_fn(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            print('epoch', epoch, 'loss', total_loss)
        
    def save_policy(self, path='policy.pth'):
        # It is good practice to save the state_dict rather than the entire object
        torch.save(self.policy_net.state_dict(), path)
        print(f'Policy network saved to {path}')

    def load_policy(self, path='policy.pth'):
        # map_location ensures we can load a GPU model onto a CPU if needed
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint)
        
        # Set to eval mode (important for inference)
        self.policy_net.eval() 
        print(f'Policy network loaded from {path}')

if __name__ == '__main__':

    p1, p2 = (5, 5), (5, 5)
    t = 10 * 60 # 10 mins
    depth = (24 * 60 * 60) // t # 10 min steps -> 144 layers

    # Setup transition info
    zones = load_zones('../data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('../data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('../data/networks/example_network/base/nodes.csv', '../data/networks/example_network/base/edges.csv')
    transition = Transition(p1, p2, graph, zones, demands)

    # Create game
    game = StochasticGame(p1, p2, depth, t, transition=transition)
    
    deepcfr = DeepCFR(game)
    time_func(deepcfr.train, {'iters': 5, 'traversals_per_iter': 100})

    # Save the policy
    deepcfr.save_policy('example_network_5c5z10m.pth')