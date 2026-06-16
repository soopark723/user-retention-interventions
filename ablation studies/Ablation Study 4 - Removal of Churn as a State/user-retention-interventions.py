# 1. IMPORTS

import numpy as np
import random
import json
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# 2. PARAMETERS

T = 50
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 50000
EPISODES = 5000
TARGET_UPDATE = 50

# Dynamics parameters: how users behave
beta = 0.36      # Fatigue penalty
eta = 0.15       # Mean reversion
sigma = 0.07     # Noise
gamma_f = 0.68   # Fatigue decay

# Rewards parameters: what agent optimizes
lambda_f = 0.32     # Fatigue weight in reward
kappa = 7.0         # Churn penalty 

# Actions
# 0 none
# 1 notify
# 2 feature
# 3 incentive
# 4 notify + feature
# 5 notify + incentive

ACTIONS = [0, 1, 2, 3, 4, 5]
NUM_ACTIONS = len(ACTIONS)

# Engagement boost
alpha_a = {
    0: 0.00,
    1: 0.04,
    2: 0.07,
    3: 0.11,
    4: 0.10,
    5: 0.15
}

# Fatigue
delta_a = {
    0: 0.00,
    1: 0.08,
    2: 0.08,
    3: 0.15,
    4: 0.18,
    5: 0.26
}

# Action cost
cost_a = {
    0: 0.00,
    1: 0.03,
    2: 0.05,
    3: 0.10,
    4: 0.14,
    5: 0.20
}

baseline_map = {
    "moderately engaged": 0.5,
    "active": 0.7,
    "declining": 0.4
}


# 3. STATE REPRESENTATION

# State: s_t = (engagement, fatigue, user_type, churn_counter)
# User type is one-hot encoded.
# churn_counter ∈ {0, 1, 2, 3, 4, 5}

user_type_encoding = {
    "moderately engaged": [1, 0, 0],
    "active": [0, 1, 0],
    "declining": [0, 0, 1]
}

def encode_state(e, f, u, c):
    """
    Encode state into continuous vector representation.
    Returns 6-dimensional vector:
    - engagement (1 dim)
    - fatigue (1 dim)
    - churn_counter normalized (1 dim)
    - user_type one-hot (3 dims)
    """
    return np.array([
        e,
        f,
        *user_type_encoding[u]
    ], dtype=np.float32)


# 4. TRANSITION FUNCTION

def transition(e, f, u, c, a):
    """
    Compute state transition based on MDP dynamics.
    
    Returns:
        e_next: next engagement [0,1]
        f_next: next fatigue [0,1]
        u_next: next user stage {moderately engaged, active, declining}
        c_next: next churn counter {0,1,2,3,4,5}
        churn: whether user churned (0 or 1)
    """
    
    e_bar = baseline_map[u]

    fatigue_multiplier = (1 - 0.6 * f)   # 0.5 to 1.0 based on fatigue

    # Engagement dynamics: e_{t+1} ~ N(μ_t, σ²)
    # μ_t = e_t + α_a(1 - f_t) - β·f_t - η·(e_t - ē_u)
    mu = (
        e
        + alpha_a[a] * fatigue_multiplier
        - beta * (f ** 1.05)     # nonlinear fatigue term
        - eta * (e - e_bar)
    )

    e_next = np.random.normal(mu, sigma)
    e_next = np.clip(e_next, 0, 1)

    # Fatigue dynamics: f_{t+1} = min(1, γ·f_t + δ_a)
    f_next = gamma_f * f + delta_a[a]
    f_next = np.clip(f_next, 0, 1)

    # User stage transition based on engagement
    if e_next > 0.6:
        u_next = "active"
    elif e_next > 0.3:
        u_next = "moderately engaged"
    else:
        u_next = "declining"

    # Churn counter logic
    if e_next < 0.25:
        c_next = c + 1
    else:
        c_next = 0

    # Churn occurs if counter reaches 5
    churn = int(c_next >= 5)

    return e_next, f_next, u_next, c_next, churn


# 5. REWARD FUNCTION

def reward(e_next, f_next, a, churn):
    """
    Compute immediate reward: r_t = e_{t+1} - λ·f_{t+1} - c(a) - κ·1_churn
    
    Args:
        e_next: next engagement
        f_next: next fatigue
        a: action taken
        churn: whether churn occurred (0 or 1)
    
    Returns:
        scalar reward value
    """
    r = (
        e_next                  # Engagement reward
        - lambda_f * f_next     # Penalty, fatigue weight
        - cost_a[a]             # Costs
        - kappa * churn         # Churn penalty
    )

    return r


# 6. DQN NETWORK

class DQN(nn.Module):
    """
    Deep Q-Network with 2 hidden layers of 64 units each.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# 7. REPLAY BUFFER

replay_buffer = deque(maxlen=BUFFER_SIZE)


# 8. INITIALIZE NETWORKS

policy_net = DQN(5, NUM_ACTIONS)
target_net = DQN(5, NUM_ACTIONS)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()


# 9. EPSILON-GREEDY POLICY

def select_action(state, epsilon=0.1):
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: encoded state vector
        epsilon: exploration probability
    
    Returns:
        action index
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return torch.argmax(q_values).item()


# 10. DQN TRAINING STEP

def train_step():
    """
    Perform one training step using experience replay.
    Implements standard DQN update with target network.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return None

    batch = random.sample(replay_buffer, BATCH_SIZE)

    states = torch.FloatTensor(np.array([x[0] for x in batch]))
    actions = torch.LongTensor(np.array([x[1] for x in batch]))
    rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
    next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
    dones = torch.FloatTensor(np.array([x[4] for x in batch]))

    # Compute current Q-values
    current_q = policy_net(states).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze()

    # Compute target Q-values
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    # Compute loss and update
    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 11. GENERATE EXPERIENCE DATA

print("\nTraining DQN with online environment interaction...")

all_data = []

for episode in range(EPISODES):

    # Initialize user state
    e = np.random.uniform(0.3, 0.8)
    f = np.random.uniform(0.0, 0.2)

    if e > 0.6:
        u = "active"
    elif e > 0.3:
        u = "moderately engaged"
    else:
        u = "declining"

    c = 0

    epsilon = max(0.02, 0.20 * (0.995 ** episode))

    episode_reward = 0

    for t in range(T):

        state = encode_state(e, f, u, c)

        # Agent selects action
        action = select_action(state, epsilon)

        # Environment transition
        e_next, f_next, u_next, c_next, churn = transition(
            e, f, u, c, action
        )

        next_state = encode_state(
            e_next,
            f_next,
            u_next,
            c_next
        )

        done = churn

        r = reward(e_next, f_next, action, churn)

        episode_reward += r

        # Store transition
        replay_buffer.append((
            state,
            action,
            r,
            next_state,
            done
        ))

        # Train immediately
        loss = train_step()

        # Save trajectory data
        all_data.append({
            "episode": episode,
            "t": t,
            "engagement_t": float(e),
            "fatigue_t": float(f),
            "user_type_t": u,
            "low_engagement_count_t": int(c),
            "action": int(action),
            "reward": float(r),
            "engagement_t1": float(e_next),
            "fatigue_t1": float(f_next),
            "user_type_t1": u_next,
            "low_engagement_count_t1": int(c_next),
            "churn": int(churn),
            "done": int(done)
        })

        # Advance state
        e, f, u, c = e_next, f_next, u_next, c_next

        if done:
            break

    # Target network update
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Progress logging
    if episode % 50 == 0:
        print(
            f"Episode {episode} | "
            f"Epsilon {epsilon:.3f} | "
            f"Reward {episode_reward:.2f}"
        )

print("Training complete!")


# 12. SAVE SIMULATED DATA

with open("simulated_data.json", "w") as f:
    json.dump(all_data, f, indent=2)

print("Saved simulated_data.json")


# 13. EXTRACT POLICY

def get_policy_action(e, f, u, c):
    """
    Get the learned policy's action for a given state.
    
    Args:
        e: engagement
        f: fatigue
        u: user type
        c: churn counter
    
    Returns:
        action index
    """
    state = encode_state(e, f, u, c)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return torch.argmax(q_values).item()


# 14. VISUALIZATIONS

print("\nGenerating visualizations...")

# (A) Retention Curve

episode_last_t = {}

for d in all_data:
    eid = d["episode"]
    t = d["t"]
    episode_last_t[eid] = max(episode_last_t.get(eid, 0), t)

retention = []

for t in range(T):
    active = sum(
        1 for eid in episode_last_t
        if episode_last_t[eid] >= t
    )
    retention.append(active / EPISODES)

plt.figure(figsize=(7,5))
plt.plot(retention, linewidth=2)
plt.title("User Retention Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Retention Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("retention_curve.png", dpi=150)
print("Saved: retention_curve.png")
plt.close()

# (B) Average Engagement

engagement_by_t = [[] for _ in range(T)]

for d in all_data:
    engagement_by_t[d["t"]].append(d["engagement_t"])

avg_engagement = [
    np.mean(x) if len(x) > 0 else 0
    for x in engagement_by_t
]

plt.figure(figsize=(7,5))
plt.plot(avg_engagement, linewidth=2, color='green')
plt.title("Average Engagement Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Engagement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("avg_engagement.png", dpi=150)
print("Saved: avg_engagement.png")
plt.close()

# (C) Average Fatigue

fatigue_by_t = [[] for _ in range(T)]

for d in all_data:
    fatigue_by_t[d["t"]].append(d["fatigue_t"])

avg_fatigue = [
    np.mean(x) if len(x) > 0 else 0
    for x in fatigue_by_t
]

plt.figure(figsize=(7,5))
plt.plot(avg_fatigue, linewidth=2, color='red')
plt.title("Average Fatigue Over Time")
plt.xlabel("Time (days)")
plt.ylabel("Fatigue")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("avg_fatigue.png", dpi=150)
print("Saved: avg_fatigue.png")
plt.close()

# (D) Learned Policy Heatmap

print("\nLearned policy samples (engagement, fatigue, action):")

sample_states = []

for e in np.linspace(0, 1, 10):
    for f in np.linspace(0, 1, 10):
        a = get_policy_action(e, f, "moderately engaged", 0)
        sample_states.append((round(e, 2), round(f, 2), a))

# Print first 20 samples
for x in sample_states[:20]:
    action_names = ['none', 'notify', 'feature', 'incentive', 'notify+feature', 'notify+incentive']
    print(f"e={x[0]:.2f}, f={x[1]:.2f} → action: {action_names[x[2]]}")

# Create policy heatmap
policy_grid = np.zeros((10, 10))
for i, e in enumerate(np.linspace(0, 1, 10)):
    for j, f in enumerate(np.linspace(0, 1, 10)):
        a = get_policy_action(e, f, "moderately engaged", 0)
        policy_grid[9-j, i] = a  # Flip vertically for correct orientation

plt.figure(figsize=(8, 6))
im = plt.imshow(policy_grid, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(im, label='Action', ticks=range(6))
plt.title("Learned Policy Heatmap (User Type: New)")
plt.xlabel("Engagement")
plt.ylabel("Fatigue")
plt.xticks(range(10), [f"{x:.1f}" for x in np.linspace(0, 1, 10)])
plt.yticks(range(10), [f"{x:.1f}" for x in np.linspace(1, 0, 10)])
plt.tight_layout()
plt.savefig("policy_heatmap.png", dpi=150)
print("Saved: policy_heatmap.png")
plt.close()


# 15. STATISTICS

print("\n" + "="*60)
print("SIMULATION STATISTICS")
print("="*60)

total_transitions = len(all_data)
total_churn = sum(d["churn"] for d in all_data)
churn_rate = total_churn / EPISODES

print(f"Total transitions: {total_transitions}")
print(f"Total episodes: {EPISODES}")
print(f"Churn events: {total_churn}")
print(f"Churn rate: {churn_rate:.2%}")
print(f"Average trajectory length: {total_transitions / EPISODES:.2f} days")

action_counts = {}
for d in all_data:
    a = d["action"]
    action_counts[a] = action_counts.get(a, 0) + 1

print("\nAction distribution:")
action_names = ['none', 'notify', 'feature', 'incentive', 'notify+feature', 'notify+incentive']
for a in sorted(action_counts.keys()):
    print(f"  {action_names[a]:20s}: {action_counts[a]:6d} ({action_counts[a]/total_transitions:.2%})")

avg_reward = np.mean([d["reward"] for d in all_data])
print(f"\nAverage reward: {avg_reward:.4f}")

print("\n" + "="*60)


# 16. EXPECTED RESULTS

print("-" * 60)
print("Expected outcomes:")
print("  • Retention at day 50: ~55-70% (realistic churn)")
print("  • Engagement: rises early, then declines to ~0.45-0.55")
print("  • Fatigue: builds up to ~0.25-0.35 (meaningful accumulation)")
print("  • Policy: diverse actions")
print("    - Low engagement + low fatigue → incentive (yellow)")
print("    - Medium engagement → feature/notify (green/teal)")
print("    - High engagement + low fatigue → none (purple)")
print("    - High fatigue → none/notify (purple/blue)")
print("="*60)