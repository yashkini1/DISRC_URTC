# ================================================================
# DISRC DQN on MiniGrid-LavaCrossingS9N1-v0
# This script trains a Deep Q-Network agent augmented with 
# a biologically-inspired surprise regularization mechanism (DISRC)
# Authors: Yash Kini, Shiv Davay, Shreya Polavarapu
# ================================================================

# --- Imports ---
import gymnasium as gym
import minigrid  # Registers MiniGrid environments
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ================================================================
# Reproducibility
# Set deterministic seeds for random, NumPy, and PyTorch
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# Preprocessing for MiniGrid
# Extracts and normalizes the image component of observations
# ================================================================
def preprocess_obs(obs):
    img = obs['image']  # Shape: (7, 7, 3)
    flat = img.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]
    return flat  # Shape: (147,)

# ================================================================
# Utility Functions
# Includes tensor reshaping and weight initialization
# ================================================================
def ensure_2d_tensor(tensor):
    """Ensure input is a 2D PyTorch tensor."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def init_weights(m):
    """Apply Xavier initialization to linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ================================================================
# DISRC Controller
# Implements surprise-based intrinsic reward modulation
# ================================================================
class DISRCController:
    def __init__(self, state_dim, alpha=0.03, beta_start=0.04):
        self.alpha = alpha  # Learning rate for moving average of latent setpoint
        self.beta_start = beta_start  # Initial bonus scale
        self.setpoint = np.zeros(state_dim, dtype=np.float32)  # Running latent setpoint

    def compute_bonus(self, encoded_state, episode_ratio):
        """
        Compute negative surprise bonus based on latent deviation.
        Bonus decays over time via episode_ratio.
        """
        s_np = encoded_state.detach().cpu().numpy().squeeze()
        s_norm = s_np / (np.linalg.norm(s_np) + 1e-8)
        sp_norm = self.setpoint / (np.linalg.norm(self.setpoint) + 1e-8)
        deviation = np.linalg.norm(s_norm - sp_norm)
        beta = self.beta_start * (1.0 - episode_ratio ** 1.2)
        bonus = -beta * deviation
        self.setpoint = (1.0 - self.alpha) * self.setpoint + self.alpha * s_np
        return bonus

# ================================================================
# Model Architectures
# Encoder maps input to latent space; DQN outputs Q-values
# ================================================================
class DISRCStateEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, encoded_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class DISRC_DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        return self.net(x)

# ================================================================
# Action Selection (Exploration Strategy)
# ================================================================
def epsilon_greedy(model, state, epsilon, action_space):
    """Random action with probability epsilon; greedy otherwise."""
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return int(torch.argmax(q_values).item())

def soft_update(target, source, tau=0.005):
    """Soft update of target model parameters."""
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ================================================================
# Training Loop
# Core logic for training DISRC-DQN using experience replay
# ================================================================
def train_dqn(env, model, target_model, encoder,
              encoder_optimizer, model_optimizer,
              disrc_controller,
              num_episodes=800, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.1,
              batch_size=128):
    replay_buffer = deque(maxlen=50000)
    all_rewards, losses = [], []
    epsilon = epsilon_start
    reward_norm = 1.0  # Running reward normalization baseline

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=SEED)
        state_arr = preprocess_obs(obs)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)
        done, total_reward = False, 0

        while not done:
            # Select and take action
            encoded_state = encoder(state_tensor)
            action = epsilon_greedy(model, encoded_state, epsilon, env.action_space.n)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Preprocess next observation
            next_state_arr = preprocess_obs(next_obs)
            next_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0)

            # Shape reward using DISRC bonus
            reward_norm = 0.99 * reward_norm + 0.01 * abs(reward)
            normed_reward = reward / (reward_norm + 1e-8)
            bonus = disrc_controller.compute_bonus(encoded_state, episode / num_episodes)
            shaped_reward = np.clip(normed_reward + 0.4 * bonus, -1.0, 1.0)

            # Store transition
            replay_buffer.append((state_tensor.detach(), action, shaped_reward,
                                  next_tensor.detach(), float(done)))
            state_tensor = next_tensor

            # Learning from experience replay
            if len(replay_buffer) >= 5000:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat([ensure_2d_tensor(s) for s in states])
                next_states = torch.cat([ensure_2d_tensor(ns) for ns in next_states])
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                # Encode states
                s_enc = encoder(states)
                ns_enc = encoder(next_states)

                # Compute target Q-values (Double DQN-style)
                with torch.no_grad():
                    next_q = model(ns_enc)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(ns_enc)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                # Compute loss and update model
                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                loss = (q_vals - targets).pow(2).mean()
                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
                model_optimizer.step()
                encoder_optimizer.step()
                losses.append(loss.item())

                # Soft update target model
                soft_update(target_model, model, tau=0.005)

        # Logging
        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * 0.997)

        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"[Ep {episode:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                  f"Eps: {epsilon:.3f} | Loss: {mean_loss:.4f}")

    return all_rewards, losses

# ================================================================
# Main Execution Logic
# Creates environment, initializes models, trains, and plots results
# ================================================================
def main():
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")  # Sparse reward navigation task
    input_dim = 7 * 7 * 3
    action_dim = env.action_space.n

    # Model instantiation
    encoder = DISRCStateEncoder(input_dim=input_dim, encoded_dim=64)
    model = DISRC_DQN(input_size=64, action_size=action_dim)
    target_model = DISRC_DQN(input_size=64, action_size=action_dim)
    target_model.load_state_dict(model.state_dict())

    # Weight initialization
    encoder.apply(init_weights)
    model.apply(init_weights)
    target_model.apply(init_weights)

    # Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # DISRC controller with stronger modulation
    disrc_controller = DISRCController(state_dim=64, alpha=0.1, beta_start=0.08)

    # Train the agent
    rewards, losses = train_dqn(env, model, target_model, encoder,
                                encoder_optimizer, model_optimizer,
                                disrc_controller,
                                num_episodes=1200)  # 1200 episodes total

    # === Metrics ===
    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:])
    success_threshold = 0.8
    episodes_to_threshold = next((i+1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards))
    loss_variance = float(np.var(losses)) if losses else 0.0
    reward_std = float(np.std(rewards))
    auc_reward = float(np.trapezoid(rewards, dx=1))

    # Metric Output
    print("===== METRICS =====")
    print(f"Mean reward in final {final_window} episodes: {mean_final_reward:.2f}")
    print(f"Episodes to first success (>{success_threshold}): {episodes_to_threshold}")
    print(f"Loss variance: {loss_variance:.6f}")
    print(f"Reward standard deviation: {reward_std:.2f}")
    print(f"AUC: {auc_reward:.2f}")
    print("===================")

    # === Plotting ===
    def smooth(data, w=0.9):
        """Exponential smoothing of data."""
        sm, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            sm.append(last)
        return sm

    # Reward and Loss Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss")
    plt.title("Mini-Batch Loss During Training")
    plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("DISRC_MiniGrid_LavaCrossing_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'DISRC_MiniGrid_LavaCrossing_results.png'")

# Entrypoint
if __name__ == "__main__":
    main()
