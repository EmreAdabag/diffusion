#!/usr/bin/env python
# coding: utf-8

# Simple Flow Matching Benchmark that outputs results to a text file

import os
import numpy as np
import torch
import collections
import time
from tqdm.auto import tqdm
import argparse
import pickle

# Add safe global for numpy scalar
torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])

# Import directly from your existing fm.py implementation
import fm
from fm import (
    PushTEnv, 
    PushTImageEnv,
    FlowMatchingScheduler,
    normalize_data, 
    unnormalize_data
)

class FlowMatchingModel:
    def __init__(self, 
                 obs_horizon=2, 
                 pred_horizon=16, 
                 action_dim=2, 
                 num_particles=1,
                 seed=None):
        """
        Initialize the Flow Matching model.
        Using existing components from fm.py
        """
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_particles = num_particles
        
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Create flow matching scheduler
        self.flow_scheduler = FlowMatchingScheduler()
        
        # Flag to check if model is reset
        self.is_reset = False
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def reset(self):
        """Reset the model state"""
        self.is_reset = True
        
    def load_model(self, checkpoint_path):
        """Load model weights from checkpoint"""
        # Set weights_only to False to handle numpy scalar objects in the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize vision encoder if not already created
        if not hasattr(self, 'vision_encoder'):
            self.vision_encoder = fm.get_resnet('resnet18')
            self.vision_encoder = fm.replace_bn_with_gn(self.vision_encoder)
            
        # Initialize noise prediction network if not already created
        if not hasattr(self, 'noise_pred_net'):
            # ResNet18 has output dim of 512
            vision_feature_dim = 512
            # agent_pos is 2 dimensional
            lowdim_obs_dim = 2
            # observation feature has 514 dims in total per step
            obs_dim = vision_feature_dim + lowdim_obs_dim
            
            self.noise_pred_net = fm.ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=obs_dim*self.obs_horizon
            )
        
        # Load state dicts from checkpoint
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
            
            # Extract vision_encoder and noise_pred_net parts
            if 'vision_encoder.' in next(iter(model_dict)):
                # The state dict has prefixes
                self.vision_encoder.load_state_dict({k.replace('vision_encoder.', ''): v for k, v in model_dict.items() 
                                            if k.startswith('vision_encoder.')})
                self.noise_pred_net.load_state_dict({k.replace('noise_pred_net.', ''): v for k, v in model_dict.items() 
                                            if k.startswith('noise_pred_net.')})
            else:
                # Direct loading
                nets = torch.nn.ModuleDict({
                    'vision_encoder': self.vision_encoder,
                    'noise_pred_net': self.noise_pred_net
                })
                nets.load_state_dict(model_dict)
                
        # Move models to device
        self.vision_encoder = self.vision_encoder.to(self.device)
        self.noise_pred_net = self.noise_pred_net.to(self.device)
        
        # Set models to evaluation mode
        self.vision_encoder.eval()
        self.noise_pred_net.eval()
        
        # Also load the data stats if available in the checkpoint
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
        
    def sample(self, shape, image=None, agent_pos=None, global_cond=None, steps=100):
        """
        Sample action sequences using flow matching.
        """
        assert self.is_reset, "Model must be reset before sampling"
        
        B = shape[0]
        
        with torch.no_grad():
            # Prepare observation conditioning
            if global_cond is not None:
                # Using state-based approach
                obs_cond = global_cond
            elif image is not None and agent_pos is not None:
                # Using vision-based approach
                # Get image features
                image_features = self.vision_encoder(image.flatten(end_dim=1))
                image_features = image_features.reshape(*image.shape[:2], -1)
                
                # Concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, agent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
            else:
                raise ValueError("Either global_cond or (image, agent_pos) must be provided")
                
            # Initialize x_t from Gaussian noise (x0)
            x_t = torch.randn(shape, device=self.device)
            
            # Set up flow matching integration
            self.flow_scheduler.set_timesteps(steps)
            
            # Integrate the ODE: dx/dt = v(x,t)
            # Using Euler method
            for i, t in enumerate(self.flow_scheduler.timesteps):
                # Create batch of current timestep
                t_batch = torch.ones((B,), device=self.device) * t
                
                # Predict velocity at current point
                v_t = self.noise_pred_net(
                    sample=x_t,
                    timestep=t_batch,
                    global_cond=obs_cond
                )
                
                # Euler integration step
                if i < len(self.flow_scheduler.timesteps) - 1:
                    # dt = t_{i+1} - t_i
                    dt = self.flow_scheduler.timesteps[i + 1] - t
                    # x_{t+dt} = x_t + v_t * dt
                    x_t = x_t + dt * v_t
            
            # The final state is our generated action sequence
            return x_t

def benchmark_flow_matching(checkpoint_path, diffusion_steps=16, mode='vision', episodes=100):
    """
    Benchmark function for flow matching model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        diffusion_steps: Number of steps for the flow matching integration
        mode: 'vision' or 'state' - which observation type to use
        episodes: Number of episodes to run
        
    Returns:
        Dictionary with average reward, average steps, and average inference time
    """
    print(f"Benchmarking flow matching with {diffusion_steps} steps in {mode} mode")
    
    # Load model and statistics
    if mode == 'vision':
        try:
            with open('./vision_data_stats.pkl', 'rb') as f:
                data_stats = pickle.load(f)
        except:
            # If we can't load the stats, we'll try to get them from the checkpoint
            data_stats = None
    else:  # state mode
        try:
            with open('./pusht_data_stats.pkl', 'rb') as f:
                data_stats = pickle.load(f)
        except:
            # If we can't load the stats, we'll try to get them from the checkpoint
            data_stats = None
    
    # Create flow matching model
    model = FlowMatchingModel(
        obs_horizon=2,
        pred_horizon=16,
        action_dim=2,
        num_particles=4,
        seed=12345
    )
    
    # Load checkpoint
    model.load_model(checkpoint_path)
    
    # If we couldn't load the stats file, use the stats from the checkpoint
    if data_stats is None:
        if hasattr(model, 'stats'):
            data_stats = model.stats
        else:
            raise ValueError("Could not find data stats in checkpoint or file")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # limit environment interaction to 350 steps before termination
    max_steps = 350
    
    # Create environment
    if mode == 'state':
        env = PushTEnv(seed=12345)
    elif mode == 'vision':
        env = PushTImageEnv(seed=12345)
    
    # Initialize arrays to store results
    steps = np.zeros(episodes)
    times = np.zeros(episodes)
    rewards = np.zeros(episodes)
    
    with tqdm(total=episodes, desc=f"Evaluating Flow Matching ({diffusion_steps} steps)") as pbar:
        for ep in range(episodes):
            # Reset model
            model.reset()
            
            # Get first observation
            obs, info = env.reset()
            
            # Keep a queue of last 2 steps of observations
            obs_deque = collections.deque([obs] * model.obs_horizon, maxlen=model.obs_horizon)
            
            done = False
            step_idx = 0
            ep_times = []
            
            while not done:
                if mode == 'state':
                    # Stack the last obs_horizon (2) number of observations
                    obs_seq = np.stack(obs_deque)
                    # Normalize observation
                    nobs = normalize_data(obs_seq, stats=data_stats['obs'])
                    # Device transfer
                    nobs = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(device, dtype=torch.float32)
                    
                    # Infer action
                    with torch.no_grad():
                        start_time = time.time()
                        r = model.sample(
                            shape=(1, model.pred_horizon, model.action_dim), 
                            global_cond=nobs, 
                            steps=diffusion_steps
                        )
                        end_time = time.time()
                        ep_times.append(1000 * (end_time - start_time))
                        
                elif mode == 'vision':
                    # Stack images and agent positions
                    images = np.stack([x['image'] for x in obs_deque])
                    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                    
                    # Normalize agent positions
                    nagent_poses = normalize_data(agent_poses, stats=data_stats['agent_pos'])
                    
                    # Images are already normalized in [0,1]
                    images = torch.from_numpy(images).unsqueeze(0).to(device, dtype=torch.float32)
                    nagent_poses = torch.from_numpy(nagent_poses).unsqueeze(0).to(device, dtype=torch.float32)
                    
                    # Infer action
                    with torch.no_grad():
                        start_time = time.time()
                        r = model.sample(
                            shape=(1, model.pred_horizon, model.action_dim), 
                            image=images, 
                            agent_pos=nagent_poses, 
                            steps=diffusion_steps
                        )
                        end_time = time.time()
                        ep_times.append(1000 * (end_time - start_time))
                
                # Unnormalize action
                naction = r.detach().to('cpu').numpy()
                naction = naction[0]
                if mode == 'state':
                    action_pred = unnormalize_data(naction, stats=data_stats['action'])
                elif mode == 'vision':
                    action_pred = unnormalize_data(naction, stats=data_stats['action'])
                
                # Only take action_horizon number of actions
                start = model.obs_horizon - 1
                end = start + 8  # action_horizon = 8
                action = action_pred[start:end, :]
                
                # Execute action_horizon number of steps without replanning
                for i in range(len(action)):
                    # Stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # Save observations
                    obs_deque.append(obs)
                    
                    step_idx += 1
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
            
            steps[ep] = step_idx
            times[ep] = np.mean(ep_times)
            rewards[ep] = reward
            
            pbar.update(1)
            pbar.set_postfix(avg_reward=sum(rewards[:ep+1])/(ep+1))
    
    results = {
        'avg_reward': float(np.mean(rewards)),
        'avg_steps': float(np.mean(steps)),
        'avg_time_ms': float(np.mean(times)),
        'success_rate': float(np.mean(rewards == 1.0))  # Added success rate metric
    }
    
    return results

def main():
    """Main function to run benchmarks with different configurations"""
    parser = argparse.ArgumentParser(description='Flow Matching Benchmark')
    parser.add_argument('--mode', type=str, choices=['vision', 'state'], default='vision',
                        help='Observation mode: vision or state')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32],
                        help='Number of flow matching steps to benchmark')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--checkpoint_epoch', type=int, nargs='+', default=[40, 60, 80, 100],
                        help='Which checkpoint epochs to evaluate')
    parser.add_argument('--output', type=str, default='flow_matching_results.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Open output file
    with open(args.output, 'w') as f:
        f.write(f"Flow Matching Benchmark Results\n")
        f.write(f"Mode: {args.mode}, Episodes: {args.episodes}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Epoch':<10}{'Steps':<10}{'Avg Reward':<15}{'Avg Steps':<15}{'Avg Time (ms)':<15}{'Success Rate':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Run benchmarks for each epoch and step count
        for epoch in args.checkpoint_epoch:
            checkpoint_path = f'/home/imahajan/diffusion/checkpoints/flow_matching_epoch_{epoch}.pt'
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} not found, skipping...")
                continue
            
            for steps in args.steps:
                print(f"\n===== Evaluating checkpoint from epoch {epoch} with {steps} steps =====")
                
                try:
                    results = benchmark_flow_matching(
                        checkpoint_path=checkpoint_path,
                        diffusion_steps=steps,
                        mode=args.mode,
                        episodes=args.episodes
                    )
                    
                    # Print and write results
                    f.write(f"{epoch:<10}{steps:<10}{results['avg_reward']:<15.4f}{results['avg_steps']:<15.1f}{results['avg_time_ms']:<15.2f}{results['success_rate']:<15.4f}\n")
                    f.flush()  # Make sure it's written immediately
                    
                    print(f"Epoch {epoch}, Steps {steps}:")
                    print(f"  Average Reward: {results['avg_reward']:.4f}")
                    print(f"  Average Steps: {results['avg_steps']:.1f}")
                    print(f"  Average Inference Time: {results['avg_time_ms']:.2f} ms")
                    print(f"  Success Rate: {results['success_rate']:.4f}")
                    
                except Exception as e:
                    error_msg = f"Error evaluating epoch {epoch} with {steps} steps: {str(e)}"
                    print(error_msg)
                    f.write(f"{epoch:<10}{steps:<10}ERROR: {str(e)}\n")
                    f.flush()
        
        f.write("=" * 100 + "\n")
        f.write(f"Benchmark completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()