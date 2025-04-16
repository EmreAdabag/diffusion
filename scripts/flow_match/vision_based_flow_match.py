"""flow_matching_vision_inference.py

Training and inference for vision-based flow matching model.
"""

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import zarr
from tqdm import tqdm
import collections
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import os
from pusht_env import PushTImageEnv
#from flow_matching_vision_based import VisionEncoder, ConditionalUnet1D, ShortcutModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        print("Available keys in dataset:", list(dataset_root.keys()))
        print("Available keys in data:", list(dataset_root['data'].keys()) if 'data' in dataset_root else "No data group found")
        
        # All demonstration episodes are concatenated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'image': dataset_root['data']['img'][:],  # Changed from 'images' to 'img'
            'agent_pos': dataset_root['data']['state'][:,0:2]  # First two dimensions of state are agent position
        }
        print("Loaded data shapes:")
        print("- action:", train_data['action'].shape)
        print("- image:", train_data['image'].shape)
        print("- agent_pos:", train_data['agent_pos'].shape)
        
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        
        # Normalize actions
        stats['action'] = get_data_stats(train_data['action'])
        normalized_train_data['action'] = normalize_data(train_data['action'], stats['action'])
        
        # Normalize images to [-1, 1]
        normalized_train_data['image'] = train_data['image'].astype(np.float32) / 127.5 - 1.0
        
        # Normalize agent positions
        stats['agent_pos'] = get_data_stats(train_data['agent_pos'])
        normalized_train_data['agent_pos'] = normalize_data(train_data['agent_pos'], stats['agent_pos'])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon]
        return nsample

def create_sample_indices(episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

class VisionEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=512, obs_horizon=2):
        super().__init__()
        self.obs_horizon = obs_horizon
        
        # Improved CNN encoder with better architecture
        self.encoder = nn.Sequential(
            # Input: (batch_size, channels, image_size, image_size)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.Mish(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.Mish(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, obs_horizon, height, width, channels) or (batch_size, height, width, channels)
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            # Process each timestep independently
            features = []
            for t in range(T):
                # Permute dimensions to (batch_size, channels, height, width)
                x_t = x[:, t].permute(0, 3, 1, 2)
                feat = self.encoder(x_t)
                features.append(feat)
            # Stack features along time dimension
            x = torch.stack(features, dim=1)
            # Flatten time and feature dimensions
            x = x.reshape(B, -1)
        else:
            # Single image case
            # Permute dimensions to (batch_size, channels, height, width)
            x = x.permute(0, 3, 1, 2)
            x = self.encoder(x)
        return x

class VisionShortcutModel():
    def __init__(self, model=None, vision_encoder=None, num_steps=1000, device='cuda'):
        self.model = model
        self.vision_encoder = vision_encoder
        self.N = num_steps
        self.device = device

    def save_checkpoint(self, epoch, optimizer, scheduler, loss, path='checkpoint.pt'):
        """Save a checkpoint of the training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'vision_encoder_state_dict': self.vision_encoder.state_dict() if self.vision_encoder is not None else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """Load a checkpoint and restore the training state"""
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(path, weights_only=True)
        except Exception as e:
            print(f"Warning: Could not load checkpoint with weights_only=True. Attempting with weights_only=False (less secure).")
            print(f"Error was: {str(e)}")
            # Fall back to weights_only=False if needed
            checkpoint = torch.load(path, weights_only=False)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.vision_encoder is not None and checkpoint['vision_encoder_state_dict'] is not None:
            self.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def get_train_tuple(self, z0=None, z1=None, obs=None):
        """
        Get training tuple for flow matching.
        
        Args:
            z0: Noise tensor of shape (batch_size, sequence_length, feature_dim)
            z1: Target action tensor of shape (batch_size, sequence_length, feature_dim)
            obs: Observation tensor of shape (batch_size, obs_horizon, height, width, channels)
            
        Returns:
            z_t: Noisy action at time t
            t: Time of interpolation (batch_size,)
            target: Target vector field
            distance: Zero distance for regular flow matching 
            global_cond: Encoded observations
        """
        B = z0.shape[0]
        distance = torch.zeros((B,), device=z0.device)
        # Sample t uniformly from [0, 1]
        t = torch.rand((B,), device=z0.device)
        # Linearly interpolate between z0 and z1 at time t
        t_view = t.view(B, 1, 1)
        z_t = t_view * z1 + (1 - t_view) * z0
        # Target vector field is (z1 - z0)
        target = z1 - z0
        
        # Encode observations
        global_cond = self.vision_encoder(obs)
        
        return z_t, t, target, distance, global_cond

    def get_shortcut_train_tuple(self, z0=None, z1=None, obs=None):
        """
        Get training tuple for shortcut flow matching.
        
        Args:
            z0: Noise tensor of shape (batch_size, sequence_length, feature_dim)
            z1: Target action tensor of shape (batch_size, sequence_length, feature_dim)
            obs: Observation tensor of shape (batch_size, obs_horizon, height, width, channels)
            
        Returns:
            z_t: Noisy action at time t
            t: Time of interpolation (batch_size,)
            target: Target vector field
            distance: Shortcut distance (batch_size,)
            global_cond: Encoded observations
        """
        B = z0.shape[0]
        # Sample log distance for shortcut training
        log_distance = torch.randint(low=0, high=7, size=(B,), device=z0.device).float()
        distance = torch.pow(2, -1 * log_distance).to(z0.device)
        
        # Sample t uniformly from [0, 1]
        t = torch.rand((B,), device=z0.device)
        t_view = t.view(B, 1, 1)
        z_t = t_view * z1 + (1 - t_view) * z0
        
        # Encode observations
        global_cond = self.vision_encoder(obs)
        
        # Predict vector field at t
        s_t = self.model(z_t, t, distance=distance, global_cond=global_cond)
        
        # Advance to t + distance
        z_tpd = z_t + s_t * distance.view(B, 1, 1)
        tpd = t + distance
        
        # Predict vector field at t + distance
        s_tpd = self.model(z_tpd, tpd, distance=distance, global_cond=global_cond)
        
        # Target is average of the two vector fields
        target = (s_t.detach().clone() + s_tpd.detach().clone()) / 2

        return z_t, t, target, distance * 2, global_cond

    @torch.no_grad()
    def sample_ode_shortcut(self, z0=None, obs=None, N=None):
        """
        Sample trajectory using shortcut ODE solver.
        
        Args:
            z0: Initial noise tensor of shape (batch_size, sequence_length, feature_dim)
            obs: Observation tensor of shape (batch_size, obs_horizon, height, width, channels)
            N: Number of steps to take
            
        Returns:
            traj: List of tensors representing the trajectory
        """
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []
        z = z0.detach().clone()
        B = z.shape[0]
        
        # Create a tensor for dt with the right shape and device
        dt_tensor = torch.ones((B,), device=self.device) * dt
        
        # Encode observations
        global_cond = self.vision_encoder(obs)
        
        traj.append(z.detach().clone())
        
        for i in range(N):
            t = torch.ones((B,), device=self.device) * i / N
            # Predict vector field at current t
            pred = self.model(z, t, distance=dt_tensor, global_cond=global_cond)
            # Update z using Euler method
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        return traj

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim=2, global_cond_dim=None):
        super().__init__()
        
        # Save input parameters
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        
        # Size of hidden layers
        hidden_dim = 256
        
        # Conditioning dimensions
        cond_dim = 128
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Distance embedding (for shortcut model)
        self.dist_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Global conditioning projection
        if global_cond_dim is not None:
            self.global_proj = nn.Sequential(
                nn.Linear(global_cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        
        # Embed action
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Combine embeddings and process with MLP
        combined_dim = hidden_dim + cond_dim
        if global_cond_dim is not None:
            combined_dim += cond_dim
            
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t, distance=None, global_cond=None):
        # Get batch size and sequence length
        batch_size = x.shape[0]
        seq_len = x.shape[1] if x.dim() > 2 else 1
        feature_dim = x.shape[-1]
        
        # If action is 3D (batch, seq, feat), flatten sequence and feature for processing
        # Otherwise, keep as is (batch, feat)
        if x.dim() == 3:
            # Reshape to (batch*seq, feat)
            x_flat = x.reshape(-1, feature_dim)
        else:
            x_flat = x
        
        # Embed time and expand if needed
        t_embed = self.time_embed(t.view(-1, 1))  # (batch, embed_dim)
        
        # If processing sequence, expand time embedding to match
        if x.dim() == 3:
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, embed_dim)
            t_embed = t_embed.reshape(-1, t_embed.size(-1))  # (batch*seq, embed_dim)
        
        # Embed distance if provided (for shortcut model)
        if distance is not None:
            d_embed = self.dist_embed(distance.view(-1, 1))  # (batch, embed_dim)
            
            # If processing sequence, expand distance embedding to match
            if x.dim() == 3:
                d_embed = d_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, embed_dim)
                d_embed = d_embed.reshape(-1, d_embed.size(-1))  # (batch*seq, embed_dim)
                
            t_embed = t_embed + d_embed
            
        # Project global conditioning if provided
        if global_cond is not None and self.global_cond_dim is not None:
            g_embed = self.global_proj(global_cond)  # (batch, embed_dim)
            
            # If processing sequence, expand global conditioning to match
            if x.dim() == 3:
                g_embed = g_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, embed_dim)
                g_embed = g_embed.reshape(-1, g_embed.size(-1))  # (batch*seq, embed_dim)
        else:
            g_embed = None
        
        # Embed action
        h = self.input_embed(x_flat)  # (batch*seq, hidden_dim) or (batch, hidden_dim)
        
        # Combine embeddings
        h_list = [h, t_embed]
        if g_embed is not None:
            h_list.append(g_embed)
        
        h = torch.cat(h_list, dim=-1)
        
        # Process with MLP
        out = self.mlp(h)
        
        # If input was 3D, reshape output back to match
        if x.dim() == 3:
            out = out.reshape(batch_size, seq_len, feature_dim)
            
        return out

def main():
    # Parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_epochs = 20
    batch_size = 256
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    checkpoint_dir = 'checkpoints'
    checkpoint_freq = 5

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataset
    try:
        dataset = PushTImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        print(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize models
    vision_encoder = VisionEncoder(
        input_channels=3,
        output_dim=512,  # Increased from 32 to match the improved architecture
        obs_horizon=obs_horizon  # Pass obs_horizon to VisionEncoder
    ).to(device)
    vision_encoder.train()

    # Calculate the actual global conditioning dimension
    # When obs_horizon=2, and each image produces 512 features,
    # the total flattened feature dim will be 512*2 = 1024
    global_cond_dim = 512 * obs_horizon
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,  # action dimension
        global_cond_dim=global_cond_dim  # Match vision encoder output dim × obs_horizon
    ).to(device)
    noise_pred_net.train()

    # Create flow matching model with vision encoder
    flow_model = VisionShortcutModel(
        model=noise_pred_net,
        vision_encoder=vision_encoder,
        num_steps=100,
        device=device
    )

    # Create optimizer with parameters from both model and vision encoder
    optimizer = torch.optim.AdamW(
        list(noise_pred_net.parameters()) + list(vision_encoder.parameters()),
        lr=1e-4,
        weight_decay=1e-1
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Try to load existing checkpoint
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}")
        start_epoch, best_loss = flow_model.load_checkpoint(
            checkpoint_path, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch} with loss {best_loss}")

    # Training loop
    try:
        avg_loss = float('inf')  # Initialize avg_loss
        with tqdm(range(start_epoch, num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                rf_epoch_loss = []
                sc_epoch_loss = []

                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # Move data to device
                        nimage = nbatch['image'].to(device)  # Shape: (B, T, H, W, C)
                        naction = nbatch['action'].to(device)  # Shape: (B, seq_len, 2)
                        
                        # First batch - print shapes for debugging
                        if epoch_idx == 0 and len(rf_epoch_loss) == 0:
                            print("\nFirst batch shapes:")
                            print(f"nimage: {nimage.shape}")
                            print(f"naction: {naction.shape}")
                            print(f"global_cond_dim: {global_cond_dim}")
                        
                        B = nimage.shape[0]
                        pred_horizon_len = naction.shape[1]

                        # Sample noise
                        noise = torch.randn(naction.shape, device=device)

                        # Regular flow matching step
                        z_t, t, target, distance, global_cond = flow_model.get_train_tuple(
                            z0=noise, z1=naction, obs=nimage)
                            
                        # First batch - print shapes for debugging
                        if epoch_idx == 0 and len(rf_epoch_loss) == 0:
                            print(f"z_t: {z_t.shape}")
                            print(f"t: {t.shape}")
                            print(f"target: {target.shape}")
                            print(f"distance: {distance.shape}")
                            print(f"global_cond: {global_cond.shape}")
                            
                        pred = flow_model.model(
                        z_t, t, distance=distance,
                        global_cond=global_cond)
                            
                        # First batch - print shapes for debugging
                        if epoch_idx == 0 and len(rf_epoch_loss) == 0:
                             print(f"pred: {pred.shape}")
                            
                        loss = nn.functional.mse_loss(pred, target)
                        loss.backward()

                        # Shortcut training step
                        noise_shortcut = torch.randn(naction.shape, device=device)
                        z_t, t, target, distance, global_cond = flow_model.get_shortcut_train_tuple(
                            z0=noise_shortcut, z1=naction, obs=nimage)
                        pred_shortcut = flow_model.model(
                            z_t, t, distance=distance,
                            global_cond=global_cond)
                        loss_shortcut = nn.functional.mse_loss(pred_shortcut, target)
                        loss_shortcut.backward()

                        # Optimize
                        optimizer.step()
                        optimizer.zero_grad()

                        # Logging
                        rf_loss_cpu = loss.item()
                        sc_loss_cpu = loss_shortcut.item()
                        rf_epoch_loss.append(rf_loss_cpu)
                        sc_epoch_loss.append(sc_loss_cpu)
                        tepoch.set_postfix(
                            rf_loss=rf_loss_cpu,
                            sc_loss=sc_loss_cpu
                        )

                # Step scheduler
                scheduler.step()

                # Calculate average loss
                avg_loss = np.mean(rf_epoch_loss) + np.mean(sc_epoch_loss)

                tglobal.set_postfix(
                    rf_loss=np.mean(rf_epoch_loss),
                    sc_loss=np.mean(sc_epoch_loss),
                    lr=optimizer.param_groups[0]['lr']
                )

                # Save checkpoint if best so far
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    flow_model.save_checkpoint(
                        epoch_idx + 1, optimizer, scheduler, avg_loss,
                        os.path.join(checkpoint_dir, 'best.pt')
                    )

                # Save periodic checkpoint
                if (epoch_idx + 1) % checkpoint_freq == 0:
                    flow_model.save_checkpoint(
                        epoch_idx + 1, optimizer, scheduler, avg_loss,
                        os.path.join(checkpoint_dir, f'epoch_{epoch_idx+1}.pt')
                    )

                # Always save latest checkpoint
                flow_model.save_checkpoint(
                    epoch_idx + 1, optimizer, scheduler, avg_loss,
                    os.path.join(checkpoint_dir, 'latest.pt')
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise  # Re-raise the exception to see the full traceback
    finally:
        if 'epoch_idx' in locals() and 'avg_loss' in locals():
            # Save final model only if we have completed at least one epoch
            flow_model.save_checkpoint(
                epoch_idx + 1, optimizer, scheduler, avg_loss,
                os.path.join(checkpoint_dir, 'final.pt')
            )

def evaluate():
    # Evaluation parameters
    max_steps = 300
    obs_horizon = 2
    action_horizon = 8
    pred_horizon = 16

    # Calculate the global conditioning dimension
    global_cond_dim = 512 * obs_horizon

    # Load models
    vision_encoder = VisionEncoder(
        input_channels=3,
        output_dim=512,  # Match the training architecture
        obs_horizon=obs_horizon
    ).to(device)
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=global_cond_dim  # Match vision encoder output dim × obs_horizon
    ).to(device)

    # Create flow model
    flow_model = VisionShortcutModel(
        model=noise_pred_net,
        vision_encoder=vision_encoder,
        num_steps=100,
        device=device
    )

    # Try to load the best checkpoint
    checkpoint_path = os.path.join('checkpoints', 'best.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join('checkpoints', 'final.pt')
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found for evaluation")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    flow_model.load_checkpoint(checkpoint_path)
    flow_model.model.eval()
    flow_model.vision_encoder.eval()

    env = PushTImageEnv()
    env.seed(100000)

    obs, info = env.reset()
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = []
    done = False
    step_idx = 0

    # Get dataset for normalization stats
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    try:
        dataset = PushTImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon
        )
    except Exception as e:
        print(f"Failed to load dataset for stats: {e}")
        return

    with tqdm(total=max_steps, desc="Evaluating") as pbar:
        while not done:
            # Stack observations
            obs_stack = []
            for obs_dict in obs_deque:
                img = obs_dict['image']
                obs_stack.append(img)
            
            # Convert to torch tensor
            obs_tensor = torch.from_numpy(np.stack(obs_stack)).float().to(device)
            # Add batch dimension
            obs_tensor = obs_tensor.unsqueeze(0)  # Shape: [1, obs_horizon, H, W, C]

            # Sample actions
            with torch.no_grad():
                # Generate noise as starting point
                noisy_action = torch.randn(
                    (1, pred_horizon, 2),  # [batch_size, sequence_length, action_dim]
                    device=device
                )
                
                # Generate trajectory using the model
                ode_traj = flow_model.sample_ode_shortcut(
                    z0=noisy_action,
                    obs=obs_tensor
                )
                
                # Get final prediction
                naction = ode_traj[-1]  # Shape: [1, pred_horizon, 2]

            # Convert to numpy and unnormalize
            naction_np = naction[0].cpu().numpy()  # Remove batch dimension
            action_pred = unnormalize_data(
                naction_np,
                dataset.stats['action']
            )

            # Select actions to execute
            start = obs_horizon - 1
            end = start + action_horizon
            action_sequence = action_pred[start:end, :]  # Shape: [action_horizon, 2]

            # Execute each action
            for i in range(len(action_sequence)):
                action = action_sequence[i]
                obs, reward, done, _, info = env.step(action)
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    print('Max reward:', max(rewards))
    print('Average reward:', np.mean(rewards))
    try:
        from skvideo.io import vwrite
        vwrite('vis.mp4', imgs)
        print("Saved evaluation video to vis.mp4")
    except ImportError:
        print("scikit-video not available, video not saved")
    except Exception as e:
        print(f"Error saving video: {e}")

if __name__ == "__main__":
    import os
    if os.path.exists(os.path.join('checkpoints', 'best.pt')):
        print("Found saved checkpoint, running evaluation...")
        evaluate()
    else:
        print("No checkpoint found, starting training...")
        main() 