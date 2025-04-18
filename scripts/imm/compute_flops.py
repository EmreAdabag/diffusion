import torch
from thop import profile

# Import the model from state_nn
from state_nn import RoboIMM as ImmState
from vision_nn import RoboIMM as ImmVision

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    immstate = ImmState(
        sigma_data=1.0,
        obs_horizon=2,
        pred_horizon=16,
        num_particles=4
    )

    # Create sample inputs
    batch_size = 1
    obs_horizon = immstate.obs_horizon
    pred_horizon = immstate.pred_horizon

    device = torch.device('cuda')
    noised_action = torch.randn((batch_size, pred_horizon, immstate.action_dim), device=device)
    obs = torch.zeros((batch_size, obs_horizon, immstate.obs_dim), device=device).flatten(start_dim=1)
    diffusion_iter = torch.zeros((batch_size,), device=device)
    diffusion_iter_s = torch.zeros((batch_size,), device=device)
    
    
    # Compute FLOPs
    macs, params = profile(immstate.model, (noised_action, diffusion_iter, diffusion_iter_s, obs))
    print(macs)
    print(params)

    
    # Get total GFLOPs
    gflops = macs / 1e9
    
    print(f"\nState IMM Total GFLOPs: {gflops:.4f}")

    immvision = ImmVision(
        sigma_data=1.0,
        obs_horizon=2,
        pred_horizon=16,
        num_particles=4
    )
    

    vision_obs = torch.zeros((1,2,514), device=device)
    image = torch.zeros((1, obs_horizon,3,96,96), device=device)

    # For ResNet, we need to pass a single argument, not a tuple
    vision_encoder = immvision.nets['vision_encoder']
    vision_input = image.flatten(end_dim=1)  # Flatten the batch and time dimensions
    vision_encoder_macs, vision_encoder_params = profile(vision_encoder, inputs=(vision_input,))
    
    noise_net_macs, noise_net_params = profile(immvision.nets['noise_net'], inputs=(noised_action, diffusion_iter, diffusion_iter_s, vision_obs.flatten(start_dim=1)))
    
    total_macs = vision_encoder_macs + noise_net_macs
    total_params = vision_encoder_params + noise_net_params
    gflops = total_macs / 1e9
    
    print(f"\nVision IMM Total GFLOPs: {gflops:.4f}")

    

if __name__ == "__main__":
    main() 