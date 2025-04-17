import torch
import torch.nn as nn
import torchvision
import math
from typing import Union, Callable
from tqdm.auto import tqdm

class IMMloss(nn.Module):
    """
    IMM loss function using the Laplace kernel.
    """

    def __init__(self, obs_horizon, pred_horizon, num_particles):
        super(IMMloss, self).__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_particles = num_particles
        
    def laplace_kernel(self, x, y, w_scale, eps=0.006, dim_normalize=True):
        """
        Laplace kernel: exp(w_scale * max(||x-y||_2, eps)/D)
        
        Args:
            x, y: input tensors
            w_scale: scaling factor (time-dependent)
            eps: small constant to avoid undefined gradients
            dim_normalize: whether to normalize by dimensionality D
        """
        D = x.shape[-1] if dim_normalize else 1.0
        distance = torch.norm(x - y, p=2, dim=-1)
        # Apply max to avoid zero gradients
        distance = torch.clamp(distance, min=eps)
        return torch.exp(-w_scale * distance / D)
    
    def forward(self, model_outputs, time_weights, stop_gradient_outputs):
        """
        Compute the IMM loss for a batch of model outputs.
        
        Args:
            model_outputs: Dictionary containing:
                - ys_t: outputs from time t to s [B, self.pred_horizon, self.obs_horizon]
                - ys_r: outputs from time r to s [B, self.pred_horizon, self.obs_horizon]
                - w_scale: time-dependent scaling factors [B]
            time_weights: w(s,t) weights [B/M]
            stop_gradient_outputs: Optional dictionary with same structure as model_outputs
                                   containing the detached outputs (θ-)
        """
        
        # Extract batch size and reshape for group processing
        batch_size = model_outputs['ys_t'].shape[0]
        M = self.num_particles
        num_groups = batch_size // M
        
        # Flatten pred_horizon and obs_horizon dimensions before reshaping
        # Reshape tensors to [num_groups, M, D]
        ys_t = model_outputs['ys_t'].reshape(batch_size, -1).reshape(num_groups, M, -1)
        ys_r_stop = stop_gradient_outputs['ys_r'].reshape(batch_size, -1).reshape(num_groups, M, -1)
        w_scale = model_outputs['w_scale'].reshape(num_groups, M)
        

        
        # Reshape time weights to [num_groups] by extracting the first element of each group
        time_weights = time_weights.reshape(num_groups, M)[:,0].reshape(-1)
        
        total_loss = 0.0
        for i in range(num_groups):
            group_loss = 0.0
            
            # Compute the kernel matrices
            for j in range(M):
                for k in range(M):
                    # First term: k(f_s,t^θ(x_t^(i,j)), f_s,t^θ(x_t^(i,k)))
                    term1 = self.laplace_kernel(ys_t[i, j], ys_t[i, k], w_scale[i, j])
                    
                    # Second term: k(f_s,r^θ-(x_r^(i,j)), f_s,r^θ-(x_r^(i,k)))
                    term2 = self.laplace_kernel(ys_r_stop[i, j], ys_r_stop[i, k], w_scale[i, j])
                    
                    # Third term: -2k(f_s,t^θ(x_t^(i,j)), f_s,r^θ-(x_r^(i,k)))
                    term3 = -2.0 * self.laplace_kernel(ys_t[i, j], ys_r_stop[i, k], w_scale[i, j])
                    
                    # Sum up the terms
                    group_loss += term1 + term2 + term3
            
            # Apply time-dependent weighting
            group_loss = group_loss * time_weights[i] / (M * M)
            total_loss += group_loss
        
        # Average over the number of groups
        return total_loss / num_groups

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Second encoder for timestep s (for IMM)
        diffusion_step_encoder_s = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
            
        # Total conditioning dimensions: t embedding + s embedding + global conditioning
        cond_dim = dsed * 2 + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.diffusion_step_encoder_s = diffusion_step_encoder_s
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float],
            timestep_s: Union[torch.Tensor, float],
            global_cond):
        """
        x: (B,T,input_dim)
        timestep: (B,), diffusion step
        timestep_s: (B,), diffusion step s (for IMM)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        # timesteps = timestep
        # if not torch.is_tensor(timesteps):
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)
        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.diffusion_step_encoder(timestep)
        s_emb = self.diffusion_step_encoder_s(timestep_s)

        global_feature = torch.cat([t_emb, s_emb], dim=-1)


        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    

''' Vision Encoder '''
def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class RoboIMM:
    """
    Simplified class for IMM with the robotics UNet.
    """
    def __init__(
        self,
        sigma_data,
        obs_horizon,
        pred_horizon,
        num_particles
    ):
        """
        Initialize the IMM sampler.
        """
        self.sigma_data = sigma_data
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_particles = num_particles
        
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        vision_encoder = get_resnet('resnet18')

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        # agent_pos is 2 dimensional
        lowdim_obs_dim = 2
        # observation feature has 514 dims in total per step
        self.obs_dim = vision_feature_dim + lowdim_obs_dim
        self.action_dim = 2

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_net': noise_pred_net
        })
        device = torch.device('cuda')
        nets = nets.to(device)
        
        self.nets = nets

        self.loss = IMMloss(obs_horizon=obs_horizon, pred_horizon=pred_horizon, num_particles=num_particles)

    def get_alpha_sigma(self, t):
        """Get alpha and sigma values for time t."""
        # Using the "flow matching" schedule
        alpha_t = (1 - t)
        sigma_t = t
        return alpha_t, sigma_t
    
    def ddim(self, yt, y, s, t):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)

        alpha_s = alpha_s.reshape(-1,1,1)
        sigma_s = sigma_s.reshape(-1,1,1)
        alpha_t = alpha_t.reshape(-1,1,1)
        sigma_t = sigma_t.reshape(-1,1,1)
        
        ys = (alpha_s -   alpha_t * sigma_s / sigma_t) * y + sigma_s / sigma_t * yt
        return ys
    
    def sample(self, shape, image, agent_pos, steps=20, sampling_method="ddim"):
        """
        Generate samples using IMM sampling.
        
        Args:
            shape: Shape of the samples to generate
            steps: Number of sampling steps
            global_cond: Global conditioning
            sampling_method: "ddim"
            
        Returns:
            Generated samples
        """
        device = next(self.nets.parameters()).device
        self.nets.eval()
        image = image.to(device)
        agent_pos = agent_pos.to(device)
        
        # vision encoder
        image_features = self.nets['vision_encoder'](image.flatten(end_dim=1))
        
        # (2,512)
        image_features = image_features.reshape(*image.shape[:2],-1)
        
        # (1,2,512)
        
        obs = torch.cat([image_features, agent_pos],dim=-1)
        # (1,2,514)

        x = torch.randn(shape, device=device) * self.sigma_data
        
        # Define time steps (uniform steps from 1 to 0)
        times = torch.linspace(0.994, 0.006, steps + 1, device=device)
                
        for i in range(steps):
            t = times[i]
            s = times[i + 1]
            
            # Create batched time tensors
            t_batch = torch.full((shape[0],), t, device=device)
            s_batch = torch.full((shape[0],), s, device=device)
            
            # Run model forward
            with torch.no_grad():
                x = self.predict(x, t_batch, s_batch, obs.flatten(start_dim=1))
                # noise = nets['noise_pred_net'](
                #     sample=noised_action,
                #     timestep=diffusion_iter,
                #     timestep_s=diffusion_iter,
                #     global_cond=)
            
            # Apply sampling function based on method
            # if sampling_method == "ddim":
            #     x = self.ddim(x, pred, s_batch.view(-1, 1, 1), t_batch.view(-1, 1, 1))
            # else:
            #     raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        return x

    def calculate_weights(self, s_times, t_times):
        """
        Calculate the time-dependent weighting function w(s,t)
        """
        b = 5  # Hyperparameter from paper
        a = 1    # Hyperparameter from paper (a ∈ {1, 2})

        alpha_t, sigma_t = self.get_alpha_sigma(t_times)
        
        # Calculate log-SNR values
        log_snr_t = 2 * torch.log(alpha_t / sigma_t)
        dlog_snr_t = 2 / (torch.square(t_times) - t_times)
        
        # Calculate coefficient based on equation 13
        sigmoid_term = torch.sigmoid(b - log_snr_t)
        
        snr_term = (alpha_t ** a) / (alpha_t ** 2 + sigma_t ** 2)
        
        return 0.5 * sigmoid_term * -1.0 * dlog_snr_t * snr_term
    
    def predict(self, xt, t, s, obs_cond):
        pass
        c = 1000.0
        cskip = 1.0
        cout = -(t-s) * self.sigma_data
        c_timestep = c * t
        c_timestep_s = c * s
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        c_in = (torch.pow(alpha_t, 2) + torch.pow(sigma_t, 2)).rsqrt() / self.sigma_data
        xs = self.nets['noise_net'](xt*c_in.reshape(-1,1,1), c_timestep, c_timestep_s, obs_cond)
        return cskip * xt + cout.reshape(-1,1,1) * xs

    def train(self, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda'):
        """
        Train the model.
        """
        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=lr, weight_decay=0)

        # Cosine LR schedule
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        self.nets.train()
        
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                # batch loop
                with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        nimage = nbatch['image'][:,:self.obs_horizon].to(device)
                        nagent_pos = nbatch['agent_pos'][:,:self.obs_horizon].to(device)
                        naction = nbatch['action'].to(device)
                        B = nagent_pos.shape[0]

                        # get vision features
                        image_features = self.nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(*nimage.shape[:2],-1)

                        # get times
                        num_groups = B // self.num_particles
                        s_times = torch.rand(num_groups, device=device)
                        t_times = s_times + (1 - s_times) * torch.rand(num_groups, device=device)
                        r_times = s_times + (t_times - s_times) * torch.rand(num_groups, device=device)

                        # times need to be shape (B,), currently they are shape (num_groups,)
                        s_times = s_times.reshape(-1,1).expand(num_groups, self.num_particles).reshape(-1)
                        t_times = t_times.reshape(-1,1).expand(num_groups, self.num_particles).reshape(-1)
                        r_times = r_times.reshape(-1,1).expand(num_groups, self.num_particles).reshape(-1)

                        noise = torch.randn_like(naction) * self.sigma_data
                        
                        x_t = self.ddim(yt=noise, y=naction, s=t_times, t=torch.ones_like(t_times))
                        x_r = self.ddim(yt=x_t, y=naction, s=r_times, t=t_times)
                        

                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        

                        optimizer.zero_grad()
                        pred_grad = self.predict(x_t, t_times, s_times, obs_cond)

                        with torch.no_grad():
                            pred_nograd = self.predict(x_r, r_times, s_times, obs_cond)
    
                        time_weights = self.calculate_weights(s_times, t_times)
                        
                        # Reshape predictions to match expected dimensions
                        # Assuming pred_grad and pred_nograd are [B, sequence_length, action_dim]
                        # Reshape to [B, self.pred_horizon, self.obs_horizon]
                        model_outputs = {
                            'ys_t': pred_grad.reshape(B, self.pred_horizon, self.obs_horizon),
                            'w_scale': 1.0 / torch.abs((t_times - s_times) * self.sigma_data)
                        }
                        
                        stop_gradient_outputs = {
                            'ys_r': pred_nograd.reshape(B, self.pred_horizon, self.obs_horizon).detach()
                        }
                        
                        loss = self.loss(model_outputs, time_weights, stop_gradient_outputs)
                        
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                        if batch_idx % 10 == 0:
                            print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item()}")
                        if batch_idx == 0 and epoch_idx % 10 == 0:
                            # save model checkpoint
                            torch.save({
                                'epoch': epoch_idx,
                                'model_state_dict': self.nets.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss.item()
                            }, f"ckpts/model_checkpoint_{epoch_idx}_{batch_idx}_{loss.item()}.pth")
    
    def load_model(self, path):
        state_dict = torch.load(path, map_location='cuda')
        self.nets.load_state_dict(state_dict['model_state_dict'])