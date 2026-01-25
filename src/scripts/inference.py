from torch import nn
import torch
import os
from tqdm import tqdm
from scripts.nn_blocks import device
from scripts.u_net import UNet
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

class InferenceHelper:
    def __init__(self):
        self.T = 1000
        self.num_classes = 10
        self.unet_base_channel = 128
        self.checkpoint_epoch = 25

        self.alphas = torch.linspace(start=0.9999, end=0.98, steps=self.T, dtype=torch.float32).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.label_to_name_map = {
            0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
            5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot",
        }
        self.class_list = list(self.label_to_name_map.values())

        self.emb = nn.Embedding(self.num_classes, self.unet_base_channel*4).to(device)
        self.unet = UNet(
            source_channel=1,
            unet_base_channel=self.unet_base_channel,
            num_norm_groups=32,
        ).to(device)

        # --- Load Weights ---
        try:
            # Get absolute path relative to this file's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_dir = os.path.join(script_dir, "..", "..", "checkpoints")
            unet_path = os.path.join(checkpoint_dir, f"guided_unet_{self.checkpoint_epoch}.pt")
            emb_path = os.path.join(checkpoint_dir, f"guided_embedding_{self.checkpoint_epoch}.pt")
            
            self.unet.load_state_dict(torch.load(unet_path, map_location=device))
            self.emb.load_state_dict(torch.load(emb_path, map_location=device))
            
            self.unet.eval()
            print(f"Successfully loaded checkpoints from '{unet_path}'")
        except FileNotFoundError:
            print(f"Error: Checkpoints not found. Please check the path.")

    def run_inference_ddim(self, class_name, s, n_steps=50, num_row=5, num_col=5, return_images=False, progress_callback=None):
        """
        Run DDIM inference for image generation.
        
        Args:
            class_name: Name of the class to generate (e.g., "Sneaker")
            s: Guidance scale (w parameter)
            n_steps: Number of DDIM steps
            num_row: Number of rows in the grid
            num_col: Number of columns in the grid
            return_images: If True, returns numpy array instead of showing plot
            progress_callback: Optional callback function for progress updates (receives current_step, total_steps)
        
        Returns:
            If return_images=True: numpy array of shape (num_row*num_col, 32, 32) with values in [0, 1]
            If return_images=False: displays matplotlib plot
        """
        self.unet.eval()

        # Time subsequence
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))
        timesteps = sorted(timesteps, reverse=True)
        
        # Start with pure noise
        x = torch.randn(num_row*num_col, 1, 32, 32).to(device)

        with torch.no_grad():
            # Prepare embeddings
            label_idx = list(self.label_to_name_map.values()).index(class_name)
            y = torch.tensor([label_idx] * (num_row * num_col)).to(device)
            
            y_emb_cond = self.emb(y)
            y_emb_uncond = torch.zeros_like(y_emb_cond)
            y_emb_batch = torch.cat([y_emb_cond, y_emb_uncond], dim=0) 

            iterator = tqdm(timesteps, desc="DDIM") if progress_callback is None else enumerate(timesteps)
            
            for i, t in enumerate(timesteps):
                if progress_callback is not None:
                    progress_callback(i, len(timesteps))
                    
                # Time batch
                t_batch = torch.full((num_row*num_col*2,), t, device=device, dtype=torch.long)
                x_batch = torch.cat([x, x], dim=0)
                
                # Predict noise
                eps_batch = self.unet(x_batch, t_batch, y_emb_batch)
                eps_cond, eps_uncond = torch.split(eps_batch, num_row*num_col, dim=0)
                
                # Guidance
                eps = (1.0 + s) * eps_cond - s * eps_uncond

                # Calculate update variables
                alpha_bar_t = self.alpha_bars[t]
                prev_t = t - step_size
                if prev_t >= 0:
                    alpha_bar_prev = self.alpha_bars[prev_t]
                else:
                    alpha_bar_prev = torch.tensor(1.0).to(device)
                
                # 1. Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
                
                # 2. Point to x_{t-1} (Deterministic)
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
                
                # 3. Update x
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        # Convert to numpy array
        x = x.permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy()
        
        if return_images:
            # Return images as numpy array (N, 32, 32)
            return x[:, :, :, 0]
        
        # Plot
        fig, axes = plt.subplots(num_row, num_col, figsize=(6,6))
        fig.suptitle(f"DDIM Results: {class_name} (s={s}, steps={n_steps})")
        for i in range(num_row * num_col):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(x[i, :, :, 0], cmap='gray')
            ax.axis('off')
        plt.show()

    def run_inference_dps(self, class_name, s, measurement, mask, zeta=1.0, steps=50, return_images=False, progress_callback=None):
        """
        Runs Diffusion Posterior Sampling for INPAINTING.
        measurement: Tensor [B, 1, 32, 32] (Range 0-1)
        mask: Tensor [B, 1, 32, 32] (1 = keep, 0 = drop)
        return_images: If True, returns numpy arrays instead of plotting
        progress_callback: Optional callback function for progress updates
        
        Returns (if return_images=True):
            tuple: (reconstruction, measurement_np, mask_np) as numpy arrays
        """
        print(f"Running DPS Inpainting... (Class: {class_name}, Scale: {s})")
        self.unet.eval()

        n = 1
        # Start with pure noise
        x = torch.randn(n, 1, 32, 32).to(device)
        label_idx = list(self.label_to_name_map.values()).index(class_name)
        y = torch.tensor([label_idx] * (n * n)).to(device)
            
        y_emb_cond = self.emb(y)
        y_emb_uncond = torch.zeros_like(y_emb_cond)
        y_emb_batch = torch.cat([y_emb_cond, y_emb_uncond], dim=0) # [Cond, Uncond]

        # Setup Measurement
        y_meas = measurement.to(device).view(1, 1, 32, 32)
        mask_tensor = mask.to(device).view(1, 1, 32, 32)
        
        # Create DDIM Time Schedule (e.g., 999 -> 0 in 50 steps)
        T = 1000
        skip = T // steps
        time_seq = list(reversed(range(0, T, skip)))
            
        # Reverse Loop
        for i,t in enumerate(time_seq):
            if progress_callback is not None:
                progress_callback(i, len(time_seq))
                
            x = x.detach().requires_grad_(True)
            # Time batch
            t_batch = torch.full((n*n*2,), t, device=device, dtype=torch.long)
            
            # Input batch (doubled for CFG)
            x_batch = torch.cat([x, x], dim=0)
            
            # Predict noise
            eps_batch = self.unet(x_batch, t_batch, y_emb_batch)
            eps_cond, eps_uncond = torch.split(eps_batch, n*n, dim=0)
            
            # Guidance
            noise_pred = (1.0 + s) * eps_cond - s * eps_uncond
        
        # --- B. DPS: Gradient Calculation ---
            # 1. Estimate x0 (using current alpha_bar)
            alpha_bar_t = self.alpha_bars[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            # Tweedie's Formula to estimate clean image
            x0_hat = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

            x0_hat = x0_hat.clamp(0, 1)

            # 2. Calculate Loss (Compare x0_hat to measurement)
            # We only look at errors inside the mask
            difference = mask_tensor * (x0_hat - y_meas)
            loss = torch.norm(difference)**2
            
            # 3. Compute Gradient
            grad = torch.autograd.grad(loss, x)[0]
            
            # --- C. DPS: Apply Nudge ---
            # We subtract the gradient from x. 
            # Note: We detach() here because we are done with gradients for this step.
            x_nudged = x.detach() - zeta * grad

            if i % 10 == 0:
                print(f"Step {i}/{steps} | t={t} | Loss: {loss.item():.4f} | |Grad|: {grad.norm().item():.4f} | x_range: [{x.min():.2f}, {x.max():.2f}]")
            
            # --- D. DDIM Update (Deterministic) ---
            # Now we proceed with the standard DDIM update using x_nudged
            
            # Re-calculate prediction on the NUDGED x if desired (Optional but strictly more correct)
            # For speed, we often reuse 'noise_pred' but treating x_nudged as the new base.
            # Let's do the fast approximation: assume noise_pred is still valid for x_nudged.
            
            # Get Previous Timestep
            if i == len(time_seq) - 1:
                t_prev = -1
                alpha_bar_prev = torch.tensor(1.0).to(device)
            else:
                t_prev = time_seq[i+1]
                alpha_bar_prev = self.alpha_bars[t_prev]

            # 1. Re-estimate x0 using the nudged x
            # (This ensures the update trajectory respects the nudge)
            pred_x0 = (x_nudged - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            pred_x0 = pred_x0.clamp(0, 1)
            
            # 2. Point to next step (Direction to x_{t-1})
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            
            # 3. Move to x_{t-1}
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            x = x_prev
        
        # Convert to numpy
        x = x.detach().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        
        if return_images:
            return x[0, :, :, 0]  # Return (32, 32) numpy array
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        fig.suptitle(f"DPS Inpainting Results: {class_name} (s={s}, steps={steps})")
        
        # Measurement (Ground Truth masked)
        axes[0].imshow(measurement.permute(0, 2, 3, 1).cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Measurement")
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask.permute(0, 2, 3, 1).cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Mask")
        axes[1].axis('off')
        
        # Reconstruction
        axes[2].imshow(x[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Reconstruction")
        axes[2].axis('off')
        
        # plt.savefig(f"dps_inpainting_{class_name}_s{s}_steps{steps}.png")
        plt.show()
    
    def load_dataset(self):
        """Load the FashionMNIST dataset and return it."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "..", "data")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),  # Pad 28x28 to 32x32
        ])
        
        dataset = datasets.FashionMNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        return dataset
    
    def get_sample_image(self, dataset, index):
        """
        Get a sample image from the dataset.
        
        Args:
            dataset: FashionMNIST dataset
            index: Index of the image to get
            
        Returns:
            tuple: (image_np, class_name, image_tensor) where image_np is (32, 32) numpy array
        """
        image, label = dataset[index]
        class_name = self.label_to_name_map[label]
        image_np = image.squeeze().numpy()  # (32, 32)
        return image_np, class_name, image
    
    def get_samples_by_class(self, dataset, class_name, num_samples=10):
        """
        Get sample indices for a specific class.
        
        Args:
            dataset: FashionMNIST dataset
            class_name: Name of the class to get samples for
            num_samples: Number of sample indices to return
            
        Returns:
            list: List of indices for samples of the specified class
        """
        label_idx = list(self.label_to_name_map.values()).index(class_name)
        indices = []
        for i, (_, label) in enumerate(dataset):
            if label == label_idx:
                indices.append(i)
            if len(indices) >= num_samples:
                break
        return indices
    
    def create_mask(self, mask_type, mask_size=8, pos_x=None, pos_y=None):
        """
        Create different types of masks for inpainting.
        
        Args:
            mask_type: Type of mask ("center", "top", "bottom", "left", "right", "random", "custom")
            mask_size: Size of the masked region
            pos_x: X position for custom mask (0-31)
            pos_y: Y position for custom mask (0-31)
            
        Returns:
            torch.Tensor: Mask tensor of shape (1, 1, 32, 32), 1=keep, 0=drop
        """
        mask = torch.ones(1, 1, 32, 32)
        
        if mask_type == "center":
            start = (32 - mask_size) // 2
            end = start + mask_size
            mask[:, :, start:end, start:end] = 0.0
        elif mask_type == "top":
            mask[:, :, 2:2+mask_size, :] = 0.0
        elif mask_type == "bottom":
            mask[:, :, 30-mask_size:30, :] = 0.0
        elif mask_type == "left":
            mask[:, :, :, 2:2+mask_size] = 0.0
        elif mask_type == "right":
            mask[:, :, :, 30-mask_size:30] = 0.0
        elif mask_type == "random":
            # Random scattered mask
            random_mask = torch.rand(1, 1, 32, 32) > 0.3
            mask = random_mask.float()
        elif mask_type == "custom":
            # Custom movable square mask
            if pos_x is None:
                pos_x = 12
            if pos_y is None:
                pos_y = 12
            # Clamp positions to valid range
            end_y = min(pos_y + mask_size, 32)
            end_x = min(pos_x + mask_size, 32)
            mask[:, :, pos_y:end_y, pos_x:end_x] = 0.0
        
        return mask