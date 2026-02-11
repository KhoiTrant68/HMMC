import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms

# Ensure these match your file structure
from model.hmmc import HMMC

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HMMC MoE Router Decisions")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth.tar)")
    parser.add_argument("--save_dir", type=str, default="vis_results", help="Directory to save results")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the expert overlay (0.0 to 1.0)")
    return parser.parse_args()

def load_image(path, device):
    """Loads image, pads to 64, returns tensor and ORIGINAL size."""
    img = Image.open(path).convert("RGB")
    original_size = (img.height, img.width) # (H, W)
    
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Pad to multiple of 64
    h, w = x.shape[2:]
    p_h = (64 - (h % 64)) % 64
    p_w = (64 - (w % 64)) % 64
    
    # F.pad expects (left, right, top, bottom)
    if p_h > 0 or p_w > 0:
        x = F.pad(x, (0, p_w, 0, p_h), mode='reflect')
        
    return x, original_size

def clean_state_dict(state_dict):
    """Removes 'module.' prefix if present (from DataParallel/Accelerate)."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def get_model_params_from_state_dict(state_dict):
    """Infers N and M from the checkpoint weights."""
    try:
        # HMMC specific: 
        # h_a.0.conv_down.weight is [N, M, 5, 5]
        # g_a.6.weight is [M, 256, 5, 5] (usually)
        if "h_a.0.conv_down.weight" in state_dict:
            N = state_dict["h_a.0.conv_down.weight"].shape[0]
            M = state_dict["h_a.0.conv_down.weight"].shape[1]
            print(f"Detected hyperparameters from checkpoint: N={N}, M={M}")
            return N, M
    except Exception as e:
        print(f"Could not infer params: {e}")
    
    print("Warning: Using default N=192, M=320. If incorrect, reconstruction will be garbage.")
    return 192, 320

def get_expert_colormap(num_experts=4):
    # Distinct colors: Red, Green, Blue, Yellow
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00'] 
    return mcolors.ListedColormap(colors[:num_experts])

def overlay_expert_map(img_tensor, expert_indices, alpha=0.4, num_experts=4):
    """
    img_tensor: [1, 3, H, W] (The Reconstruction)
    expert_indices: [H_feat, W_feat]
    """
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # H, W, 3
    H, W, _ = img_np.shape

    # Resize expert map to Image Size
    expert_map_tensor = expert_indices.unsqueeze(0).unsqueeze(0).float()
    expert_map_up = F.interpolate(expert_map_tensor, size=(H, W), mode='nearest')
    expert_map_up = expert_map_up.squeeze().cpu().numpy()

    # Colorize
    cmap = get_expert_colormap(num_experts)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_experts + 0.5, 1), cmap.N)
    colored_map = cmap(norm(expert_map_up)) # [H, W, 4] (RGBA)
    
    overlay_rgb = colored_map[..., :3]
    
    # Blend
    blended = (1 - alpha) * img_np + alpha * overlay_rgb
    blended = np.clip(blended, 0, 1)
    
    return blended, expert_map_up

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Device Setup (CRITICAL FOR HMMC)
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. HMMC (Mamba/Triton) usually fails on CPU.")
        device = "cpu"
    else:
        device = "cuda"
    print(f"Using device: {device}")

    # 2. Load Checkpoint & Clean
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)

    # 3. Initialize Model with Correct Params
    N, M = get_model_params_from_state_dict(state_dict)
    model = HMMC(N=N, M=M)
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed, trying strict=False (Risky):\n{e}")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    # 4. Update Scale Tables (Important for correct Gaussian params)
    model.update(force=True)

    # 5. Load Image
    x, (orig_h, orig_w) = load_image(args.image, device)

    # 6. Inference
    print("Running inference...")
    with torch.no_grad():
        # Using 'ste' mimics the rounding during compression
        out = model(x, training_mode="ste")
        
        # Get raw reconstruction
        x_hat_padded = out["x_hat"]
        
        # CROP back to original size (remove padding)
        x_hat = x_hat_padded[:, :, :orig_h, :orig_w]
        x_orig_cropped = x[:, :, :orig_h, :orig_w]
        
        router_logits = out["router_logits"]

    if router_logits is None:
        print("No router logits returned. Check if the model has MoE layers activated.")
        return

    # 7. Visualization
    num_layers = len(router_logits)
    print(f"Visualizing {num_layers} MoE layers...")

    # Layout calculation
    cols = 4
    # Layers + (Original, Recon, Legend)
    rows = ((num_layers + 2) // cols) + 1
    if (num_layers + 2) % cols != 0: rows += 1
    
    fig = plt.figure(figsize=(20, 5 * rows))

    # Plot 1: Original
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(x_orig_cropped.squeeze().permute(1, 2, 0).cpu().numpy())
    ax.set_title("Original Input")
    ax.axis('off')

    # Plot 2: Reconstruction
    ax = fig.add_subplot(rows, cols, 2)
    x_hat_np = x_hat.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    ax.imshow(x_hat_np)
    ax.set_title(f"Reconstruction\nBPP: {out.get('bpp_loss', 0):.3f}")
    ax.axis('off')
    
    # Plot 3: Legend
    ax = fig.add_subplot(rows, cols, 3)
    cmap = get_expert_colormap(4)
    patches = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(4)]
    ax.legend(patches, [f"Expert {i}" for i in range(4)], loc='center')
    ax.axis('off')
    ax.set_title("Expert Map Legend")

    # Plot MoE Layers
    layer_names = [f"Std Slice {i}" for i in range(num_layers - 2)] + ["Anchor", "Non-Anchor"]
    
    for i, layer_data in enumerate(router_logits):
        if layer_data is None: continue
            
        # Data format from HMMC: (logits, indices)
        # indices shape: [B, H, W, K]
        logits, indices = layer_data
        
        # Get Top-1 Expert Index
        # We need to reshape/permute to match image orientation if needed, 
        # but usually [B, H, W, K] corresponds to spatial layout.
        expert_choice_map = indices[0, :, :, 0] # Take 1st batch, Top-1 expert
        
        # Overlay on the RECONSTRUCTION (x_hat), not padded x_hat
        # Note: expert map is based on PADDED feature map.
        # We overlay on padded reconstruction first, then crop result.
        blended_padded, raw_map = overlay_expert_map(x_hat_padded.clamp(0,1), expert_choice_map, alpha=args.alpha)
        
        # Crop the blended image
        blended_cropped = blended_padded[:orig_h, :orig_w, :]
        
        ax = fig.add_subplot(rows, cols, i + 4)
        ax.imshow(blended_cropped)
        
        # Usage stats
        unique, counts = np.unique(raw_map, return_counts=True)
        total = raw_map.size
        stats = ", ".join([f"E{int(u)}:{c/total:.0%}" for u, c in zip(unique, counts)])
        
        name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        ax.set_title(f"{name}\n{stats}")
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, f"vis_{os.path.basename(args.image)}")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()