import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms

# Import your model
from model.hmmc import HMMC

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HMMC MoE Router Decisions")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth.tar)")
    parser.add_argument("--save_dir", type=str, default="vis_results", help="Directory to save results")
    parser.add_argument("--alpha", type=float, default=0.4, help="Opacity of the expert overlay (0.0 to 1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_image(path, device):
    """Loads an image and pads it to be divisible by 64 (standard for VAEs)."""
    img = Image.open(path).convert("RGB")
    original_size = img.size
    
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Pad to multiple of 64 to avoid shape mismatch errors during downsampling
    h, w = x.shape[2:]
    p_h = (64 - (h % 64)) % 64
    p_w = (64 - (w % 64)) % 64
    if p_h > 0 or p_w > 0:
        x = F.pad(x, (0, p_w, 0, p_h), mode='reflect')
        
    return x, original_size

def get_expert_colormap(num_experts=4):
    """Defines distinct colors for each expert."""
    # Expert 0: Red, 1: Green, 2: Blue, 3: Yellow
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    
    # Extend if more experts
    if num_experts > 4:
        # Use a qualitative colormap from matplotlib
        cmap = plt.get_cmap('tab10')
        colors = [mcolors.rgb2hex(cmap(i)) for i in range(num_experts)]
        
    return mcolors.ListedColormap(colors[:num_experts])

def overlay_expert_map(img_tensor, expert_indices, alpha=0.4, num_experts=4):
    """
    Overlays expert indices onto the image.
    img_tensor: [1, 3, H, W]
    expert_indices: [H_feat, W_feat] (Integer tensor 0..N)
    """
    # 1. Convert Image to Numpy (H, W, 3)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    H, W, _ = img_np.shape

    # 2. Resize Expert Map to Image Size (Nearest Neighbor to keep integers)
    # expert_indices is [H_feat, W_feat] -> needs to be [1, 1, H_feat, W_feat] for interpolate
    expert_map_tensor = expert_indices.unsqueeze(0).unsqueeze(0).float()
    
    # Upsample
    expert_map_up = F.interpolate(expert_map_tensor, size=(H, W), mode='nearest')
    expert_map_up = expert_map_up.squeeze().cpu().numpy() # [H, W]

    # 3. Create Color Overlay
    cmap = get_expert_colormap(num_experts)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_experts + 0.5, 1), cmap.N)
    
    # Apply colormap to indices -> [H, W, 4] (RGBA)
    colored_map = cmap(norm(expert_map_up))
    
    # 4. Blend
    # Convert RGBA to RGB for blending
    overlay_rgb = colored_map[..., :3]
    
    # Simple alpha blending
    blended = (1 - alpha) * img_np + alpha * overlay_rgb
    blended = np.clip(blended, 0, 1)
    
    return blended, expert_map_up

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Attempt to infer N/M from checkpoint if possible, else default
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    model = HMMC(N=192, M=320) # Adjust M/N if your config differs
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    # 2. Load Image
    x, (orig_w, orig_h) = load_image(args.image, args.device)
    
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        out = model(x, training_mode="ste") # Use 'ste' to get discrete codes if needed, or 'noise'
        x_hat = out["x_hat"]
        router_logits = out["router_logits"] 
        # router_logits structure: Tuple of (logits, indices)
        # indices shape: [B, H, W, K]

    if router_logits is None:
        print("Error: Model did not return router_logits. Ensure 'router_logits' is in the return dict of HMMC.")
        return

    # 4. Visualization
    num_layers = len(router_logits)
    print(f"Found {num_layers} MoE layers.")
    
    # Prepare Plot
    # Row 1: Original | Reconstruction
    # Row 2+: MoE Maps
    
    cols = 4 
    rows = (num_layers // cols) + 2
    
    fig = plt.figure(figsize=(20, 5 * rows))
    
    # Original
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(x.squeeze().permute(1, 2, 0).cpu().numpy())
    ax.set_title("Original Input")
    ax.axis('off')

    # Reconstruction
    ax = fig.add_subplot(rows, cols, 2)
    ax.imshow(x_hat.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy())
    ax.set_title(f"Reconstruction\nBPP Loss: {out.get('bpp_loss', 0):.4f}")
    ax.axis('off')
    
    # Legend
    ax = fig.add_subplot(rows, cols, 3)
    cmap = get_expert_colormap(4)
    patches = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(4)]
    ax.legend(patches, [f"Expert {i}" for i in range(4)], loc='center', fontsize='large')
    ax.axis('off')
    ax.set_title("Expert Color Legend")

    # 5. Process Each MoE Layer
    # HMMC Structure: Standard Slices -> Anchor -> Non-Anchor
    layer_names = [f"Standard Slice {i}" for i in range(num_layers - 2)] + ["Anchor Slice", "Non-Anchor Slice"]
    
    for i, layer_data in enumerate(router_logits):
        if layer_data is None: 
            continue
            
        logits, indices = layer_data
        # indices shape: [B, H_feat, W_feat, K=2]
        
        # We visualize the PRIMARY expert (k=0)
        primary_expert_indices = indices[0, :, :, 0] # [H_feat, W_feat]
        
        blended_img, raw_map = overlay_expert_map(x_hat.clamp(0,1), primary_expert_indices, alpha=args.alpha)
        
        # Plot
        ax = fig.add_subplot(rows, cols, i + 5) # Start after row 1
        ax.imshow(blended_img)
        
        # Calculate usage stats for title
        unique, counts = np.unique(raw_map, return_counts=True)
        total_pixels = raw_map.size
        stats = ", ".join([f"E{int(u)}:{c/total_pixels:.0%}" for u, c in zip(unique, counts)])
        
        title = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        ax.set_title(f"{title}\nTop-1 Choice\n({stats})")
        ax.axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(args.save_dir, f"moe_vis_{os.path.basename(args.image)}")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()