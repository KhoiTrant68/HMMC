import torch
from ptflops import get_model_complexity_info

from model.hmmc import HMMC


# ------------------------------
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output["x_hat"]
        return output


def test_dcae_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    try:
        model = HMMC(N=192, M=320).to(device).eval()
        # CompressAI-like models require CDF update; keeping your logic here
        if hasattr(model, "update"):
            model.update(force=True)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    wrapped_model = Wrapper(model)

    print("\n--- 1. Input/Output Check ---")
    print(f"Input shape: {input_tensor.shape}")
    with torch.no_grad():
        out = wrapped_model(input_tensor)

    print(f"Output shape: {out.shape}")
    if out.shape == input_tensor.shape:
        print("✅ Shapes match!")
    else:
        print(f"❌ Shape Mismatch! Got {out.shape}, expected {input_tensor.shape}")

    print("\n--- 2. Parameters Count ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} Million")

    # --- 3. FLOPs / Complexity ---
    print("\n--- 3. FLOPs / Complexity ---")

    try:
        macs, params = get_model_complexity_info(
            wrapped_model,
            input_res=(3, 256, 256),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="pytorch",
        )

        # Convert to GFLOPs
        # MACs = Multiply-Accumulates. 1 MAC ≈ 2 FLOPs
        gflops = (macs * 2) / 1e9

        print(f"MACs: {macs / 1e9:.2f} G")
        print(f"GFLOPs (approx): {gflops:.2f} G")
        print(f"Params (ptflops): {params / 1e6:.2f} M")

    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        print("Tip: Ensure ptflops is installed and compatible with your model layers.")

    print("Note: Calculated on 256x256 resolution.")
    print("-----------------------------------------")


if __name__ == "__main__":
    test_dcae_model()
