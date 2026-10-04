"""Export the unchanged desktop checkpoint to browser-readable JSON tensors."""

import json
import os

import torch


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "flappy_bc.pth")
    output_path = os.path.join(model_dir, "flappy_bc.json")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    expected_shapes = {
        "network.0.weight": (128, 4),
        "network.0.bias": (128,),
        "network.2.weight": (128, 128),
        "network.2.bias": (128,),
        "network.4.weight": (2, 128),
        "network.4.bias": (2,),
    }
    exported_state = {}
    for name, shape in expected_shapes.items():
        tensor = state_dict[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        exported_state[name] = {
            "shape": list(shape),
            "data": tensor.contiguous().view(-1).tolist(),
        }

    browser_model = {
        "architecture": [4, 128, 128, 2],
        "state_dict": exported_state,
    }
    with open(output_path, "w", encoding="utf-8") as model_file:
        json.dump(browser_model, model_file, separators=(",", ":"))
    print(f"Exported {model_path} to {output_path}")


if __name__ == "__main__":
    main()
