"""Pure-Python inference for the exported 4 -> 128 -> 128 -> 2 policy."""

import json
import struct


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def normalize_observation(obs):
    """Apply the exact scaling and clipping used by the desktop policy."""
    return [
        max(-1.0, min(1.0, _float32(float(obs[0]) / 512.0))),
        max(-1.0, min(1.0, _float32(float(obs[1]) / 10.0))),
        max(-1.0, min(1.0, _float32(float(obs[2]) / 288.0))),
        max(-1.0, min(1.0, _float32(float(obs[3]) / 512.0))),
    ]


class DQN:
    """Dense neural network with the same layer dimensions and ReLU activations."""

    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = []
        self.training = True

    def load_state_dict(self, path):
        with open(path, "r", encoding="utf-8") as model_file:
            checkpoint = json.load(model_file)
        dimensions = checkpoint.get("architecture")
        if dimensions != [self.state_dim, 128, 128, self.action_dim]:
            raise ValueError(f"Unexpected model architecture: {dimensions!r}")

        state = checkpoint["state_dict"]
        expected = (
            ("network.0.weight", "network.0.bias", self.state_dim, 128),
            ("network.2.weight", "network.2.bias", 128, 128),
            ("network.4.weight", "network.4.bias", 128, self.action_dim),
        )
        layers = []
        for weight_name, bias_name, input_size, output_size in expected:
            weights = state[weight_name]["data"]
            biases = state[bias_name]["data"]
            if len(weights) != input_size * output_size or len(biases) != output_size:
                raise ValueError(f"Invalid tensor sizes for {weight_name}")
            matrix = [
                weights[row * input_size : (row + 1) * input_size]
                for row in range(output_size)
            ]
            layers.append((matrix, biases))
        self.layers = layers

    def eval(self):
        self.training = False
        return self

    def __call__(self, values):
        if len(values) != self.state_dim or len(self.layers) != 3:
            raise ValueError("Policy is not loaded or state has the wrong size")
        activations = [float(value) for value in values]
        for layer_index, (weights, biases) in enumerate(self.layers):
            outputs = []
            for row, bias in zip(weights, biases):
                value = bias + sum(
                    weight * activation
                    for weight, activation in zip(row, activations)
                )
                if layer_index < 2:
                    value = max(0.0, value)
                outputs.append(value)
            activations = outputs
        return activations


def argmax(values):
    """Return the index of the largest network output, matching torch.argmax."""
    if not values:
        raise ValueError("Cannot select an action from empty network output")
    return max(range(len(values)), key=values.__getitem__)
