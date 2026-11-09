import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int = 8, 
        lora_alpha: float = 16.0, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0

        # Ensure the base layer is frozen
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            # Dimensions for matmul: (in_features, r) and (r, out_features)
            self.lora_A = nn.Parameter(torch.empty(self.base_layer.in_features, r))
            self.lora_B = nn.Parameter(torch.empty(r, self.base_layer.out_features))
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # Initialize A with Kaiming uniform (similar to standard linear layers). lora_A is stored as
            # (in_features, r), so init its transpose: kaiming takes fan_in from dim 1, which must be in_features
            # (as for nn.Linear(in_features, r)), not r -- otherwise the init is ~sqrt(in/r) times too large.
            nn.init.kaiming_uniform_(self.lora_A.T, a=math.sqrt(5))
            # Initialize B to zero so the initial forward pass is identical to the base layer
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard forward pass through the frozen base layer
        result = self.base_layer(x)
        
        # Add the LoRA adapter output
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            result = result + lora_out * self.scaling   # out-of-place: safe for autograd / views
            
        return result


def apply(
    backbone: nn.Module, 
    r: int = 8, 
    lora_alpha: float = 16.0, 
    lora_dropout: float = 0.05, 
    target_modules: list[str] = None
) -> None:
    """
    Wraps nn.Linear layers in the backbone with LoRALinear.
    
    Args:
        backbone: The PyTorch model to adapt.
        r: Low-rank dimension.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout probability for the adapter.
        target_modules: List of module name substrings to target (e.g., ["q_proj", "v_proj"]).
                        If None, applies to all nn.Linear layers except the tied LM head.
    """
    # 1. Freeze all parameters in the entire backbone
    for param in backbone.parameters():
        param.requires_grad = False
    # The LM head shares its weight with the token embedding and is not used for NER features: never wrap it
    tied = {id(m.weight) for m in backbone.modules() if isinstance(m, nn.Embedding)}

    # 2. Recursively find and replace targeted nn.Linear layers
    def replace_layers(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if we should adapt this specific linear layer
                if id(child.weight) in tied:
                    continue
                if target_modules is None or any(target in full_name for target in target_modules):
                    lora_layer = LoRALinear(
                        base_layer=child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )
                    # Replace the original layer with the LoRA-wrapped layer in-place
                    setattr(module, name, lora_layer)
            else:
                # Traverse deeper into the module tree
                replace_layers(child, prefix=full_name)

    replace_layers(backbone)