import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRMALinear(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int = 8, 
        lorma_alpha: float = 16.0, 
        lorma_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lorma_alpha = lorma_alpha
        self.scaling = self.lorma_alpha / self.r if self.r > 0 else 0.0

        # Ensure the base layer is frozen
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            self.lorma_dropout = nn.Dropout(p=lorma_dropout) if lorma_dropout > 0 else nn.Identity()
            
            # LoRMA (Low-Rank Multiplicative Adaptation) applies the update multiplicatively:
            # W_new = (I + (alpha/r) * B @ A) @ W_0 
            # Where W_0 has shape (out_features, in_features).
            # B is (out_features, r) and A is (r, out_features). 
            # When computing x @ W_new^T = x @ W_0^T @ (I + (alpha/r) * A^T @ B^T), 
            # the low-rank projection operates on the output dimension of the base layer.
            
            # lorma_A represents A^T and lorma_B represents B^T
            self.lorma_A = nn.Parameter(torch.empty(self.base_layer.out_features, r))
            self.lorma_B = nn.Parameter(torch.empty(r, self.base_layer.out_features))
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lorma_A'):
            # Initialize A (lorma_A) with Kaiming uniform. lorma_A is stored as (out_features, r) and acts on the
            # base output (dim out_features), so init its transpose: kaiming takes fan_in from dim 1, which must be
            # out_features, not r -- otherwise the init is ~sqrt(out/r) times too large.
            nn.init.kaiming_uniform_(self.lorma_A.T, a=math.sqrt(5))
            # Initialize B (lorma_B) to zero. 
            # This ensures the initial transformation is an identity matrix (I + 0)
            # resulting in the base model's initial behavior.
            nn.init.zeros_(self.lorma_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base projection without bias (x @ W_0^T): the multiplicative update acts on W_0 only,
        # i.e. y = (I + s*B@A) W_0 x + b, so the bias must not be multiplied
        base_out = F.linear(x, self.base_layer.weight)

        # Apply the LoRMA+ multiplicative adapter (base_out + base_out @ A^T @ B^T * scaling)
        if self.r > 0:
            lorma_out = self.lorma_dropout(base_out) @ self.lorma_A @ self.lorma_B
            base_out = base_out + lorma_out * self.scaling

        if self.base_layer.bias is not None:
            base_out = base_out + self.base_layer.bias
        return base_out


def apply(
    backbone: nn.Module, 
    r: int = 8, 
    lorma_alpha: float = 16.0, 
    lorma_dropout: float = 0.05, 
    target_modules: list[str] = None
) -> None:
    """
    Wraps nn.Linear layers in the backbone with LoRMALinear.
    
    Args:
        backbone: The PyTorch model to adapt.
        r: Low-rank dimension.
        lorma_alpha: Scaling factor.
        lorma_dropout: Dropout probability for the adapter.
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
                    lorma_layer = LoRMALinear(
                        base_layer=child,
                        r=r,
                        lorma_alpha=lorma_alpha,
                        lorma_dropout=lorma_dropout
                    )
                    # Replace the original layer with the LoRMA-wrapped layer in-place
                    setattr(module, name, lorma_layer)
            else:
                # Traverse deeper into the module tree
                replace_layers(child, prefix=full_name)

    replace_layers(backbone)