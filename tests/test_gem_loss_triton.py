import torch
import torch.nn.functional as F

# Add project root to Python path
import os 
import sys 
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.gem_triton_loss import GEMLoss

class ReferenceGEMLoss(torch.nn.Module):
    def forward(self, logits, labels, beta=0.7, ignore_index=-100, h="linear", reduction="mean"):
        """Reference implementation of GEM loss"""
        mask = labels != ignore_index
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                masked_logits, dim=-1, index=masked_labels.unsqueeze(-1)
            ).squeeze(-1)
            logits_diff = masked_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            else:
                raise ValueError(f"Unsupported h function: {h}")

        gene_log_probs = F.log_softmax(masked_logits, dim=-1)
        with torch.no_grad():
            q_probs = torch.exp(F.log_softmax(masked_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=masked_labels.unsqueeze(-1)
        )

        if reduction == "mean":
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()
        elif reduction == "sum":
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs)
            )
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        return loss

def test_gem_loss():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 10
    vocab_size = 120000
    beta = 0.7
    ignore_index = -100

    # Create random inputs
    logits = torch.randn(batch_size, vocab_size, device='cuda', requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size,), device='cuda')
    # Add some ignored indices
    # labels[0] = ignore_index
    
    # Create loss functions
    triton_loss_fn = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="sum")
    ref_loss_fn = ReferenceGEMLoss()
    
    # Forward pass
    triton_loss = triton_loss_fn(logits, labels)
    ref_loss = ref_loss_fn(logits, labels, beta=beta, ignore_index=ignore_index, h="linear", reduction="sum")

    test_failed = False
    # Check forward pass results
    print("*" * 100)
    print("triton_loss:", triton_loss)
    print("ref_loss:", ref_loss)
    print("Forward pass difference:", torch.abs((triton_loss - ref_loss)).mean().item())
    try:    
        torch.testing.assert_close(triton_loss, ref_loss, rtol=1e-4, atol=1e-4)
    except Exception as e:
        print(e)
        test_failed = True
    
    # Backward pass
    triton_logits = logits.clone().detach().requires_grad_(True)
    ref_logits = logits.clone().detach().requires_grad_(True)
    
    triton_loss = triton_loss_fn(triton_logits, labels)
    ref_loss = ref_loss_fn(ref_logits, labels, beta=beta, ignore_index=ignore_index, h="linear", reduction="sum")
    
    triton_loss.mean().backward()
    ref_loss.mean().backward()
    
    # Check backward pass results
    print("*" * 100)
    print("Max gradient difference:", torch.max(torch.abs(triton_logits.grad - ref_logits.grad)).item())
    try:
        torch.testing.assert_close(triton_logits.grad, ref_logits.grad, rtol=1e-4, atol=1e-4)
    except Exception as e:
        print(e)
        test_failed = True
    
    if test_failed:
        print("Test failed!")
    else:
        print("All tests passed!")

if __name__ == "__main__":
    test_gem_loss()
    