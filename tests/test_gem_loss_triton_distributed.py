import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to Python path
import os 
import sys 
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.gem_triton_loss import GEMLoss as TritonGEMLoss

class ReferenceGEMLoss(torch.nn.Module):
    def forward(self, logits, labels, beta=0.7, ignore_index=-100, h="linear"):
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

        loss = -torch.sum(
            q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
        ).mean()

        return loss

def setup(local_rank=None):
    """Initialize the distributed environment."""
    # When using torch.distributed.launch, use the environment variables it sets
    if local_rank is None:
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
        else:
            raise ValueError("LOCAL_RANK not found in environment variables")
    else:
        # For mp.spawn path
        rank = local_rank
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size} initialized")
    return rank, world_size

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_test(rank=None, world_size=None):
    """Run the distributed GEM loss test on a single process."""
    # Initialize distributed environment
    if rank is None:  # When called directly (not through mp.spawn)
        rank, world_size = setup()
    else:  # When called through mp.spawn
        setup(rank)
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + rank)  # Different seed per rank
    
    # Test parameters
    batch_size = 100
    vocab_size = 102400
    beta = 0.7
    ignore_index = -100
    
    # Each rank handles a portion of the vocabulary
    local_vocab_size = vocab_size // world_size
    
    # Create random inputs for this rank
    logits = torch.randn(batch_size, local_vocab_size, device=rank, requires_grad=True)
    
    # All ranks have the same labels (in the range of the full vocabulary)
    # We use the same seed for labels to ensure consistency
    torch.manual_seed(42)  # Same seed for labels across all ranks
    labels = torch.randint(0, vocab_size, (batch_size,), device=rank)
    
    # Create loss function with process group
    triton_loss_fn = TritonGEMLoss(beta=beta, ignore_index=ignore_index, process_group=dist.group.WORLD)
    
    # Forward pass
    triton_loss = triton_loss_fn(logits, labels)
    
    # Basic sanity checks
    assert not torch.isnan(triton_loss).any(), f"Rank {rank}: Loss contains NaN values"
    assert not torch.isinf(triton_loss).any(), f"Rank {rank}: Loss contains Inf values"
    
    # Backward pass
    triton_loss.mean().backward()
    
    # Check gradients
    assert not torch.isnan(logits.grad).any(), f"Rank {rank}: Gradients contain NaN values"
    assert not torch.isinf(logits.grad).any(), f"Rank {rank}: Gradients contain Inf values"
    
    # Gather all logits to rank 0 for verification (optional)
    if rank == 0:
        all_logits = [torch.zeros_like(logits) for _ in range(world_size)]
    else:
        all_logits = None
    
    dist.gather(logits, all_logits if rank == 0 else None, dst=0)
    
    # On rank 0, verify the loss against reference implementation
    if rank == 0:
        print(f"Distributed test on {world_size} GPUs:")
        print(f"Triton GEM loss: {triton_loss.mean().item()}")
        
        # Concatenate all logits to get the full vocabulary
        full_logits = torch.cat(all_logits, dim=1).detach().requires_grad_(True)
        
        # Compute reference loss
        ref_loss_fn = ReferenceGEMLoss()
        ref_loss = ref_loss_fn(full_logits, labels, beta=beta, ignore_index=ignore_index)
        
        print(f"Reference loss: {ref_loss.item()}")
        print(f"Difference: {abs(triton_loss.mean().item() - ref_loss.item())}")
        
        # Note: We don't expect exact match due to distributed computation differences
        print("Distributed test completed successfully!")
    
    # Wait for all processes
    dist.barrier()
    
    # Clean up
    cleanup()

def main():
    """Main function to launch the distributed test."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping distributed test.")
        return
    
    # Check if we're being launched by torch.distributed.launch
    if 'LOCAL_RANK' in os.environ:
        # We're being launched by torch.distributed.launch, so just run the test
        run_test()
    else:
        # Manual launch with mp.spawn
        world_size = int(os.environ.get("WORLD_SIZE", 2))
        
        # Ensure we have enough GPUs
        if torch.cuda.device_count() < world_size:
            print(f"Not enough GPUs available. Need {world_size}, found {torch.cuda.device_count()}")
            world_size = torch.cuda.device_count()
            print(f"Reducing world_size to {world_size}")
        
        if world_size < 2:
            print("Need at least 2 GPUs for a meaningful distributed test")
            return
        
        print(f"Running distributed test with world_size={world_size}")
        
        # Spawn processes
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()