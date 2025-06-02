import argparse
import os
import torch
import numpy as np
from transformers import AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_das_params


def rotation_sim(rot_a, rot_b):
    """Compute Frobenius norm distance between rotation matrices."""
    return torch.norm(rot_a - rot_b, p='fro').item()


def random_rotation_matrix(n):
    """Generate a random rotation matrix of size n x n."""
    random_matrix = torch.randn(n, n)
    q, r = torch.qr(random_matrix)
    d = torch.diag(r).prod().sign()  # Check sign of determinant
    rotation_matrix = q * d          # Ensure det(Q) = 1
    return rotation_matrix


def load_alignment_rotation(align_path, config):
    """Load rotation matrix from alignment file."""
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Alignment file not found: {align_path}")
    
    rotation_matrix = get_das_params(align_path, config)
    return rotation_matrix.detach()


def compare_rotations(args):
    """Main function to compare rotation matrices from source and target alignments."""
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model_name)
    
    # Use specified token position
    token_pos = args.token_position
    print(f"Using token position: {token_pos}")
    
    # Construct alignment file paths
    align_path_source = args.alignment_path_source.format(
        layer=args.layer_source, 
        token=token_pos
    )
    align_path_target = args.alignment_path_target.format(
        layer=args.layer_target, 
        token=token_pos
    )
    
    print(f"Loading source alignment from: {align_path_source}")
    print(f"Loading target alignment from: {align_path_target}")

    # Load rotation matrices
    try:
        rotation_source = load_alignment_rotation(align_path_source, config)
        rotation_target = load_alignment_rotation(align_path_target, config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate random baseline
    rotation_random = random_rotation_matrix(rotation_source.shape[0])
    
    # Store rotations and labels
    rotations = [rotation_source, rotation_target, rotation_random]
    labels = [
        f"Source (Layer {args.layer_source})",
        f"Target (Layer {args.layer_target})",
        "Random Baseline"
    ]
    
    # Compute similarity matrix
    n_rotations = len(rotations)
    similarity_matrix = np.zeros((n_rotations, n_rotations))
    
    print("\nComputing rotation similarities...")
    for i in range(n_rotations):
        for j in range(n_rotations):
            similarity_matrix[i, j] = rotation_sim(rotations[i], rotations[j])
    
    # Print results
    print("\nSimilarity Matrix (Frobenius norm distances):")
    print("=" * 50)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print(f"{label_i} vs {label_j}: {similarity_matrix[i, j]:.4f}")
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt='.4f',
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis'
    )
    plt.title('Rotation Matrix Similarity (Frobenius Distance)')
    plt.tight_layout()
    
    if args.save_path:
        save_path = os.path.join(args.save_path, f"rotation_similarity_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        save_path_txt = os.path.join(args.save_path, f"rotation_similarity_matrix.txt")
        np.savetxt(
            save_path_txt,
            similarity_matrix,
            fmt='%.4f',
            header=' '.join(labels),
            comments=''
        )
        print(f"Similarity matrix saved to: {save_path_txt}")
    else:
        plt.show()
    
    return similarity_matrix, labels


def main():
    parser = argparse.ArgumentParser(
        description="Compare rotation matrices from source and target model alignments"
    )
    
    # Required arguments
    parser.add_argument(
        "--alignment_path_source", 
        type=str, 
        required=True,
        help="Path template for source alignment (use {layer} and {token} "
             "placeholders)"
    )
    parser.add_argument(
        "--alignment_path_target", 
        type=str, 
        required=True,
        help="Path template for target alignment (use {layer} and {token} "
             "placeholders)"
    )
    parser.add_argument(
        "--layer_source", 
        type=int, 
        required=True,
        help="Layer number for source alignment"
    )
    parser.add_argument(
        "--layer_target", 
        type=int, 
        required=True,
        help="Layer number for target alignment"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name for loading config"
    )
    
    # Optional arguments
    parser.add_argument(
        "--token_position", 
        type=int, 
        default=-1,
        help="Token position to analyze (default: -1 for last token)"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None,
        help="Path to save plot (if not specified, plot will be displayed)"
    )
    
    args = parser.parse_args()
    
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    
    print("Rotation Matrix Comparison")
    print("=" * 30)
    print(f"Model: {args.model_name}")
    print(f"Comparing Source Layer {args.layer_source} vs Target Layer {args.layer_target}")
    print(f"Token position: {args.token_position}")
    
    # Run comparison
    compare_rotations(args)


if __name__ == "__main__":
    main()