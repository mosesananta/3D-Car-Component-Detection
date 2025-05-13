import torch
import torch.nn.functional as F
import torch.nn as nn


def info_nce_loss(projections, labels, temperature=0.07):
    """
    InfoNCE loss for contrastive learning, using component states to determine positive pairs.
    
    Args:
        projections: Batch of embeddings from projection head [N, D]
        labels: Batch of component state labels [N, C]
        temperature: Temperature parameter for softmax scaling
    """
    # Convert labels to string representation for comparison
    batch_size = labels.size(0)
    label_strs = [''.join(map(str, l.int().tolist())) for l in labels]
    
    # Normalize projections to unit sphere
    projections = F.normalize(projections, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create mask for positive pairs (same component states)
    mask_positives = torch.zeros_like(similarity_matrix)
    
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j and label_strs[i] == label_strs[j]:
                mask_positives[i, j] = 1
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute log_prob
    exp_logits = torch.exp(logits)
    
    # Create mask to exclude self-similarity
    mask_self = torch.eye(batch_size, device=similarity_matrix.device)
    
    # Calculate positive pairs sum (numerator)
    pos_exp_logits = torch.sum(exp_logits * mask_positives, dim=1)
    
    # Calculate denominator (all except self)
    denominator = torch.sum(exp_logits * (1 - mask_self), dim=1)
    
    # Calculate loss for each element in batch
    log_prob = torch.log(pos_exp_logits / denominator + 1e-8)
    
    # Handle cases with no positive pairs in the batch
    mask_valid = torch.sum(mask_positives, dim=1) > 0
    
    # Return mean loss over valid samples
    if torch.sum(mask_valid) > 0:
        return -torch.sum(log_prob * mask_valid) / torch.sum(mask_valid)
    else:
        return torch.tensor(0.0, device=similarity_matrix.device)
    

# Focal Loss implementation for uncertainty-aware learning
class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance and hard examples.
    
    Args:
        alpha: Weight for positive class (typically the minority class)
        gamma: Focusing parameter that reduces the loss contribution from easy examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Standard binary cross entropy with logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities for the ground truth class
        pt = torch.exp(-BCE_loss)
        
        # Apply focal weighting: (1-pt)^gamma reduces the loss for easy examples
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing: alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute final focal loss
        focal_loss = alpha_t * focal_weight * BCE_loss
        
        return focal_loss.mean()