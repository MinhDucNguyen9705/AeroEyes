import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class NoisyTokenChoiceRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(NoisyTokenChoiceRouter, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
    
    def importance_auxiliary_loss(self, gates):
        axis = tuple(range(gates.ndim - 1))
        importance_per_expert = gates.sum(dim=axis)
        std = importance_per_expert.std()
        mean = importance_per_expert.mean()
        return (std / (mean + 1e-8)) ** 2
    
    def load_auxiliary_loss(self, logits, logits_noisy, noise_std, num_selected_experts):

        thresholds = torch.topk(logits_noisy, num_selected_experts, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )

        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std 

        normal = Normal(loc=0.0, scale=1.0)
        p = 1.0 - normal.cdf(noise_required_to_win)

        p_mean = p.mean(dim=tuple(range(p.ndim - 1)))

        return (p_mean.std() / (p_mean.mean() + 1e-8)) ** 2
    
    def forward(self, x):
        gates_logits = self.gate(x)
        gates_softmax = F.softmax(gates_logits, dim=-1)
        noise_std = (1.0 / self.num_experts)
        logits_noise = noise_std * torch.randn_like(gates_logits)
        gates_logits_noisy = gates_logits + logits_noise
        gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)
        top_k_values, top_k_indices = torch.topk(gates_softmax_noisy, self.top_k, dim=-1)
        
        denominator = top_k_values.sum(dim=-1, keepdim=True) + 1e-20
        top_k_values = top_k_values / denominator 

        if self.training:
            # Importance loss to encourage balanced expert usage
            importance_loss = self.importance_auxiliary_loss(gates_softmax)

            # Load balancing loss to prevent expert overload
            load_loss = self.load_auxiliary_loss(gates_logits, gates_logits_noisy, noise_std, self.top_k)

            # Combine losses
            aux_loss = 0.5 * (importance_loss + load_loss)
        else:
            aux_loss = 0

        return top_k_values, top_k_indices, aux_loss

class VMoEBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=1, capacity_factor=1.0, expert_dim=None, return_shape='1d'):
        
        super(VMoEBlock, self).__init__()
        self.num_experts = num_experts
        self.top_k = int(top_k)
        self.capacity_factor = capacity_factor
        self.return_shape = return_shape

        if expert_dim is None:
            expert_dim = input_dim * 4
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim, output_dim) for _ in range(num_experts)])
        self.router = NoisyTokenChoiceRouter(input_dim, num_experts, self.top_k)
        self.norm = nn.LayerNorm(input_dim)
    
    def compute_capacity(self, num_tokens, num_experts, capacity_factor, multiple_of=4):
        capacity = int(round(num_tokens / num_experts * capacity_factor))
        capacity += (-capacity) % multiple_of
        return capacity

    def forward(self, x):
        h, w = None, None

        if len(x.shape) == 4:
            h, w = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)
        
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        num_tokens = batch_size * seq_len

        x = x.view(-1, input_dim)
        x = self.norm(x)

        top_k_values, top_k_indices, aux_loss = self.router(x)

        flat_indices = top_k_indices.view(-1)
        flat_gates_softmax = top_k_values.view(-1)

        sample_indices = torch.arange(num_tokens, device=device)[:, None]
        sample_indices = sample_indices.expand(-1, self.top_k).flatten()

        unprocessed_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)

        outputs = torch.zeros(num_tokens, self.experts[0].fc2.out_features, device=device)
        capacity = self.compute_capacity(num_tokens, self.num_experts, self.capacity_factor)

        for idx in range (self.num_experts):

            expert_mask = (flat_indices == idx)
            if not expert_mask.any():
                continue

            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_gates_softmax[expert_mask]

            if expert_samples.numel() > capacity:
                #Take the top 'capacity' samples based on their weights
                most_important = torch.topk(expert_weights, capacity)
                expert_weights = most_important.values
                expert_samples = expert_samples[most_important.indices]
            
            #Mark processed tokens    
            unprocessed_mask[expert_samples] = True
            expert_inputs = x[expert_samples]
            expert_outputs = self.experts[idx](expert_inputs)
            weighted_expert_outputs = expert_outputs * expert_weights.unsqueeze(-1)

            outputs.index_add_(0, expert_samples, weighted_expert_outputs)

        #Skip connecting unprocessed tokens
        unprocessed_indices = torch.nonzero(~unprocessed_mask, as_tuple=False).squeeze(-1)
        outputs.index_add_(0, unprocessed_indices, x[unprocessed_indices])

        outputs = outputs.view(batch_size, seq_len, -1)

        if self.return_shape == '2d':
            outputs = outputs.transpose(1, 2).view(batch_size, -1, h, w)
        elif self.return_shape == '1d':
            outputs = outputs
        
        return outputs, aux_loss
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=4, top_k=1, capacity_factor=1.0, dropout=0.0, batch_first=True):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.ffn = VMoEBlock(d_model, d_model, num_experts, top_k, capacity_factor, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        skip = x
        x = self.norm1(x)
        x = self.attn(x, x, x, attn_mask=src_mask)[0]
        x = skip + self.dropout1(x)

        skip = x
        x = self.norm2(x)
        x, aux_loss = self.ffn(x)
        x = skip + self.dropout2(x)

        return x, aux_loss

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=4, top_k=1, capacity_factor=1.0, dropout=0.0, batch_first=True):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.ffn = VMoEBlock(d_model, d_model, num_experts, top_k, capacity_factor, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)[0]
        x = skip + self.dropout1(x)

        skip = x
        x = self.norm2(x)
        x = self.multihead_attn(x, memory, memory, attn_mask=memory_mask)[0]
        x = skip + self.dropout2(x)

        skip = x
        x = self.norm3(x)
        x, aux_loss = self.ffn(x)
        x = skip + self.dropout3(x)

        return x, aux_loss