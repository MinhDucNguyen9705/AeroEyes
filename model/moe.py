import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class GatedExpert(nn.Module):
    def __init__(self, input_dim, intermediate_dim):
        super(GatedExpert, self).__init__()
        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, input_dim, bias=False)
        self.act = nn.GELU()
    
    def forward(self, x):
        down_proj = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class SoftMLPMoEBlock(nn.Module):
    def __init__(self, input_dim, num_experts, num_slots, expert_dim, output_dim, dropout=0.):
        super(SoftMLPMoEBlock, self).__init__()
        self.num_experts = num_experts
        if expert_dim is None:
            expert_dim = input_dim * 4
        self.slot_norm = nn.LayerNorm(input_dim)
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, input_dim))
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim, output_dim, dropout) for _ in range(num_experts)])
        # self.experts = nn.ModuleList([GatedExpert(input_dim, expert_dim) for _ in range(num_experts)])
        # self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):        
        # x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        logits = torch.einsum('b n d, e s d -> b n e s', x, slot_embeds)  # (batch_size, seq_len, num_experts, num_slots)
        
        dispatch_weights = F.softmax(logits, dim=1)  # (batch_size, seq_len, num_experts, num_slots)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)') # (batch_size, seq_len, num_experts * num_slots)
        combine_weights = F.softmax(combine_weights, dim=-1)  # (batch_size, seq_len, num_experts * num_slots)

        slots = torch.einsum('b n d, b n e s -> b e s d', x, dispatch_weights)  # (batch_size, num_experts, num_slots, input_dim)
        
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(slots[:, i, :, :])  # (batch_size, num_slots, input_dim)
            outputs.append(expert_out)
        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_experts, num_slots, input_dim)
        outputs = rearrange(outputs, 'b e s d -> b (e s) d')  # (batch_size, num_experts * num_slots, input_dim)
        outputs = torch.einsum('b n s, b s d -> b n d', combine_weights, outputs)  # (batch_size, seq_len, input_dim)

        return outputs
    
class TransformerMoEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts, num_slots, dim_feedforward, dropout):
        super(TransformerMoEEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SoftMLPMoEBlock(d_model, num_experts, num_slots, dim_feedforward, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        skip = x
        x = self.attn(x, x, x, attn_mask=src_mask)[0]

        x = skip + self.dropout1(x)
        x = self.norm1(x)

        skip = x
        x = self.ffn(x)
        x = skip + self.dropout2(x)
        x = self.norm2(x)

        return x
    
class TransformerMoEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts, num_slots, dim_feedforward, dropout):
        super(TransformerMoEDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SoftMLPMoEBlock(d_model, num_experts, num_slots, dim_feedforward, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        skip = x
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)[0]

        x = skip + self.dropout1(x)
        x = self.norm1(x)

        skip = x
        x = self.multihead_attn(x, memory, memory, attn_mask=memory_mask)[0]
        x = skip + self.dropout2(x)
        x = self.norm2(x)

        skip = x
        x = self.ffn(x)
        x = skip + self.dropout3(x)
        x = self.norm3(x)

        return x