#
# probe_model.py
#
import torch
import torch.nn as nn

class StructuralProbe(nn.Module):
    def __init__(self, model_dim, probe_rank):
        """
        Initializes the structural probe.
        model_dim (int): The dimension of BERT's hidden states (e.g., 768).
        probe_rank (int): The dimension of the subspace to project into.
        """
        super(StructuralProbe, self).__init__()
        # This matrix B is the only learnable parameter
        self.probe_rank = probe_rank
        self.projection = nn.Parameter(data=torch.randn(model_dim, probe_rank), requires_grad=True)

    def forward(self, batch_hidden_states):
        """
        Projects hidden states into the lower-dimensional space.
        """
        return torch.matmul(batch_hidden_states, self.projection)