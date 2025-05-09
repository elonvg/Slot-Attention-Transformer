import torch
from torch import nn
from torch.nn import init
import math
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

#--------------------------------SLOT-ATTN---------------------------------------------------------------------------------------------------


class SlotAttention(nn.Module):
    def __init__ (self, num_slots, dim, iters=3, epsilon=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim # Dimensionality of feature vectors for both inputs and slots
        self.num_slots = num_slots # Number of slots or objects to detect
        self.iters = iters # How many iterations for attention
        self.epsilon = epsilon # Small number added for stability
        self.scale = dim ** -0.5 # Scaling factor used in attention, useful for stabilizing training

        self.slots_mean = nn.Parameter(torch.randn(1, 1, dim)) # Learnable parameters for initializing slots

        self.slots_logvar = nn.Parameter(torch.zeros(1, 1, dim)) # Learnable parameters for initializing slots
        init.xavier_uniform_(self.slots_logvar) # Xavier initialization for logvar, type of weight initialization

        self.to_q = nn.Linear(dim, dim) # Slots -> Queries (Linear layer)
        self.to_k = nn.Linear(dim, dim) # Input features -> Keys (Linear layer)
        self.to_v = nn.Linear(dim, dim) # Input features -> Values (Linear layer)

        self.gru = nn.GRUCell(dim, dim) # Gated Recurrent Unit - Used to iterativley update slots - input + slot_prev -> slot_new

        hidden_dim = max(dim, hidden_dim) # Size of hidden layer in MLP

        self.mlp =  nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        ) # MLP applied to slots after GRU update - refines slot representation

        # Normalization layers to stabilize training
        self.norm_input = nn.LayerNorm(dim) # Applied to input
        self.norm_slots = nn.LayerNorm(dim) # Applied to slot before query
        self.norm_pre_ff = nn.LayerNorm(dim) # Applied to slot before MLP (feed forward)

        # Forward method - how information flows through network
        # Inputs from CNN - shape: (batch_size, num_input_features, dim_features)
    def forward(self, inputs, num_slots=None):
        batch_size, num_input_features, dim_features, device, dtype = *inputs.shape, inputs.device, inputs.dtype

        n_s = num_slots if num_slots is not None else self.num_slots # Number of slots used for batch

        mean = self.slots_mean.expand(batch_size, n_s, -1) # Initialization of slots (mean)
        var = self.slots_logvar.exp().expand(batch_size, n_s, -1) # Initialization of slots (var)

        slots = mean + var * torch.randn(mean.shape, device=device, dtype=dtype) # Initialization of slots with random gaussian

        inputs = self.norm_input(inputs) # Normalize input features
        k, v = self.to_k(inputs), self.to_v(inputs) # Produce keys and values - shape: (batch_size, num_input_features, dim_features) - Onlu computed once, depend only on inputs

        # Iterative attention loop 
        for _ in range(self.iters):
            slots_prev = slots # Store current state of slots - for GRU

            slots = self.norm_slots(slots) # Normalize slots
            q = self.to_q(slots) # Produce queries - shape: (batch_size, num_slots=n_s, dim_features)

            # Attention mechanism
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # Computes attn scores - dots shape: (batch_size, num_slots=n_s, dim_features) - Scales resuls afterwards
            attn = dots.softmax(dim=1) + self.epsilon # Apply softmax along n_s dimension - for each feature assign prob to fit to specific slot

            updates = torch.einsum('bjd,bij->bid', v, attn) # Compute how much slots shoud update

            # Update slots
            slots = self.gru(
                updates.reshape(-1, dim_features),
                slots_prev.reshape(-1, dim_features)
            )

            slots = slots.reshape(batch_size, -1, dim_features) # Reshape slots back to normal after GRU
            slots = slots + self.mlp(self.norm_pre_ff(slots)) # Updates are passed through a MLP - with residual connections, takes info from input (+=)

        return slots, attn

#--------------------------------MULTIHEAD---------------------------------------------------------------------------------------------------

# Different heads learn different aspects of image
class MultiHeadSlotAttention(nn.Module):
    def __init__ (self, num_slots, dim, heads=2, dim_head=32, iters=3, epsilon=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim # Dimensionality of feature vectors for both inputs and slots
        self.num_slots = num_slots # Number of slots or objects to detect
        self.iters = iters # How many iterations for attention
        self.epsilon = epsilon # Small number added for stability
        self.scale = dim ** -0.5 # Scaling factor used in attention, useful for stabilizing training

        self.slots_mean = nn.Parameter(torch.randn(1, 1, dim)) # Learnable parameters for initializing slots

        self.slots_logvar = nn.Parameter(torch.zeros(1, 1, dim)) # Learnable parameters for initializing slots
        init.xavier_uniform_(self.slots_logvar) # Xavier initialization for logvar, type of weight initialization

        dim_inner = dim_head * heads # Total dimension after concatenating heads

        # Linear layers now project dim -> dim_inner - so resulting vector can be split among the heads
        self.to_q = nn.Linear(dim, dim_inner) # Slots -> Queries (Linear layer)
        self.to_k = nn.Linear(dim, dim_inner) # Input features -> Keys (Linear layer)
        self.to_v = nn.Linear(dim, dim_inner) # Input features -> Values (Linear layer)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads) # Splits last dim of K, Q and V to h heads and d dim_head
        # Size change: (batch_size, num_elements, dim_inner) -> (batch_size, num_heads, num_elements, dim_head)

        self.merge_heads = Rearrange('b h n d -> b n (h d)') # Reverse of split_heads (concattenates) - used after attention is calculated per head
        # Size change: (batch_size, num_heads, num_slots, dim_head) -> (batch_size, num_slots, dim_inner)
        self.combine_heads = nn.Linear(dim_inner, dim) # Projects back dim_inner -> dim

        self.gru = nn.GRUCell(dim, dim) # Gated Recurrent Unit - Used to iterativley update slots - input + slot_prev -> slot_new

        hidden_dim = max(dim, hidden_dim) # Size of hidden layer in MLP

        self.mlp =  nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        ) # MLP applied to slots after GRU update - refines slot representation

        # Normalization layers to stabilize training
        self.norm_input = nn.LayerNorm(dim) # Applied to input
        self.norm_slots = nn.LayerNorm(dim) # Applied to slot before query
        self.norm_pre_ff = nn.LayerNorm(dim) # Applied to slot before MLP (feed forward)

        # Forward method - how information flows through network with multiple heads
        # Inputs from CNN - shape: (batch_size, num_input_features, dim_features)
    def forward(self, inputs, num_slots=None):

        batch_size, num_input_features, dim_features, device, dtype = *inputs.shape, inputs.device, inputs.dtype

        n_s = num_slots if num_slots is not None else self.num_slots # Number of slots used for batch

        mean = repeat(self.slots_mean, '1 1 d -> b s d', b=batch_size, s=n_s)
        var = repeat(self.slots_logvar.exp(), '1 1 d -> b s d', b=batch_size, s=n_s)

        slots = mean + var * torch.randn(mean.shape, device=device, dtype=dtype) # Initialization of slots with random gaussian

        inputs = self.norm_input(inputs) # Normalize input features
        
        k, v = self.to_k(inputs), self.to_v(inputs) # Inputs are projected to dim_inner
        k, v = map(self.split_heads, (k, v)) # Splits keys and values into different heads
        # Shape k, v: (batch_size, heads, num_input_features, dim_head)

        # Iterative attention loop 
        for _ in range(self.iters):
            slots_prev = slots # Store current state of slots - for GRU

            slots = self.norm_slots(slots) # Normalize slots

            q = self.to_q(slots) # Produce queries - shape: (batch_size, num_slots=n_s, dim_features)
            q = self.split_heads(q) # Split queries into different heads
            # Shape: (batch_size, heads, num_slots, dim_head)

            # Attention mechanism
            dots = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale # Computes attn scores using dot product - Scales resuls afterwards
            # Shape: (batch_size, heads, num_slots, num_input_features)
            
            attn = dots.softmax(dim=-2) # Apply softmax - For each input feature and each attention head distribute attn weights accross all slots - slots "compete" to explain features
            attn = F.normalize(attn + self.epsilon, p=1, dim=-1) #L1 normalization across num_input_features dim -> forces sum of attn aross batch, head and slot = 1 (across all unpit features)
            # Each slot redistributed total attn capacity across all input features based on relative "wins" from softmax

            updates = torch.einsum('... j d, ... i j -> ... i d', v, attn) # Calculated weighted sum of values using attn weights
            updates = self.merge_heads(updates) # Concatenate updates from all heads
            # Shape: (batch_size, num_slots, dim_inner)
            updates = self.combine_heads(updates) # Project back to original dim
            # Shape: (batch_size, num_slots, dim)

            updates, packed_shape = pack([updates], '* d') # Reshape before GRU
            slots_prev, _ = pack([slots_prev], '* d') # Reshape before GRU

            # Update slots
            slots = self.gru(updates, slots_prev)

            slots = unpack(slots, packed_shape, '* d')[0] # Reshape slots back to normal after GRU
            # Shape: (batch_size, num_slots, dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots)) # Updates are passed through a MLP - with residual connections, takes info from input (+=)

        return slots, attn

#--------------------------------POS-ENCODER---------------------------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, p_dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=p_dropout) # Initialize dropout layer

        pos_enc = torch.zeros(max_len, model_dim) # Initialize positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Column vector of position indices
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000) / model_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term) # Fill even with sine
        pos_enc[:, 1::2] = torch.cos(position * div_term) # Fill odd with cos
        pos_enc = pos_enc.unsqueeze(0) # Adds batch dimension
        # Shape: (1, max_len, model_dim)
        
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        num_features = x.size(1)
        x = x + self.pos_enc[:, :num_features, :]
        return self.dropout(x)

#--------------------------------ADAPTIVE-SLOTS---------------------------------------------------------------------------------------------------

# Helper function for computing log of clamped tensor - clamping: sets min and max for tensor values
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

# Helper function for generating Gimbel-distributed random noise
def gumbel_noise(t):
    noise = torch.rand_like(t) # Tensor of random numbers
    return -log(-log(noise)) # Inverse CDF transform -> Gumbel noise tensor

# Gumbel-Softmax: Way to draw samples to decide "keep" or "discard" in differentiable way - can train using backpropagation (gradient-descent)
def gumbel_softmax(logits, temperature=1.):
    dtype, size = logits.dtype, logits.shape[-1]

    # Logit - input tensor of raw unnormalized output
    # Temperature - controlls "roughness" of sample distribution

    assert temperature > 0 # Checks if temperature > 0

    scaled_logits = logits / temperature # Scales logits, lower temp -> difference in logits more clear

    # Gumbel sampling and derive one-hot
    noised_logits = scaled_logits + gumbel_noise(scaled_logits) # Gumbel-Max trick

    indices = noised_logits.argmax(dim=-1) # Perform sampling -> discrete category index, 0: discard, 1: keep

    hard_one_hot = F.one_hot(indices, size).type(dtype) # Converts indices to one-hot vector

    # Get soft for gradients
    soft = scaled_logits.softmax(dim=-1) # Standard softmax of scaled logits - is differentiable

    # Straight Through Estimator (STE)
    hard_one_hot = hard_one_hot + soft - soft.detach() # "Tricks" backpropagation into working for hard_one_hot

    # Return one-hot and indices
    return hard_one_hot, indices

# Wrapper
class AdaptiveSlotWrapper(nn.Module):
    def __init__(self, sat_model: SlotAttention | MultiHeadSlotAttention, temperature=1):
        super().__init__()

        self.sat_model = sat_model # Instance of sat_model
        dim = sat_model.dim # Dimension of slots
        self.temperature = temperature
        
        # Leanable component
        self.pred_keep_slot = nn.Linear(dim, 2, bias=False) # Slots -> 2 logits representing "keep" and "discard"

    def forward(self, x, **slot_kwargs):

        slots, attn_maps = self.sat_model(x, **slot_kwargs)
        # Shape slots: (batch_size, num_slots_init, slot_dim)

        keep_slot_logits = self.pred_keep_slot(slots)
        # Shape: (batch_size, num_slots_init, 2)

        keep_slots, _ = gumbel_softmax(keep_slot_logits, temperature=self.temperature) # Apply Gumbel-Softmax
        # Shape keep_slots: (batch_size, num_slots_init, 2)

        # Just use last column for keep mask
        keep_slots = keep_slots[..., -1] # Float["batch num_slots"] of {0., 1.}
        # Shape: (batch_size, num_slots_init)

        # Use keep_slots to mask wich slots will be sent to the decoder

        # Shapes:
        # slots: (batch_size, num_slots_init, slot_dim)
        # attn_maps: (batch_size, heads, num_slots, num_input_features)
        # keep_slots: (batch_size, num_slots_init)

        return slots, attn_maps, keep_slots


if __name__ == "__main__":
    x = torch.randn(8, 1024, 64)
    dim = x.shape[2]
    num_slots = 10

    # sat = SlotAttention(num_slots=num_slots, dim=dim)
    # y = sat(x)

    sat_mh = MultiHeadSlotAttention(num_slots=num_slots, dim=dim)
    # y, attn_map = sat_mh(x)

    adap = AdaptiveSlotWrapper(sat_mh)

    slots, attn_map, keep_slots = adap(x)

    # print(slots.shape)
    # print(attn_map.shape)
    # print(keep_slots.shape)

    print(keep_slots[1])

    active_slots = slots * keep_slots.unsqueeze(-1)

    print(active_slots[1])

    print(active_slots.shape)


