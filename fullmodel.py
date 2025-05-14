import torch
import torch.nn as nn
from CNNmodel import CNNencoder, CNNdecoder
from SATmodel import SlotAttention, MultiHeadSlotAttention, PositionalEncoding, AdaptiveSlotWrapper

class FullModel(nn.Module):
    def __init__(
            self,
            img_c,
            img_height,
            img_width,
            encoder_features,
            encoder_out_channels,
            num_slots,
            num_heads,
            dim_head,
            slot_iters,
            hidden_dim,
            temperature,
            decoder_features
        ):
        super().__init__()
        self.img_c = img_c
        self.img_height = img_height
        self.img_width = img_width
        self.num_slots = num_slots
        self.slot_dim = encoder_out_channels

        self.sampling_num = encoder_features.count("pool")


        #CNN encoder
        self.cnn_encoder = CNNencoder(img_c=img_c, encoder_features=encoder_features, out_channels=encoder_out_channels)

        # Positional encoder
        self.pos_encoder = PositionalEncoding(model_dim=encoder_out_channels, p_dropout=0.1)

        # Multihead slot-attention
        self.mh_sat = MultiHeadSlotAttention(num_slots=num_slots, dim=encoder_out_channels, heads=num_heads, dim_head=dim_head, iters=slot_iters, hidden_dim=hidden_dim)

        # Adaptive slot wrapper
        self.adaptive_slot_wrapper = AdaptiveSlotWrapper(self.mh_sat, temperature)

        # CNN decoder
        self.cnn_decoder = CNNdecoder(in_channels=encoder_out_channels+2, img_c=img_c, out_size=(img_height, img_width), decoder_features=decoder_features)

        # Attention upsampler
        self.attn_upsample = nn.Upsample(size=(self.img_height, self.img_width), mode="bilinear", align_corners=False)


    def forward(self, x):

        # batch_size, img_c, img_height, img_width = x.shape
        
        # -----------------------------------Encoder--------------------------------------------------------
        
        x_cnn = self.cnn_encoder(x)

        # Reshape output from encoder to fit into slot-attention model
        batch_size, encoder_out_channels, H_encoded, W_encoded = x_cnn.shape

        x_cnn = x_cnn.permute(0, 2, 3, 1) # Rearrange shape
        # Shape: (batch_size, height, width, out_channels)

        x_cnn = x_cnn.reshape(batch_size, H_encoded * W_encoded, encoder_out_channels) # Flatten out spatial dimenstions
        # Shape: (batch_size, height * width, out_channels)

        # -----------------------------------Positional Encoding---------------------------------------------------
        
        x_pos = self.pos_encoder(x_cnn)
        # Shape: (batch_size, height * width, out_channels)

        # -----------------------------------Slot-Attention Transformer--------------------------------------------

        # Adaptive slots
        slots, attn_maps, keep_slots = self.adaptive_slot_wrapper(x_pos)
        # Shapes:
        # slots: (batch_size, num_slots, slot_dim)
        # attn_maps: (batch_size, heads, num_slots, num_input_features)
        # keep_slots: (batch_size, num_slots)
        
        # Set non-active slots to 0
        active_slots = slots * keep_slots.unsqueeze(-1)
        # Shape: (batch_size, num_slots, slot_dim)

        # -----------------------------------Process For Decoder-------------------------------------------------------
        # Prepare input to decoder
        active_slots = active_slots.view(batch_size * self.num_slots, self.slot_dim) # Combine batches and slots into one dimension -> Larger batch where each item is a slot
        # Shape: (batch_size*num_slots, slot_dim)

        # Broadcast shape of x_slot (batch_size, num_slots, slot_dim) -> (batch_size, slot_dim, h_decode, w_decode) for each slot
        # Use broadcast to give slot vector spatial features
        H_decode = self.img_height // 2**self.sampling_num
        W_decode = self.img_width // 2**self.sampling_num

        active_slots = active_slots.view(batch_size * self.num_slots, self.slot_dim, 1, 1) # Adds spatial dimensions
        # Shape: (batch_size*num_slots, slot_dim, 1, 1)
        active_slots = active_slots.expand(-1, -1, H_decode, W_decode) # Broadcasting
        # Shape: (batch_size*num_slots, slot_dim, H_decode, W_decode

        # Concatenate with positional grid, one channel for x-coord and one for y-coord -> (slot_dim+2, h_decode, w_decode)
        # Do this to give network spatial information
        x_grid = torch.linspace(-1, 1, W_decode).view(1, -1).repeat(H_decode, 1) # Creates 2d grid of values from -1, 1 along x-coord
        y_grid = torch.linspace(-1, 1, H_decode).view(-1, 1).repeat(1, W_decode) # Creates 2d grid of values from -1, 1 along y-coord
        pos_grid = torch.stack((x_grid, y_grid), dim=0).to(active_slots.device)  # Concat x and y grid
        # Shape: (2, H_decode, W_decode)
        pos_grid = pos_grid.unsqueeze(0).expand(batch_size * self.num_slots, -1, -1, -1) # Expand to fit shape of slots
        # Shape: (batch_size * num_slots, 2, H_decode, W_decode)

        # Concatenate with slots for final input to decoder
        # input_decoder = torch.cat((pos_grid, active_slots), dim=1)
        input_decoder = active_slots
        # Shape: (batch_size * num_slots, slot_dim + 2, H_decode, W_decode))

        # -----------------------------------Decoder--------------------------------------------------------------------

        output_decoder = self.cnn_decoder(input_decoder)
        # Shape: (batch_size * num_slots, img_c, img_height, img_width)

        # Reshape (unflatten) output from decoder
        output_decoder = output_decoder.view(batch_size, self.num_slots, self.img_c, self.img_height, self.img_width)
        # Shape: (batch_size, num_slots, img_c, height, width)

        # -----------------------------------Attenion-Map Upsampling-------------------------------------------------------
        
        # Shape OG attn_maps: (batch_size, heads, num_slots, num_input_features)
        attn_maps_avg_head = attn_maps.mean(dim=1) # Average attention value over heads
        # Shape: (batch_size, num_slots, num_input_features)

        attn_maps_unsq = attn_maps_avg_head.view(batch_size, self.num_slots, H_encoded, W_encoded).unsqueeze(2)
        # Shape: (batch_size, num_slots, 1, H_encoded, W_encoded)

        # Upsampling of attention
        attn_maps_reshape = attn_maps_unsq.view(batch_size * self.num_slots, 1, H_encoded, W_encoded)
        # Shape: (batch_size * num_slots, 1, H_encoded, W_encoded)

        if self.sampling_num > 0:
            attn_maps_upsampled = self.attn_upsample(attn_maps_reshape)
        else:
            attn_maps_upsampled = attn_maps_reshape
        # Shape: (batch_size * num_slots, 1, img_height, img_width)

        attn_maps_recon = attn_maps_upsampled.view(batch_size, self.num_slots, 1, self.img_height, self.img_width)
        # Shape: (batch_size, num_slots, 1, img_height, img_width)

        # -----------------------------------Final Output------------------------------------------------------

        # Masking with attention maps
        masked_recon = output_decoder * attn_maps_recon
        # Shape: (batch_size, num_slots, img_c, img_h, img_w)

        # Sum over all slots for final out
        final_output = masked_recon.sum(dim=1)
        # Shape: (batch_size, img_c, img_h, img_w))

        return final_output, attn_maps_recon, keep_slots