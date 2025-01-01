import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        return self.linear(x)  # (batch_size, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a long enough positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it's not trained
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add position encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()  # or nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query, key, value shapes: (batch_size, seq_len, d_model)
        out, _ = self.mha(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout(out)

class ResidualConnection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer_out):
        # sublayer_out is the output of either attention or feed-forward
        return self.norm(x + sublayer_out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, nheads, dropout)
        self.res1 = ResidualConnection(d_model)

        self.ff = FeedForwardBlock(d_model, dim_feedforward, dropout)
        self.res2 = ResidualConnection(d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attention
        attn_out = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.res1(x, attn_out)

        # 2) Feed-forward
        ff_out = self.ff(x)
        x = self.res2(x, ff_out)

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention (for the decoder)
        self.self_attn = MultiHeadAttentionBlock(d_model, nheads, dropout)
        self.res1 = ResidualConnection(d_model)

        # Cross-attention (decoder attends to encoder output)
        self.cross_attn = MultiHeadAttentionBlock(d_model, nheads, dropout)
        self.res2 = ResidualConnection(d_model)

        # Feed-forward
        self.ff = FeedForwardBlock(d_model, dim_feedforward, dropout)
        self.res3 = ResidualConnection(d_model)

    def forward(self, x, enc_out, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # 1) Masked self-attention in the decoder
        _x = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.res1(x, _x)

        # 2) Cross-attention: Query = decoder states, Key/Value = encoder output
        _x = self.cross_attn(x, enc_out, enc_out, key_padding_mask=memory_key_padding_mask)
        x = self.res2(x, _x)

        # 3) Feed-forward
        _x = self.ff(x)
        x = self.res3(x, _x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.linear(x)  # (batch_size, seq_len, output_dim)


class Transformer(nn.Module):
    def __init__(
            self,
            input_dim,  # 1024 if we flatten (512 for image + 512 for pcd)
            d_model=512,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            output_dim=2  # speed + steering
    ):
        super().__init__()

        # Embedding layers for encoder and decoder
        self.encoder_embedding = InputEmbeddings(input_dim, d_model)
        self.encoder_positional_encoding = PositionalEncoding(d_model)

        self.decoder_embedding = InputEmbeddings(output_dim, d_model)
        self.decoder_positional_encoding = PositionalEncoding(d_model)

        # Encoder and Decoder
        self.encoder = Encoder(num_encoder_layers, d_model, nheads, dim_feedforward, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, nheads, dim_feedforward, dropout)

        # Final projection to speed & steering
        self.projection = ProjectionLayer(d_model, output_dim)

    def generate_subsequent_mask(self, size):
        """
        Generates an upper-triangular matrix of -inf, used for masking out future tokens.
        shape: (size, size)
        """
        attn_shape = (size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
        return subsequent_mask

    def forward(self, src, tgt):
        """
        src: (batch_size, src_seq_len, input_dim)  # e.g. 25 timesteps x 1024
        tgt: (batch_size, tgt_seq_len, output_dim) # e.g. 15 timesteps x 2
                                                   # (initial "teacher-forced" speed/steering or
                                                   #  a zero vector for autoregressive inference)
        """
        # --- Encoder ---
        # 1) Embed + position
        enc_in = self.encoder_embedding(src)  # (B, src_seq_len, d_model)
        enc_in = self.encoder_positional_encoding(enc_in)  # add positional info
        # 2) Pass through the stacked encoder layers
        memory = self.encoder(enc_in)  # (B, src_seq_len, d_model)

        # --- Decoder ---
        # 1) Embed + position
        dec_in = self.decoder_embedding(tgt)  # (B, tgt_seq_len, d_model)
        dec_in = self.decoder_positional_encoding(dec_in)

        # 2) Generate a causal mask for the target (so it can’t attend to future positions)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_subsequent_mask(tgt_seq_len).to(tgt.device)

        # 3) Pass through the stacked decoder
        dec_out = self.decoder(dec_in,
                               memory,
                               tgt_mask=tgt_mask)

        # --- Projection ---
        out = self.projection(dec_out)  # (B, tgt_seq_len, 2)
        return out


'''if __name__ == "__main__":
    # Suppose:
    # batch_size = 8
    # src_seq_len = 25  (5 seconds @ 5Hz)
    # tgt_seq_len = 15  (3 seconds @ 5Hz)
    # input_dim = 1024  (512 + 512)
    # output_dim = 2    (speed + steering)

    batch_size = 8
    src_seq_len = 25
    tgt_seq_len = 15
    input_dim = 1024
    output_dim = 2

    model = Transformer(
        input_dim=input_dim,
        d_model=512,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        output_dim=output_dim
    )

    # Dummy inputs
    src = torch.randn(batch_size, src_seq_len, input_dim)
    # For the decoder input, you typically have teacher-forcing during training:
    tgt = torch.randn(batch_size, tgt_seq_len, output_dim)

    # Forward pass
    out = model(src, tgt)  # shape: (batch_size, tgt_seq_len, 2)
    print("Output shape:", out.shape)
    print(src)'''
    # Output shape should be [8, 15, 2]