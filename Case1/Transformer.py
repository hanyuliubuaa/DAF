import torch
import torch.nn as nn
from layers.Transformer_Decoder import Decoder, DecoderLayer
from layers.SelfAttention import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    Vanilla Transformer
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out

        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_dec):
        # Embedding
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, x_mask=None)
        return dec_out

    def forward(self, x_dec):
        dec_out = self.forecast(x_dec)
        return dec_out
