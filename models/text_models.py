"""
@author: Adityam Ghosh
Date: 10-15-2023

"""
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.text_conv_models import LightWeightConvBlock
from models.text_transformers import TransformersEncoderBlock, PositionalEncoding

from transformers import ConvBertModel, ConvBertConfig


class LiteTransformerEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.embed_dim = config["embedding_dim"]
        dropout_rate = config["dropout_rate"]

        self.embedding = nn.Embedding(
            num_embeddings=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
            padding_idx=config["padding_idx"],
        )

        self.n_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "trans_encoder": TransformersEncoderBlock(
                            input_dim=config["embedding_dim"] // 2,
                            num_heads=config["transformer_encoder"]["num_heads"],
                            dropout_rate=config["transformer_encoder"]["dropout_rate"],
                        ),
                        "lconv_block": LightWeightConvBlock(
                            input_dim=config["embedding_dim"] // 2,
                            num_heads=config["lconv"]["num_heads"],
                            kernel_size=config["lconv"]["kernel_size"][i],
                            padding=config["lconv"]["padding"][i]
                            if isinstance(config["lconv"]["padding"], list)
                            else config["lconv"]["padding"],
                            weight_softmax=config["lconv"]["weight_softmax"][i]
                            if isinstance(config["lconv"]["weight_softmax"], list)
                            else config["lconv"]["weight_softmax"],
                            dropout_rate=config["lconv"]["dropout_rate"][i]
                            if isinstance(config["lconv"]["dropout_rate"], list)
                            else config["lconv"]["dropout_rate"],
                        ),
                        "layer_norm1": nn.LayerNorm(config["embedding_dim"]),
                        "layer_norm2": nn.LayerNorm(config["embedding_dim"]),
                        "ffn": nn.Sequential(
                            nn.Linear(
                                in_features=config["embedding_dim"],
                                out_features=config["ffn_embedding_dim"],
                            ),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(
                                in_features=config["ffn_embedding_dim"],
                                out_features=config["embedding_dim"],
                            ),
                        ),
                    }
                )
                for i in range(config["n_blocks"])
            ]
        )

        self.pos_encoding = PositionalEncoding(
            embed_dim=config["embedding_dim"] // 2,
            max_seq_length=config["max_seq_length"],
            dropout_rate=config["pos_encoding_dropout_rate"],
        )

        self.out_layer = nn.Linear(
            in_features=config["embedding_dim"], out_features=config["output_dim"]
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.last_layer_norm = (
            nn.LayerNorm(config["embedding_dim"]) if config["last_layer_norm"] else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embed_dim**-0.5)
        for module in self.n_blocks:
            for name, m in module["ffn"].named_modules():
                if "Linear" in name:
                    nn.init.constant_(m.bias, 0.0)
                    nn.init.xavier_uniform_(m.weight)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        An implementation of the LiteTransformerBlock Forward function

        Parameters:
        -----------
        :param x: a tensor of shape (B x T)

        Returns:
        --------
        a tensor of shape (B x T x C)
        """
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        for i, block in enumerate(self.n_blocks):
            x_left = x[:, :, : self.embed_dim // 2]
            x_right = x[:, :, self.embed_dim // 2 :]
            if i == 0:
                x_left = self.pos_encoding(x_left)

            x_left_out = block["trans_encoder"](x_left, attn_mask)
            x_right_out = block["lconv_block"](x_right, None)

            concat_x = torch.concat([x_left_out, x_right_out], dim=-1)
            concat_x = self.dropout(concat_x)
            add_norm1 = block["layer_norm1"](x + concat_x)
            ffn_out = block["ffn"](add_norm1)
            ffn_out = self.dropout(ffn_out)
            add_norm2 = block["layer_norm2"](add_norm1 + ffn_out)

            x = add_norm2

        x = self.out_layer(x[:, -1, :])
        x = F.tanh(x)
        if self.last_layer_norm is not None:
            x = self.last_layer_norm(x)

        return x


class ConvBertEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.model = None
        if config["pretrained"]:
            self.model = ConvBertModel.from_pretrained("YituTech/conv-bert-base")

        else:
            conv_bert_cfg = ConvBertConfig(
                num_hidden_layers=config["num_hidden_layers"],
                num_attention_heads=config["num_attention_heads"],
            )
            self.model = ConvBertModel(conv_bert_cfg)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.model(input_ids=x, attention_mask=attn_mask).last_hidden_state[
            :, 0, :
        ]
