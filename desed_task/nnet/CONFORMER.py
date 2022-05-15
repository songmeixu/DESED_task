import warnings

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from torchaudio.models import Conformer as ConformerAudio

from conformer.encoder import ConformerEncoder, ConformerBlock, Linear

from .CNN import CNN


class CNN_CONFORMER_BLOCK(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        n_in_channel: int = 1,
        nclass: int = 10,
        attention: bool = True,
        activation="glu",
        dropout: float = 0.5,
        encoder_dim: int = 512,
        n_conformer_blocks: int = 2,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        train_cnn=True,
        cnn_integration=False,
        freeze_bn=False,
        **kwargs,
    ):
        """
            Initialization of CNN_CONFORMER_BLOCK model

        Args:
            n_in_channel: int, number of input channel
            **kwargs: keywords arguments for CNN_CONFORMER_BLOCK.
        """
        super(CNN_CONFORMER_BLOCK, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = 1 if cnn_integration else n_in_channel

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        nb_in = self.cnn.nb_filters[-1]
        if self.cnn_integration:
            # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
            nb_in = nb_in * n_in_channel

        # self.input_projection = nn.Sequential(
        #     Linear(nb_in, encoder_dim),
        #     nn.Dropout(p=input_dropout_p),
        # )

        self.conformer_blocks = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(n_conformer_blocks)])

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(encoder_dim, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(encoder_dim, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum(p.numel for p in self.parameters())

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, x, pad_mask=None):

        x = x.transpose(1, 2).unsqueeze(1)

        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # x = self.input_projection(x)

        # conformer blocks
        for layer in self.conformer_blocks:
            x = layer(x)

        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            if pad_mask is not None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CNN_CONFORMER_BLOCK, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var, Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class CONFORMER(nn.Module):
    """
    Compare to Class `CNN_CONFORMER_BLOCK`, the difference is
    use `Conformer.ConformerEncoder` instead of the `Conformer.ConformerBlock`,
    in which include `Conv2dSubampling` and `ConformerBlock`.
    Thus, `self.cnn` is not needed here.
    """
    def __init__(
        self,
        input_dim: int = 128,
        nclass: int = 10,
        attention: bool = True,
        encoder_dim: int = 512,
        n_conformer_blocks: int = 2,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        freeze_bn=False,
        **kwargs,
    ) -> None:
        """
            Initialization of CONFORMER model

        Args:
            n_in_channel: int, number of input channel
            **kwargs: keywords arguments for CONFORMER.
        """
        super(CONFORMER, self).__init__()
        self.attention = attention
        self.freeze_bn = freeze_bn

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=n_conformer_blocks,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )

        self.dense = Linear(encoder_dim, nclass, bias=False)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = Linear(encoder_dim, nclass, bias=False)
            self.softmax = nn.Softmax(dim=-1)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, x, pad_mask=None) -> Tuple[Tensor, Tensor]:

        x = x.transpose(1, 2)

        # input size : (batch_size, n_frames, n_freq)
        bs_in, frames_in, freq_in = x.size()

        x_lengths = torch.full([bs_in, ], frames_in, device=x.device)

        x, _ = self.encoder(x, x_lengths)

        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            if pad_mask is not None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CONFORMER, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var, Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class CONFORMER_AUDIO(nn.Module):
    """
    Compare to Class `CNN_CONFORMER_BLOCK`, the difference is
    use `torchaudio.models.conformer` instead of the `Conformer.ConformerBlock`.
    """
    def __init__(
        self,
        input_dim: int = 128,
        n_in_channel: int = 1,
        activation="glu",
        dropout: float = 0.5,
        nclass: int = 10,
        attention: bool = True,
        encoder_dim: int = 512,
        n_conformer_blocks: int = 2,
        num_attention_heads: int = 8,
        conv_kernel_size: int = 31,
        input_dropout_p: float = 0.1,
        train_cnn=True,
        cnn_integration=False,
        freeze_bn=False,
        **kwargs,
    ) -> None:
        """
            Initialization of CONFORMER_AUDIO model

        Args:
            n_in_channel: int, number of input channel
            **kwargs: keywords arguments for CONFORMER_AUDIO.
        """
        super(CONFORMER_AUDIO, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = 1 if cnn_integration else n_in_channel

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        nb_in = self.cnn.nb_filters[-1]
        if self.cnn_integration:
            # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
            nb_in = nb_in * n_in_channel

        self.input_projection = nn.Sequential(
            Linear(nb_in, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )

        self.encoder = ConformerAudio(
            input_dim=input_dim,
            ffn_dim=encoder_dim,
            num_layers=n_conformer_blocks,
            num_heads=num_attention_heads,
            depthwise_conv_kernel_size=conv_kernel_size,
        )

        self.dense = Linear(encoder_dim, nclass, bias=False)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = Linear(encoder_dim, nclass, bias=False)
            self.softmax = nn.Softmax(dim=-1)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum(p.numel for p in self.parameters())

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, x, pad_mask=None) -> Tuple[Tensor, Tensor]:

        x = x.transpose(1, 2).unsqueeze(1)

        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        x = self.input_projection(x)

        x_lengths = torch.full([x.size(0), ], x.size(1), device=x.device)
        x, _ = self.encoder(x, x_lengths)

        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            if pad_mask is not None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CONFORMER_AUDIO, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var, Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
