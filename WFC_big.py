import warnings

import torch.nn as nn
import torch

from .RNN import BidirectionalGRU
from .fairseq_pjh.models import BaseFairseqModel
from .wav2vec2 import Wav2Vec2Model_big, Wav2Vec2Config #as w2v_encoder
#from .wav2vec2 import Wav2Vec2Model as w2v_encoder
from .fairseq_pjh.fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model as w2v_encoder
#from .fairseq_pjh.models.wav2vec.wav2vec import Wav2Vec2Model, Wav2Vec2Config


class Wav2Vec2Encoder(BaseFairseqModel):
    def __init__(self, cfg:Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = Wav2Vec2Model_big(cfg)
        #self.proj = nn.Linear(1024,2001)

    def forward(self,x):
        w2v_feature = self.w2v_encoder(x) #out_dim: 2001 hmm..is it additional fc layer?
    #    x = self.proj(w2v_feature)
        return w2v_feature


class WFC(nn.Module):
    def __init__(
        self,
        w2v_cfg,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        freeze_bn=False,
        **kwargs,
    ):
        """
            Initialization of WFC model
        
        Args:
            w2v_cfg: pre-trained wav2vec configuration
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            freeze_bn: 
            **kwargs: keywords arguments for CNN.
        """
        super(WFC, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration =False #cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = n_in_channel
        #self.w2v = Wav2Vec2Encoder(w2v_cfg)
        self.w2v = w2v_encoder(w2v_cfg)
        self.activation = nn.ReLU() # insert activateion between w2v and CNN, may use ReLU either
        

#        self.before_rnn = nn.Linear(768, 128) #n_RNN_cell)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell, nclass)
        #self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pad_mask=None):
        #x = x.transpose(1, 2).unsqueeze(1)
        # wav2vec model extracts feature
        feature = self.w2v(x, features_only=True)['x']
        x = self.activation(feature)



        # input size : (batch_size, n_frames, n) = [B,499,768]
#        x = self.before_rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            if not pad_mask is None:
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
        super(WFC, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
