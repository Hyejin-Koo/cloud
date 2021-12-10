import warnings

import torch.nn as nn
import torch

from .RNN import BidirectionalGRU

from .hubert import HubertModel as hubert

from .SpecAugment import SpecAugment

from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingTask,
)

class HFC(nn.Module):
    def __init__(
        self,
        hubert_cfg,
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
        freq_mask_para=18,
        time_mask_num=10,
        freq_mask_num=2,
        **kwargs,
    ):
        """
            Initialization of HFC model
        
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
        super(HFC, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = n_in_channel

        self.hubert = hubert(hubert_cfg, HubertPretrainingTask)
        self.activation = nn.ReLU() # insert activateion between w2v and CNN, may use ReLU either


#        self.freq_mask_para, self.time_mask_num, self.freq_mask_num = freq_mask_para, time_mask_num, freq_mask_num


        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768,256)
        self.fc2 = nn.Linear(256,64)
        self.dense = nn.Linear(64, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            #self.dense_softmax = nn.Linear(768, nclass)
            self.dense_softmax = nn.Linear(64, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pad_mask=None):

        #x = x.transpose(1, 2).unsqueeze(1)
        # wav2vec model extracts feature
        feature = self.hubert(x)
        #x = feature.transpose(1,2)
        x = feature['x']
        x = self.activation(x)

#        spec = SpecAugment(self.freq_mask_para, self.time_mask_num,self.freq_mask_num)
#        x = spec(x.clone())

        # input size : (batch_size, n_frames, n) = [B,499,768]
#        x = self.before_rnn(x)

        # rnn features
        #x = self.rnn(x)

        #add fc layer
        x = self.fc1(x)
        x = self.fc2(x)

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
        super(HFC, self).train(mode)
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
