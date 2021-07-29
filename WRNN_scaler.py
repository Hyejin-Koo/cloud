import warnings

import torch.nn as nn
import torch

from .RNN import BidirectionalGRU

from .wav2vec2 import Wav2Vec2Model as w2v_encoder
from desed_task.utils.scaler import TorchScaler




class WRNN(nn.Module):
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
            Initialization of WRNN model
        
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
        super(WRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.scaler = self._init_scaler()
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = n_in_channel

        self.w2v = w2v_encoder(w2v_cfg)
        self.activation = nn.ReLU() # insert activateion between w2v and CNN, may use ReLU either
        

#        self.before_rnn = nn.Linear(768, 128) #n_RNN_cell)

        if rnn_type == "BGRU":
            nb_in = n_RNN_cell
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pad_mask=None):

        #x = x.transpose(1, 2).unsqueeze(1)
        # wav2vec model extracts feature
        feature = self.w2v(x)
#        x = feature.transpose(1,2)
        x = feature['x']
        x = self.activation(x)
        x = self.scaler(x)


        # input size : (batch_size, n_frames, n) = [B,499,768]
#        x = self.before_rnn(x)

        # rnn features
        x = self.rnn(x)
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
        super(WRNN, self).train(mode)
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
                        
                        
    def _init_scaler(self):
        """ Scaler inizialization
        Raises:
            NotImplementedError: in case of not Implemented scaler
        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler
