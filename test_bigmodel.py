import fairseq #don't required
import torch

from omegaconf import OmegaConf
from desed_task.nnet.WRNN_2_ import WRNN

import pdb

model = torch.load('/home1/irteam/users/koo/trained2.pt')
big_cfg = OmegaConf.create(vars(vars(model['args'])['w2v_args']))

sed = WRNN(big_cfg)
pdb.set_trace()
sed.w2v.load_state_dict(model['model'])
