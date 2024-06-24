
import torch
import torch.nn as nn
from transformers import XLMRobertaModel,XLMRobertaConfig
from torch.nn.init import xavier_uniform_

from .encoder import TransformerInterEncoder, Classifier, RNNEncoder
from .optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class XLM_R(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, xlmr_config):
        super(XLM_R, self).__init__()
        if(load_pretrained_bert):
            self.model = XLMRobertaModel.from_pretrained("hfl/cino-large-v2")
        else:
            self.model = XLMRobertaModel(xlmr_config)

    def forward(self, x, segs, mask):

        outputs = self.model(input_ids = x,attention_mask = mask)
        last_hidden_state = outputs.last_hidden_state
        
        top_vec = last_hidden_state
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, xlmr_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.xlmr = XLM_R(args.temp_dir, load_pretrained_bert, xlmr_config)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.xlmr.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.xlmr.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.xlmr.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            xlmr_config = XLMRobertaConfig.from_pretrained("hfl/cino-large-v2")
            self.xlmr.model = XLMRobertaModel(xlmr_config)
            self.encoder = Classifier(self.xlmr.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        
        top_vec = self.xlmr(x, segs, mask)

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
    
