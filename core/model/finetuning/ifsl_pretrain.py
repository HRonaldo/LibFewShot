import torch
from torch import nn
import numpy as np

from core.utils import accuracy
from .finetuning_model import FinetuningModel


class IfslPretrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param,ifsl_pretrain_param, **kwargs):
        super(IfslPretrain, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param
        for (key,value) in ifsl_pretrain_param.items():
            setattr(self, key, value)
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()
        self.temp_features = np.zeros((self.num_class, self.feat_dim))
        self.features = np.zeros((self.num_class, self.feat_dim))
        self.counts = np.zeros((self.num_class))
        self.adds=np.zeros((self.num_class))

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)

        target = target.view(-1).to(self.device)
        feat = self.emb_func(image)
        output = self.classifier(feat)

        acc=100*torch.count_nonzero(torch.argmax(output, dim=1)==target).detach().cpu().item()/target.size(0)
        return output, acc


    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)

        if self.training is False:
            f = feat.detach().cpu().numpy()
            id=target.detach().cpu().numpy()
            self.adds[id] += 1
            self.temp_features[id] =f[id]
            for i in range(self.num_class):
                if self.adds[i]>=self.batch_num:
                    t=self.adds[i]+self.counts[i]
                    alpha=self.adds/t
                    self.features[i] =self.features[i]*(1-alpha)+self.temp_features[i]*alpha
                    self.counts[i] +=self.adds[i]
                    self.adds[i]=0
                    self.temp_features[i]=0
            np.save(self.feature_path,self.features)
        loss = self.loss_func(output, target)
        acc = 100 * torch.count_nonzero(torch.argmax(output, dim=1) == target).detach().cpu().item() / target.size(0)
        return output, acc, loss

    def set_forward_adaptation(self):
        pass

