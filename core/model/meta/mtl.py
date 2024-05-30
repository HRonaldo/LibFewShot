import torch
from torch import digamma, nn
import torch.nn.functional as F
import copy

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_mtl_module
from ...utils.ifsl_utils import IFSLUtils


class MTLBaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, ways, z_dim):
        super().__init__()
        self.ways = ways
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.ways, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.ways))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class MTL(MetaModel):
    def __init__(self, feat_dim, num_classes, inner_param, ifsl_param, use_MTL, **kwargs):
        super(MTL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.base_learner = MTLBaseLearner(self.way_num, z_dim=self.feat_dim).to(self.device)
        self.inner_param = inner_param

        self.loss_func = nn.CrossEntropyLoss()
        self.ifsl_param = ifsl_param
        
        # 调用类
        self.utils = IFSLUtils(self.emb_func, feat_dim, ifsl_param, self.device)

        convert_mtl_module(self, use_MTL)

    def set_forward(self, batch):
        """
        meta-validation
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        # 使用 IFSLUtils 进行特征处理和融合
        fused_support, fused_query, _ = self.utils.fusing(support_feat, query_feat)
        classifier, base_learner_weight = self.set_forward_adaptation(fused_support, support_target)

        output = classifier(fused_query, base_learner_weight)
        acc = accuracy(output, query_target.contiguous().reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """
        meta-train
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        # 使用 IFSLUtils 进行特征处理和融合
        fused_support, fused_query, _ = self.utils.fusing(support_feat, query_feat)
        classifier, base_learner_weight = self.set_forward_adaptation(fused_support, support_target)

        output = classifier(fused_query, base_learner_weight)
        loss = self.loss_func(output, query_target.contiguous().reshape(-1))
        acc = accuracy(output, query_target)

        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = self.base_learner
        logit = self.base_learner(support_feat)
        loss = self.loss_func(logit, support_target)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_parameters = list(
            map(
                lambda p: p[1] - 0.01 * p[0],
                zip(grad, self.base_learner.parameters()),
            )
        )

        for _ in range(1, self.inner_param["iter"]):
            logit = self.base_learner(support_feat, fast_parameters)
            loss = F.cross_entropy(logit, support_target)
            grad = torch.autograd.grad(loss, fast_parameters)
            fast_parameters = list(
                map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_parameters))
            )

        return classifier, fast_parameters
