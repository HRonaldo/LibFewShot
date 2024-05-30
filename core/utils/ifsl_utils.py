
import numpy as np
import torch
import torch.nn as nn

class IFSLUtils(nn.Module):
    def __init__(self,embed_func,feat_dim,ifsl_param,device):
        super(IFSLUtils, self).__init__()
        self.embed_func = embed_func
        self.feat_dim = feat_dim
        self.device=device
        for (key,value) in ifsl_param.items():
            setattr(self,key,value)
        self.linear=nn.Linear(feat_dim,self.class_num)
        self.softmax = nn.Softmax(dim=1)
        self.features=torch.from_numpy(self.get_pretrain_features()).float().to(self.device)
        self.mean_features=self.features.mean(dim=0)
    def forward(self,x):
        return self.embed_func(x)
    def classify(self,x,is_feature=False):
        if is_feature is True:
            return self.softmax(self.linear(x))
        return self.softmax(self.linear(self(x)))
    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized
    def fuse_proba(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == "linear_sum":
            return p1 + p2
        elif self.logit_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p /(1 + p))
    def fuse_features(self, x1, x2):
        if self.fusion == "concat":
            return torch.cat((x1, x2), dim=2)
        elif self.fusion == "+":
            return x1 + x2
        elif self.fusion == "-":
            return x1 - x2
    def get_feat_dim(self):
        split_feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "pd":
            return split_feat_dim + self.num_classes
        else:
            if self.fusion == "concat":
                return split_feat_dim * 2
            else:
                return split_feat_dim
    # def softmax_score(self,scores,c_scores):
    #     scores = self.softmax(scores)
    #     c_scores = self.softmax(c_scores)
    #     if self.use_counterfactual:
    #         scores = scores - c_scores
    #     if self.sum_log:
    #         scores = scores.log()
    #         scores = scores.mean(dim=0)
    #     else:
    #         scores = scores.mean(dim=0)
    #         scores = scores.log()
    #     return scores
    def process_score(self,a,b):
        if self.single is True:
            return a*self.temp,b*self.temp
        c= torch.ones_like(a).to(self.device)
        if self.use_x_only:
            return a*self.temp, c*self.temp
        return self.fuse_proba(a,b)*self.temp,self.fuse_proba(c,b)*self.temp

    def fusing(self,support,query):
        support = self.embed_func(support)
        query = self.embed_func(query)
        split_support, support_d, split_query, query_d = self.get_features(support, query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)
        if self.x_zero:
            c_split_query = torch.zeros_like(split_query).to(self.device)
        else:
            c_split_query = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
        c_fused_query = self.fuse_features(c_split_query, query_d)
        if self.single is True:
            return fused_support,fused_query,c_fused_query
        else:
            return  split_support,support_d,split_query,query_d

    # def get_mean_features_(self, dataloader,normalize=True):
    #     features = np.zeros((self.class_num, self.feat_dim))
    #     counts = np.zeros((self.class_num))
    #     for (data, label) in dataloader:
    #         output = self.embed_func(data.to(self.device))
    #         if normalize is True:
    #             output=self.normalize(output)
    #         label = label.numpy()
    #         output = output.cpu().detach().numpy()
    #         for j in range(output.size(0)):
    #             idx = label[j]
    #             counts[idx] += 1
    #             features[idx] += output[j]
    #     for i in range(self.class_num):
    #         if counts[i] > 0:
    #             features[i] = features[i] / counts[i]
    #     return features
    def get_pretrain_features(self):
        if self.feature_path is not None:
            return np.load(self.feature_path)
        return np.zeros((self.class_num, self.feat_dim))
    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).to(self.device)
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            split_features[i] = x[:, start_idx:end_idx]
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
        return split_features

    def nn_preprocess(self, data, center=None, preprocessing="l2n"):
        if preprocessing == "none":
            return data
        elif preprocessing == "l2n":
            return self.normalize(data)
        elif preprocessing == "cl2n":
            if self.normalize_before_center:
                data = self.normalize(data)
            centered_data = data - center
            return self.normalize(centered_data)

    def calc_pd(self, x):
        with torch.no_grad():
            proba = self.classify(x,True)
        return proba

    def get_d_feature(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "ed":
            d_feat_dim = int(self.feat_dim / self.n_splits)
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim).to(self.device)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            pd = self.calc_pd(x)
            if self.d_feature == "pd":
                d_feature[i] = pd
            else:
                d_feature[i] = torch.mm(pd, self.features)[:, start:stop]
        return d_feature

    def get_features(self, support, query):
        support_d = self.get_d_feature(support)
        query_d = self.get_d_feature(query)
        if self.normalize_ed:
            support_d = self.normalize(support_d, dim=2)
            query_d = self.normalize(query_d, dim=2)
        support_size = support.shape[0]
        query_size = query.shape[0]
        pmean_support = self.mean_features.expand((support_size, self.feat_dim))
        pmean_query = self.mean_features.expand((query_size, self.feat_dim))
        support = self.nn_preprocess(support, pmean_support, preprocessing=self.preprocess_before_split)
        query = self.nn_preprocess(query, pmean_query, preprocessing=self.preprocess_before_split)
        split_support = self.get_split_features(support, preprocess=True, center=pmean_support, preprocess_method=self.preprocess_after_split)
        split_query = self.get_split_features(query, preprocess=True, center=pmean_query,preprocess_method=self.preprocess_after_split)
        return split_support, support_d, split_query, query_d