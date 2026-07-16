
import torch
import torch.nn as nn
from torch.autograd.function import Function



class DDALoss(nn.Module):
    #0.01    7
    def __init__(self, num_classes, feat_dim, lamb=0.01, gamma=7.0):
        super(DDALoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.centerloss = CenterLossFunction.apply
        self.feat_dim = feat_dim
        self.reset_params()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()
        self.lamb = lamb
        self.gamma = gamma

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        feat = feat.view(batch_size, -1)

        # center loss 
        if feat.size(1) != self.feat_dim:
            raise ValueError("Centers' dimensions: {0} should be equal to input feature's \
                              dim: {1}".format(self.feat_dim, feat.size(1)))
        centerloss = self.centerloss(feat, label, self.centers, batch_size)

        x = self.centers.unsqueeze(1)
        y = self.centers.unsqueeze(0)
        z = (x - y).pow(2)
        z = torch.where(z < 2, z, z-z+2)
        z = z.sum() / 128 / 2
        loss = self.lamb * centerloss + self.gamma / z

        return loss


class CenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, torch.tensor(batch_size))
        centers_batch = centers.index_select(0, label.long())
        z = (feature - centers_batch).pow(2)
        z = torch.where(z < 4, z, z-z+4)
        return z.sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers, None