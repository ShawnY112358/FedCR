import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import config
import sys

conf = config.conf

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        alpha = alpha.to(conf.device)
        probs = probs.to(conf.device)
        log_p = log_p.to(conf.device)

        batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Ratio_Cross_Entropy(nn.Module):
    r"""
            This criterion is a implemenation of Ratio Loss, which is proposed to solve the imbalanced
            problem in Fderated Learning

                Loss(x, class) = - \alpha  \log(softmax(x)[class])

            The losses are averaged across observations for each minibatch.

            Args:
                alpha(1D Tensor, Variable) : the scalar factor for this criterion
                size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                    However, if the field size_average is set to False, the losses are
                                    instead summed for each minibatch.
        """

    def __init__(self, class_num, alpha=None, size_average=False):
        super(Ratio_Cross_Entropy, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        # print(inputs)
        # print(P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        alpha = alpha.to(conf.device)
        probs = probs.to(conf.device)
        log_p = log_p.to(conf.device)

        batch_loss = - alpha * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        if math.isnan(loss.item()):
            print('inputs:' + str(inputs))
            print("P:" + str(P))
            sys.exit()
        return loss
