import sys

from tqdm import tqdm
import torch

from multi_train_utils.distributed_utils import reduce_value, is_main_process
from model.contrastive_loss import *


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    # optimizer.zero_grad()
    optimizer['softmax'].zero_grad()
    # optimizer['dda'].zero_grad()
    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for  step, ((x, y), labels) in enumerate(data_loader):
        # images, labels = data
        pred, z_i, z_j, c_i, c_j = model(x.to(device), y.to(device))
        loss_instance =criterion['criterion_instance'](z_i, z_j)
        loss_cluster = criterion['criterion_cluster'](c_i, c_j)
        loss = loss_instance + loss_cluster
        # loss = loss_function(pred, labels.to(device))
        # loss.backward()
        l_softmax = criterion['softmax'](pred, labels.to(device))
        # l_added, l_center, l_dda = criterion['dda'](feat, labels.to(device))
        # loss = l_softmax + l_added
        loss += l_softmax
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # optimizer.step()
        # optimizer.zero_grad()
        optimizer['softmax'].step()
        # optimizer['dda'].step()
        optimizer['softmax'].zero_grad()
        # optimizer['dda'].zero_grad()
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, ((x, y), labels) in enumerate(data_loader):
        pred, z_i, z_j, c_i, c_j = model(x.to(device), y.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()






