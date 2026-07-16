import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.datasets as datasets

from vit_model import vit_base_patch16_224_in21k as create_model
from model.utils import train_one_epoch, evaluate
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lrf', type=float, default=0.01)

# 数据集所在根目录
parser.add_argument('--data-path', type=str,
                    default="../../../dataset/RAF")
parser.add_argument('--model-name', default='', help='create model name')

# 预训练权重路径，如果不想载入就设置为空字符
parser.add_argument('--weights', type=str, default='../../vision_transformer/weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
args = parser.parse_args()

def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter("logs")
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'test')
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             # transforms.Resize((224, 224)),
                                         transforms.RandomResizedCrop((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
                                         )
    val_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
                                       )


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    model = create_model(num_classes=7, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    gpus = [0, 1, 2, 3]
    model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc



if __name__ == '__main__':
    main()
