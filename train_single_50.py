import os
import argparse

# from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.datasets as datasets
from torch import nn
from torchvision.transforms import transforms

from model.utils import train_one_epoch, evaluate

# from model.transform import Transforms
from dan import DAN
from DDALoss import DDALoss

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lrf', type=float, default=0.0001)
# 是否启用SyncBatchNorm
parser.add_argument('--syncBN', type=bool, default=True)

# 数据集所在根目录
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
parser.add_argument('--data-path', type=str, default="/data/wdj_zhujl/RAF-DB/data")
# parser.add_argument('--data-path', type=str, default="/data/wdj_zhujl/fer2013+/data3_RAN")
# parser.add_argument('--data-path', type=str, default="/data/wdj_zhujl/AffectNet/manually")
# parser.add_argument('--data-path', type=str, default="/data/wdj_zhujl/SFEW/data_align")

parser.add_argument('--weights', type=str,
                    default='./weights/RAF_model.pth',
                    # default='./weights/ferplus_model.pth',
                    # default='./weights/affect0_model.pth',
                    # default='./weights/SFEW_model.pth',
                    # default='',
                    help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
# 不要改该参数，系统会自动分配
parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
# 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
parser.add_argument('--world-size', default=4, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()

def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # tb_writer = SummaryWriter("logs")


    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'test')

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomApply([
                                                 transforms.RandomRotation(20),
                                                 transforms.RandomCrop(224, padding=32)
                                             ], p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.RandomErasing(scale=(0.02, 0.25))
                                         ]))

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize((224, 224)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])
                                                           ]))


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=nw,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             # drop_last=True,
                                             shuffle=False,
                                             num_workers=nw,
                                             pin_memory=True)

    model = DAN(num_class=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
    #     # 删除不需要的权重
    #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #         else ['head.weight', 'head.bias', 'manet.fc_1.weight', 'manet.fc_1.bias']
    #         # else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #
    #     for k in list(weights_dict.keys()):
    #         if k.find('patch_embed.atten')!=-1:
    #             del weights_dict[k]
    #
    #     for k in del_keys:
    #         del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    # gpus = [0, 1, 2, 3]
    # model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
        'dda': DDALoss(num_classes=args.num_classes, feat_dim=128, lamb=0.01, gamma=4).to(device)
    }
    optimizer = {
                 # 'softmax': torch.optim.SGD(pg, lr=args.lr, weight_decay=1e-4, momentum=0.9),
                 'softmax': torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4),
                 'dda': torch.optim.SGD(criterion['dda'].parameters(), 0.5)
    }
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer['softmax'], lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer['softmax'], step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer['softmax'], gamma=0.9)
    # best_acc = 0.9035
    # best_acc = 0.579952
    # best_acc = 0.575
    best_acc = 0.0
    for epoch in range(args.epochs):
        # train
        # train_loss, train_acc = train_one_epoch(model=model,
        #                                         optimizer=optimizer,
        #                                         data_loader=train_loader,
        #                                         device=device,
        #                                         epoch=epoch,
        #                                         criterion=criterion)
        #
        # scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     criterion=criterion)

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # if best_acc < val_acc:
            # torch.save(model.state_dict(), "./weights/ferplus_model.pth")
            # best_acc = val_acc
        print(val_acc)



if __name__ == '__main__':
    main()
