import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from dan import DAN



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = DAN(num_head=1,num_class=7).to(device)
    weights_dict = torch.load('./weights/RAF_model.pth')
    model.load_state_dict(weights_dict, strict=False)
    # target_layers = [model.rdb]

    target_layers = [model.resnet.cv3]
    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # img_path = "/data/zhujinlin/RAF-DB/data/train/0/train_00013_aligned.jpg"
    type = ['train', 'test']
    index = 0
    for t in type:
        for i in range (7):
            list = os.listdir('/data/zhujinlin/RAF-DB/data/'+t+'/'+str(i))
            for file in list:
                img_path = '/data/zhujinlin/RAF-DB/data/'+t+'/'+str(i)+'/'+file
                assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(img_path).convert('RGB')
                resize = transforms.Resize((224, 224))
                img = resize(img)
                img = np.array(img, dtype=np.uint8)
                # img = center_crop_img(img, 224)

                # [C, H, W]
                img_tensor = data_transform(img)
                # expand batch dimension
                # [C, H, W] -> [N, C, H, W]
                input_tensor = torch.unsqueeze(img_tensor, dim=0)

                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                target_category = i  # tabby, tabby cat
                # target_category = 254  # pug, pug-dog

                grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                                  grayscale_cam,
                                                  use_rgb=True)
                # plt.imshow(visualization)
                # plt.show()
                base = '/data/zhujinlin/RAF-DB/CAM/' + t + '/' + str(i) + '/'
                if not os.path.exists(base+file.split('.')[0]):
                    os.mkdir(base+file.split('.')[0])
                plt.imsave(base+file.split('.')[0]+'/3.png',visualization)
                index = index+1
                if index % 100 == 0:
                    print(index)

if __name__ == '__main__':
    main()
