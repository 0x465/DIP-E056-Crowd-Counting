import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import cm as CM

import torch
from torchvision import transforms
from model import CANNet

import warnings
warnings.filterwarnings('ignore')

def get_videocapture(device):
    # initialise videocapture device
    cap = cv2.VideoCapture(device)
    if cap is None or not cap.isOpened():
        print('Warning: VideoCapture is None or not opened.')
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        # define codec and create VideoWriter object
        out = cv2.VideoWriter('output.avi', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, 
                            (width, height))

    return cap, out, width, height

def load_model(weight):
    # load model with trained weights
    model = CANNet()
    model = model.cuda()
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def get_transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])

    return transform

if __name__ == '__main__':
    transform = get_transform()
    model = load_model('part_B_train.pth.tar')
    # device_type = 0 for webcam 
    device_type = 'grandcentral.avi'
    cap, out, width, height = get_videocapture(device_type)

    try:
        print('Running. Press \'CTRL-C\' to exit.')
        iter_count = 0
        img_count = 1

        while(cap.isOpened()):
            _, frame = cap.read() # capture image

            if _:
                img = transform(frame).unsqueeze(0).cuda()

                # generate density
                density = model(img).data.cpu().numpy()
                pred_sum = density.sum()
                print('Sum: ', pred_sum)

                # normalize density
                density_min = np.min(density, axis=tuple(range(density.ndim-1)), keepdims=1)
                density_max = np.max(density, axis=tuple(range(density.ndim-1)), keepdims=1)
                density = (density - density_min)/ (density_max - density_min)

                # apply colormap and transparency
                im = Image.fromarray((CM.jet(density.squeeze())*255).astype(np.uint8))
                im.putalpha(128)

                # overlay background and foreground
                im = im.resize((width, height))
                frame = Image.fromarray(frame)
                frame.paste(im, (0, 0), im)
                
                # save image and video
                # first and then every 10 frames
                if iter_count == 0 or not iter_count % 10:
                    save_path = ('D:/Coding/DIP/Context-Aware-Crowd-Counting/output')
                    out_name = 'density_' + str(img_count) + '.png'
                    out_path = os.path.join(save_path, out_name)
                    frame.save(out_path)
                    out.write(cv2.cvtColor((np.array(frame)), cv2.COLOR_BGR2RGB))
                    img_count += 1

            iter_count += 1 

    except KeyboardInterrupt:
        print('\'CTRL-C\' pressed. Exiting.')
        cap.release()
        out.release()
        sys.exit(0)
