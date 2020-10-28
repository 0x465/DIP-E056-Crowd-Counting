import os
import sys
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm as CM

import torch
from torchvision import transforms
from model import CSRNet

import warnings
warnings.filterwarnings('ignore')

def parser():
    # define parsers
    main_parser = argparse.ArgumentParser(description='Image/Video/Real-Time Crowd Counnting')
    main_parser.add_argument('mode', choices=['image', 'video', 'real_time'], help='Choose mode of operation')
    main_subparser = main_parser.add_subparsers(dest='cmd')

    img_subparser = main_subparser.add_parser('img_path')
    img_subparser.add_argument('img_path', type=str, help='Image folder path')

    vid_subparser = main_subparser.add_parser('vid_path')
    vid_subparser.add_argument('vid_path', type=str, help='Video folder Path')

    rlt_subparser = main_subparser.add_parser('device_id')
    rlt_subparser.add_argument('device_id', type=int, help='VideoCapture device id')

    args = main_parser.parse_args()

    return args

def data_loader():
    # prepare data
    if args.mode == 'image':
        data = []
        [data.append(img) for img in glob.glob(os.path.join(args.img_path,'*.jpg'))]
        print('Number of images found: {}'.format(len(data)))
        task = 0
    elif args.mode == 'video':
        data = args.vid_path
        task = 1    
    elif args.mode == 'real_time':
        data = args.device_id
        task = 2

    return data, task

def load_model(weight):
    # load model with trained weights
    print('Loading model.....')
    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('Loaded.')
    
    return model

def get_videocapture(device, task):
    # initialise videocapture device
    print('Initialising VideoCapture device.....')
    cap = cv2.VideoCapture(device)
    if cap is None or not cap.isOpened():
        print('Warning: VideoCapture is None or not opened.')
    else:
        # get dimension and fps for resize and synchronization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if task == 2:
            fps = 1000
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
    
        # define codec and create VideoWriter object
        out = cv2.VideoWriter('output.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            fps, 
                            (width, height))
        print('Initialised.')
        interval = int(1000/fps)
        

    return cap, out, interval, width, height

def get_transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])

    return transform

if __name__ == '__main__':
    args = parser()
    data, task = data_loader()
    transform = get_transform()
    model = load_model('partBmodel_best.pth.tar')

    if task == 0:
        interval = 1
    elif task == 1 or task == 2:
        cap, out, interval, width, height = get_videocapture(data, task)

    try:
        print('Running. Press \'CTRL-C\' to exit.')
        iter_count = 0
        img_count = 1

        while(True):
            if task == 0:
                frame = Image.open(data[iter_count]).convert('RGB')
                width, height = frame.size
                _ = 1
            elif task == 1 or task == 2:
                _, frame = cap.read() # capture image
                frame = cv2.medianBlur(frame,5)

            if _:
                img = transform(frame).unsqueeze(0).cuda()

                # generate density
                density = model(img).data.cpu().numpy()
                pred_sum = density.sum()

                # normalize density
                density_min = np.min(density, axis=tuple(range(density.ndim-1)), keepdims=1)
                density_max = np.max(density, axis=tuple(range(density.ndim-1)), keepdims=1)
                density = (density - density_min)/ (density_max - density_min)

                # apply colormap and transparency
                im = Image.fromarray((CM.jet(density.squeeze())*255).astype(np.uint8))
                im.putalpha(128)

                # overlay background and foreground
                im = im.resize((width, height))
                if task == 1:
                    frame = Image.fromarray(frame)
                if task == 2:
                    Image.fromarray(frame).paste(im, (0, 0), im)
                else:
                    frame.paste(im, (0, 0), im)

                # write sum onto image
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                text = 'sum: ' + str(int(pred_sum))
                frame = cv2.putText(np.array(frame), text, (0,height-50), font, 2, (255,0,0), 1, cv2.LINE_AA)

                # save image/video
                if task == 1 or task == 2: 
                    if iter_count == 0 or not iter_count % 10:
                        save_path = ('D:/Coding/DIP/Final/CSRNet-pytorch/out_video')
                        out_name = 'density_' + str(img_count) + '.png'
                        out_path = os.path.join(save_path, out_name)
                        Image.fromarray(frame).save(out_path)
                        img_count += 1
                    out.write(cv2.cvtColor((np.array(frame)), cv2.COLOR_BGR2RGB))
                elif task == 0:
                    save_path = ('D:/Coding/DIP/Final/CSRNet-pytorch/out_image')
                    out_name = 'density_' + str(iter_count) + '.png'
                    out_path = os.path.join(save_path, out_name)
                    Image.fromarray(frame).save(out_path)

                # display image
                cv2.imshow('Density Map',cv2.cvtColor((np.array(frame)), cv2.COLOR_BGR2RGB))
                cv2.waitKey(interval)
                iter_count += 1
            
            else:
                print('End of video. Exiting')
                break

            if task == 0 and iter_count == len(data):
                print('All images done. Exiting.')
                break
        
        if task == 1 or task == 2:
            cap.release()
            out.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print('\'CTRL-C\' pressed. Exiting.')
        if task == 1:
            cap.release()
            out.release()
        cv2.destroyAllWindows()
        sys.exit(0)