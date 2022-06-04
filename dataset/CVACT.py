import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import scipy.io as sio

class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x
        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

# pytorch implementation of CVACT loader
class CVACT(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = '/path/of/CVACT/ANU_data_small/', same_area=True, print_bool=False, polar = '',args=None):
        super(CVACT, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar
        self.mode = mode
        self.sat_size = [256, 256]  # [512, 512]
        self.sat_size_default = [256, 256]  # [512, 512]
        self.grd_size = [112, 616]  # [224, 1232]
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [750, 750]
        self.grd_ori_size = [224, 1232]

        if args.fov != 0:
            self.transform_query = input_transform_fov(size=self.grd_size,fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)

        if len(polar) == 0:
            self.transform_reference = input_transform(size=self.sat_size)
        else:
            self.transform_reference = input_transform(size=[750,750])

        self.to_tensor = transforms.ToTensor()

        anuData = sio.loadmat(os.path.join(self.root, 'ACT_data.mat'))

        self.id_all_list = []
        self.id_idx_all_list = []
        idx = 0
        missing = 0
        for i in range(0, len(anuData['panoIds'])):
            grd_id = 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
            sat_id = 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'

            self.id_all_list.append([sat_id, grd_id])
            self.id_idx_all_list.append(idx)
            idx += 1

        if print_bool:
            print('CVACT: load',' data_size =', len(self.id_all_list))

        self.id_list = []
        self.id_idx_list = []
        self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        self.trainNum = len(self.training_inds)
        if print_bool:
            print('CVACT train:', self.trainNum)

        for k in range(self.trainNum):
            sat_id = self.id_all_list[self.training_inds[k][0]][0]
            grd_id = self.id_all_list[self.training_inds[k][0]][1]
            if not os.path.exists(os.path.join(self.root, grd_id)) or not os.path.exists(os.path.join(self.root, sat_id)):
                if print_bool:
                    print('train:',k, grd_id, sat_id)
                missing += 1
            else:
                self.id_list.append(self.id_all_list[self.training_inds[k][0]])
                self.id_idx_list.append(k)

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)
        if print_bool:
            print('CVACT val:', self.valNum)

        self.id_test_list = []
        self.id_test_idx_list = []
        for k in range(self.valNum):
            sat_id = self.id_all_list[self.val_inds[k][0]][0]
            grd_id = self.id_all_list[self.val_inds[k][0]][1]
            if not os.path.exists(os.path.join(self.root, grd_id)) or not os.path.exists(
                    os.path.join(self.root, sat_id)):
                if print_bool:
                    print('val', k, grd_id, sat_id)
                missing += 1
            else:
                self.id_test_list.append(self.id_all_list[self.val_inds[k][0]])
                self.id_test_idx_list.append(k)
        if print_bool:
            print('missing:', missing)  # may miss some images

    def __getitem__(self, index, debug=False):
        if self.mode== 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.root + self.id_list[idx][1]).convert('RGB')
            img_query = img_query.crop((0,img_query.size[1]//4,img_query.size[0],img_query.size[1]//4*3))
            img_reference = Image.open(self.root + self.id_list[idx][0]).convert('RGB')
            img_query = self.transform_query(img_query)
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','train',str(idx)+'.png')).convert('RGB')
                return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), 0, self.to_tensor(atten_sat)
            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), 0, 0

        elif 'scan_val' in self.mode:
            img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
            img_query = img_query.crop((0, img_query.size[1] // 4, img_query.size[0], img_query.size[1] // 4 * 3))
            img_query = self.transform_query(img_query)
            return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0

        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val',str(index)+'.png')).convert('RGB')
                return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
            return img_reference, torch.tensor(index), 0

        elif 'test_query' in self.mode:
            img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
            img_query = img_query.crop((0, img_query.size[1] // 4, img_query.size[0], img_query.size[1] // 4 * 3))
            img_query = self.transform_query(img_query)
            return img_query, torch.tensor(index), torch.tensor(index)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'scan_val' in self.mode:
            return len(self.id_test_list)
        elif 'test_reference' in self.mode:
            return len(self.id_test_list)
        elif 'test_query' in self.mode:
            return len(self.id_test_list)
        else:
            print('not implemented!')
            raise Exception
