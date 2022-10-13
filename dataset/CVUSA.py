import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random

class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        # print(x.shape)
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


# pytorch version of CVUSA loader
class CVUSA(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = '/path/of/CVUSA/', same_area=True, print_bool=False, polar = '',args=None): #CV-dataset
        super(CVUSA, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar
        self.mode = mode
        self.sat_size = [256, 256]
        self.sat_size_default = [256, 256]
        self.grd_size = [112, 616]
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
            self.transform_reference = input_transform(size=[750,750]) # 750, 512

        self.to_tensor = transforms.ToTensor()

        self.train_list = self.root + 'splits/train-19zl.csv'
        self.test_list = self.root + 'splits/val-19zl.csv'

        if print_bool:
            print('CVUSA: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        if print_bool:
            print('CVUSA: load', self.train_list, ' data_size =', self.data_size)
            print('CVUSA: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        if print_bool:
            print('CVUSA: load', self.test_list, ' data_size =', self.test_data_size)

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.root + self.id_list[idx][1]).convert('RGB')
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
