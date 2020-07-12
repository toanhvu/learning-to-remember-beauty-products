import os
import random
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from util import load_image_list_from_textfile
from transform import load_image, load_rgba_image, simple_transform_image
from config import IMAGE_SIZE, IMAGE_PATH

## LOADER FOR TRAINING
class BeautyData(data.Dataset):
    def __init__(self, image_links, image_labels, bg_links, transform=simple_transform_image, load_function=load_image):
        self.image_links = image_links
        self.image_labels = image_labels
        self.bg_links = bg_links
        self.transform = transform
        self.load_function = load_function

    def __getitem__(self, index):
        image_path, image_label = self.image_links[index], self.image_labels[index]        
        img = self.load_function(image_path)        # PIL image
        img, mask = self.transform(img)       # tensor
        return img, mask, torch.LongTensor([image_label])

    def __len__(self):
        return len(self.image_links)

def collate_fn(data):
    X, M, Y = zip(*data)
    X = torch.cat(X, 0)
    M = torch.cat(M, 0)
    Y = torch.cat(Y, 0)
    return X, M, Y

def get_beauty_loader(image_links, image_labels, bg_links, transform, load_function, batch_size=8, 
                        shuffle=True, num_workers=8):
    data = BeautyData(image_links, image_labels, bg_links, transform, load_function)                        
    data_loader = torch.utils.data.DataLoader(dataset = data,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn = collate_fn)
    return data_loader
