import math
import numpy as np
import random
from PIL import Image, ImageFile, ImageFilter
import torch
from torchvision import transforms as trf
from config import IMAGE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(filename):
    return Image.open(filename).convert('RGB')

def get_mask(img, thre=255):    
    if len(img.getbands()) > 3:
        mask = np.array(img)[:,:,:-1].mean(axis=2)
    else:
        mask = np.array(img)[:,:,:-1].mean(axis=2)
    mask = Image.fromarray((mask < thre).astype(np.uint8) * 255, mode='L')
    return mask

def img_to_tensor(img):
    to_tensor = trf.Compose([
        trf.ToTensor(),
        trf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return to_tensor(img)

def tensor_to_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image *= np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image

def load_rgba_image(filename):
    img = Image.open(filename).convert('RGB')
    mask = get_mask(img)
    img.putalpha(mask)
    return img

def simple_transform_image(img):
    # input: PIL Image- RGB
    img = img.convert('RGB')
    tasks = trf.Compose([trf.Resize([IMAGE_SIZE, IMAGE_SIZE]), 
            trf.ToTensor(),
            trf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])    
    img = tasks(img)[:3, :, :].unsqueeze(0)    
    return img

def normal_transform_image(img):
    # input: PIL Image- RGB
    img = img.convert('RGB')
    tasks = trf.Compose([trf.RandomHorizontalFlip(p=0.5),
                    trf.RandomVerticalFlip(p=0.2),
                    trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    trf.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8,1.0)), 
                    trf.ToTensor(),
                    trf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    #trf.RandomErasing(p=0.2),
                    ])    
    img = tasks(img)[:3, :, :].unsqueeze(0)    
    return img


class SimpleTransform(object):    
    def __init__(self):
        self.transform = trf.Resize([IMAGE_SIZE, IMAGE_SIZE])

    def __call__(self, img):
        img = self.transform(img)
        try:
            mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            mask = torch.FloatTensor(mask)
            mask = mask.view(1, 1, mask.size(0), mask.size(1))            
        except:
            mask = torch.ones(1, 1, img.size[1], img.size[0])
        img = img.convert('RGB')
        return img_to_tensor(img).unsqueeze(0), mask

class NormalTransform(object):    
    def __init__(self):
        self.transform = trf.Compose([trf.RandomPerspective(p=0.5),  
                            trf.RandomRotation(degrees=[-45, 45]),
                            trf.RandomHorizontalFlip(p=0.5),                            
                            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            trf.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8,1.0)), 
                            ])

    def __call__(self, img):
        img = self.transform(img)
        try:
            mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            mask = torch.FloatTensor(mask)
            mask = mask.view(1, 1, mask.size(0), mask.size(1))            
        except:
            mask = torch.ones(1, 1, img.size[1], img.size[0])
        img = img.convert('RGB')
        return img_to_tensor(img).unsqueeze(0), mask

class ComplexTransform(object):
    def __init__(self, bg_images):
        self.bg_images = bg_images
        self.bg_transform = trf.Compose([
            trf.RandomHorizontalFlip(p=0.5),                                
            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            trf.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0))
            ])        
        
    def _get_transform(self):
        transform = trf.Compose([trf.RandomPerspective(p=0.5),  
                            trf.RandomRotation(degrees=[-45, 45]),
                            trf.RandomHorizontalFlip(p=0.5),                            
                            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            trf.RandomResizedCrop(size=random.randint(int(IMAGE_SIZE * 0.3), int(IMAGE_SIZE * 0.9)), scale=(0.8,1.0)),
                            ])        
        return transform

    def __call__(self, img):
        if random.random() > 0.05:
            img_bg = load_image(random.choice(self.bg_images))    # load a random background image 
            img_bg = self.bg_transform(img_bg)
        else:
            bg_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            img_bg = Image.new('RGB',(IMAGE_SIZE, IMAGE_SIZE), bg_color)

        transform = self._get_transform()
        img = transform(img)
        # paste
        im_w, im_h = img.size
        bg_w, bg_h = img_bg.size
        onset_h = random.randint(0, bg_h - im_h - 1)
        onset_w = random.randint(0, bg_w - im_w - 1)

        if len(img.split()) == 4:
            img_mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            img_mask = torch.FloatTensor(img_mask)
            img_bg.paste(img, [onset_w, onset_h], img)            
        else:
            img_bg.paste(img, [onset_w, onset_h])
            img_mask = torch.ones(im_h, im_w)

        img_bg = img_bg.convert('RGB')                
        mask = torch.zeros(1, 1, img_bg.size[1], img_bg.size[0])
        mask[:, :, onset_h:(onset_h + im_h), onset_w:(onset_w + im_w)] = img_mask
        return img_to_tensor(img_bg).unsqueeze(0), mask


###
#   TRANSFORM WITH THE USE OF EDGE 
###

class AddEdge(object):
    def __call__(self, img):
        r, g, b = img.split()
        edge = img.filter(ImageFilter.FIND_EDGES).convert('L')
        img.putalpha(edge)
        return img

def img_with_edge_to_tensor(img):
    to_tensor = trf.Compose([ 
        AddEdge(),       
        trf.ToTensor(),
        trf.Normalize((0.485, 0.456, 0.406, 0.5), (0.229, 0.224, 0.225, 0.25)),        
        ])
    return to_tensor(img)

class SimpleTransformEdge(object):    
    def __init__(self):
        self.transform = trf.Resize([IMAGE_SIZE, IMAGE_SIZE])

    def __call__(self, img):
        img = self.transform(img)
        try:
            mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            mask = torch.FloatTensor(mask)
            mask = mask.view(1, 1, mask.size(0), mask.size(1))
        except:
            mask = torch.ones(1, 1, img.size[1], img.size[0])        
        img = img.convert('RGB')        
        return img_with_edge_to_tensor(img).unsqueeze(0), mask

class NormalTransformEdge(object):    
    def __init__(self):
        self.transform = trf.Compose([trf.RandomPerspective(p=0.5),  
                            trf.RandomRotation(degrees=[-45, 45]),
                            trf.RandomHorizontalFlip(p=0.5),                            
                            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            trf.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8,1.0)), 
                            ])

    def __call__(self, img):
        img = self.transform(img)
        try:
            mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            mask = torch.FloatTensor(mask)
            mask = mask.view(1, 1, mask.size(0), mask.size(1))
        except:
            mask = torch.ones(1, 1, img.size[1], img.size[0])
        img = img.convert('RGB')        
        return img_with_edge_to_tensor(img).unsqueeze(0), mask

class ComplexTransformEdge(object):
    def __init__(self, bg_images):
        self.bg_images = bg_images
        self.bg_transform = trf.Compose([
            trf.RandomHorizontalFlip(p=0.5),                                
            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            trf.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0))
            ])        
        
    def _get_transform(self):
        transform = trf.Compose([trf.RandomPerspective(p=0.5),  
                            trf.RandomRotation(degrees=[-45, 45]),
                            trf.RandomHorizontalFlip(p=0.5),                            
                            trf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            trf.RandomResizedCrop(size=random.randint(int(IMAGE_SIZE * 0.3), int(IMAGE_SIZE * 0.9)), scale=(0.8,1.0)),
                            ])        
        return transform

    def __call__(self, img):
        if random.random() > 0.05:
            img_bg = load_image(random.choice(self.bg_images))    # load a random background image 
            img_bg = self.bg_transform(img_bg)
        else:
            bg_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            img_bg = Image.new('RGB',(IMAGE_SIZE, IMAGE_SIZE), bg_color)

        transform = self._get_transform()
        img = transform(img)
        # paste
        im_w, im_h = img.size
        bg_w, bg_h = img_bg.size
        onset_h = random.randint(0, bg_h - im_h - 1)
        onset_w = random.randint(0, bg_w - im_w - 1)
        if len(img.split()) == 4:
            img_mask = (np.array(img.getchannel(3)) >= 128).astype(np.float)
            img_mask = torch.FloatTensor(img_mask)
            img_bg.paste(img, [onset_w, onset_h], img)
        else:
            img_bg.paste(img, [onset_w, onset_h])
            img_mask = torch.ones(im_h, im_w)

        img_bg = img_bg.convert('RGB')                
        mask = torch.zeros(1, 1, img_bg.size[1], img_bg.size[0])
        mask[:, :, onset_h:(onset_h + im_h), onset_w:(onset_w + im_w)] = img_mask
        return img_with_edge_to_tensor(img_bg).unsqueeze(0), mask

