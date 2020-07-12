import os
from PIL import Image, ImageFile
import numpy as np
import torch

def load_image_list(folder):
    EXTENSIONS = ("jpg", "JPG", "png", "PNG", "jpeg", "JPEG")
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)\
                    if f.endswith(EXTENSIONS)]
    return image_files    

def load_image_list_from_a_bigfol(folder):
    subfols = [os.path.join(folder, f) for f in os.listdir(folder) 
                        if os.path.isdir(os.path.join(folder, f))]
    image_files = []
    for subfol in subfols:
        tmp_image_files = load_image_list(subfol)
        image_files += tmp_image_files
    return image_files

def save_image_list(folder, saved_file="data/images.txt"):
    '''
        Save image list into a text file for a fixed list
    '''
    EXTENSIONS = ("jpg", "JPG", "png", "PNG", "jpeg", "JPEG")
    image_files = [f for f in os.listdir(folder)\
                    if f.endswith(EXTENSIONS)]
    with open(saved_file, 'w') as f:
        for image_file in image_files:
            f.write(image_file + '\n')    

def load_image_list_from_textfile(txtfile, origin_path=None):    
    f = open(txtfile, 'r')
    if origin_path:
        image_files = [os.path.join(origin_path, line.strip()) for line in f if len(line.strip()) > 0]
    else:
        image_files = [line.strip() for line in f if len(line.strip()) > 0]
    classes = np.arange(len(image_files))
    return image_files, classes

def write_text(filename, s, mode='a'):
    with open(filename, mode) as f:
        f.write(s)

def find_best_models(fol):
    model_files = [f for f in os.listdir(fol) if f.startswith("model")]
    best_acc = 0
    ep = None
    best_model = None
    for model_file in model_files:
        tmp_parts = os.path.splitext(model_file)[0].split('_')
        tmp_ep = int(tmp_parts[1])
        tmp_acc = float(tmp_parts[-1])
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            best_model = model_file
            ep = tmp_ep
    if best_model:
        best_mem = best_model.replace('model', 'memory')
        return ep, best_model, best_mem
    else:
        return None
