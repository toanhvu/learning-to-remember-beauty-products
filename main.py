import os
import sys
import random
import argparse
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from util import load_image_list_from_textfile, save_image_list, load_image_list, \
                    load_image_list_from_a_bigfol, write_text, find_best_models
from transform import SimpleTransform, NormalTransform, ComplexTransform, load_image, load_rgba_image, \
                    SimpleTransformEdge, NormalTransformEdge, ComplexTransformEdge
from loader import  get_beauty_loader
from models import Dense121, Dense121Edge, MultiScaleDense121, MultiScaleDense121Edge
from memory import Memory
from config import IMAGE_PATH, IMAGE_SIZE, FEAT_DIM, TOPK

np.random.seed(12345)
torch.random.manual_seed(12345)

parser = argparse.ArgumentParser(description='AI Meets Beauty 2020')
parser.add_argument('--model', type=str, default='Dense121', metavar='M',
                    help='model name including Dense121, Dense121Edge, MultiScaleDense121, MultiscaleDense121Edge')
parser.add_argument('--batch-size', type=int, default=32, metavar='BZ',
                    help='input batch size for training (default: 32)')
parser.add_argument('--no-proc', type=int, default=32, metavar='PZ',
                    help='input batch size for training (default: 32)')        
parser.add_argument('--epochs', type=int, default=100, metavar='EP',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                    help='weight decay (default: 0.0001)')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PROC = args.no_proc
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

save_fol = "data"
if not os.path.exists(save_fol):
    os.mkdir(save_fol)

# create image list
data_file = os.path.join(save_fol, "images.txt")
if not os.path.exists(data_file):
    print("Create a fixed list of images in the dataset")
    save_image_list(folder=IMAGE_PATH, 
                saved_file=data_file)
    print("done!")

# load image_files, and labels
print("Load images and their labels")
image_files, labels = load_image_list_from_textfile(data_file, IMAGE_PATH)
print("done!")

# models        
if args.model=='Dense121':
    model = Dense121(feat_dim=FEAT_DIM, pretrained=True, frozen=False)  
elif args.model=='Dense121Edge':
    model = Dense121Edge(feat_dim=FEAT_DIM, pretrained=True, frozen=False)  
elif args.model=='MultiScaleDense121':
    model = MultiScaleDense121(feat_dim=FEAT_DIM, pretrained=True, frozen=False)      
elif args.model=='MultiScaleDense121Edge':
    model = MultiScaleDense121Edge(feat_dim=FEAT_DIM, pretrained=True, frozen=False)  
else:
    print("Model should be in [Dense121, Dense121Edge, MultiScaleDense121, MultiScaleDense121Edge]");
    exit()

model = model.to(device)

memory = Memory(mem_size=len(labels), feat_dim=FEAT_DIM, margin=1, topk=100, update_rate=0.2)
memory = memory.to(device)
for p in memory.parameters():
    p.requires_grad = False

save_fol = os.path.join(save_fol, model.__class__.__name__)
if not os.path.exists(save_fol):
    os.mkdir(save_fol)
    shutil.copy(data_file, save_fol)

model_file = os.path.join(save_fol, "model_{}_{:.3f}_{:.3f}_{:.3f}.pkl")
mem_file = os.path.join(save_fol, "memory_{}_{:.3f}_{:.3f}_{:.3f}.pkl")
log_file = os.path.join(save_fol, "logfile.txt")

START_EP = 0

try:
    START_EP, saved_best_model, saved_best_memory = find_best_models(save_fol)
    print("Load model... %s"%saved_best_model)
    model.load_state_dict(torch.load(os.path.join(save_fol, saved_best_model)))
    print("Load memory... %s"%saved_best_memory)
    memory.load_state_dict(torch.load(os.path.join(save_fol, saved_best_memory)))
    START_EP += 1
except:
    START_EP = 0
    print("No pretrained models!")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# load bg list
bg_fol1 = "/media/moon/data/VOCdevkit/VOC2012/JPEGImages"
bg_fol2 = "/media/moon/data/coco/coco/images/train2017"
bg_fol3 = "/media/moon/data/imagenet/imagenet_images"

bg_list1 = load_image_list(bg_fol1)
bg_list2 = load_image_list(bg_fol2)
bg_list3 = load_image_list_from_a_bigfol(bg_fol3)

bg_list = bg_list1 + bg_list2 + bg_list3

# Image transform
if 'Edge' in args.model:
    simple_image_transform = SimpleTransformEdge()
    normal_image_transform = NormalTransformEdge()
    complex_image_transform = ComplexTransformEdge(bg_list)
else:
    simple_image_transform = SimpleTransform()
    normal_image_transform = NormalTransform()
    complex_image_transform = ComplexTransform(bg_list)

## Eval function
def estimate_topk_accuracy(model, memory, loader, k=TOPK, no_batch=1000):
    model.eval()
    memory.eval()
    pbar = tqdm(enumerate(loader))    
    acc = 0.0
    count = 0.0
    for i ,(miniX, _, miniY) in pbar:        
        miniX, miniY = miniX.to(device), miniY.to(device)
        with torch.no_grad():
            feat, _ = model(miniX) 
            distances, indices = memory.search_l2(feat, k)
        miniY = miniY.unsqueeze(1).expand(miniX.size(0), k)
        acc += ((miniY == indices).float().sum() / miniX.size(0)).item()
        count += 1
        if count > no_batch:
            break
    pbar.close()
    return acc / count

def adjust_learning_rate(optimizer, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= ratio


## TRAIN
best_acc = 0.
is_reloaded = False

for ep in range(START_EP, START_EP + NUM_EPOCHS):
    att_train = False
    frozen_model = False
    if ep == 0:
        loader = get_beauty_loader(image_files, labels, bg_list, simple_image_transform, load_image,
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)
    elif ep == 1:
        loader = get_beauty_loader(image_files, labels, bg_list, normal_image_transform, load_image,
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)
    elif ep < max(50, START_EP + 20) and best_acc < 97.0:
        frozen_model = False        
        if random.random() > 0.3:
            print("Using Complex Transform...")
            loader = get_beauty_loader(image_files, labels, bg_list, complex_image_transform, 
                            load_rgba_image if random.random() > 0.3 else load_image,
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)                    
            att_train = True
        else:
            print("Using Simple/Normal Transform...")
            loader = get_beauty_loader(image_files, labels, bg_list, 
                        normal_image_transform if random.random() > 0.3 else simple_image_transform, 
                        load_image,
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)            
            att_train = False
    else:
        frozen_model = True
        image_transform = random.choice([simple_image_transform, normal_image_transform, complex_image_transform])
        loader = get_beauty_loader(image_files, labels, bg_list, 
                    image_transform, 
                    random.choice([load_image, load_rgba_image]),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)            
        att_train = False

    soft_train = False if ep > 6 else True
    
    # load best model and memory for once
    if frozen_model and not is_reloaded:        
        is_reloaded = True        
        _, saved_best_model, saved_best_memory = find_best_models(save_fol)
        print("Load best model... %s"%saved_best_model)
        model.load_state_dict(torch.load(os.path.join(save_fol, saved_best_model)))
        print("Load best memory... %s"%saved_best_memory)
        memory.load_state_dict(torch.load(os.path.join(save_fol, saved_best_memory)))
        memory.update_rate = 0.05

    pbar = tqdm(enumerate(loader))    
    for i ,(miniX, miniM, miniY) in pbar:        
        miniX, miniM, miniY = miniX.to(device), miniM.to(device), miniY.to(device)                 
        if ep==0:
            model.eval()
            with torch.no_grad():
                feat, att, miniM = model.forward_train(miniX, miniM, soft_train)            
                memory.update_mem(feat, miniY)                  
            pbar.set_description("Epoch {}  {}/{}".format(ep, i, len(loader)))

        elif ep > 0 and not frozen_model:            
            model.train()            
            optimizer.zero_grad()        
            feat, att, miniM = model.forward_train(miniX, miniM, soft_train)            
            att_loss = torch.nn.BCELoss()(att, miniM)
            loss, min_loss, max_loss = memory.compute_l2loss(feat, miniY) 
            if ep >= 2 and att_train:
                loss = loss + att_loss
            
            loss.backward() 
            optimizer.step() 
            memory.update_mem(feat, miniY)           
            pbar.set_description("Epoch {}  {}/{}  loss={:3f} ~~ min_loss={:3f}, max_loss={:.3f}, att_loss={:.3f}".format(ep, i, len(loader), 
                                loss.item(), min_loss.item(), max_loss.item(), att_loss.item()))
        else:
            model.eval()
            with torch.no_grad():
                feat, att, miniM = model.forward_train(miniX, miniM, soft_train)            
                att_loss = torch.nn.BCELoss()(att, miniM)
                loss, min_loss, max_loss = memory.compute_l2loss(feat, miniY)            
                memory.update_mem(feat, miniY)           
                pbar.set_description("Epoch {}  {}/{}  loss={:3f} ~~ min_loss={:3f}, max_loss={:.3f}, att_loss={:.3f}".format(ep, i, len(loader), 
                                    loss.item(), min_loss.item(), max_loss.item(), att_loss.item()))

    pbar.close()

    if (ep - START_EP + 1) % 10 == 0 and frozen_model:
        adjust_learning_rate(optimizer, 0.5)
        memory.update_rate *= 0.5
        
    if ep > 2 and ep % 3 == 0:        
        ## random eval    
        no_batch = int(0.2 * len(image_files) / BATCH_SIZE)    
        # simple
        eval_loader = get_beauty_loader(image_files, labels, bg_list, simple_image_transform, load_image, 
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)
        simple_val_acc = estimate_topk_accuracy(model, memory, eval_loader, k=TOPK, no_batch=no_batch)
        print("\n\n---- Simple Val Acc = {}".format(simple_val_acc))
        write_text(log_file, "Epoch {} : simple val acc {} \n".format(ep, simple_val_acc))

        # normal
        eval_loader = get_beauty_loader(image_files, labels, bg_list, normal_image_transform, load_image, 
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)
        normal_val_acc = estimate_topk_accuracy(model, memory, eval_loader, k=TOPK, no_batch=no_batch)
        print("---- Simple Val Acc = {}".format(normal_val_acc))
        write_text(log_file, "Epoch {} : normal val acc {} \n".format(ep, normal_val_acc))

        # Complex 1
        eval_loader = get_beauty_loader(image_files, labels, bg_list, complex_image_transform, load_rgba_image,
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)   
        complex_val_acc1 = estimate_topk_accuracy(model, memory, eval_loader, k=TOPK, no_batch=no_batch)
        print("---- Complex Val Acc 1 with load_rgba = {}".format(complex_val_acc1))
        write_text(log_file, "Epoch {} : complex val acc 1 with load_rgba {} \n".format(ep, complex_val_acc1))

        # Complex 2
        eval_loader = get_beauty_loader(image_files, labels, bg_list, complex_image_transform, load_image,
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROC)   
        complex_val_acc2 = estimate_topk_accuracy(model, memory, eval_loader, k=TOPK, no_batch=no_batch)
        print("---- Complex Val Acc 2 with load_image = {}".format(complex_val_acc2))
        write_text(log_file, "Epoch {} : complex val acc 2  with load_image {} \n".format(ep, complex_val_acc2))

        val_acc = (simple_val_acc + normal_val_acc + complex_val_acc1 + complex_val_acc2) / 4.0
        if val_acc > best_acc:
            best_acc = val_acc  
        print("===== Vac Acc {}".format(val_acc))
        write_text(log_file, "Epoch {} : VAL ACC {} \n".format(ep, val_acc))
        torch.save(model.state_dict(), model_file.format(ep, (simple_val_acc + normal_val_acc)/2, (complex_val_acc1 + complex_val_acc2) / 2, val_acc))
        torch.save(memory.state_dict(), mem_file.format(ep, (simple_val_acc + normal_val_acc)/2, (complex_val_acc1 + complex_val_acc2) / 2, val_acc))
