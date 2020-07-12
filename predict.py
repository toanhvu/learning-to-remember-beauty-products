import os
import sys
import numpy as np
import torch
from util import load_image_list_from_textfile, write_text
from transform import SimpleTransform, SimpleTransformEdge, load_image
from loader import get_image_loader, get_beauty_loader
from models import MultiScaleDense121, MultiScaleDense121Edge
from memory import Memory
from config import FEAT_DIM

fol_data = "data"

# load image_files, and labels
print("Load images and their labels")
image_files, labels = load_image_list_from_textfile(os.path.join(fol_data, "images.txt"), None)
print("done!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL and MEM 1
print("Load model and memory 1 (MultiScaleDense121)...")
model1 = MultiScaleDense121(feat_dim=FEAT_DIM, pretrained=False, frozen=False)  
model1 = model1.to(device)
memory1 = Memory(mem_size=len(labels), feat_dim=FEAT_DIM, margin=1, topk=100)
memory1 = memory1.to(device)
model1.load_state_dict(torch.load(os.path.join("models", model1.__class__.__name__, "model.pkl"), map_location=device))
memory1.load_state_dict(torch.load(os.path.join("models", model1.__class__.__name__, "memory.pkl"), map_location=device))
model1.eval()
memory1.eval()
transform1 = SimpleTransformEdge() if 'Edge' in model1.__class__.__name__ else SimpleTransform()
print("done!")

# MODEL and MEM 2
print("Load model and memory 2 (MultiScaleDense121Edge)...")
model2 = MultiScaleDense121Edge(feat_dim=FEAT_DIM, pretrained=False, frozen=False)  
model2 = model2.to(device)
memory2 = Memory(mem_size=len(labels), feat_dim=FEAT_DIM, margin=1, topk=100)
memory2 = memory2.to(device)
model2.load_state_dict(torch.load(os.path.join("models", model2.__class__.__name__, "model.pkl"), map_location=device))
memory2.load_state_dict(torch.load(os.path.join("models", model2.__class__.__name__, "memory.pkl"), map_location=device))
model2.eval()
memory2.eval()
transform2 = SimpleTransformEdge() if 'Edge' in model2.__class__.__name__ else SimpleTransform()
print("done!")

def search_image(model, memory, transform, image_path, topk=7):
    model.eval()
    memory.eval()
    img = load_image(image_path)
    img, _ = transform(img)
    img = img.to(device)
    feat, _ = model(img) 
    distances, indices = memory.search_l2(feat, topk)
    distances, indices = distances.data.squeeze().cpu().numpy(), indices.data.squeeze().cpu().numpy()
    return distances, indices

def find_element(d, l):
    return np.where(l==d)[0]

def combine_search_image(model1, memory1, transform1, \
                        model2, memory2, transform2, \
                        image_path, topk=7):
    distances1, indices1 = search_image(model1, memory1, transform1, image_path, topk)
    distances2, indices2 = search_image(model2, memory2, transform2, image_path, topk)
    distances = np.concatenate([distances1, distances2], 0)
    distances.sort()
    final_indices = []
    final_distances = []
    for d in distances:
        tmp_indices = [indices1[i] for i in find_element(d, distances1)] + [indices2[i] for i in find_element(d, distances2)]
        for a in tmp_indices:
            if a not in final_indices:
                final_indices.append(a)
                final_distances.append(d)
    return final_distances[0:topk], final_indices[0:topk]


model_name = sys.argv[1]    # MultiScaleDense121, MultiScaleDense121Edge, Combine
test_fol = sys.argv[2]      # /testset
save_file = sys.argv[3]     # /result/predictions.csv

write_text(save_file, "Validation Image ID,Training Image ID\n", mode='w')
# image_list = [os.path.join(test_fol, f) for f in os.listdir(test_fol) if f.endswith(('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'))]
image_list = [os.path.join(test_fol, f) for f in os.listdir(test_fol)]
image_list.sort()

if model_name in ["MultiScaleDense121", "model1"]:
    print("*** Generate predictions by MultiScaleDense121\n\n")
elif model_name in ["MultiScaleDense121Edge", "model2"]:
    print("*** Generate predictions by MultiScaleDense121Edge\n\n")
else:
    print("*** Generate predictions by the combination of (MultiScaleDense121 + MultiScaleDense121Edge)\n\n")

for tmp_image in image_list:
    print(tmp_image)
    try:
        if model_name in ["MultiScaleDense121", "model1"]:            
            tmp_distances, tmp_indices = search_image(model1, memory1, transform1, tmp_image, topk=7)
        elif model_name in ["MultiScaleDense121Edge", "model2"]:            
            tmp_distances, tmp_indices = search_image(model2, memory2, transform2, tmp_image, topk=7)
        else:            
            tmp_distances, tmp_indices = combine_search_image(model1, memory1, transform1, \
                                                model2, memory2, transform2, tmp_image, topk=7)
    except:
        continue
    s = os.path.split(tmp_image)[-1] 
    for tmp_index in tmp_indices:
        s += ',' + image_files[tmp_index]
    s += '\n'
    write_text(save_file, s, mode='a')
