# Learning to Remember Beauty Products
by Toan Vu, An Dang, Jia-Ching Wang
## Descriptions
This is the implementation of our submissions at [AI Meets Beauty Challenge 2020](https://challenge2020.perfectcorp.com/)

We develop a deep learning model for the beauty product image retrieval problem. The proposed model has two main components- an encoder and a memory. The encoder extracts and aggregates features from a deep convolutional neural network at multiple scales to get feature embeddings. With the use of an attention mechanism and a data augmentation method, it learns to focus on foreground objects and neglect background on images, so can it extract more relevant features. The memory consists of representative states of all database images as its stacks, and it can be updated during training process. Based on the memory, we introduce a distance loss to regularize embedding vectors from the encoder to be more discriminative. Our model is fully end-to-end, requires no manual feature aggregation and post-processing.

<img src="./data/model.png" width="100%" alt="model">

## Usage
### Train
You can train a model for a given architecture as follows
```bash
python main.py --model=MultiScaleDense121 --batch-size=16 
```

### Predict
You can get retrieval results given an image folder
```bash
python predict.py model_name ./test_data ./result/predictions.csv
```

## Our results
<img src="./results/v000164.jpg" width="100%" alt="v000164.jpg">
<img src="./results/v000185.jpg" width="100%" alt="v000185.jpg">
<img src="./results/v000141.jpg" width="100%" alt="v000141.jpg">
<img src="./results/v000160.jpg" width="100%" alt="v000160.jpg">
<img src="./results/v000145.jpg" width="100%" alt="v000145.jpg">
<img src="./results/v000148.jpg" width="100%" alt="v000148.jpg">
<img src="./results/v000154.jpg" width="100%" alt="v000154.jpg">
<img src="./results/v000126.jpg" width="100%" alt="v000126.jpg">
<img src="./results/v000181.jpg" width="100%" alt="v000181.jpg">
<img src="./results/v000103.jpg" width="100%" alt="v000103.jpg">
<img src="./results/v000172.jpg" width="100%" alt="v000172.jpg">
<img src="./results/v000194.jpg" width="100%" alt="v000194.jpg">
<img src="./results/v000155.jpg" width="100%" alt="v000155.jpg">
<img src="./results/v000177.jpg" width="100%" alt="v000177.jpg">
<img src="./results/v000175.jpg" width="100%" alt="v000175.jpg">
<img src="./results/v000162.jpg" width="100%" alt="v000162.jpg">
<img src="./results/v000113.jpg" width="100%" alt="v000113.jpg">
<img src="./results/v000116.jpg" width="100%" alt="v000116.jpg">
<img src="./results/v000188.jpg" width="100%" alt="v000188.jpg">
<img src="./results/v000158.jpg" width="100%" alt="v000158.jpg">
<img src="./results/v000173.jpg" width="100%" alt="v000173.jpg">
<img src="./results/v000152.jpg" width="100%" alt="v000152.jpg">
<img src="./results/v000184.jpg" width="100%" alt="v000184.jpg">
<img src="./results/v000124.jpg" width="100%" alt="v000124.jpg">
<img src="./results/v000195.jpg" width="100%" alt="v000195.jpg">
<img src="./results/v000150.jpg" width="100%" alt="v000150.jpg">
<img src="./results/v000120.jpg" width="100%" alt="v000120.jpg">
<img src="./results/v000129.jpg" width="100%" alt="v000129.jpg">
<img src="./results/v000179.jpg" width="100%" alt="v000179.jpg">
<img src="./results/v000196.jpg" width="100%" alt="v000196.jpg">
<img src="./results/v000118.jpg" width="100%" alt="v000118.jpg">
<img src="./results/v000130.jpg" width="100%" alt="v000130.jpg">
<img src="./results/v000111.jpg" width="100%" alt="v000111.jpg">
<img src="./results/v000138.jpg" width="100%" alt="v000138.jpg">
<img src="./results/v000167.jpg" width="100%" alt="v000167.jpg">
<img src="./results/v000176.jpg" width="100%" alt="v000176.jpg">
<img src="./results/v000132.jpg" width="100%" alt="v000132.jpg">
<img src="./results/v000104.jpg" width="100%" alt="v000104.jpg">
<img src="./results/v000161.jpg" width="100%" alt="v000161.jpg">
<img src="./results/v000136.jpg" width="100%" alt="v000136.jpg">
<img src="./results/v000192.jpg" width="100%" alt="v000192.jpg">
<img src="./results/v000107.jpg" width="100%" alt="v000107.jpg">
<img src="./results/v000189.jpg" width="100%" alt="v000189.jpg">
<img src="./results/v000151.jpg" width="100%" alt="v000151.jpg">
<img src="./results/v000174.jpg" width="100%" alt="v000174.jpg">
<img src="./results/v000159.jpg" width="100%" alt="v000159.jpg">
<img src="./results/v000143.jpg" width="100%" alt="v000143.jpg">
<img src="./results/v000121.jpg" width="100%" alt="v000121.jpg">
<img src="./results/v000102.jpg" width="100%" alt="v000102.jpg">
<img src="./results/v000199.jpg" width="100%" alt="v000199.jpg">
<img src="./results/v000101.jpg" width="100%" alt="v000101.jpg">
<img src="./results/v000165.jpg" width="100%" alt="v000165.jpg">
<img src="./results/v000182.jpg" width="100%" alt="v000182.jpg">
<img src="./results/v000171.jpg" width="100%" alt="v000171.jpg">
<img src="./results/v000139.jpg" width="100%" alt="v000139.jpg">
<img src="./results/v000109.jpg" width="100%" alt="v000109.jpg">
<img src="./results/v000183.jpg" width="100%" alt="v000183.jpg">
<img src="./results/v000110.jpg" width="100%" alt="v000110.jpg">
<img src="./results/v000133.jpg" width="100%" alt="v000133.jpg">
<img src="./results/v000168.jpg" width="100%" alt="v000168.jpg">
<img src="./results/v000122.jpg" width="100%" alt="v000122.jpg">
<img src="./results/v000123.jpg" width="100%" alt="v000123.jpg">
<img src="./results/v000114.jpg" width="100%" alt="v000114.jpg">
<img src="./results/v000131.jpg" width="100%" alt="v000131.jpg">
<img src="./results/v000140.jpg" width="100%" alt="v000140.jpg">
<img src="./results/v000190.jpg" width="100%" alt="v000190.jpg">
<img src="./results/v000178.jpg" width="100%" alt="v000178.jpg">
<img src="./results/v000186.jpg" width="100%" alt="v000186.jpg">
<img src="./results/v000144.jpg" width="100%" alt="v000144.jpg">
<img src="./results/v000191.jpg" width="100%" alt="v000191.jpg">
<img src="./results/v000156.jpg" width="100%" alt="v000156.jpg">
<img src="./results/v000193.jpg" width="100%" alt="v000193.jpg">
<img src="./results/v000200.jpg" width="100%" alt="v000200.jpg">
<img src="./results/v000146.jpg" width="100%" alt="v000146.jpg">
<img src="./results/v000163.jpg" width="100%" alt="v000163.jpg">
<img src="./results/v000125.jpg" width="100%" alt="v000125.jpg">
<img src="./results/v000170.jpg" width="100%" alt="v000170.jpg">
<img src="./results/v000149.jpg" width="100%" alt="v000149.jpg">
<img src="./results/v000169.jpg" width="100%" alt="v000169.jpg">
<img src="./results/v000127.jpg" width="100%" alt="v000127.jpg">
<img src="./results/v000105.jpg" width="100%" alt="v000105.jpg">
<img src="./results/v000198.jpg" width="100%" alt="v000198.jpg">
<img src="./results/v000147.jpg" width="100%" alt="v000147.jpg">
<img src="./results/v000108.jpg" width="100%" alt="v000108.jpg">
<img src="./results/v000153.jpg" width="100%" alt="v000153.jpg">
<img src="./results/v000142.jpg" width="100%" alt="v000142.jpg">
<img src="./results/v000115.jpg" width="100%" alt="v000115.jpg">
<img src="./results/v000119.jpg" width="100%" alt="v000119.jpg">
<img src="./results/v000197.jpg" width="100%" alt="v000197.jpg">
<img src="./results/v000187.jpg" width="100%" alt="v000187.jpg">
<img src="./results/v000112.jpg" width="100%" alt="v000112.jpg">
<img src="./results/v000106.jpg" width="100%" alt="v000106.jpg">
<img src="./results/v000157.jpg" width="100%" alt="v000157.jpg">
<img src="./results/v000128.jpg" width="100%" alt="v000128.jpg">
<img src="./results/v000180.jpg" width="100%" alt="v000180.jpg">
<img src="./results/v000134.jpg" width="100%" alt="v000134.jpg">
<img src="./results/v000135.jpg" width="100%" alt="v000135.jpg">
<img src="./results/v000137.jpg" width="100%" alt="v000137.jpg">
<img src="./results/v000117.jpg" width="100%" alt="v000117.jpg">
<img src="./results/v000166.jpg" width="100%" alt="v000166.jpg">


## Contact
If your have any suggestion or questions, please send email to toanvuhong@gmail.com

## Citation
If you find this code useful for your work, please cite our paper
'''
@inproceedings{10.1145/3394171.3416281,
author = {Vu, Toan H. and Dang, An and Wang, Jia-Ching},
title = {Learning to Remember Beauty Products},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3416281},
doi = {10.1145/3394171.3416281},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {4728–4732},
numpages = {5},
location = {Seattle, WA, USA},
series = {MM '20}
}
'''
