# 🚦 Traffic Sign Recognition based on CLIP

👉 **[Results](https://j4vixq.github.io/Traffic-Sign-Recognition-CLIP/)**

## Motivation
Based on CLIP, build a model that can recognize signs from unseen countries using given prompts — aiming for better performance than the original CLIP (domain adaptation).

## Data Sources
- https://nlpr.ia.ac.cn/pal/trafficdata/recognition.html  
- https://en.wikipedia.org/wiki/Road_signs_in_China  
- https://www.kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes  
- https://www.kaggle.com/datasets/dataclusterlabs/indian-sign-board-image-dataset/  
- https://www.kaggle.com/datasets/spareaccount2/indian-road-traffic-signs-dataset-classified  
- https://benchmark.ini.rub.de/gtsrb_dataset.html  
- https://synset.de/datasets/synset-signset-ger/  
- https://en.wikipedia.org/wiki/Road_signs_in_Germany  

## Image Augmentation
- **Random Augmentation**: brightness, contrast, noise, blur, zoom, rotation  
- **AIGC**: use ChatGPT to create traffic signs in diverse backgrounds  
- **Manual Work**: labeling, cropping, resizing  

## Prompts

Multiple prompts are crafted for each category:
  - **Visual traits** (e.g., "a triangular traffic sign with a yellow background")
  - **Symbolic content** (e.g., "the center shows a black cross")
  - **Common labels** (e.g., "a cross road sign")

Prompt quality **directly impacts performance**
Prompts act as a way to **"teach" the model** the meaning of each class

## Finetuning

...

## Steps
1. **Zero-Shot**:  
   Use original CLIP with images and prompts (no training)

2. **Finetune with 1 set → Test on 2 sets**:  
   Finetune CLIP on one country's dataset and prompts, test on the other two

3. **Finetune with 2 sets → Test on 1 set**:  
   Finetune CLIP on two countries, test on the third

4. **Finetune with 3 sets**
   If the previous steps show increasing accuracy on the test domain, then this finetuning method can improve the generalization ability.

**Expected Result**:  
Finetuning should improve cross-domain generalization.  

**Current Progress**
Step 2 results are already better than zero-shot baseline (Step 2 > Step 1).
Step 3 results are already better than finetune with 1 country results (Step 3 > Step 2).