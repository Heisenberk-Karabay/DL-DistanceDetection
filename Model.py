# decied to use Inception v4 model. Because it is trained by google and probably its the best model for human detection
# either gonna use pretrained model or train the model with https://www.kaggle.com/constantinwerner/human-detection-dataset dataset.

import timm
import torch
import cv2
import numpy as np
import torchvision
import urllib


test_img = cv2.imread('human.jpg')

#dimension_correction = np.expand_dims(test_img, 1)

down_width = 299
down_height = 299
down_points = (down_width,down_height)
resized = cv2.resize(test_img, down_points, interpolation= cv2.INTER_LINEAR)

resized = resized.transpose(2,0,1)


tensor_img = torch.from_numpy(resized)
tensor_img = tensor_img.float()

model = timm.create_model('inception_v4', pretrained=True)
model.eval()



with torch.no_grad():
    out = model(tensor_img.unsqueeze(0))
probabilities = torch.nn.functional.softmax(out[0], dim=0)
#print(probabilities)


# pretrained olarak imagenet datasetini ve classlarini kullaniyor kendi datasetimle modeli tekrar egitmek mecburi


url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())