

import numpy as np
import cv2
import os

# On va créer le chemin d'accès au dataset
train_path = 'C:/Users/GK/OneDrive/Bureau/dataset/train/'
categories = os.listdir(train_path)

# Nombre de catégories dans les données d'entraînement
nbr_categories = len(categories)
# print(nbr_categories)

image_paths = []
image_classes = []
class_id = 0  

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]  # Récupère la liste des images

for categorie in categories:
    dir = os.path.join(train_path, categorie)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)  
    class_id += 1
print("Nombre total d'images :", len(image_paths))
print("Nombre de catégories :", nbr_categories)

# On passe pour extraction des features de en utilisant algorithm shift

# from skimage.feature import SIFT
from PIL import Image
des_list = []
descriptor_extractor = cv2.SIFT_create()
for image_path in image_paths :
    im = np.array(Image.open(image_path).convert('L').resize((128,128)))
    kpts,des = descriptor_extractor.detectAndCompute(im,None)
   # kpts = descriptor_extractor.keypoints
   # des = descriptor_extractor.descriptors
    des_list.append((image_path,des))
    
# Empiler tous les descripteurs verticalement dans un tableau numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Création du Bag-Of-Features (BOF)  
# kmeans ne fonctionne que sur les types float
descriptors_float = descriptors.astype(float)
# affectuer Kmeans pour les clustering k-means et la quantification vectorielle 
from scipy.cluster.vq import kmeans , vq
k = 200 #k-means avec 200 clusters
codebook, variance = kmeans(descriptors_float, k, 1) 
# Calculer l'histogramme des caractéristiques et les représenter sous forme de vecteur
#vq Attribue des codes du codebook à des observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], codebook)
    for w in words:
        im_features[i][w] += 1
# Normaliser les features en supprimant la moyenne et en mettant à l'échelle la variance unitaire
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Application de random Forest sur un dataset ecommerce : 
