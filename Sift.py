import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import rotate, rescale
# from skimage.feature  import match_descriptors,draw_matches

# Chargement de l'image originale
img_originale = Image.open('../images/Lenna.png')

# Conversion en niveaux de gris
img = np.array(img_originale.convert('L'))

# Création de l'objet SIFT
sift = cv2.SIFT_create()

# Détection des points clés et calcul des descripteurs
keypoints, descriptors = sift.detectAndCompute(img, None)

# Dessin des points clés sur l'image
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Affichage du nombre de points clés détectés
print("Nombre de points clés (original) :", len(keypoints))

# Affichage de l'image avec les points clés
plt.imshow(img_sift, cmap='gray')
plt.title("Points clés détectés avec SIFT")
plt.show()

# Dimensions de l'image
(h, w) = img.shape[:2]

# Application des transformations : rotation 45 degrés, translation et mise à l'échelle
img_rotate = rotate(img, angle=45, resize=True)

# Création de la matrice de transformation pour la translation
matrice_trans = np.float32([[1, 0, 20], [0, 1, 30]])
img_translate = cv2.warpAffine(img_rotate, matrice_trans, (img_rotate.shape[1], img_rotate.shape[0]))

# Mise à l'échelle (zoom x1.5)
img_transform = rescale(img_translate, scale=1.5, anti_aliasing=True)
img_transform = (img_transform * 255).astype(np.uint8)

# Détection des points clés après transformation
keypoints2, descriptors2 = sift.detectAndCompute(img_transform, None)

# Affichage du nombre de points clés détectés après transformation
print("Nombre de points clés (après transformation) :", len(keypoints2))

# Dessin des points clés après transformation
img_sift1 = cv2.drawKeypoints(img_transform, keypoints2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Affichage de l'image transformée avec les points clés
plt.imshow(img_sift1, cmap='gray')
plt.title("Points clés détectés après transformations")
plt.show()

# dans la premier fois on detecter 238 point interet
# 1-2 apres les transformation sur image originale on detecter 473 point 
# oui , il ya une changement dans le nombre des point interet detecter et aussi dans certains position les point sont differents.

# 1.3 Correspondance des points clés entre deux images :
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors, descriptors2)

# Tri des correspondances en fonction de la distance
matches = sorted(matches, key=lambda x: x.distance)

# Dessin des correspondances
img_matches = cv2.drawMatches(img, keypoints, img_transform, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Affichage des correspondances
plt.figure(figsize=(12, 6))
plt.imshow(img_matches, cmap='gray')
plt.title("Correspondance des points clés entre l'image originale et transformée")
plt.show()
