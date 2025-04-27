import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from skimage import exposure
from PIL import Image 
import matplotlib.pyplot as plt

# ouvrir image et convertir en niveaux de gris 
img = Image.open('../images/lenna.png')

# generer la matrice de image
img = np.array(img)

# redimentionner les images 
img_resize = resize(img,(128*4,64*4))

# application de calcul descripteur hog
fd , img_hog = hog(img_resize, orientations = 9 , pixels_per_cell = (8,8) ,cells_per_block=(2,2), visualize=True,channel_axis = 2)

# Amélioration de la visibilité du HOG
hog_image = exposure.rescale_intensity(img_hog, out_range=(0, 255))
# print(fd.shape)

# Affichage
plt.imshow(hog_image, cmap='gray')
plt.title("Image des descripteurs HOG")
plt.show()

# 2.2 Analyse des parametres de hog
# 1- fd : taille de descripteur des images obtenus. / img_hog est image obtenus apres calcul descripteur hog
# 2- impact parametres channel_axis representer les couleurs de image soit RGB ou niveaux de gris ,mais dans cette cas on peut ignorer.
# 3- nombre de directions ou orientation utiliser pour quantifier les gradient de chaque cellule
# 4- -- --
# 5- tester .