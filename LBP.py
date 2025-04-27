import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.util import view_as_blocks
from PIL import Image

# Charger l'image Lenna en niveaux de gris
img = np.array(Image.open("../images/Lenna.png").convert('L'))

# Paramètres LBP
P = 8  # Nombre de voisins
R = 1  # Rayon

# Calcul du LBP
descriptor_lbp = local_binary_pattern(img, P, R, method="uniform")

# Affichage de l'image LBP
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(descriptor_lbp, cmap='gray')
plt.title("Image LBP")
plt.axis("off")

# Affichage de l'histogramme
plt.subplot(1,2,2)
plt.hist(descriptor_lbp.ravel(), bins=256, range=(0, 256), color='gray')
plt.title("Histogramme du LBP")
plt.show()

# ============================
# Segmentation en blocs (16x16)
# ============================
block_size = (16, 16)

# Ajuster la taille de l'image pour être multiple de 16
height, width = img.shape
new_height = height - (height % 16)
new_width = width - (width % 16)
img_cropped = img[:new_height, :new_width]

# Diviser l'image en blocs de 16x16
blocks = view_as_blocks(img_cropped, block_size)
num_blocks_x, num_blocks_y = blocks.shape[:2]

# Extraction des descripteurs LBP pour chaque bloc
lbp_descriptors = []
for i in range(num_blocks_x):
    for j in range(num_blocks_y):
        block = blocks[i, j]
        lbp = local_binary_pattern(block, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
        lbp_descriptors.append(hist)

# Concaténation de tous les descripteurs
final_descriptor = np.concatenate(lbp_descriptors)

# Affichage de la taille finale du descripteur
print(f"Taille du descripteur final après concaténation : {final_descriptor.shape[0]}")
