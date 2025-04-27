from PIL import Image,ImageFilter
import numpy as np
import random

img = Image.open("../images/Lenna.png").convert('L')

# Ajouter le bruits gaussienne a une image.
def add_gauss_noise(img, variance):
    img = np.array(img)
    noise = np.random.normal(0, variance, img.shape)
    noisy_img = Image.fromarray(img + noise).convert('L')
    return noisy_img

# Ajouter le Filter implusionnel
def add_imp_noise(img,density):
    row,col=img.size
    number_of_pixels = int(density*100)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img.putpixel((y_coord,x_coord), 255)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img.putpixel((y_coord,x_coord), 0)
    return img

# utiliser les deux images pour generer des images bruitees
img_bruit_gauss = add_gauss_noise(img, 50)
img_bruit_gauss.show()

img_bruit_impl = add_imp_noise(img, 10)
img_bruit_impl.show()

# en utilisant les methode MedianFilter et GaussienFilter de classe ImageFilter.
img_medianFiler = img_bruit_gauss.filter(ImageFilter.MedianFilter(size=3))
img_medianFiler.show()

img_gaussFilter = img_bruit_gauss.filter(ImageFilter.GaussianBlur(radius=0.2))
img_gaussFilter.show()

""" cette partie concernant le deuxiemme activite"""

#générer un filtre gaussien 
Im_flitree = img_bruit_gauss.filter(ImageFilter.GaussianBlur(radius = 0.3)).show()

#générer le noyau d’un filtre moyen
image_filtree = img_bruit_gauss.filter(ImageFilter.Kernel((3, 3), 
								(1,1,1,1,1,1,1,1,1)))
image_filtree.show()
