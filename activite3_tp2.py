from PIL import Image,ImageFilter
import numpy as np
import random

"""
les filters pass haut Sobel et Previtt sont utiliser pour detection des contours 

"""
img = Image.open("../images/Lenna.png").convert('L')


def seuillage(img,seuil ) :
    pixel = img.load()   # acces direct aux intensite d'image.
    col , row = img.size
    for i in range(col):
         for j in range(row):
             if pixel[i,j] >= seuil:
                 pixel[i,j]=255
             else:
                 pixel[i,j]=0
    return img

img_seuil = seuillage(img, 130).show()

# utiliser les fonctions ImageFilter.Kernel detecter les conteurs par sobel
sobel_gv = img.filter(ImageFilter.Kernel((3,3), (-1,0,1,-2,0,2,-1,0,1),1,0))
sobel_gv.show()
sobel_gh = img.filter(ImageFilter.Kernel((3,3), (-1,-1,-1,0,0,0,1,1,1),1,0))
sobel_gh.show()
addition_s = np.asarray(sobel_gv) +  np.asarray(sobel_gh)
img_sobel = Image.fromarray(addition_s)
img_sobel.show()

# utiliser les fonctions ImageFilter.Kernel detecter conteur par prewitt
prewitt_gh = img.filter(ImageFilter.Kernel((3,3), (-1,0,1,-1,0,1,-1,0,1),1,0))
prewitt_gh.show()
prewitt_gv = img.filter(ImageFilter.Kernel((3,3), (-1,-1,-1,0,0,0,1,1,1),1,0))
prewitt_gv.show()
addition_p = np.asarray(prewitt_gv) +  np.asarray(prewitt_gh)
img_prewitt = Image.fromarray(addition_p)
img_prewitt.show() 

 
