from PIL import Image, ImageOps
import matplotlib.pyplot as plt

img = Image.open("../images/lenna.png")
img.show()
# format, size et mode sont des données (attributs) de la classe Image
print(img.format, img.size, img.mode)
print("Fichier :",img.filename)
# Affichage des dimensions de l'image
print("L=",img.width, "x H =",img.height)
# Calcul et Affichage du nombre de pixels de l'image
print("Nombre de pixels =", (img.width*img.height) , "pixels")
# resize() est une fonction (méthode) de la classe Image qui permet de modifier la taille d'une image et d'en obtenir une nouvelle
# im2 est un objet de la classe Image
size = (228,192)
im2 = img.resize(size)
# save() est une fonction (méthode) de la classe Image qui sauvegarder l'image dans un fichier (ici, l'extension précisera le format de l'image)
path = "../images/lena_228x192.png"
im2.save(path)
# Afficher toutes les attributs precedant da im2
print(im2.format, im2.size, im2.mode)
print("Nombre de pixels =", "L =", im2.width, "x H =", im2.height, "=", (im2.width*im2.height) , "pixels")
im2.show()
# size est un tuple (les tuples sont des séquences qu'on ne pourra plus modifier) 
size = (144,96) 
#= les parenthèses ne sont pas obligatoires 
width, height = size
im3 = img.resize(size)
im3.save("../images/lena_144x96.png")
print(im3.format, im3.size, im3.mode)
print("Nombre de pixels =", "L =", im3.width, "x H =", im3.height, "=", (im3.width*im3.height) , "pixels")
im3.show()
