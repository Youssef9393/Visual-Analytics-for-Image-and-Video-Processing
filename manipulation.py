from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image
img = Image.open("C:/Users/GK/OneDrive/Bureau/WISD/S2/VisualAnalytics/images/lenna.png")

# Affiche Format Image
print("Format: ",img.format,", size: ",img.size," mode: ",img.mode)
# Afficher le fichier de image 
print("Fichier: ",img.filename)
# Afficher les dimentions de images 
print("Dimention: H=",img.height,"L=",img.width)
# Afficher Image 
img.show()
# calculer et Afficher le nombre de pixel de Images
print("le nombre de pixel est : ",img.height*img.width)
# resize() est une fonction (méthode) de la classe Image qui permet de modifier la taille d'une image et d'en obtenir une nouvelle
# im2 est un objet de la classe Image
img2 = img.resize((228,192))
img2.save("C:/Users/GK/OneDrive/Bureau/WISD/S2/VisualAnalytics/images/lennaM2.png")
print(img2.format, img2.size, img2.mode)
print("Nombre de pixels : ", "L =", img2.width, "x H =", img2.height, "=", (img2.width*img2.height) , "pixels")
img2.show()
# size est un tuple (les tuples sont des séquences qu'on ne pourra plus modifier) 
size = (144,96) 
#= les parenthèses ne sont pas obligatoires 
width, height = size
print(size, width, height)
im3 = img.resize(size)
im3.save("C:/Users/GK/OneDrive/Bureau/WISD/S2/VisualAnalytics/images/lennaM3.png")
print(im3.format, im3.size, im3.mode)
print("Nombre de pixels : ", "L =", im3.width, "x H =", im3.height, "=", (im3.width*im3.height) , "pixels")
im3.show()

#
# Convertir en niveaux de gris
# plt.imshow(img)
# plt.axis("off")  # Cacher les axes
# plt.show()
#
