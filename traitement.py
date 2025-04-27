from PIL import Image, ImageOps
import matplotlib.pyplot as plt

img = Image.open("../images/simba_surexp.png")
img.show()

img_niveaux_gris = ImageOps.grayscale(img)
img_niveaux_gris.show()

# application de autocontrast sur les deux images 
img1_contraste = ImageOps.autocontrast(img_niveaux_gris);
img1_contraste.show()

# afficher histogramme avant et apres operation contraste
hist = img_niveaux_gris.histogram()
plt.figure(figsize=(10,5))
plt.bar(range(256),hist,color='black')
plt.xlabel("le niveaux de gris")
plt.ylabel("nombre pixel ")
plt.title("histogramme avant contraste")
plt.show()

# afficher histogramme apres operation contraste
hist1 = img1_contraste.histogram()
plt.figure(figsize=(10,5))
plt.bar(range(256),hist1,color='black')
plt.xlabel("le niveaux de gris")
plt.ylabel("nombre pixel ")
plt.title("histogramme apres contraste")
plt.show()

# egalisation de image 
img_egaliser = ImageOps.equalize(img_niveaux_gris)
hist_eg = img_egaliser.histogram()
plt.figure(figsize=(10,5))
plt.bar(range(256),hist_eg,color='black')
plt.xlabel("le niveaux de gris")
plt.ylabel("nombre pixel ")
plt.title("histogramme apres egalisation")
plt.show()

# Explorer les methodes invert de module ImageOps
img_invert = ImageOps.invert(img_niveaux_gris)
img_invert.show()

# Afficher Histogramme apres Exploration invert.
hist_mihist_invert = img_invert.histogram()
plt.figure(figsize=(20,5))
plt.bar(range(256), hist_invert, color='black')
plt.xlabel("le niveaux de gris")
plt.ylabel("nombre pixel ")
plt.title("histogramme apres invert")
plt.show()  

# Explorer les methodes mirror de module ImageOps
img_mirror = ImageOps.mirror(img_niveaux_gris)
img_mirror.show()

# Afficher Histogramme apres Exploration mirror.
hist_mi = img_mirror.histogram()
plt.figure(figsize=(20,5))
plt.bar(range(256), hist_mi, color='black')
plt.xlabel("le niveaux de gris")
plt.ylabel("nombre pixel ")
plt.title("histogramme apres mirror")
plt.show()  
