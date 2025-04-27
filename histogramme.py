from PIL import Image
import matplotlib.pyplot as plt

# lire une image 
img = Image.open("../images/Lenna.png")
img.show()

# convertir image aux niveaux de gris 
img_niveaux_gris = img.convert('L')
img_niveaux_gris.show()

# Calculer l'histogramme
hist = img_niveaux_gris.histogram()

# Afficher Histogramme de image en niveaux de gris 
plt.figure(figsize=(10, 5))
plt.bar(range(256), hist, color='black')
plt.xlabel("Niveaux de gris")
plt.ylabel("Nombre de pixels")
plt.title("Histogramme des niveaux de gris")

# séparer les trois couches 
r, g, b = img.split()
r.show()
g.show()
b.show()

# Calculer les histogrammes
hist_r = r.histogram()
hist_g = g.histogram()
hist_b = b.histogram()

# Afficher les histogrammes sur le même graphique
plt.figure(figsize=(10, 5))
plt.bar(range(256),hist_r, color='red', alpha=0.3, label='R')
plt.bar(range(256),hist_g, color='green', alpha=0.3, label='G')
plt.bar(range(256),hist_b, color='blue', alpha=0.3, label='B')
plt.xlabel("Valeur des pixels")
plt.ylabel("Frequence")
plt.title("Histogramme des canaux R, G et B")
plt.legend()
plt.show()