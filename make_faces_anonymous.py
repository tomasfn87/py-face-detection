import cv2
import numpy as np

# caminho da imagem
imagePath = './smiling_guy.jpg'
# leia a imagem
image = cv2.imread(imagePath)

cascadePath = './haarcascade_frontalface_default.xml'

# vamos utilizar novamente o classificador provido pelo opencv
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# a detecção de face é feita em imagem em escala de cinza
# vamos convert a imagem original
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# vamos utilizar o método detectMultiScale com 
faces = faceCascade.detectMultiScale(
    gray, 
    minSize = (40, 40),
    flags = cv2.CASCADE_SCALE_IMAGE
)

# imprimindo quantas faces foram encontradas
sufixo_plural = ""
if not len(faces):
    print("No face was found!")
else:
    if len(faces) > 1:
        sufixo_plural = "s"
    print("Found {} face{}!".format(len(faces), sufixo_plural))

# Desenhando o retangulo ao redor de cada face encontrada
# (255,150,150) é a cor do retangulo (azul)
# 3 é a grossura da linha
# w é a largura do quadrado da face
# h é a altura  "     "     "   "
# x é o ponto inicial no eixo horizontal
# y é o ponto inicial no eixo vertical

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 150, 150), 3)

imageEmbacado = image.copy()
for (x, y, w, h) in faces:

    ROI = image[y:y+h, x:x+w, :]
    
    # crie um filtro (no caso esse embaça a imagem) e aplique
    # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel = np.ones((5, 5), np.float32) / 25

    embacado = cv2.filter2D(ROI, -1, kernel)
    for i in range(1, 500):
        embacado = cv2.filter2D(embacado, -1, kernel)
    
    # LIMITE O VALOR PARA 255
    embacado[embacado > 255] = 255

    imageEmbacado[y:y+h, x:x+w] = embacado
    
cv2.imshow('imageEmbacado', imageEmbacado)
cv2.waitKey()
