import cv2 
import numpy as np

#caminho da imagem
imagePath = './rock.jpg'

#link para a imagem utilizar - https://www.jbtc.com/foodtech/wp-content/uploads/sites/2/2021/08/Fruit-Vegetable-Processing.jpg

#lendo a imagem 
image = cv2.imread(imagePath)

#criando image imagem do mesmo tamanho para ser o 'brilho'
bright = np.zeros_like(image)
#adicionaremos um brilho de 100 em cada pixel
bright[:,:,:] = [0,75,0] # (b,g,r)
#some 100 em cada pixel da imagem
brilho = cv2.add(image, bright)
#caso a soma ultrapasse 255 (máximo), limite-o a 255 (branco)
brilho[brilho > 255] = 255

#mostre a imagem original
cv2.imshow('image', image)
#mostre a imagem de brilho
cv2.imshow('brilho', brilho)
#espere um comando
cv2.waitKey()
