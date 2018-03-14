import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import features

def storage_information(file_name, descriptors, image_name):
  file = open(f"./db/{file_name}.txt", 'a')
  file.write(image_name + '=')
  file.write(str(descriptors.tolist()))
  file.write('\n')
  file.close()

# cargar imagenes
images = os.listdir('./siluetas')

for image in images:
  img = cv2.imread(f"./siluetas/{image}", cv2.IMREAD_GRAYSCALE)
  # umbralización e invertir la imagen
  retval, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
  # obtener descriptores de fourier
  descriptors = features.find_descriptors(th)
  fourier_filt = descriptors[0:21]
  # Obtener momentos Hu invariantes
  huMoments = features.find_huMoments(th)
  # Obtener nombre de la imagen
  image_name, image_extension, *rest = image.split('.')
  # Almacenar momento HU
  storage_information("huMoments_DB", huMoments, image_name)
  # Almacenar descriptores de fourier
  storage_information("fourierDescriptors_DB", fourier_filt, image_name)

  
