import cv2
import numpy as np
import math, cmath
import features

IMAGES_DB = 323
HU_MOMENTS = 7
FOURIER_DESCRIPTORS = 21

class Image:
  def __init__(self, image_name, distance):
    self.image_name = image_name
    self.distance = distance
  
  def __repr__(self):
    return f"({self.image_name}, {self.distance})"



def calculate_euclidean_distance(huMoments_db, inHuMoments):
  result_sum = 0
  for i in range(HU_MOMENTS):
    result_sum += ((huMoments_db[i] - inHuMoments[i])**2)

  return math.sqrt(result_sum)


def compare_huMoments(huMoments):
  distances = []
  file = open('./db/huMoments_DB.txt', 'r')

  for i in range(IMAGES_DB):
    line = file.readline()
    # Obtener nombre de la imagen y vector de huMoments desde el archivo de texto
    image_name, huMoments_db, *rest = line.split('=')
    # Transformarlo a numpy array
    huMoments_db = huMoments_db[1:]
    huMoments_db = huMoments_db[:-2]
    huMoments_db = np.fromstring(huMoments_db, sep=',')
    ed = calculate_euclidean_distance(huMoments_db, huMoments)
    e = Image(image_name, ed)
    distances.append(e)
  
  distances_sorted = sorted(distances, key=lambda eu: eu.distance)
  candidate_image = distances_sorted[0]
  if(candidate_image.distance >= 0.0 and candidate_image.distance <= 0.01):
    return f"La imagen es similar a: {candidate_image.image_name}"
  else:
    return "La imagen no es lo suficiente similar a alguno de los objetos"
  
  file.close()



def calculate_distance_fourier(descriptors_db, inDescriptors):
  # Calcular razón normalizada
  sum_db = 0
  sum_in = 0

  for i in range(FOURIER_DESCRIPTORS):
    sum_db += (descriptors_db[i]**2)
    sum_in += (inDescriptors[i]**2)

  for j in range(FOURIER_DESCRIPTORS):
    descriptors_db[j] /= cmath.sqrt(sum_db)
    inDescriptors[j] /= cmath.sqrt(sum_in)

  # Calcular distancia
  sum_vectors = 0

  for x in range(FOURIER_DESCRIPTORS):
    sum_vectors += (descriptors_db[x] * inDescriptors[x])

  distance = cmath.acos(cmath.sqrt(sum_vectors))
  return distance



def compare_fourierDescriptors(descriptors):
  distances = []
  descriptors_filt = descriptors[0:21]
  file = open('./db/fourierDescriptors_DB.txt', 'r')

  for i in range(IMAGES_DB):
    line = file.readline()
    image_name, descriptors_db, *rest = line.split('=')
    descriptors_db = descriptors_db[1:]
    descriptors_db = descriptors_db[:-2]
    descriptors_db = descriptors_db.replace('(', '')
    descriptors_db = descriptors_db.replace(')', '')
    array_complex = descriptors_db.split(',')
    descriptors_db = []
    for num in array_complex:
      descriptors_db.append(complex(num))
    
    descriptors_db = np.array(descriptors_db)
    if(descriptors_db.size == FOURIER_DESCRIPTORS):
      distance = calculate_distance_fourier(descriptors_db, descriptors_filt)
      fourier = Image(image_name, distance)
      distances.append(fourier)

  distances_sorted = sorted(distances, key=lambda f: f.distance.real)
  candidate_image = distances_sorted[0]
  if(candidate_image.distance.real >= 0 and  candidate_image.distance.real <= 0.0001):
    return f"La imagen es similar a: {candidate_image.image_name}"
  else:
    return "La imagen no es lo suficiente similar a alguno de los objetos"

  file.close()



image = input('Introduzca una imagen: ')
# Binarizar la imagen
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# Umbralización de la imagen
retval, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Encontrar descriptores
descriptors = features.find_descriptors(th)
huMoments = features.find_huMoments(th)

res1 = compare_huMoments(huMoments)
print('HuMoments')
print(res1)

res2 = compare_fourierDescriptors(descriptors)
print('\nFourier')
print(res2)
