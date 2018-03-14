import cv2
import numpy as np

def find_descriptors(img):
  image, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  array_contours = contours[0][:, 0, :]
  complex_contours = np.empty(array_contours.shape[:-1], dtype=complex)
  complex_contours.real = array_contours[:, 0]
  complex_contours.imag = array_contours[:, 1]
  fourier_descriptors = np.fft.fft(complex_contours)
  return fourier_descriptors

def find_huMoments(img):
  huMoments = cv2.HuMoments(cv2.moments(img))
  return huMoments.flatten()
