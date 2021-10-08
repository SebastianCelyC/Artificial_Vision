#importar libreria OpenCv
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#3. Leer imagenes
print("*****************************")
print("***   Visión Artificial   ***")
print("**   Parcial Primer Corte  **")
print("*****************************")

img1 = cv2.imread("bubbles.PNG")
img2 = cv2.imread("celulas.tiff")
img3 = cv2.imread("xray-chest.png")

chimg1 = cv2.resize(img1, dsize=(230, 230))
chimg2 = cv2.resize(img2, dsize=(230, 230))
chimg3 = cv2.resize(img3, dsize=(230, 230))

#4. Determinar tamaño de las imágenes -------------------------------------------------------------------------------
print("\n Tamaño de las imágenes originales (en pixeles)\n")
print(" Burbújas")
print(img1.shape[0:2])
print("\n Células")
print(img2.shape[0:2])
print("\n Rayos X")
print(img3.shape[0:2])
print("\n Intensidad de las imágenes\n")
print("¿ Desea ver los histogramas de intensidad de las imágenes ?\n")
itn =  int(input("1. Si \n" + "2. No \n" ))

if itn == 1:
    color = ('b', 'g', 'r')
    for i,col in enumerate(color):
        hist_Xray = cv2.calcHist([img1], [i],None,[256],[0,255])
        hist_bub = cv2.calcHist([img2], [i],None,[256],[0,255])
        hist_cell = cv2.calcHist([img3], [i],None,[256],[0,255])
        plt.subplot(1,3,1), plt.plot(hist_Xray,color = col)
        plt.xlabel('intensidad de iluminacion'), plt.ylabel('cantidad de pixeles'), plt.xlim([0,255])
        plt.subplot(1,3,2), plt.plot(hist_bub,color = col)
        plt.xlabel('intensidad de iluminacion'), plt.ylabel('cantidad de pixeles'), plt.xlim([0, 255])
        plt.subplot(1,3,3), plt.plot(hist_cell,color = col)
        plt.xlabel('intensidad de iluminacion'), plt.ylabel('cantidad de pixeles'), plt.xlim([0, 255])
        plt.show()
#5. Cambiar tamaño de las imágenes -------------------------------------------------------------------------
print("\n Se modificó el tamaño de las imágenes \n")
print(" Burbújas")
print(    chimg1.shape[0:2])
print("\n Células")
print(    chimg2.shape[0:2])
print("\n Rayos X")
print(    chimg3.shape[0:2])

print("\n¿ Desea ver las imágenes modificadas ?")
opc =  int(input("1. Si \n" + "2. No \n" ))

if opc == 1:

    plt.subplot(142), plt.imshow(chimg1), plt.title('Burbujas'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(chimg2), plt.title('Celulas'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(chimg3), plt.title('Rayos X'), plt.xticks([]), plt.yticks([])
    plt.show()

#6. gamma ------------------------------------------------------------------------------------------
print("\nModificacion del parametro Gamma\n")

#Factor gamma < 1
# Factor gamma < 1
contrast0_img1 = cv2.addWeighted(chimg1, 0.5, np.zeros(chimg1.shape, chimg1.dtype), 0, 0)
contrast0_img2 = cv2.addWeighted(chimg2, 0.5, np.zeros(chimg2.shape, chimg2.dtype), 0, 0)
contrast0_img3 = cv2.addWeighted(chimg3, 0.5, np.zeros(chimg3.shape, chimg3.dtype), 0, 0)

#Factor gamma = 1
contrast1_img1 = cv2.addWeighted(chimg1, 1, np.zeros(chimg1.shape, chimg1.dtype), 0, 0)
contrast1_img2 = cv2.addWeighted(chimg2, 1, np.zeros(chimg2.shape, chimg2.dtype), 0, 0)
contrast1_img3 = cv2.addWeighted(chimg3, 1, np.zeros(chimg3.shape, chimg3.dtype), 0, 0)

#Factor gamma > 1
contrast2_img1 = cv2.addWeighted(chimg1, 1.5, np.zeros(chimg1.shape, chimg1.dtype), 0, 0)
contrast2_img2 = cv2.addWeighted(chimg2, 1.5, np.zeros(chimg2.shape, chimg2.dtype), 0, 0)
contrast2_img3 = cv2.addWeighted(chimg3, 1.5, np.zeros(chimg3.shape, chimg3.dtype), 0, 0)

print("\n¿ Desea ver la modificacion del parametro gamma ?")
gamma =  int(input("1. Si \n" + "2. No \n" ))

if gamma == 1:
    plt.subplot(141), plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(contrast0_img1), plt.title('Gamma < 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(contrast1_img1), plt.title('Gamma = 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(contrast2_img1), plt.title('Gamma > 1'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(141), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(contrast0_img2), plt.title('Gamma < 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(contrast1_img2), plt.title('Gamma = 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(contrast2_img2), plt.title('Gamma > 1'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(141), plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(contrast0_img3), plt.title('Gamma < 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(contrast1_img3), plt.title('Gamma = 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(contrast2_img3), plt.title('Gamma > 1'), plt.xticks([]), plt.yticks([])
    plt.show()

# 6. Histograma  ------------------------------------------------------------------------------------------
print("Histograma de las imagenes\n")

print("¿ Desea ver los histogramas ?")
his = int(input("1. Si \n" + "2. No \n" ))

hist1 = cv2.calcHist([chimg1], [0], None, [256], [0, 256]) / 91000
hist2 = cv2.calcHist([chimg2], [0], None, [256], [0, 256]) / 49138
hist3 = cv2.calcHist([chimg3], [0], None, [256], [0, 256]) / 13666

if his == 1:
    hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    histn = hist / 91000
    plt.plot(hist, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Histograma Burbujas')
    plt.show()

    plt.plot(histn, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Histograma Burbujas normalizado')
    plt.show()


    hist1 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist1n = hist1 / 50000
    plt.plot(hist1, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Células')
    plt.show()

    plt.plot(hist1n, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Histograma Células normalizado')
    plt.show()

    hist2 = cv2.calcHist([img3], [0], None, [256], [0, 256])
    hist2n = hist2 / 14000
    plt.plot(hist2, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Rayos X')
    plt.show()

    plt.plot(hist2n, color='gray' ) 
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title('Histograma Rayos X normalizado')
    plt.show()

#8 - MP ------------------------------------------------------------------------------------------
print("Variaciones de intensidad\n")
# Lectura de la imagen - Burbujas
image = cv2.imread('bubbles.PNG')

# Se realizan matrices de intensidad y se multiplican por constantes
MI = np.ones(img2.shape, dtype="uint8") * 60
MI1 = np.ones(img2.shape, dtype="uint8") * 100
print(MI)

# Se suma o se resta la matriz de intensidad a las imágenes
image1 = cv2.add(img2, MI)
image2 = cv2.subtract(img2, MI)
image3 = cv2.add(img2, MI1)
image4 = cv2.subtract(img2, MI1)

# Se realizan matrices de intensidad y se multiplican por constantes
MI0 = np.ones(img3.shape, dtype="uint8") * 30
MI10 = np.ones(img3.shape, dtype="uint8") * 60
print(MI0)

# Se suma o se resta la matriz de intensidad a las imágenes
image10 = cv2.add(img3, MI0)
image20 = cv2.subtract(img3, MI0)
image30 = cv2.add(img3, MI10)
image40 = cv2.subtract(img3, MI10)

# Se realizan matrices de intensidad y se multiplican por constantes
MI2 = np.ones(img1.shape, dtype="uint8") * 50
MI12 = np.ones(img1.shape, dtype="uint8") * 100
print(MI2)

# Se suma o se resta la matriz de intensidad a las imágenes
image12 = cv2.add(img1, MI2)
image22 = cv2.subtract(img1, MI2)
image32 = cv2.add(img1, MI12)
image42 = cv2.subtract(img1, MI12)


print("¿ Desea ver las variaciones de intensidad ?")

vari = int(input("1. Si \n" + "2. No \n" ))
if vari == 1:
    #Se muestra la imagen original y las 3 variaciones de intensidad
    plt.subplot(141), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image1), plt.title('Variación 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(image3), plt.title('Variación 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(image4), plt.title('Variación 3'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(141), plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image10), plt.title('Variación 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(image20), plt.title('Variación 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(image30), plt.title('Variación 3'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(141), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image12), plt.title('Variación 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(image22), plt.title('Variación 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(image42), plt.title('Variación 3'), plt.xticks([]), plt.yticks([])
    plt.show()


#9 - MP - Filtros ------------------------------------------------------------------------------------------
#Filtro pasabajos
print("Aplicación de filtros a las imágenes originales")

# Lectura de la imagen
#img1 = bubbles, img2 = celulas, img3 = xray

#Opcion 1 - Filtro pasabajos
#pasabajos Burbujas ------------------------------
# Creación de un kernel 3x3
kernel = np.ones((3,3), np.float32)/9
# Aplicación de convolución entre la imagen y el kernel 3x3
lpf1b = cv2.filter2D(img1, -1, kernel)
# Opción 2 - Filtro pasabajos
lpf2b = cv.blur(img1, (3, 3))
#FIn burbujas ---------------------------

#pasabajos celulas ----------------------
lpf1c = cv2.filter2D(img2, -1, kernel)
lpf2c = cv.blur(img2, (3, 3))
#fin celulas ----------------------------

#pasabajos X-Ray ------------------------
lpf1x = cv2.filter2D(img3, -1, kernel)
lpf2x = cv.blur(img3, (3, 3))
#fin x-ray ------------------------------


print("¿ Desea ver la aplicación del filtro pasabajos ?")

filtro = int(input("1. Si \n" + "2. No \n" ))
if filtro == 1:
# Imagen original, imagen procesada con kernel e imagen procesada con comando blur
    #Burbujas
    plt.subplot(131), plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(lpf1b), plt.title('FPB 1 - Kernel'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(lpf2b), plt.title('FPB 2 -  Comando "blur"'), plt.xticks([]), plt.yticks([])
    plt.show()
    #celulas
    plt.subplot(131), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(lpf1c), plt.title('FPB 1 - Kernel'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(lpf2c), plt.title('FPB 2 -  Comando "blur"'), plt.xticks([]), plt.yticks([])
    plt.show()
    #x-ray
    plt.subplot(131), plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(lpf1x), plt.title('FPB 1 - Kernel'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(lpf2x), plt.title('FPB 2 -  Comando "blur"'), plt.xticks([]), plt.yticks([])
    plt.show()

#FILTRO PASAALTOS
# Filtro pasa altos - deteccion de bordes
# Kernel de 3x3
kernelPA = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
# Division del kernel entre la suma de los datos de la matriz
kernelPA = kernelPA/(np.sum(kernelPA) if np.sum(kernelPA)!=0 else 1)
# Aplicación de la convolución entre iamgen original y el kernel
image_pb = cv2.filter2D(img1,-1,kernelPA)
image_pc = cv2.filter2D(img2,-1,kernelPA)
image_px = cv2.filter2D(img3,-1,kernelPA)
# Imagen original e imagen convolucionada

print("¿ Desea ver la aplicación del filtro pasa altos ?")

pa = int(input("1. Si \n" + "2. No \n" ))
if pa == 1:
    plt.subplot(121), plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_pb), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(121),plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(image_pc), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(121),plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(image_px), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.show()

#FILTRO HIGH BOOST

# Filtro High-Boost - Bubbles
# Filtro high-boost - original - filtro pasa bajos
# Resta de imagen original a imagen filtrada con filtro pasa bajos

kernelhb = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
# Division del kernel entre la suma de los datos de la matriz
kernelhb = kernelhb/(np.sum(kernelhb) if np.sum(kernelhb)!=0 else 1)

#Burbujas
lpf2hbb = cv2.blur(img1, (3, 3))
HBb = (img1 * 2) - lpf2hbb
image_phb = cv2.filter2D(img1,-1,kernelhb)

#Celulas
lpf2hbc = cv2.blur(img2, (3, 3))
HBc = (img2 * 2) - lpf2hbc
image_phbc = cv2.filter2D(img2,-1,kernel)

#X-Ray
lpf2hbx = cv2.blur(image, (3, 3))
HBx = (image * 2) - lpf2hbx
image_phbx = cv2.filter2D(img3,-1,kernel)

print("¿ Desea ver la aplicación del filtro high boost ?")

hb = int(input("1. Si \n" + "2. No \n" ))
if hb == 1:
    plt.subplot(131), plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_phb), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(HBb), plt.title('FHB'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(131), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])   
    plt.subplot(132), plt.imshow(image_phbc), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(HBc), plt.title('FHB'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(131), plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_phbx), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(HBx), plt.title('FPA'), plt.xticks([]), plt.yticks([])
    plt.show()

#Filtro laplaciano -----------------------------------------------------

# Filtro Laplaciano - Burbujas2
# Lectura de la imagen
laplacianb = cv2.Laplacian(img1,cv2.CV_64F)
laplacianc = cv2.Laplacian(img2,cv2.CV_64F)
laplacianx = cv2.Laplacian(img3,cv2.CV_64F)

print("¿ Desea ver la aplicación del filtro Laplaciano ?")
lpc = int(input("1. Si \n" + "2. No \n" ))
if lpc == 1:
    plt.subplot(321), plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(laplacianb), plt.title('FL'), plt.xticks([]), plt.yticks([])
    plt.subplot(323), plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(324), plt.imshow(laplacianc), plt.title('FL'), plt.xticks([]), plt.yticks([])
    plt.subplot(325), plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(326), plt.imshow(laplacianx), plt.title('FL'), plt.xticks([]), plt.yticks([])
    plt.show()

#10 MP -----------------------------------------------------------------------------------------

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')
# plt.show()
blur = cv2.GaussianBlur(gray, (7, 7), 250)
# plt.imshow(blur, cmap='gray')
# plt.show()
canny = cv2.Canny(blur, 100, 200, 3)
# plt.imshow(canny, cmap='gray')
# plt.show()
dilated = cv2.dilate(canny, (1, 1), iterations=2)
# plt.imshow(dilated, cmap='gray')
# plt.show()
kernelf = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
# Division del kernel entre la suma de los datos de la matriz
kernelf = kernelf/(np.sum(kernelf) if np.sum(kernelf)!=0 else 1)
opening = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernelf)
# plt.imshow(opening, cmap='gray')
# plt.show()
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
print("¿ Desea ver el resultado de la segmentación ? ")
r = int(input("1. Si \n" + "2. No \n" ))
if r == 1:
    plt.imshow(rgb)
    plt.show()
    print('Celulas in the image: ', len(cnt))
#FIN 10 ----------------------------------------------------------------------------------------
