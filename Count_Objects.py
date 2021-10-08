# Cargamos la imagen
original = cv2.imread("Segmentacion.jpg")
cv2.imshow("original", original)
kernel = np.ones((5,5),np.uint8) 

# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("grises", gris)
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
cv2.imshow("suavizado", gauss)
 
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 0, 255)
 
cv2.imshow("canny", canny)

#------------------
erosion = cv2.dilate(canny,kernel,iterations = 1)
cv2.imshow("erosion", erosion)
 
# Buscamos los contornos
(contornos,_) = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Mostramos el número de monedas por consola
print("\n He encontrado {} objetos\n".format(len(contornos)))
 
contornos = cv2.drawContours(original,contornos,-1,(0,0,255), 2)
cv2.imshow("contornos", contornos)
 
cv2.waitKey(0)
