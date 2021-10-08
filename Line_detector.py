import numpy as np 
import cv2
#CREAR  OBJETO PARA LA ADQUISICION DE IMAGENES
video = cv2.VideoCapture('lane_detection_video.mp4') 
while   True:
    #METODO READ PARA LA LECTURA DE LAS IMAGENES DADOS POR EL OBJETO LLAMADO video
    ret,frame1 = video.read()
    #RGB-GRISES 
    frame = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    #RECOMENDABLE REALIZAR UN PROCESO DE FILTRADO 
    #CONVERSION DE DATO DE TIPO ENTERO 8bit EN FLOTANTES
    frame_float = frame.astype(float)
    #KERNEL DE SOBEL
    kernel = np.ones((9,9),np.uint8) 
    Hsx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Hsy = np.transpose(Hsx)
    #BORDES  EN  LAS  DIRECCIONES  HORIZONTALES  Y  VERTICALES
    bordex = cv2.filter2D(frame_float,-1,Hsx)
    bordey = cv2.filter2D(frame_float,-1,Hsy)
     #CALCULO DE LA MAGNITUD DEL GRADIENTE 
    Mxy = bordex**2+bordey**2 #OPERACION PIXEL POR PIXEL
    Mxy = np.sqrt(Mxy)
    #NORMALIZACION
    Mxy = Mxy/np.max(Mxy)
    #SEGMENTACION
    mask = np.where(Mxy>0.1,255,0)
    mask = np.uint8(mask)
    gauss = cv2.GaussianBlur(mask, (9,9), 0)
    canny = cv2.Canny(gauss, 200, 255)
    erosion = cv2.dilate(canny,kernel,iterations = 1)

    (contornos,_) = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = cv2.drawContours(mask,contornos,-1,(0,0,255), 2)
    cv2.imshow('BORDES',contornos)
    

    k = cv2.waitKey(1)&0xFF
    if(k == ord('p')):
        print('ACABO EL PROGRAMA')
        break
video.release()
cv2.destroyAllWindows()  
