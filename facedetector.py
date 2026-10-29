import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


ac = cv2.VideoCapture(0)

while True:
    c_squ,d_img = ac.read()
    e = cv2.cvtColor(d_img,cv2.COLOR_BGR2GRAY)
    f = face.detectMultiScale(e,1.3,5)

    for (x1,y1,w1,h1) in f:
        cv2.rectangle(d_img,(x1,y1),(x1+w1,y1+h1),(128,0,128),10 )#color and width



    cv2.imshow('img',d_img)
    h = cv2.waitKey(1) & 0xFF
    if h == 27:
        break



ac.release()
cv2.destroyAllWindows()
