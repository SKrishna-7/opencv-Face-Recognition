
import cv2
import os

HarFile="haarcascade_frontalface_default.xml"
DataSet="DataSet"
Person=input("\nEnter Person Name : ")
SubDir=Person
print(os.path)
path=os.path.join(DataSet,SubDir)
print(path)
if not os.path.isdir(path):
    os.mkdir(path)
width,height=(130,100)

Classifier=cv2.CascadeClassifier(HarFile)
Capture=cv2.VideoCapture(2)
count=1
while count<31:
    print(count)
    _,Img=Capture.read()
    GrayImg=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    Faces=Classifier.detectMultiScale(GrayImg,1.3,4)
    for x,y,w,h in Faces:
        cv2.rectangle(Img,(x,y),(x+w,y+h),(255,0,0),2)
        Face=GrayImg[y:y+h ,x:x+w]
        Face_resize=cv2.resize(Face,(width,height))
        cv2.imwrite("%s/%s.png"%(path,count),Face_resize)
        count+=1
    cv2.imshow("DataSet",Img)
    key=cv2.waitKey(1)
    if key==ord("q") &0xFF:
        break
Capture.release()
cv2.destroyAllWindows()
