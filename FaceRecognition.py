
import cv2
import numpy
import os

HaarFile="haarcascade_frontalface_default.xml"
DataSet="DataSet"
Classifier=cv2.CascadeClassifier(HaarFile)
print("Training.....")

(images,labels,names,id)=([],[],{},0)

for (SubDir,Dirs,Files) in os.walk(DataSet):
    for SubDir in Dirs:
        names[id]=SubDir
        SubjectPath=os.path.join(DataSet,SubDir)
        for filename in os.listdir(SubjectPath):
            path=SubjectPath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
           # print(names[id])
        id+=1

images,labels=[numpy.array(lis) for lis in [images,labels]]
#print(images,labels)

(width,height)=(130,100)
Model=cv2.face.LBPHFaceRecognizer_create()
#Model=cv2.face.FisherFaceRecognizer_create()

Model.train(images,labels)

Capture=cv2.VideoCapture(2)
cnt=0

while True:
    _,Img=Capture.read()
    Gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    Faces=Classifier.detectMultiScale(Gray,1.3,5)
    for(x,y,w,h) in Faces:
        cv2.rectangle(Img,(x,y),(x+w,y+h),(0,255,0),2)
        face=Gray[y:y+h,x:x+w]
        faceResize=cv2.resize(face,(width,height))

        Prediction=Model.predict(faceResize)
        cv2.rectangle(Img,(x,y),(x+w,y+h),(0,0,255),2)
        if Prediction[1]<800:
            cv2.putText(Img,"%s-%.0f"%(names[Prediction[0]],Prediction[1]),(x-10,y-10),1,cv2.FONT_HERSHEY_PLAIN,(0,0,255),4)
            print(names[Prediction[0]])
        else:
            cnt+=1
            cv2.putText(Img,"Unknown",(x-10,y-10),cv2.FONT_HERSHEY_DUPLEX,(0,0,255),2)
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg",Img)
                cnt=0
    cv2.imshow("FaceRecognizer",Img)
    key=cv2.waitKey(1)
    if key==ord("q") &0xFF:
        break
Capture.release()
cv2.destroyAllWindows()
     
