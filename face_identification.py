#importing libraries
import cv2
import face_recognition as face_rec
import numpy


#function
def resize(img,size):
    width=int(img.shape[1]*size)
    height=int(img.shape[0]*size)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)

#image declaration
simran=face_rec.load_image_file('C:\Program Files\Python38\code\sample_images\simran.jpg')
simran=cv2.cvtColor(simran,cv2.COLOR_BGR2RGB)
simran=resize(simran,0.50)
taruna=face_rec.load_image_file(r"C:\Program Files\Python38\code\sample_images\taruna.jpg")
taruna=cv2.cvtColor(taruna,cv2.COLOR_BGR2RGB)
taruna=resize(taruna,0.50)
sanvi=face_rec.load_image_file(r"C:\Program Files\Python38\code\sample_images\sanvi.jpg")
sanvi=cv2.cvtColor(sanvi,cv2.COLOR_BGR2RGB)
sanvi=resize(sanvi,0.50)
eshwari=face_rec.load_image_file(r"C:\Program Files\Python38\code\sample_images\eshwari.jpg")
eshwari=cv2.cvtColor(eshwari,cv2.COLOR_BGR2RGB)
eshwari=resize(eshwari,0.50)
prakriti=face_rec.load_image_file(r"C:\Program Files\Python38\code\sample_images\prakriti.jpg")
prakriti=cv2.cvtColor(prakriti,cv2.COLOR_BGR2RGB)
prakriti=resize(prakriti,0.50)



#finding face location
facelocation_simran= face_rec.face_locations(simran)[0]
encode_simran=face_rec.face_encodings(simran)[0]
cv2.rectangle(simran,(facelocation_simran[3],facelocation_simran[0]),(facelocation_simran[1],facelocation_simran[2]),(255,0,0),3)
facelocation_taruna= face_rec.face_locations(taruna)[0]
encode_taruna=face_rec.face_encodings(taruna)[0]
cv2.rectangle(taruna,(facelocation_taruna[3],facelocation_taruna[0]),(facelocation_taruna[1],facelocation_taruna[2]),(255,0,0),3)
facelocation_sanvi= face_rec.face_locations(sanvi)[0]
encode_sanvi=face_rec.face_encodings(sanvi)[0]
cv2.rectangle(sanvi,(facelocation_sanvi[3],facelocation_sanvi[0]),(facelocation_sanvi[1],facelocation_sanvi[2]),(255,0,0),3)
facelocation_eshwari= face_rec.face_locations(eshwari)[0]
encode_eshwari=face_rec.face_encodings(eshwari)[0]
cv2.rectangle(eshwari,(facelocation_eshwari[3],facelocation_eshwari[0]),(facelocation_eshwari[1],facelocation_eshwari[2]),(255,0,0),3)
facelocation_prakriti= face_rec.face_locations(prakriti)[0]
encode_prakriti=face_rec.face_encodings(prakriti)[0]
cv2.rectangle(prakriti,(facelocation_prakriti[3],facelocation_prakriti[0]),(facelocation_prakriti[1],facelocation_prakriti[2]),(255,0,0),3)

#print(encode_simran)
#print(encode_taruna)
#print(encode_sanvi)


result=face_rec.compare_faces([encode_taruna],encode_prakriti)
print(result)
cv2.putText(taruna,f'{result}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('main_img2',taruna)
cv2.imshow('main_img5',prakriti)
cv2.waitKey(0)
cv2.destroyAllWindows()
