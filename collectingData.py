import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

rec = cv2.VideoCapture(0)
detect = FaceDetector()

# O'zgaruvchan qiymatlar
###################################################################
save = True
BlurThreashold = 35 # Kattaroq qiymat fokus uchun kerak
confidenceValue  = 0.8  # Agar rasm yoki videoda insonni 80% yuqori  darajada aniqlamasa yuzni qidirish
#                         funksiyasi ishlamaydi.Faqat inson ko'rinsa yuzni aniqlaydi
offsetPercW = 10 # boshlangich boksni kengligi
offsetPercH = 20 # boshlangich boksni balandligi

classid = 1   # 0 = soxta rasmlar uchun ; 1 = haqiqiy uchun

debug = False

###########################################################
direct = "Datasets/Real"

# ramni hajmi  uchun qiymatlar
recH,recW  = 640,  480
floatinPoint = 6

while True:
    success,image  = rec.read()
    imageshow = image.copy()
    image,bbox = detect.findFaces(image,draw=False)

    BlurList = [] #rasmlarni xira yoki tiniqlarini ajratadi
    InfoList  = [] # normallashtirilgan qiymatlar va datasetni class nomlari yoziladi

    #  Agar biz data uchun kopgina yuzlardan foydalansak, "if bbox : " qismidan tashqarida
    #  list yaratib olamiz .Bolmasa, agar yuz bir marta aniqlab
    #  yana shu yuz xira ko'ringanda bu yuzni aniqlamaydi va oldingi rasmni
    #  parametrlarini saqlab qoyadi

    if bbox:
        # tortburchakni kattalashtiramiz yuzni toliq korsatishi uchun
        for box in bbox:
            x,y,w,h = box['bbox']
            acc_score = box['score'][0]

            # Insonning Aniqlik darajasi tekshirish
            if acc_score > confidenceValue:
                # Rasmga chegara chizish
                offsetKeng = (offsetPercW/100)*w  # yuzni aniqlovchi 4 burchakni kengligi
                x = int(x-offsetKeng)
                w = int(w+offsetKeng*2)

                offsetBal = (offsetPercH / 100) * h # yuzni aniqlovchi 4 burchakni balandligi
                y = int(y - offsetBal*3)
                h = int(w + offsetBal * 3.5)

                # Qiymatlar 0 dan kichiklashib ketmasligi uchun

                if x < 0: x = 0
                if y < 0: y = 0
                if h < 0: h = 0
                if w < 0: w = 0

               # Xiralikni yo'qotish
                faceplace = image[y:y+h, x:x+w]
                cv2.imshow("Face",faceplace)
                BlurVal = int(cv2.Laplacian(faceplace,cv2.CV_64F).var())
                if BlurVal>BlurThreashold:
                    BlurList.append(True)
                else:
                    BlurList.append(False)

                # normallashtirish
                h_im , w_im , _ = image.shape
                xc,yc = x+w/2,y+h/2
                print(xc,yc)
                xcn,ycn = round(x/w_im,floatinPoint),round(y/h_im,floatinPoint)
                wn, hn = round(w / w_im, floatinPoint), round(h / h_im, floatinPoint)

                # Qiymatlar 1 dan kattalashib ketmasligi uchun

                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if hn > 1: hn = 1
                if wn > 1: wn = 1

                InfoList.append(f"{classid} {xcn} {ycn} {wn} {hn}\n")

                # Rasmga to'rtburchak chizish va matn yozish
                cv2.rectangle(imageshow,(x,y,w,h),(0,0,255),3)
                cvzone.putTextRect(imageshow,f"Score {int(acc_score*100)}% blur {BlurVal}",(x,y-20),scale=2,thickness=3)

                if debug:
                    cv2.rectangle(image, (x, y, w, h), (0, 0, 255), 3)
                    cvzone.putTextRect(image, f"Score {int(acc_score * 100)}% blur {BlurVal}", (x, y - 20), scale=2,
                                       thickness=3)

        # Rasmlarni saqlash uchun
        if save:
            if  all(BlurList) and BlurList!=[]:
                #  vaqt orqali rasmni saqlab qoyamiz
                timeN = time()
                timeN = str(timeN).split(".")
                timeN = timeN[0]+timeN[1]
                print(timeN)
                cv2.imwrite(f"{direct}/{timeN}.jpg",image)
                # rasmlarni nomi bilan saqlaymiz
                for info in InfoList:
                    file = open(f"{direct}/{timeN}.txt", "a")
                    file.write(info)
                    file.close()

    cv2.imshow("webcam_done",imageshow)
    cv2.waitKey(1)
