import os
import random
import shutil
from itertools import islice

outPutFpath = "Datasets/SplitNewData"
inputFpath = "Datasets/CollectedData"
cls = ["fake" ,"real"]
try :
    shutil.rmtree(outPutFpath)
    print("Directory removed")
except OSError as e :
    os.mkdir(outPutFpath)

###### Ma'lumotlarni taqsimlash
os.makedirs(f"{outPutFpath}/train/images",exist_ok=True)
os.makedirs(f"{outPutFpath}/train/labels",exist_ok=True)
os.makedirs(f"{outPutFpath}/val/images",exist_ok=True)
os.makedirs(f"{outPutFpath}/val/labels",exist_ok=True)
os.makedirs(f"{outPutFpath}/test/images",exist_ok=True)
os.makedirs(f"{outPutFpath}/test/labels",exist_ok=True)

## ----Listni nomini olish---
getListName = os.listdir(inputFpath)
UniqueValues = []

ratio = {"train" : 0.7 , "val" : 0.2 , "test":0.1}
for values in getListName:
    UniqueValues.append(values.split('.')[0])
UniqueValues = list(set(UniqueValues))

### Rasmlarni aralashtirish

random.shuffle(UniqueValues)

### Har bir fayldagi rasmlarni soni aniqlash

LenData = len(UniqueValues)

LenTrain =int(LenData*ratio['train'])
Lenval =int(LenData*ratio['val'])
LenTest =int(LenData*ratio['test'])

## Rasmlarni sonini taqsimlash, ratiodan ortib qolgan rasmlar "Train" qo'shib yuboriladi

lenRatio = LenTrain+Lenval+LenTest
if LenData!=lenRatio:
    left = LenData - lenRatio
    LenTrain += left

# Rasmlarni soniga asosan ajratish

LenghtSplit = [LenTrain,Lenval,LenTest]
Input = iter(UniqueValues)
Output = [list(islice(Input,i)) for i in LenghtSplit]
print(f"Total data: {len(Output)};\nTrain/Validation/Test: {len(Output[0])},{len(Output[1])},{len(Output[2])}")

# ma'lumotlarni kochirib train/test/val fayllariga joylash

filenames = ['train',"val",'test']
for s,out in enumerate(Output):
    for fname in out:
        shutil.copy(f"{inputFpath}/{fname}.jpg", f'{outPutFpath}/{filenames[s]}/images/{fname}.jpg')
        shutil.copy(f"{inputFpath}/{fname}.txt", f'{outPutFpath}/{filenames[s]}/labels/{fname}.txt')
print("Data has been Splitted!")

## dataYaml fayl yaratish bunda fayl joylashgan o'rini yozib qo'yiladi
datayaml = f'path : ../data\n\
train : ../train/images\n\
val : ../val/images\n\
test : ../test/images \n\
\n\
number_of_classes : {len(cls)}\n\
names : {cls}'


file = open(f"{outPutFpath}/data.yaml", "a")
file.write(datayaml)
file.close()