from Pic2NpArray import ResizeAndConToNumpyArray as tna
import Image
import random




class PicSet:
    def __init__(self):
        self.MainTraingData = []
        self.RandomData = []      
        
        ####


    def add(self,PicDir,index=0):
        data = Image.open(PicDir)
        NArray = tna(data,28,28)
        self.MainTraingData.append([NArray,index])
        ####
    def show(self):
        print self.MainTraingData[0]

    def random(self,index=-1):
        self.RandomData = self.MainTraingData
        random.shuffle(self.RandomData)
        if index == -1:
            return self.RandomData
        else:
            return self.RandomData[index]
        
MP = PicSet()
MP.add('TraingData/0/11.jpg',0)
MP.add('TraingData/0/6.jpg',1)
MP.add('TraingData/0/21.jpg',2)
print MP.random(0)[0]




