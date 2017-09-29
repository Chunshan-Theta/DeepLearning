from Pic2NpArray import ResizeAndConToNumpyArray as tna
import Image
import random




class PicSet:
    def __init__(self):
        self.MainTraingData = []                                            #PIC Set
        self.RandomData = []                                                #PIC Set with Random
        self.PushIndex = 0                                                  #record of output  
        ####


    def add(self,PicDir,ans=-1):                                            #add pic's data to PIC Set
        data = Image.open(PicDir)                                           #read a pic from dir
        NArray = tna(data,28,28)                                            #resize pic
        self.MainTraingData.append([NArray,ans])                            #add data of pic and Ans to Set
        ####
    def show(self):
        print self.MainTraingData                                           #print data of PIC Set                                       

    def random(self,index=-1):                                              #upset the index of the set
        self.RandomData = self.MainTraingData                               #copy a set with MainTraingData
        random.shuffle(self.RandomData)                                     #upset the set
        if index == -1:
            return self.RandomData
        else:
            return self.RandomData[index]

    def batch(self,Num):
        if self.RandomData == []:                                           #initial Random array
            MP.random()
        q_set=[]
        a_set=[]
        if self.PushIndex+Num>len(self.RandomData):                         #confirm the number of pics is enough
            print "ERROR : Set only have "+str(len(self.RandomData))+" pic"
            return
        for i in self.RandomData[self.PushIndex:self.PushIndex+Num]:
            q_set.append(i[0])
            a_set.append(i[1])
        self.PushIndex += Num
        return q_set,a_set
MP = PicSet()
MP.add('TraingData/0/1.jpg',1)
MP.add('TraingData/0/6.jpg',2)
MP.add('TraingData/0/11.jpg',3)
MP.add('TraingData/0/16.jpg',4)
MP.add('TraingData/0/21.jpg',5)
MP.add('TraingData/0/26.jpg',6)
a1,a2 = MP.batch(3)
print a1[0]
a1,a2 = MP.batch(4)
print a1[0]



