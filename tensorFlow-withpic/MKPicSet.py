from Pic2NpArray import ResizeAndConToNumpyArray as tna
import Image
import random
import os



class PicSet:
    def __init__(self):
        self.MainTraingData = []                                            #PIC Set
        self.RandomData = []                                                #PIC Set with Random
        self.PushIndex = 0                                                  #record of output  
        ####


    def add(self,PicDir,ans=-1,x_size=28,y_size=28):                        #add pic's data to PIC Set
        data = Image.open(PicDir)                                           #read a pic from dir
        NArray = tna(data,x_size,y_size)                                    #resize pic
        self.MainTraingData.append([NArray,ans])                            #add data of pic and Ans to Set
        ####
    def AddDir(self,picdir= "./TraingData/",x_size=28,y_size=28):
        for dirPath, dirNames, fileNames in os.walk(picdir):
            for f in fileNames:                        
                ans=[0,0,0,0,0,0,0,0,0,0]
                ans[int(dirPath[len(picdir):])]+=1
                self.add(str(os.path.join(dirPath, f)),ans,x_size,y_size)
                print "add pic:",os.path.join(dirPath, f),"The Pic's Ans:",ans
        ####
    def show(self):
        print self.MainTraingData                                           #print data of PIC Set                                       
        ####

    def random(self,index=-1):                                              #upset the index of the set
        self.RandomData = self.MainTraingData                               #copy a set with MainTraingData
        random.shuffle(self.RandomData)                                     #upset the set
        if index == -1:
            return self.RandomData
        else:
            return self.RandomData[index]
        ####


    def batch(self,Num):
        if self.RandomData == []:                                           #if haven't upset pic set, initial Random array
            self.random()
        q_set=[]                                                            #data of pic
        a_set=[]                                                            #Answer of pic
        if self.PushIndex+Num>len(self.RandomData):                         #confirm the number of pics is enough
            print "ERROR : Pic Set only have "+str(len(self.RandomData))+" pics"
            return 
        for i in self.RandomData[self.PushIndex:self.PushIndex+Num]:
            q_set.append(i[0])
            a_set.append(i[1])
        self.PushIndex += Num
        return q_set,a_set
        ####
    def reset(self):
        self.RandomData = []                                                #PIC Set with Random
        self.PushIndex = 0                                                  #record of output  
        
        ####
'''
MP = PicSet()
MP.add('TraingData/0/1.jpg',1)
MP.add('TraingData/0/6.jpg',2)
MP.add('TraingData/0/11.jpg',3)
MP.add('TraingData/0/16.jpg',4)
MP.add('TraingData/0/21.jpg',5)
MP.add('TraingData/0/26.jpg',6)
MP.add('TraingData/0/31.jpg',7)
MP.add('TraingData/0/36.jpg',8)
MP.add('TraingData/0/41.jpg',9)
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2
####


MP = PicSet()
MP.AddDir('./TraingData/')
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2
a1,a2 = MP.batch(3)
print a2

'''

