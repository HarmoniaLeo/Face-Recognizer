from controller import controller
import os
import random
import time

times=10
acc=0
cos=0
ctr=controller()
model="dist"
for i in range(times):
    size=40
    names=os.listdir("lfw_funneled")
    names=random.choices(names,k=size)
    trains=[]
    tests=[]
    for name in names:
        if not os.path.isdir("lfw_funneled/"+name):
            names.remove(name)
    for name in names:
        faces=os.listdir("lfw_funneled/"+name)
        train=random.choice(faces)
        test=train
        while(test==train):
            test=random.choice(faces)
        trains.append("lfw_funneled/"+name+"/"+train)
        tests.append("lfw_funneled/"+name+"/"+test)
    ctr.reset()
    correct=0
    TP=0
    FP=0
    FN=0
    size=len(trains)
    for i in range(size):
        img,vec=ctr.getLocalImg(trains[i])
        for v in vec:
            ctr.addImg(v,names[i])
    ctr.trainModel(model)
    ticks = time.time()
    for i in range(size):
        img,vec=ctr.getLocalImg(tests[i])
        label,p=ctr.predict(vec,model)
        #print(label,names[i])
        if(label[0]==names[i]):
            correct+=1
            TP+=1
        
    ticks = time.time()-ticks
    print("Accuracy: {0:.3f}".format(correct/size))
    acc+=correct/size
    print("Average time: {0:.3f}s".format(ticks/size))
    cos+=ticks/size
print("\nAccuracy: {0:.3f}".format(acc/times))
print("Average time: {0:.3f}".format(cos/times))