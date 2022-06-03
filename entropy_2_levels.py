import torch.nn as nn
import torch

def entropy2lvl(outputs, labels):
    loss = nn.CrossEntropyLoss()
    coarse = []
    real_superclass = [None]*len(outputs)
    #print(real_superclass)
    func=max

    for i in range(len(outputs)):
        coarse.append([])
        ind=labels[i]
        if (ind in [72, 4, 95, 30, 55]):
          real_superclass[i]=0
        elif ind in [73, 32, 67, 91, 1]:
          real_superclass[i]=1
        elif ind in [92, 70, 82, 54, 62]:
          real_superclass[i]=2
        elif ind in [16, 61, 9, 10, 28]:
          real_superclass[i]=3
        elif ind in [51, 0, 53, 57, 83]:
          real_superclass[i]=4
        elif ind in [40, 39, 22, 87, 86]:
          real_superclass[i]=5
        elif ind in [20, 25, 94, 84, 5]:
          real_superclass[i]=6
        elif ind in [14, 24, 6, 7, 18]:
          real_superclass[i]=7
        elif ind in [43, 97, 42, 3, 88]:
          real_superclass[i]=8
        elif ind in [37, 17, 76, 12, 68]:
          real_superclass[i]=9
        elif ind in [49, 33, 71, 23, 60]:
          real_superclass[i]=10
        elif ind in [15, 21, 19, 31, 38]:
          real_superclass[i]=11
        elif ind in [75, 63, 64, 66, 34]:
          real_superclass[i]=12
        elif ind in [77, 26, 45, 99, 79]:
          real_superclass[i]=13
        elif ind in [11, 2, 35, 46, 98]:
          real_superclass[i]=14
        elif ind in [29, 93, 27, 78, 44]:
          real_superclass[i]=15
        elif ind in [65, 50, 74, 36, 80]:
          real_superclass[i]=16
        elif ind in [56, 52, 59, 47, 96]:
          real_superclass[i]=17
        elif ind in [8, 58, 90, 13, 48]:
          real_superclass[i]=18
        elif ind in [81, 69, 41, 89, 85]:
          real_superclass[i]=19
    #print(real_superclass)

    for i in range(len(outputs)):
        coarse[i].append(func([outputs[i][72], outputs[i][4], outputs[i][95], outputs[i][30], outputs[i][55]]))
        coarse[i].append(func([outputs[i][73], outputs[i][32], outputs[i][67], outputs[i][91], outputs[i][1]]))
        coarse[i].append(func([outputs[i][92], outputs[i][70], outputs[i][82], outputs[i][54], outputs[i][62]]))
        coarse[i].append(func([outputs[i][16], outputs[i][61], outputs[i][9], outputs[i][10], outputs[i][28]]))
        coarse[i].append(func([outputs[i][51], outputs[i][0], outputs[i][53], outputs[i][57], outputs[i][83]]))
        coarse[i].append(func([outputs[i][40], outputs[i][39], outputs[i][22], outputs[i][87], outputs[i][86]]))
        coarse[i].append(func([outputs[i][20], outputs[i][25], outputs[i][94], outputs[i][84], outputs[i][5]]))
        coarse[i].append(func([outputs[i][14], outputs[i][24], outputs[i][6], outputs[i][7], outputs[i][18]]))
        coarse[i].append(func([outputs[i][43], outputs[i][97], outputs[i][42], outputs[i][3], outputs[i][88]]))
        coarse[i].append(func([outputs[i][37], outputs[i][17], outputs[i][76], outputs[i][12], outputs[i][68]]))
        coarse[i].append(func([outputs[i][49], outputs[i][33], outputs[i][71], outputs[i][23], outputs[i][60]]))
        coarse[i].append(func([outputs[i][15], outputs[i][21], outputs[i][19], outputs[i][31], outputs[i][38]]))
        coarse[i].append(func([outputs[i][75], outputs[i][63], outputs[i][64], outputs[i][66], outputs[i][34]]))
        coarse[i].append(func([outputs[i][77], outputs[i][26], outputs[i][45], outputs[i][99], outputs[i][79]]))
        coarse[i].append(func([outputs[i][11], outputs[i][2], outputs[i][35], outputs[i][46], outputs[i][98]]))
        coarse[i].append(func([outputs[i][29], outputs[i][93], outputs[i][27], outputs[i][78], outputs[i][44]]))
        coarse[i].append(func([outputs[i][65], outputs[i][50], outputs[i][74], outputs[i][36], outputs[i][80]]))
        coarse[i].append(func([outputs[i][56], outputs[i][52], outputs[i][47], outputs[i][59], outputs[i][96]]))
        coarse[i].append(func([outputs[i][8], outputs[i][58], outputs[i][90], outputs[i][13], outputs[i][48]]))
        coarse[i].append(func([outputs[i][81], outputs[i][69], outputs[i][41], outputs[i][89], outputs[i][85]]))
        
    #print('outputs: ', outputs[0])
    #print('outputs of superclasses', coarse[0])
    #print('real superclass:', real_superclass)
    l1=loss(outputs, labels)
    l2=loss(torch.tensor(coarse), torch.tensor(real_superclass))
    #print("class loss =", l1, "; superclass loss =", l2)
    return 0.3*l1+0.7*l2


def modifiedEntropy2lvl(outputs, labels):
    loss = nn.CrossEntropyLoss()
    
    coarse = []
    outputs=outputs.softmax(dim=1)
    for i in range(len(labels)):
        coarse.append([])
        for j in range(20):
            coarse[i].append(sum(outputs[i][j*5:(j+1)*5]))
        #coarse[i]=torch.tensor(coarse[i]).softmax(dim=0)
    coarse=torch.tensor(coarse).softmax(dim=1)
    #print("coarse with softmax:", coarse)
        
    
    real_superclass = torch.tensor([labels[i]//5 for i in range(len(labels))])
    
    #for i in range(10):
        #print(i, ":")
        #print("label:", labels[i], "; real_superclass:", real_superclass[i])
        #print("outputs:", outputs[i], "; coarse:", coarse[i])
        #print()
    
    l1=loss(outputs, labels)
    #l2=loss(torch.cat(coarse, 0), real_superclass)
    l2=loss(coarse, real_superclass)
    #print("class loss =", l1, "; superclass loss =", l2)
    
    #c=input()
    
    return 0.3*l1+0.7*l2

