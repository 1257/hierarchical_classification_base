import torch.nn as nn
import torch

def entropy2lvl(outputs, labels, class_labels, use_superclasses, use_classes):
    indices = []
    #loss = nn.CrossEntropyLoss(ignore_index=indices)
    loss = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    coarse = []
    classes = []
    real_classes = []
    real_superclass = [None]*len(outputs)
    func=max
    print("start outputs: ", outputs)
    
    for i in range(len(outputs)):
        coarse.append([])

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
            
    print(torch.tensor(coarse, requires_grad=True))        
    l1=loss(torch.tensor(coarse, requires_grad=True).cuda(), labels)    #loss on superclasses
    
    mask = class_labels >= 0
    indices = torch.nonzero(mask)
    
    outs = outputs[indices]
    outs1 = []*len(indices)
    targs = class_labels[indices]
    targs1 = []*len(indices)
    
    for i in range(len(labels)):
        if i not in indices:
            outs1[i] = outs[i][0]
            targs1[i] = targs[i][0]
    outs1=torch.tensor(outs1).cuda()
    targs1=torch.tensor(targs1).cuda()
    
    
    res=[outputs[i] for i in range(len(labels)) if labels[i]>=0]
    #res=torch.tensor(res, requires_grad=True)
    res= torch.tensor([item.cpu().detach().numpy() for item in res]).cuda() 
    print("result size of comprehension output: ", res.size())
    labs=[labels[i] for i in range(len(labels)) if labels[i]>=0]
    labs=torch.tensor(labs).cuda()
    print("labs size of comprehension output: ", labs.size())
        
    #outs.reshape([ len(indices), 100])
    #outs = (torch.tensor(outs)).cuda()
    #print("outputs size:", outputs.size())
    
    #print("outputs: ", outputs)
    #print("outs: ", outs)
    #new_labels = torch.tensor(class_labels[indices]).cuda()
    #new_labels = new_labels.reshape([len(indices)])
    #print("new_labels size:", new_labels.size())
    #l2=loss(outputs[indices], class_labels[indices])   #loss on classes
    #l2=loss(outs1, targs1)
    l2=loss(res, labs)
    
    if use_superclasses==True and use_classes==True:
        print("loss 1 and 2:", 0.7*l1+0.3*l2)
        return 0.7*l1+0.3*l2
    elif use_superclasses==True:
        print("loss 1:", l1)
        return l1
    elif use_superclasses==True:
        print("loss 2:", l2)
        return l2
    else:
        print("none of parts included")


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

