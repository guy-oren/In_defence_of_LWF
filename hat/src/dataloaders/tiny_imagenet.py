import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torchvision.datasets import ImageFolder


class TinyImagenet(ImageFolder):
    def __init__(self, root, train=True, **kwargs):
        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")

        super().__init__(root, **kwargs)

        self.data = [s[0] for s in self.samples]


def get(seed=0, task_size=20, pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]

    classes_order = list(shuffle(np.arange(200), random_state=seed))
    print('Classes order =', classes_order)

    n_tasks = 200 // task_size

    dat_dir_name = '../dat/binary_tiny_imagenet_t_{}_s_{}/'.format(task_size, seed)
    if not os.path.isdir(dat_dir_name):
        os.makedirs(dat_dir_name)

        mean=[0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5]

        # TinyImagenet
        dat={}
        dat['train']=TinyImagenet('../dat/tiny-imagenet-200',train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=TinyImagenet('../dat/tiny-imagenet-200',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        for n in range(n_tasks):
            data[n]={}
            data[n]['name']='tiny_imagenet'
            data[n]['ncla']=task_size
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
            data[n]['classes'] = classes_order[n * task_size: (n + 1) * task_size]
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                n=classes_order.index(n)
                nn=(n//task_size)
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n%task_size)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(dat_dir_name),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(dat_dir_name),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    for i in range(n_tasks):
        data[i] = dict.fromkeys(['name','ncla','train','test', 'classes'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(dat_dir_name),'data'+str(i)+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(dat_dir_name),'data'+str(i)+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='tiny_imagenet-'+str(i)
        data[i]['classes'] = classes_order[i * task_size: (i + 1) * task_size]

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
