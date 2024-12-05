import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import random
from data.common_dataloader import CommonDataloader
from torchnet import meter
from config import opt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from models.borderf import BorderF
import os
from tqdm import tqdm



def load_model_weights(old_model,new_model):
    if torch.cuda.is_available():
        new_model.cuda()
    pretrained_dict = old_model.state_dict()
    substitute_dict = new_model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in substitute_dict}
    substitute_dict.update(pretrained_dict)
    new_model.load_state_dict(substitute_dict)
    return new_model

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class CombineTrainer():
    def __init__(self, opt):
        self.train_data_root = opt.train_data_root
        self.val_data_root = opt.val_data_root
        self.test_data_root1 = opt.test_data_root_cnn
        self.model_name      = opt.model_name

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.constant(m.weight, 1e-2)
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias,0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")
            # nn.init.constant(m.weight, 1e-3)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 2e-1)
            nn.init.constant(m.bias, 0)
    
    def test(self,realname,fakename):

        model = eval(self.model_name)(opt)
        loaded_model = eval(opt.loaded_model_name)(opt)
        self.print_parameters(model)
        model.apply(self.weight_init)
        for e in [13]:
            l_m_p = opt.load_model_path+'__0429_'+str(e)+'.pth'
            print(l_m_p)
            if opt.load_model and os.path.exists(l_m_p):
                loaded_model.load(l_m_p)
                model = load_model_weights(loaded_model, model)
                print("done")
            if torch.cuda.is_available():
                model.cuda()
        
            model.eval()
            
            for gan in ['biggan1','deepfake','gaugan','stargan','cyclegan','stylegan','stylegan2','progan']:
                if gan =="cyclegan":
                    k=-1
                    for n in ['zebra','horse','orange','apple','winter','summer']:
                        data_root1 = self.test_data_root1+'/cyclegan1/'+n
                        print('正在计算：%s %s'%(gan,n))
                        k+=1
                        test_data   = CommonDataloader(data_root1,  noise=opt.test_noise, train=False,test=True,real_name=realname,fake_name=fakename)
                        test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
                        score,label,cm_value=self.val(model,test_dataloader)
                        if k ==0:
                            a_score =score
                            a_label = label
                            a_cm = cm_value
                        else:
                            a_score = a_score+score
                            a_label = a_label+label
                            a_cm = a_cm+cm_value
                    print(len(a_score))
                    print(len(a_label))
                    acc,auc,ap = self.result(a_score,a_label,a_cm)
                    f=open('./data/%s.txt'%gan,mode='a')
                    f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                    f.close()
                elif gan=="stylegan":
                    k=-1
                    for n in ['bedroom','car','cat']:
                        data_root1 = self.test_data_root1+'/stylegan/'+n
                        print('正在计算：%s %s'%(gan,n))
                        k+=1
                        test_data   = CommonDataloader(data_root1, noise=opt.test_noise, train=False,test=True,real_name=realname,fake_name=fakename)
                        test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
                        score,label,cm_value=self.val(model,test_dataloader)
                        if k ==0:
                            a_score =score
                            a_label = label
                            a_cm = cm_value
                        else:
                            a_score = a_score+score
                            a_label = a_label+label
                            a_cm = a_cm+cm_value
                    print(len(a_score))
                    print(len(a_label))
                    acc,auc,ap = self.result(a_score,a_label,a_cm)
                    f=open('./data/%s.txt'%gan,mode='a')
                    f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                    f.close()
                elif gan=="stylegan2":
                    k=-1
                    for n in ['church','horse','car','cat']:
                        data_root1 = self.test_data_root1+'/stylegan2/'+n
                        print('正在计算：%s %s'%(gan,n))
                        k+=1
                        test_data   = CommonDataloader(data_root1, noise=opt.test_noise, train=False,test=True,real_name=realname,fake_name=fakename)
                        test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
                        score,label,cm_value=self.val(model,test_dataloader)
                        if k ==0:
                            a_score =score
                            a_label = label
                            a_cm = cm_value
                        else:
                            a_score = a_score+score
                            a_label = a_label+label
                            a_cm = a_cm+cm_value
                    print(len(a_score))
                    print(len(a_label))
                    acc,auc,ap = self.result(a_score,a_label,a_cm)
                    f=open('./data/%s.txt'%gan,mode='a')
                    f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                    f.close()
                elif gan=="progan":
                    k=-1
                    for n in ['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']:
                        data_root1 = self.test_data_root1+'/progan/'+n
                        print('正在计算：%s %s'%(gan,n))
                        k+=1
                        test_data   = CommonDataloader(data_root1, noise=opt.test_noise, train=False,test=True,real_name=realname,fake_name=fakename)
                        test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
                        score,label,cm_value=self.val(model,test_dataloader)
                        if k ==0:
                            a_score =score
                            a_label = label
                            a_cm = cm_value
                        else:
                            a_score = a_score+score
                            a_label = a_label+label
                            a_cm = a_cm+cm_value
                    print(len(a_score))
                    print(len(a_label))
                    acc,auc,ap = self.result(a_score,a_label,a_cm)
                    f=open('./data/%s.txt'%gan,mode='a')
                    f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                    f.close()

                else:
                    data_root1 = self.test_data_root1+'/'+gan
                    print('正在计算：%s'%gan)
                    test_data   = CommonDataloader(data_root1, noise=opt.test_noise, train=False,test=True,real_name=realname,fake_name=fakename)
                    test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
                    score,label,cm_value=self.val(model,test_dataloader)
                    acc,auc,ap = self.result(score,label,cm_value)
                    f=open('./data/%s.txt'%gan,mode='a')
                    f.write('%s: acc %.2f ,auc %.2f ,ap %.2f \n'%(gan,acc,auc,ap))
                    f.close()

        return
    
    def val(self,model_1,dataloader):
        model_1.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        score_all = []
        label_all = []
        with t.no_grad():
            for ii,(rgb,_,label) in tqdm(enumerate(dataloader),total=len(dataloader), desc='testing'):
                valrgb_input = Variable(rgb)
                val_label = Variable(label.type(t.LongTensor))
                if torch.cuda.is_available():
                    valrgb_input = valrgb_input.cuda()
                    val_label = val_label.cuda()
                score,_ = model_1(valrgb_input)
                confusion_matrix.add(score.data, label.long())
                score_all.extend(score[:,1].detach().cpu().numpy())
                label_all.extend(label)
        cm_value = confusion_matrix.value()
        return score_all,label_all,cm_value
    
    def result(self,score_all,label_all,cm_value):

        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /(cm_value.sum())
        auc = roc_auc_score(label_all, score_all)
        ap = average_precision_score(label_all,score_all)
        print(f"val acc.{accuracy} | AUC {auc} | AP {ap*100} |", end='')
        print('\n')
        return accuracy,auc*100,ap*100

    def print_parameters(self,model):
        pid = os.getpid()
        total_num = sum(i.numel() for i in model.parameters())
        trainable_num = sum(i.numel() for i in model.parameters() if i.requires_grad)

        print("=========================================")
        print("PID:",pid)
        print("\nNum of parameters:%i"%(total_num))
        print("Num of trainable parameters:%i"%(trainable_num))
        print("Save model path:",opt.save_model_path)
        print("learning rate:",opt.lr)

        print("batch_size:",opt.batch_size)
        print("=========================================")

