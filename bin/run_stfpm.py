"""STFPM baseline using the WM811K preprocessing and evaluation protocol."""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset
from patchcore.datasets.synthetic_masks import DatasetSplit as SyntheticSplit, SyntheticMaskPatchCoreDataset
from torchvision.models import wide_resnet50_2

class Hooks(torch.nn.Module):
    def __init__(self, net, layers):
        super().__init__(); self.net=net; self.layers=layers; self.out={}
        for name in layers: net._modules[name].register_forward_hook(lambda _,__,o,n=name:self.out.__setitem__(n,o))
    def forward(self,x): self.out={}; self.net(x); return self.out

def maps(teacher,student,x,layers,size):
    with torch.no_grad(): t=teacher(x)
    s=student(x)
    score=0
    for layer in layers:
        a=F.normalize(t[layer],dim=1); b=F.normalize(s[layer],dim=1)
        score += F.interpolate(((a-b)**2).sum(1,keepdim=True),size=size,mode='bilinear',align_corners=False)
    return score.squeeze(1)

def auc(maps,masks):
    y=(np.asarray(masks).ravel()>0).astype(int); x=np.asarray(maps).ravel()
    return float(roc_auc_score(y,x)) if np.unique(y).size==2 else float('nan')

def evaluate(loader,teacher,student,layers,device):
    scores=[]; labels=[]; allmaps=[]; masks=[]
    teacher.eval(); student.eval()
    with torch.no_grad():
        for batch in loader:
            pred=maps(teacher,student,batch['image'].to(device),layers,batch['mask'].shape[-2:]).cpu().numpy()
            pred=np.asarray([ndimage.gaussian_filter(v,sigma=4) for v in pred])
            allmaps.extend(pred); masks.extend(batch['mask'].numpy()[:,0]); scores.extend(pred.reshape(len(pred),-1).max(1)); labels.extend(batch['is_anomaly'].numpy())
    return np.asarray(scores),np.asarray(labels),np.asarray(allmaps),np.asarray(masks)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_root',required=True); p.add_argument('--synthetic_root',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--resize',type=int,default=128); p.add_argument('--imagesize',type=int,default=128); p.add_argument('--max_train_images',type=int,default=0); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
 torch.manual_seed(a.seed); np.random.seed(a.seed); d=torch.device('cuda:0'); root=Path(a.dataset_root); kw=dict(resize=a.resize,imagesize=a.imagesize,transform_mode='resize_pad',apply_filter=True,filter_window_size=3,filter_threshold=1.25)
 train=wm811kDataset(str(root.parent),root.name,split=DatasetSplit.TRAIN,**kw); test=wm811kDataset(str(root.parent),root.name,split=DatasetSplit.TEST,**kw); pixel=SyntheticMaskPatchCoreDataset(a.synthetic_root,'all',split=SyntheticSplit.TEST,**kw)
 if a.max_train_images: train=Subset(train,torch.randperm(len(train),generator=torch.Generator().manual_seed(a.seed))[:min(a.max_train_images,len(train))].tolist())
 loaders=[DataLoader(x,batch_size=a.batch_size,shuffle=i==0,num_workers=0) for i,x in enumerate((train,test,pixel))]
 layers=['layer1','layer2','layer3']; teacher=Hooks(wide_resnet50_2(weights='IMAGENET1K_V1'),layers).to(d).eval(); student=Hooks(wide_resnet50_2(weights=None),layers).to(d)
 for q in teacher.parameters(): q.requires_grad=False
 opt=torch.optim.Adam(student.parameters(),lr=a.lr)
 for epoch in range(a.epochs):
  student.train(); total=0
  for batch in loaders[0]:
   x=batch['image'].to(d); t=teacher(x); s=student(x); loss=sum(F.mse_loss(F.normalize(s[k],dim=1),F.normalize(t[k],dim=1)) for k in layers); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
  print(f'epoch={epoch+1} loss={total/len(loaders[0]):.6f}',flush=True)
 scores,labels,_,_=evaluate(loaders[1],teacher,student,layers,d); _,_,pm,gm=evaluate(loaders[2],teacher,student,layers,d); anomalous=np.asarray(gm).reshape(len(gm),-1).sum(1)>0
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); result={'instance_auroc':float(roc_auc_score(labels,scores)),'full_pixel_auroc':auc(pm,gm),'anomaly_pixel_auroc':auc(pm[anomalous],gm[anomalous])}
 with (out/'metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=result);w.writeheader();w.writerow(result)
 with (out/'image_scores.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['image_score','is_anomaly']);w.writeheader();w.writerows({'image_score':float(x),'is_anomaly':int(y)} for x,y in zip(scores,labels))
 torch.save(student.net.state_dict(),out/'student_wrn50.pt'); print(result)
if __name__=='__main__': main()
