# -*- coding: utf-8 -*-
"""检查作为 answer 提交的生成器(ckpt_fc.pth / ckpt_dc.pth)是否模式崩溃。
生成 2000 张, 用已训练的 FashionMNIST 分类器统计类别覆盖与熵。"""
import os, numpy as np, torch
from models import Z_DIM, FCGenerator, DCGenerator
from mc_train import SmallCNN
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda" if torch.cuda.is_available() else "cpu"
clf = SmallCNN().to(DEV); clf.load_state_dict(torch.load(os.path.join(HERE,"ckpt_classifier.pth"),map_location=DEV)); clf.eval()
classes = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
@torch.no_grad()
def cov(G, n=2000):
    G.eval(); c = np.zeros(10,int); done=0
    while done<n:
        b=min(256,n-done); z=torch.randn(b,Z_DIM,device=DEV)
        for p in clf(G(z)).argmax(1).cpu().numpy(): c[p]+=1
        done+=b
    p=c/c.sum(); ent=float(-(p[p>0]*np.log(p[p>0])).sum())
    return c, ent
for tag, Gc, ck in [("FC(answer)",FCGenerator(),"ckpt_fc.pth"),("DC(answer)",DCGenerator(),"ckpt_dc.pth")]:
    G=Gc.to(DEV); G.load_state_dict(torch.load(os.path.join(HERE,ck),map_location=DEV))
    c,ent=cov(G)
    nz=int((c>20).sum())
    print(f"{tag}: entropy={ent:.3f}/{np.log(10):.3f}  classes>1%={nz}/10")
    print("   ", {classes[i]:int(c[i]) for i in range(10)})
