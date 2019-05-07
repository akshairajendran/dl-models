#!/usr/bin/env python
# coding: utf-8

# In[23]:


from path import Path
from fastai.vision import *
import numpy as np
import pandas as pd
from PIL import Image
import json
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from cycler import cycler
from typing import List, Union, Tuple, Callable, Optional


# In[20]:


class StdConv(nn.Module):
    def __init__(self, nin: int, nout: int, stride: int=2, dropout: float=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor):
        return self.drop(self.bn(F.relu(self.conv(x))))
    
def flatten_conv(x: torch.Tensor, k: int):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs, -1, nf//k)

class OutConv(nn.Module):
    def __init__(self, k: int, nin: int, nclasses: int, bias: float):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, nclasses * self.k, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, 4*self.k, 3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self, x: torch.Tensor):
        return [flatten_conv(self.oconv1(x), self.k), flatten_conv(self.oconv2(x), self.k)]

class SSD_MultiHead(nn.Module):
    def __init__(self, k: int, num_classes: int, bias: float, dropout: float=0.1):
        super().__init__()
        self.k, self.num_classes = k, num_classes
        self.drop = nn.Dropout(dropout)
        self.sconv0 = StdConv(512, 256, stride=1, dropout=dropout)
        self.sconv1 = StdConv(256, 256, dropout=dropout)
        self.sconv2 = StdConv(256, 256, dropout=dropout)
        self.sconv3 = StdConv(256, 256, dropout=dropout)
        self.out1 = OutConv(self.k, 256, self.num_classes, bias)
        self.out2 = OutConv(self.k, 256, self.num_classes, bias)
        self.out3 = OutConv(self.k, 256, self.num_classes, bias)
        
    def forward(self, x: torch.Tensor):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out1(x)
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        return [torch.cat([o1c, o2c, o3c], dim=1),
                torch.cat([o1l, o2l, o3l], dim=1)]


# In[21]:


def one_hot_embedding(labels: torch.Tensor, num_classes: int):
    """Returns a one hot embedded matrix
    """
    return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
    """Binary cross entropy loss
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        t = one_hot_embedding(targ, self.num_classes)
        t = t[:,:-1].contiguous()
        x = pred[:, :-1].cpu()
        w = (self.get_weight(x, t)).detach()
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)/self.num_classes
    
    def get_weight(self, x: torch.Tensor, t: torch.Tensor):
        return None

class FocalLoss(BCE_Loss):
    """Focal loss modifies BCE_Loss by adding weighting, details to come.
    """
    def get_weight(self, x: torch.Tensor, t: torch.Tensor):
        alpha,gamma = 0.25, 1
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt).pow(gamma) 


# In[27]:


class SingleShotDetector():
    """Single shot detector model for multiple object detection
       Takes databunch object and creates learner for training and inference
    """
    def __init__(self, data: ObjectItemList, anc_grids: List[int]=[4, 2, 1], anc_zooms: List[float]=[.7, 1., 1.3],
                       anc_ratios: List[Tuple[float, float]]=[(1., 1.), (1., 0.5), (0.5, 1.)],
                       base_arch: Callable=models.resnet34, dropout: float=0.1, bias: float=-4.,
                       loss_func: nn.Module=FocalLoss, thresh: float=.4,
                       num_color: int=12, verbose: bool=False):
        
        ###Init Data###
        self.data = data
        self.num_classes = self.data.c
        
        ###Init Anchors###
        #default settings give a 4x4, 2x2 and 1x1 grid for a total of 16 + 4 + 1 = 21 anchor points
        #3 zooms and 3 ratios for a total of 9 boxes per anchor point
        #9 x 21 = 189 total anchor boxes
        self.anc_grids, self.anc_zooms, self.anc_ratios = anc_grids, anc_zooms, anc_ratios
        self.anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i, j) in anc_ratios] #len(zooms) * len(ratios)
        self.k = len(self.anchor_scales)
        self.anc_offsets = [1/(o*2) for o in self.anc_grids] #scale grids to [0,1]
        self.anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(self.anc_offsets, self.anc_grids)])
        self.anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(self.anc_offsets, self.anc_grids)])
        self.anc_ctrs = np.repeat(np.stack([self.anc_x, self.anc_y], axis=1), self.k, axis=0)
        self.anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in self.anchor_scales]) for ag in self.anc_grids])
        self.grid_sizes = torch.tensor(np.concatenate([np.array([1/ag for i in range(ag*ag) for o,p in self.anchor_scales])for ag in self.anc_grids])).unsqueeze(1)
        self.anchors = torch.tensor(np.concatenate([self.anc_ctrs, self.anc_sizes], axis=1))
        self.anchor_cnr = self._hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        
        ###Init Model###
        self.base_arch, self.dropout, self.bias, self.thresh = base_arch, dropout, bias, thresh
        self.loss_func = loss_func(self.num_classes)
        self.head = SSD_MultiHead(self.k, self.num_classes, self.bias, self.dropout)
        self.learn = cnn_learner(self.data, self.base_arch, custom_head=self.head)
        
        ###Init Drawing###
        self.num_color = num_color
        cmap = self._get_cmap(self.num_color)
        self.color_list = [cmap(float(x)) for x in range(self.num_color)]
        
        ###Init Misc###
        self.verbose = verbose
    
    def plot_anchors(self):
        """Plots anchor points on blank plot
        """
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        plt.scatter(self.anc_x, self.anc_y)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    
    def _ssd_1_loss(self, b_c: torch.Tensor, b_bb: torch.Tensor, bbox: torch.Tensor, clas: torch.Tensor, 
                    thresh: Optional[float]=None, verbose: Optional[bool]=None):
        """Computes loss for one image and set of bbox predictions
           Takes b_c and b_bb which are predicted classes and bounding boxes (output from NN)
           and bbox and clas which are ground truth bounding box and classes for image
        """
        thresh = self.thresh if not thresh else thresh
        verbose = self.verbose if not verbose else verbose
        bbox, clas = self._get_y(bbox, clas) #reformat gt bbox and gt clas
        if len(bbox) == 0: return 0, 0 #if there's no bbox, we can't use this in our loss
        a_ic = self._actn_to_bb(b_bb, self.anchors) #map activation b_bb to anchor points (defined above)
        overlaps = self._jaccard(bbox.data, anchor_cnr.data.cuda().float()) #get overlaps between gt bbox and anchor boxes
        gt_overlap, gt_idx = self._map_to_ground_truth(overlaps, verbose)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > thresh
        pos_idx = torch.nonzero(pos)[:, 0] #get indices of anchor boxes with sufficient overlap with GT
        gt_clas[1-pos] = 0 #assign background class
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean() #get average absolute loss between coordinates
        clas_loss = self.loss_func(b_c, gt_clas) #get cross entropy loss on predicted classes
        return loc_loss, clas_loss

    def _ssd_loss(self, pred: torch.Tensor, targ0: torch.Tensor, targ1: torch.Tensor, 
                        verbose: Optional[bool]=None):
        """For a batch of predictions and a batch of (targ0, targ1) where targ0 is ground truth bbox and
           targ1 is ground truth classes compute the loss. We do this by iterating over the batch and computing
           loss on each individual image and summing the results. I believe the losses must be placed on the CPU
           due to the way fastai is handling the loss function
        """
        verbose = self.verbose if not verbose else verbose
        lcs,lls=0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred, targ0, targ1):
            if len(bbox) == 0:
                continue
            loc_loss, clas_loss = self._ssd_1_loss(b_c, b_bb, bbox, clas, verbose=verbose)
            lls += loc_loss
            lcs += clas_loss
        if verbose:
            print("loc: {0}, clas: {1}".format(lls.data[0], lcs.data[0]))
        return lls.cpu() + lcs.cpu() #need to revisit why cpu is necessary
    
    def _show_ground_truth(self, ax, im, bbox: torch.Tensor, clas: Optional[torch.Tensor]=None, 
                                 prs: Optional[torch.Tensor]=None, thresh: Optional[float]=None):
        """Displays anchor box, predicted activation and recentered activation
        """
        thresh = self.thresh if not thresh else thresh
        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = self._show_img(im, ax=ax)
        k=0
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                k+=1
                self._draw_rect(ax, b, color=self.color_list[i%self.num_color])
                txt = f'{k}: '
                if isinstance(c, str):
                    txt += c
                else:
                    if c is not None: txt += str(self.data.classes[c])
                if pr is not None: txt += f' {pr:.2f}'
                self._draw_text(ax, b[:2] + np.array([0, np.random.randint(0, 40)]), txt, color=self.color_list[i%self.num_color])

    def _torch_gt(sefl, ax, ima, bbox: torch.Tensor, clas: torch.Tensor, 
                        prs: Optional[torch.Tensor]=None, thresh: Optional[float]=None, 
                        ignore_bg: bool=False):
        """Show ground truth given image and torch bbox
        """
        thresh = self.thresh if not thresh else thresh
        return self._show_ground_truth(ax, ima, to_np((bbox*224).long()),
             to_np(clas), to_np(prs) if prs is not None else None, thresh, ignore_bg=ignore_bg)
    
    def _actn_to_bb(actn: torch.Tensor, anchors: torch.Tensor):
        """Mapping activation to anchor boxes
        """
        actn_bbs = torch.tanh(actn) #map activations to [-1,1]
        actn_centers = ((actn_bbs[:, :2]/2).float() * self.grid_sizes.cuda().float()) + anchors[:,:2].cuda().float() #center relative to anchor box
        actn_hw = ((actn_bbs[:, 2:]/2).float() + 1) * anchors[:,2:].cuda().float() #get height and width relative to anchor box
        return self._hw2corners(actn_centers, actn_hw)
    
    @staticmethod
    def _show_img(im, figsize=None, ax=None):
        """Standard func to display image
        """
        if not ax: fig,ax = plt.subplots(figsize=figsize)
        ax.imshow(im)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return ax
    
    @staticmethod
    def _bb_hw(a: torch.Tensor): 
        """Reshapes bounding box from bottom left (x,y), top right (x,y) to
           bottom left y, bottom left x, height, width
        """
        return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])

    @staticmethod
    def _draw_outline(o, lw):
        """Adds black border around line
        """
        o.set_path_effects([patheffects.Stroke(
            linewidth=lw, foreground='black'), patheffects.Normal()])
    
    @staticmethod
    def _draw_rect(ax, b, color='white'):
        """Draws bounding rectangle
        """
        patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
        self._draw_outline(patch, 4)

    @staticmethod
    def _draw_text(ax, xy, txt, sz=14, color='white'):
        """Draws text inside bounding rectangle
        """
        text = ax.text(*xy, txt,
            verticalalignment='top', color=color, fontsize=sz, weight='bold')
        self._draw_outline(text, 1)

    @staticmethod
    def _get_cmap(N):
        """Gets colors
        """
        color_norm  = mcolors.Normalize(vmin=0, vmax=N-1)
        return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba
    
    @staticmethod
    def _get_y(bbox: torch.tensor, clas: torch.tensor):
        """Returns tensor with 0 rows removed
        """
        if len(bbox.shape) == 1:
            bbox.unsqueeze_(0)
        bbox = (bbox + 1.) / 2. #normalize, remove from here?
        idx_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[idx_keep], clas[idx_keep]
    
    @staticmethod
    def _hw2corners(ctr: torch.Tensor, hw: torch.Tensor): 
        """Converts array of x,y and height,width to top-left, bottom-right
        """
        return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
    
    @staticmethod
    def _intersection(box_a: torch.Tensor, box_b: torch.Tensor):
        """Computes intersection between two boxes
        """
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]
    
    @staticmethod
    def _box_sz(b: torch.Tensor):
        """Computes box size of b
        """
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) #height x width

    @staticmethod
    def _jaccard(box_a: torch.Tensor, box_b: torch.Tensor):
        """Computes jaccard distance between two boxes
        """
        box_a = box_a.cuda().float()
        box_b = box_b.cuda().float()
        inter = intersection(box_a, box_b)
        union = (box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter)
        return inter/union

    @staticmethod
    def _map_to_ground_truth(overlaps: torch.Tensor, verbose: bool=False):
        """Takes an array of IOU overlaps between anchor boxes and the ground truth bounding boxes
           Yields an array of overlaps by anchor box and indices of the class with which the anchor box overlaps
        """
        prior_overlap, prior_idx = overlaps.max(1)
        if verbose: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap, gt_idx


# In[ ]:



