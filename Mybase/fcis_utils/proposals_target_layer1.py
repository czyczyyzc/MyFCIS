import cv2
import numpy as np
import tensorflow as tf
from Mybase.comp_utils import *
from Mybase.losses import *
from .bbox import *
from .rois_pooling_layer import *
#from shapely.geometry import Polygon

class ProposalsTargetLayer(object):
    
    def __init__(self, roi_cls_num=21, img_shp=None):
        
        self.img_shp     = img_shp
        self.roi_cls_num = roi_cls_num
        self.roi_fbg_num = 256
        self.roi_fgd_frc = 0.25
        self.roi_fgd_ovp = 0.5
        self.roi_bgd_ovp = 0.5
        self.roi_crw_max = 0.001   #crowd_thresh
        self.roi_fgd_num = int(self.roi_fbg_num * self.roi_fgd_frc)
        self.roi_bfg_rat = 1.0 / self.roi_fgd_frc - 1.0
        self.roi_msk_siz = [21, 21]
        self.roi_msk_min = 0.4
        self.RP          = RoisPoolingLayer(self.roi_cls_num, self.img_shp)
        
    #如果是Faster RCNN，就用roi_imxs把每幅图片的rois合并起来!!!    
    def sample_rois_img(self, elems=None):
        
        rois, roi_prbs_pst, roi_prds_pst, roi_num, gbxs, gmks, gbx_num = elems
        rois = rois[0:roi_num]
        gbxs = gbxs[0:gbx_num]
        gmks = gmks[0:gbx_num]
        
        crw_idxs = tf.where(gbxs[:, 4]< 0)                   #crowd_idx
        ncw_idxs = tf.where(gbxs[:, 4]>=0)                   #no_crowd_idx #最后一个是无效边框
        gbxs_tmp = tf.gather_nd(gbxs, crw_idxs)
        gbxs     = tf.gather_nd(gbxs, ncw_idxs)
        gmks     = tf.gather_nd(gmks, ncw_idxs)
        #rois    = tf.concat([rois, gbxs[:-1, :-1]], axis=0) #不能让有些rois的长或宽为0
        
        ###################计算CROWD IOU##################
        #roi_iscs = bbox_intersects(rois, gbxs_tmp[:, 0:4])
        #max_iscs = tf.reduce_max(roi_iscs, axis=1)
        roi_ovps = bbox_overlaps(rois, gbxs_tmp[:, 0:4])
        max_ovps = tf.reduce_max(roi_ovps, axis=1)
        ncw_msks = max_ovps < self.roi_crw_max
        
        #################计算NO CROWD IOU#################
        roi_ovps = bbox_overlaps(rois, gbxs[:, 0:4])
        roi_amxs = tf.argmax(roi_ovps, axis=1)
        max_ovps = tf.reduce_max(roi_ovps, axis=1)
        
        roi_gtas = tf.zeros(shape=[tf.shape(rois)[0]], dtype=tf.int64) - 2
        
        ######################选前景######################
        fgd_idxs0 = tf.where(max_ovps>=self.roi_fgd_ovp)[:, 0]  #无效边框不影响
        fgd_idxs1 = tf.argmax(roi_ovps, axis=0)                 #最后一个是无效边框
        fgd_idxs1 = fgd_idxs1[:-1]                              #剔除最后一个无效边框匹配的anchor
        fgd_idxs  = tf.concat([fgd_idxs0, fgd_idxs1], axis=0)
        fgd_idxs, idxs = tf.unique(fgd_idxs)
        fgd_idxs  = tf.expand_dims(fgd_idxs, axis=-1)
        #fgd_idxs = tf.where(max_ovps>=self.roi_fgd_ovp)
        fgd_num   = tf.shape(fgd_idxs)[0]
        fgd_num   = tf.minimum(fgd_num, self.roi_fgd_num)
        fgd_idxs  = tf.random_shuffle(fgd_idxs)[:fgd_num]
        fgd_gtas  = tf.gather_nd(roi_amxs, fgd_idxs)            #和fgd_idxs相对应
        roi_gtas  = tensor_update(roi_gtas, fgd_idxs, fgd_gtas) #这里的fgd_idxs可以乱序
        
        ######################选背景######################
        bgd_idxs     = tf.where(tf.logical_and(tf.logical_and(max_ovps<self.roi_bgd_ovp, tf.equal(roi_gtas, -2)), ncw_msks))
        bgd_num      = tf.shape(bgd_idxs)[0]
        #Hard Negative Mining
        bgd_num      = tf.minimum(bgd_num, tf.cast(tf.cast(fgd_num, tf.float32)*self.roi_bfg_rat, tf.int32))
        #roi_prbs_tmp= tf.gather_nd(roi_prbs_pst, bgd_idxs)      #和bgd_idxs对应
        bgd_rois     = tf.gather_nd(rois, bgd_idxs)
        roi_prbs_tmp, _, _ = self.RP.rois_pooling_img_vld(roi_prbs_pst, roi_prds_pst, bgd_rois)
        roi_prbs_tmp = tf.stop_gradient(roi_prbs_tmp)
        roi_prbs_pre = tf.zeros(shape=[tf.shape(roi_prbs_tmp)[0]], dtype=tf.int32)
        roi_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=roi_prbs_pre, logits=roi_prbs_tmp) #和bgd_idxs对应
        _, idxs      = tf.nn.top_k(roi_prbs_los, k=bgd_num, sorted=False) #Hard Negative Mining
        bgd_idxs     = tf.gather(bgd_idxs, idxs)
        roi_gtas     = tensor_update(roi_gtas, bgd_idxs, -1)       #这里的bgd_idxs可以乱序
        '''
        bgd_num   = tf.minimum(bgd_num, self.roi_fbg_num-fgd_num)
        #bgd_num  = tf.minimum(bgd_num, self.roi_fbg_num-fgd_num)
        bgd_num   = tf.minimum(bgd_num, tf.cast(tf.cast(fgd_num, tf.float32)*self.roi_bfg_rat, tf.int32))
        bgd_idxs  = tf.random_shuffle(bgd_idxs)[:bgd_num]
        roi_gtas  = tensor_update(roi_gtas, bgd_idxs, -1)       #这里的bgd_idxs可以乱序
        '''
        ####################整合前背景#####################
        fgd_idxs = tf.where(roi_gtas>=0)
        bgd_idxs = tf.where(tf.equal(roi_gtas, -1))
        fbg_idxs = tf.concat([fgd_idxs, bgd_idxs], axis=0)      #确保fgd_idxs/bgd_idxs/fbg_idxs三者的顺序相对应
        fgd_gtas = tf.gather_nd(roi_gtas, fgd_idxs)             #和fgd_idxs对应
        fbg_gtas = tf.gather_nd(roi_gtas, fbg_idxs)             #和fbg_idxs对应
        fgd_num  = tf.shape(fgd_idxs)[0]
        bgd_num  = tf.shape(bgd_idxs)[0]
        fbg_num  = tf.shape(fbg_idxs)[0]
        fbg_rois = tf.gather_nd(rois, fbg_idxs)                 #和fbg_idxs对应
        
        ####################Get Target###################
        ###Get类别###
        fgd_prbs_pre = tf.gather(gbxs[:, 4], fgd_gtas)                         #和fgd_idxs对应
        fgd_prbs_pre = tf.cast(fgd_prbs_pre, dtype=tf.int32)
        bgd_prbs_pre = tf.zeros(shape=[bgd_num], dtype=tf.int32)
        fbg_prbs_pre = tf.concat([fgd_prbs_pre, bgd_prbs_pre], axis=0)         #和fbg_idxs对应
        ###Get坐标###
        #注意是定类还是不定类
        fgd_prds_tmp = tf.gather(gbxs[:, 0:4], fgd_gtas)        #和fgd_idxs对应
        fgd_rois     = tf.gather_nd(rois, fgd_idxs)             #和fgd_idxs对应
        fgd_prds_pre = bbox_transform(fgd_rois, fgd_prds_tmp)   #和fgd_idxs对应
        ###Get MASK###
        #ROIS相对于gt_boxes的位置
        beg = tf.stack([fgd_prds_tmp[:, 0], fgd_prds_tmp[:, 1], \
                        fgd_prds_tmp[:, 0], fgd_prds_tmp[:, 1]], axis=-1)
        leh = tf.stack([fgd_prds_tmp[:, 2]-fgd_prds_tmp[:, 0], \
                        fgd_prds_tmp[:, 3]-fgd_prds_tmp[:, 1],
                        fgd_prds_tmp[:, 2]-fgd_prds_tmp[:, 0],
                        fgd_prds_tmp[:, 3]-fgd_prds_tmp[:, 1]], axis=-1)
        fgd_rois = (fgd_rois - beg) / leh
        
        gmks         = tf.expand_dims(gmks, axis=-1) #(M, H, W, 1)
        fgd_gtas     = tf.cast(fgd_gtas, dtype=tf.int32)
        fgd_msks_pre = tf.image.crop_and_resize(gmks, fgd_rois, fgd_gtas, self.roi_msk_siz, method='bilinear')
        fgd_msks_pre = tf.squeeze(fgd_msks_pre, axis=-1)
        fgd_msks_pre = tf.cast(fgd_msks_pre>=self.roi_msk_min, dtype=tf.int32) #(M0, 21, 21)
        
        paddings     = [[0, self.roi_fbg_num-fbg_num], [0, 0]]     #fbg_num/fgd_num要是整型!!!
        fbg_rois     = tf.pad(fbg_rois,     paddings, "CONSTANT")
        paddings     = [[0, self.roi_fbg_num-fbg_num]]
        fbg_prbs_pre = tf.pad(fbg_prbs_pre, paddings, "CONSTANT")
        paddings     = [[0, self.roi_fgd_num-fgd_num], [0, 0]]
        fgd_prds_pre = tf.pad(fgd_prds_pre, paddings, "CONSTANT")
        paddings     = [[0, self.roi_fgd_num-fgd_num], [0, 0], [0, 0]]
        fgd_msks_pre = tf.pad(fgd_msks_pre, paddings, "CONSTANT")
        return fbg_rois, fbg_prbs_pre, fgd_prds_pre, fgd_msks_pre, fgd_num, fbg_num
    
    ############使用sample_rois主要还是为了减少内存消耗#############
    def sample_rois(self, rois, roi_prbs_pst, roi_prds_pst, roi_nums, gbxs, gmks, gbx_nums):
        
        elems = [rois, roi_prbs_pst, roi_prds_pst, roi_nums, gbxs, gmks, gbx_nums]
        self.fbg_rois, self.fbg_prbs_pre, self.fgd_prds_pre, self.fgd_msks_pre, self.fgd_nums, self.fbg_nums = \
            tf.map_fn(self.sample_rois_img, elems, dtype=(tf.float32, tf.int32, tf.float32, tf.int32, tf.int32, tf.int32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return self.fbg_rois, self.fbg_nums                       #和fbg_idxs对应
        
    
    def generate_roi_loss_img(self, elems=None):
        
        fbg_prbs_pre, fbg_prbs_pst, fgd_prds_pre, fbg_prds_pst, fgd_msks_pre, fbg_msks_pst, fgd_num, fbg_num = elems
        #####################计算损失######################
        ###Get Class Loss###
        fbg_prbs_pre = fbg_prbs_pre[0:fbg_num]                    #(M)
        fbg_prbs_pst = fbg_prbs_pst[0:fbg_num]                    #(M, C)
        roi_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fbg_prbs_pre, logits=fbg_prbs_pst) #和fbg_idxs对应
        roi_prbs_los = tf.reduce_sum(roi_prbs_los)
        ###Get Mask Loss###
        fgd_prbs_pre = fbg_prbs_pre[0:fgd_num]                    #(M0)
        fgd_msks_pre = fgd_msks_pre[0:fgd_num]                    #(M0, 21, 21)
        fgd_idxs     = tf.range(fgd_num)
        fgd_idxs     = tf.stack([fgd_idxs, fgd_prbs_pre], axis=-1)
        fgd_msks_pst = tf.gather_nd(fbg_msks_pst, fgd_idxs)       #(M0, 21, 21, 2) #(M, C, 21, 21, 2)
        roi_msks_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fgd_msks_pre, logits=fgd_msks_pst) #和fgd_idxs对应
        roi_msks_los = tf.reduce_mean(roi_msks_los, axis=[1, 2])  #reduce_mean还是reduce_sum
        roi_msks_los = tf.reduce_sum(roi_msks_los)
        ###Get Bbox loss###
        fgd_prds_pre = fgd_prds_pre[0:fgd_num]                    #(M0, 4)
        fgd_prds_pst = fbg_prds_pst[0:fgd_num]                    #(M0, 4)
        roi_prds_los = smooth_l1(1.0, fgd_prds_pst, fgd_prds_pre)
        #roi_prds_los = tf.reduce_mean(roi_prds_los, axis=1)      #reduce_mean还是reduce_sum
        roi_prds_los = tf.reduce_sum(roi_prds_los)
        return roi_prbs_los, roi_prds_los, roi_msks_los
        
        
    def generate_roi_loss(self, roi_prbs_pst=None, roi_prds_pst=None, roi_msks_pst=None):
        
        elems = [self.fbg_prbs_pre, roi_prbs_pst, self.fgd_prds_pre, roi_prds_pst, \
                 self.fgd_msks_pre, roi_msks_pst, self.fgd_nums, self.fbg_nums]
        roi_prbs_los, roi_prds_los, roi_msks_los = \
            tf.map_fn(self.generate_roi_loss_img, elems, dtype=(tf.float32, tf.float32, tf.float32),
                      parallel_iterations=10, back_prop=True, swap_memory=True, infer_shape=True)
        roi_prbs_los = tf.reduce_sum(roi_prbs_los)
        roi_prds_los = tf.reduce_sum(roi_prds_los)
        roi_msks_los = tf.reduce_sum(roi_msks_los)
        fgd_num      = tf.cast(tf.reduce_sum(self.fgd_nums), dtype=tf.float32)
        fbg_num      = tf.cast(tf.reduce_sum(self.fbg_nums), dtype=tf.float32)
        fgd_rat      = fgd_num / fbg_num
        roi_prbs_los = tf.cond(fgd_num>0, lambda: roi_prbs_los/fgd_num, lambda: tf.constant(0.0))
        roi_prds_los = tf.cond(fgd_num>0, lambda: roi_prds_los/fgd_num, lambda: tf.constant(0.0))
        roi_msks_los = tf.cond(fgd_num>0, lambda: roi_msks_los/fgd_num, lambda: tf.constant(0.0)) * 20.0
        return roi_prbs_los, roi_prds_los, roi_msks_los, fgd_rat