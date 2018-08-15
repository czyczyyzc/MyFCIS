import cv2
import numpy as np
import tensorflow as tf
from Mybase.comp_utils import *
from Mybase.losses import *
from .bbox import *

class AnchorsTargetLayer(object):
    
    def __init__(self, rpns=None, rpn_cls_num=None):
        
        self.rpns        = rpns
        self.rpn_cls_num = rpn_cls_num
        self.rpn_fbg_num = 512  #256                    #1024
        self.rpn_fgd_frc = 0.25 #0.5  #rpn_fg_fraction  #0.25
        self.rpn_fgd_ovp = 0.5  #0.7  #rpn_fg_overlap   #0.5
        self.rpn_bgd_ovp = 0.5  #0.3  #rpn_bg_overlap   #0.5
        self.rpn_crw_max = 0.001      #crowd_thresh
        self.rpn_fgd_num = int(self.rpn_fbg_num * self.rpn_fgd_frc)
        self.rpn_bfg_rat = 1.0 / self.rpn_fgd_frc - 1.0
        
    def generate_rpn_loss_img(self, elems=None):
        
        rpn_prbs_pst, rpn_prds_pst, gbxs, gbx_num = elems
        
        gbxs     = gbxs[0:gbx_num] #最后一个是无效边框
        crw_idxs = tf.where(gbxs[:, 4]< 0) #crowd_idx
        ncw_idxs = tf.where(gbxs[:, 4]>=0) #no_crowd_idx #最后一个是无效边框
        gbxs_tmp = tf.gather_nd(gbxs, crw_idxs)
        gbxs     = tf.gather_nd(gbxs, ncw_idxs)
        
        ###################计算CROWD IOU##################
        #rpn_iscs = bbox_intersects(self.rpns, gbxs_tmp[:, 0:4])
        #max_iscs = tf.reduce_max(rpn_iscs, axis=1)
        rpn_ovps = bbox_overlaps(self.rpns, gbxs_tmp[:, 0:4])
        max_ovps = tf.reduce_max(rpn_ovps, axis=1)
        ncw_msks = max_ovps < self.rpn_crw_max

        #################计算NO CROWD IOU#################
        rpn_ovps = bbox_overlaps(self.rpns, gbxs[:, 0:4])
        rpn_amxs = tf.argmax(rpn_ovps, axis=1)
        max_ovps = tf.reduce_max(rpn_ovps, axis=1)
        
        rpn_gtas = tf.zeros(shape=[tf.shape(self.rpns)[0]], dtype=tf.int64) - 2
        
        ######################选前景######################
        fgd_idxs0 = tf.where(max_ovps>=self.rpn_fgd_ovp)[:, 0]  #无效边框不影响
        fgd_idxs1 = tf.argmax(rpn_ovps, axis=0)                 #最后一个是无效边框
        fgd_idxs1 = fgd_idxs1[:-1]                              #剔除最后一个无效边框匹配的anchor
        fgd_idxs  = tf.concat([fgd_idxs0, fgd_idxs1], axis=0)
        fgd_idxs, idxs = tf.unique(fgd_idxs)
        fgd_idxs  = tf.expand_dims(fgd_idxs, axis=-1)
        #fgd_idxs = tf.where(max_ovps>=self.rpn_fgd_ovp)
        fgd_num   = tf.shape(fgd_idxs)[0]
        fgd_num   = tf.minimum(fgd_num, self.rpn_fgd_num)
        fgd_idxs  = tf.random_shuffle(fgd_idxs)[:fgd_num]
        rpn_amxs  = tf.gather_nd(rpn_amxs, fgd_idxs)
        rpn_gtas  = tensor_update(rpn_gtas, fgd_idxs, rpn_amxs)  #这里的fgd_idxs可以乱序

        ######################选背景######################
        bgd_idxs     = tf.where(tf.logical_and(tf.logical_and(max_ovps<self.rpn_bgd_ovp, tf.equal(rpn_gtas, -2)), ncw_msks))
        bgd_num      = tf.shape(bgd_idxs)[0]
        #Hard Negative Mining
        bgd_num      = tf.minimum(bgd_num, tf.cast(tf.cast(fgd_num, tf.float32)*self.rpn_bfg_rat, tf.int32))
        rpn_prbs_tmp = tf.gather_nd(rpn_prbs_pst, bgd_idxs)      #和bgd_idxs对应
        rpn_prbs_tmp = tf.stop_gradient(rpn_prbs_tmp)
        rpn_prbs_pre = tf.zeros(shape=[tf.shape(rpn_prbs_tmp)[0]], dtype=tf.int32)
        rpn_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_prbs_pre, logits=rpn_prbs_tmp) #和bgd_idxs对应
        _, idxs      = tf.nn.top_k(rpn_prbs_los, k=bgd_num, sorted=False) #Hard Negative Mining
        bgd_idxs     = tf.gather(bgd_idxs, idxs)
        rpn_gtas     = tensor_update(rpn_gtas, bgd_idxs, -1)     #这里的bgd_idxs可以乱序
        '''
        #bgd_num     = tf.minimum(bgd_num, self.rpn_fbg_num-fgd_num)
        bgd_num      = tf.minimum(bgd_num, tf.cast(tf.cast(fgd_num, tf.float32)*self.rpn_bfg_rat, tf.int32))
        bgd_idxs     = tf.random_shuffle(bgd_idxs)[:bgd_num]
        rpn_gtas     = tensor_update(rpn_gtas, bgd_idxs, -1)     #这里的bgd_idxs可以乱序
        '''
        ####################整合前背景#####################
        fgd_idxs = tf.where(rpn_gtas>=0)
        bgd_idxs = tf.where(tf.equal(rpn_gtas, -1))
        fbg_idxs = tf.concat([fgd_idxs, bgd_idxs], axis=0)       #确保fgd_idxs/bgd_idxs/fbg_idxs三者的顺序相对应
        fgd_gtas = tf.gather_nd(rpn_gtas, fgd_idxs)              #和fgd_idxs对应
        fbg_gtas = tf.gather_nd(rpn_gtas, fbg_idxs)              #和fbg_idxs对应
        fgd_num  = tf.shape(fgd_idxs)[0]
        fbg_num  = tf.shape(fbg_idxs)[0]
        ###################计算分类损失####################
        '''
        fgd_prbs_pre = tf.gather(gbxs[:,   4], fgd_gtas)         #和fgd_idxs对应
        fgd_prbs_pre = tf.cast(fgd_prbs_pre, dtype=tf.int32)
        bgd_prbs_pre = tf.zeros(shape=[tf.shape(bgd_idxs)[0]], dtype=tf.int32)
        fbg_prbs_pre = tf.concat([fgd_prbs_pre, bgd_prbs_pre], axis=0) #和fbg_idxs对应
        '''
        fbg_prbs_pre = tf.cast(fbg_gtas>=0, dtype=tf.int32)
        fbg_prbs_pst = tf.gather_nd(rpn_prbs_pst, fbg_idxs)      #和fbg_idxs对应
        rpn_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fbg_prbs_pre, logits=fbg_prbs_pst)
        rpn_prbs_los = tf.reduce_sum(rpn_prbs_los)
        
        ###################计算回归损失####################
        fgd_prds_pre = tf.gather(gbxs[:, 0:4], fgd_gtas)         #和fgd_idxs对应
        fgd_rpns     = tf.gather_nd(self.rpns, fgd_idxs)         #和fgd_idxs对应
        fgd_prds_pre = bbox_transform(fgd_rpns, fgd_prds_pre)    #和fgd_idxs对应
        fgd_prds_pst = tf.gather_nd(rpn_prds_pst, fgd_idxs)      #和fgd_idxs对应
        rpn_prds_los = smooth_l1(1.0, fgd_prds_pst, fgd_prds_pre)
        #rpn_prds_los = tf.reduce_mean(rpn_prds_los, axis=1)     #reduce_mean还是reduce_sum
        rpn_prds_los = tf.reduce_sum(rpn_prds_los)
        return rpn_prbs_los, rpn_prds_los, fgd_num, fbg_num
        

    def generate_rpn_loss(self, rpn_prbs_pst=None, rpn_prds_pst=None, gbxs=None, gbx_nums=None):
        
        elems = [rpn_prbs_pst, rpn_prds_pst, gbxs, gbx_nums]
        rpn_prbs_los, rpn_prds_los, fgd_num, fbg_num = \
            tf.map_fn(self.generate_rpn_loss_img, elems, dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
                      parallel_iterations=10, back_prop=True, swap_memory=True, infer_shape=True)
        rpn_prbs_los = tf.reduce_sum(rpn_prbs_los)
        rpn_prds_los = tf.reduce_sum(rpn_prds_los)
        fgd_num      = tf.cast(tf.reduce_sum(fgd_num), dtype=tf.float32)
        fbg_num      = tf.cast(tf.reduce_sum(fbg_num), dtype=tf.float32)
        fgd_rat      = fgd_num / fbg_num
        rpn_prbs_los = tf.cond(fgd_num>0, lambda: rpn_prbs_los/fgd_num, lambda: tf.constant(0.0))
        rpn_prds_los = tf.cond(fgd_num>0, lambda: rpn_prds_los/fgd_num, lambda: tf.constant(0.0))
        return rpn_prbs_los, rpn_prds_los, fgd_rat
