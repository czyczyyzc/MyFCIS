import numpy as np
import tensorflow as tf
from .bbox import *

###########For RCNN##########
class ProposalsLayer(object):
    
    def __init__(self, mod_tra=True, rpns=None, img_shp=None):
        
        self.rpns        = rpns
        self.img_shp     = img_shp
        self.mod_tra     = mod_tra
        self.roi_siz_min = 16 * 6
        self.roi_prb_min = None                            #0.5
        self.roi_nms_pre = 12000 if self.mod_tra else 6000 #None
        self.roi_nms_pst =  2000 if self.mod_tra else 300  #200
        self.roi_nms_max = 0.7                             #0.4

    def generate_rois_img(self, elems=None):
        
        rpn_prbs_pst, rpn_prds_pst = elems
        roi_prbs = rpn_prbs_pst[:, -1] #for RCNN
        
        #设置一个rpn索引，避免大量的gather操作(prds、msks)，节省内存，提升速度
        kep_idxs = tf.range(tf.shape(self.rpns)[0])
        
        #剔除得分较低的roi
        if self.roi_prb_min is not None:
            idxs     = tf.where(roi_prbs>=self.roi_prb_min)
            kep_idxs = tf.gather_nd(kep_idxs, idxs)
            roi_prbs = tf.gather_nd(roi_prbs, idxs)
        #进一步剔除过多的roi
        if self.roi_nms_pre is not None:
            roi_nms_pre    = tf.minimum(self.roi_nms_pre, tf.shape(kep_idxs)[0])
            roi_prbs, idxs = tf.nn.top_k(roi_prbs, k=roi_nms_pre, sorted=True)
            kep_idxs       = tf.gather(kep_idxs, idxs)
        #根据kep_idxs进行剩余的gather操作
        rpns         = tf.gather(self.rpns,    kep_idxs)
        rpn_prds_pst = tf.gather(rpn_prds_pst, kep_idxs)
        #还原出roi以进行后续的滤除
        rois     = bbox_transform_inv(rpns, rpn_prds_pst)
        rois     = bbox_clip(rois, [0.0, 0.0, self.img_shp[0]-1.0, self.img_shp[1]-1.0])
        #剔除过小的roi
        idxs     = bbox_filter(rois, self.roi_siz_min)
        rois     = tf.gather_nd(rois,     idxs)
        roi_prbs = tf.gather_nd(roi_prbs, idxs)
        #进行非极大值抑制操作
        idxs     = tf.image.non_max_suppression(rois, roi_prbs, self.roi_nms_pst, self.roi_nms_max)
        rois     = tf.gather(rois,     idxs)
        roi_prbs = tf.gather(roi_prbs, idxs)
        roi_num  = tf.shape(rois)[0]
        #roi_num = tf.Print(roi_num, [roi_num], message=None, first_n=None, summarize=None)
        paddings = [[0, self.roi_nms_pst-roi_num], [0, 0]]
        rois = tf.pad(rois, paddings, "CONSTANT")
        paddings = [[0, self.roi_nms_pst-roi_num]]
        roi_prbs = tf.pad(roi_prbs, paddings, "CONSTANT")
        return rois, roi_prbs, roi_num
        
    def generate_rois(self, rpn_prbs_pst=None, rpn_prds_pst=None):
        
        elems = [rpn_prbs_pst, rpn_prds_pst]
        rois, roi_prbs, roi_nums = \
            tf.map_fn(self.generate_rois_img, elems, dtype=(tf.float32, tf.float32, tf.int32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return rois, roi_prbs, roi_nums
