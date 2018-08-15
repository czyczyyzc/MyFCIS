import numpy as np
import tensorflow as tf
from .bbox import *

class RoisPoolingLayer(object):
    
    def __init__(self, roi_cls_num=21, img_shp=None):
        
        self.img_shp     = img_shp
        self.roi_cls_num = roi_cls_num
        self.roi_pol_num = 300
        self.roi_pol_siz = 21
        self.roi_grp_siz = 7
        self.roi_bin_siz = self.roi_pol_siz // self.roi_grp_siz
    
    def rois_pooling_img_vld(self, roi_scrs_pst=None, roi_prds_pst=None, rois=None):
        
        ###################Get Assemble####################
        #rois (M, 4)
        #roi_scrs_pst (7*7, H, W, C) roi_prds_pst (7*7, H, W, 4)
        p = self.roi_pol_siz
        g = self.roi_grp_siz
        b = self.roi_bin_siz
        roi_num  = tf.shape(rois)[0]
        roi_ycds = tf.cumsum(tf.concat([rois[:, 0::4], tf.tile((rois[:, 2::4]-rois[:, 0::4])/g, [1, g])], axis=-1), axis=-1)
        roi_xcds = tf.cumsum(tf.concat([rois[:, 1::4], tf.tile((rois[:, 3::4]-rois[:, 1::4])/g, [1, g])], axis=-1), axis=-1)
        roi_ymns = roi_ycds[:, :-1] #(M, 7)
        roi_ymxs = roi_ycds[:, 1: ] #(M, 7)
        roi_xmns = roi_xcds[:, :-1] #(M, 7)
        roi_xmxs = roi_xcds[:, 1: ] #(M, 7)
        roi_ycds = tf.stack([roi_ymns, roi_ymxs], axis=-1) #(M, 7, 2) (ymn, ymx)
        roi_xcds = tf.stack([roi_xmns, roi_xmxs], axis=-1) #(M, 7, 2) (xmn, xmx)
        roi_ycds = tf.reshape(tf.tile(roi_ycds, [1, 1, g]), [-1, g*g, 2])  #(M, 7,|7|,2) #(M, 7*7, 2) (ymn, ymx)
        roi_xcds = tf.reshape(tf.tile(roi_xcds, [1, g, 1]), [-1, g*g, 2])  #(M,|7|,7, 2) #(M, 7*7, 2) (xmn, xmx)
        roi_crds = tf.stack([roi_ycds[..., 0], roi_xcds[..., 0], 
                             roi_ycds[..., 1], roi_xcds[..., 1]], axis=-1) #(M, 7*7, 4) (ymn, xmn, ymx, xmx)
        roi_crds = tf.reshape(roi_crds, [-1, 4])     #(M*7*7, 4)
        img_leh  = np.concatenate([self.img_shp, self.img_shp], axis=0) - 1.0
        roi_crds = roi_crds / img_leh #归一化坐标     #(M*7*7, 4)
        roi_idxs = tf.tile(tf.range(g*g), [roi_num]) #(M*7*7)
        ##########这样pooling真的对吗?#############
        roi_scrs_pst  = tf.image.crop_and_resize(roi_scrs_pst, roi_crds, roi_idxs, [b, b], method='bilinear') #(M*7*7, 3, 3, C)
        roi_prds_pst  = tf.image.crop_and_resize(roi_prds_pst, roi_crds, roi_idxs, [b, b], method='bilinear') #(M*7*7, 3, 3, 4)
        roi_scrs_pst  = tf.transpose(tf.reshape(roi_scrs_pst, [roi_num, g, g, b, b, self.roi_cls_num*2]), \
                                     [0, 1, 3, 2, 4, 5])              #(M, 7, 3, 7, 3, C)
        roi_prds_pst  = tf.transpose(tf.reshape(roi_prds_pst, [roi_num, g, g, b, b, 4]), \
                                     [0, 1, 3, 2, 4, 5])              #(M, 7, 3, 7, 3, 4)
        roi_scrs_pst  = tf.reshape(roi_scrs_pst, [roi_num, p, p, self.roi_cls_num*2]) #(M, 21, 21, C)
        roi_prds_pst  = tf.reshape(roi_prds_pst, [roi_num, p, p, 4]) #(M, 21, 21, 4)
        ###Get Mask Predictions####
        roi_msks_pst  = tf.reshape(roi_scrs_pst, [roi_num, p, p, self.roi_cls_num, 2]) #(M, 21, 21, C//2, 2)
        roi_msks_pst  = tf.transpose(roi_msks_pst, [0, 3, 1, 2, 4])      #(M, C//2, 21, 21, 2)
        ###Get Class Predictions###
        roi_scrs_pst0 = roi_scrs_pst[..., 0::2] #inside score map  #(M, 21, 21, C//2)
        roi_scrs_pst1 = roi_scrs_pst[..., 1::2] #outside score map #(M, 21, 21, C//2)
        roi_scrs_pst  = tf.maximum(roi_scrs_pst0, roi_scrs_pst1)   #(M, 21, 21, C//2)
        roi_scrs_pst  = tf.reduce_mean(roi_scrs_pst, axis=[1, 2])  #(M, C//2)
        ###Get Bbox Predicitons###
        roi_prds_pst  = tf.reduce_mean(roi_prds_pst, axis=[1, 2])  #(M, 4)
        return roi_scrs_pst, roi_prds_pst, roi_msks_pst
        
    #如果是Faster RCNN，就用roi_imxs把每幅图片的rois合并起来!!!    
    def rois_pooling_img(self, elems=None):
        
        roi_scrs_pst, roi_prds_pst, rois, roi_num = elems
        rois = rois[0:roi_num]
        '''
        assert_op = tf.Assert(tf.size(rois)>0, [tf.shape(roi_scrs_pst)[3], tf.shape(roi_prds_pst)[3], rois, roi_num], summarize=100)
        with tf.control_dependencies([assert_op]):
            rois = tf.identity(rois)
        '''
        roi_scrs_pst, roi_prds_pst, roi_msks_pst = self.rois_pooling_img_vld(roi_scrs_pst, roi_prds_pst, rois)
        paddings = [[0, self.roi_pol_num-roi_num], [0, 0]]
        roi_scrs_pst = tf.pad(roi_scrs_pst, paddings, "CONSTANT")
        roi_prds_pst = tf.pad(roi_prds_pst, paddings, "CONSTANT")
        paddings = [[0, self.roi_pol_num-roi_num], [0, 0], [0, 0], [0, 0], [0, 0]]
        roi_msks_pst = tf.pad(roi_msks_pst, paddings, "CONSTANT")
        return roi_scrs_pst, roi_prds_pst, roi_msks_pst
        

    def rois_pooling(self, roi_scrs_pst=None, roi_prds_pst=None, rois=None, roi_nums=None):
        
        elems = [roi_scrs_pst, roi_prds_pst, rois, roi_nums]
        roi_scrs_pst, roi_prds_pst, roi_msks_pst = \
            tf.map_fn(self.rois_pooling_img, elems, dtype=(tf.float32, tf.float32, tf.float32),
                      parallel_iterations=10, back_prop=True, swap_memory=True, infer_shape=True)
        return roi_scrs_pst, roi_prds_pst, roi_msks_pst