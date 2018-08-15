import numpy as np
import tensorflow as tf
from .bbox import *

class ProposalsLayer(object):
    
    def __init__(self, mod_tra=True, rpns=None, roi_cls_num=None, img_shp=None):
        
        self.rpns        = rpns
        self.img_shp     = img_shp
        self.mod_tra     = mod_tra
        self.roi_cls_num = roi_cls_num
        self.roi_siz_min = 15
        self.roi_prb_min = 0.5
        self.roi_nms_pre = None
        self.roi_nms_pst = 200   #2000
        self.roi_nms_max = 0.4   #0.2
    
    def generate_rois_img(self, elems=None):
        
        rpn_prbs_pst, rpn_prds_pst = elems
        """
        if not is_train:
            self.roi_nms_pre = int(self.roi_nms_pre / 1) #12000
            self.roi_nms_pst = int(self.roi_nms_pst / 2) #1000
        """
        #取出最佳类的预测值
        roi_clss = tf.argmax(rpn_prbs_pst, axis=1)
        roi_clss = tf.cast(roi_clss, tf.int32)
        roi_prbs = tf.reduce_max(rpn_prbs_pst, axis=1)
        
        #设置一个rpn索引，避免大量的gather操作(prds、msks)，节省内存，提升速度
        kep_idxs = tf.range(tf.shape(self.rpns)[0])
        #剔除背景roi
        idxs     = tf.where(roi_clss>0)
        kep_idxs = tf.gather_nd(kep_idxs, idxs)
        roi_clss = tf.gather_nd(roi_clss, idxs)
        roi_prbs = tf.gather_nd(roi_prbs, idxs)
        #剔除得分较低的roi
        if self.roi_prb_min is not None:
            idxs = tf.where(roi_prbs>=self.roi_prb_min)
            kep_idxs = tf.gather_nd(kep_idxs, idxs)
            roi_clss = tf.gather_nd(roi_clss, idxs)
            roi_prbs = tf.gather_nd(roi_prbs, idxs)
        #进一步剔除过多的roi
        if self.roi_nms_pre is not None:
            roi_nms_pre = tf.minimum(self.roi_nms_pre, tf.shape(kep_idxs)[0])
            roi_prbs, idxs = tf.nn.top_k(roi_prbs, k=roi_nms_pre, sorted=True)
            kep_idxs = tf.gather(kep_idxs, idxs)
            roi_clss = tf.gather(roi_clss, idxs)
        #根据kep_idxs进行剩余的gather操作
        rpns         = tf.gather(self.rpns,    kep_idxs)
        #kep_idxs    = tf.stack([kep_idxs, roi_clss], axis=-1)  #如果roi的预测是定类的话要加上这句
        rpn_prds_pst = tf.gather(rpn_prds_pst, kep_idxs)
        #还原出roi以进行后续的滤除
        rois = bbox_transform_inv(rpns, rpn_prds_pst)
        rois = bbox_clip(rois, [0.0, 0.0, self.img_shp[0]-1.0, self.img_shp[1]-1.0])
        #剔除过小的roi
        idxs = bbox_filter(rois, self.roi_siz_min)
        rois = tf.gather_nd(rois, idxs)
        roi_clss = tf.gather_nd(roi_clss, idxs)
        roi_prbs = tf.gather_nd(roi_prbs, idxs)
        #做逐类的nms
        rois_kep     = tf.zeros(dtype=tf.float32, shape=[0, 4])
        roi_clss_kep = tf.zeros(dtype=tf.int32,   shape=[0])
        roi_prbs_kep = tf.zeros(dtype=tf.float32, shape=[0])
        
        def cond(i, rois, roi_clss, roi_prbs, rois_kep, roi_clss_kep, roi_prbs_kep):
            c = tf.less(i, self.roi_cls_num)
            return c

        def body(i, rois, roi_clss, roi_prbs, rois_kep, roi_clss_kep, roi_prbs_kep):
            #选出对应类的rois
            idxs = tf.where(tf.equal(roi_clss, i))
            rois_cls = tf.gather_nd(rois, idxs)
            roi_clss_cls = tf.gather_nd(roi_clss, idxs)
            roi_prbs_cls = tf.gather_nd(roi_prbs, idxs)
            #进行非极大值抑制操作
            idxs = tf.image.non_max_suppression(rois_cls, roi_prbs_cls, self.roi_nms_pst, self.roi_nms_max)
            rois_cls = tf.gather(rois_cls, idxs)
            roi_clss_cls = tf.gather(roi_clss_cls, idxs)
            roi_prbs_cls = tf.gather(roi_prbs_cls, idxs)
            # 保存结果
            rois_kep = tf.concat([rois_kep, rois_cls], axis=0)
            roi_clss_kep = tf.concat([roi_clss_kep, roi_clss_cls], axis=0)
            roi_prbs_kep = tf.concat([roi_prbs_kep, roi_prbs_cls], axis=0)
            return [i+1, rois, roi_clss, roi_prbs, rois_kep, roi_clss_kep, roi_prbs_kep]

        i = tf.constant(1) #要剔除背景类
        [i, rois, roi_clss, roi_prbs, rois_kep, roi_clss_kep, roi_prbs_kep] = \
            tf.while_loop(cond, body, loop_vars=[i, rois, roi_clss, roi_prbs, rois_kep, roi_clss_kep, roi_prbs_kep], \
                          shape_invariants=[i.get_shape(), rois.get_shape(), roi_clss.get_shape(), roi_prbs.get_shape(), \
                                            tf.TensorShape([None, 4]), tf.TensorShape([None]), tf.TensorShape([None])], \
                          parallel_iterations=10, back_prop=False, swap_memory=False)
            
        roi_num = tf.minimum(self.roi_nms_pst, tf.shape(rois_kep)[0])
        roi_prbs, idxs = tf.nn.top_k(roi_prbs_kep, k=roi_num, sorted=True)
        rois = tf.gather(rois_kep, idxs)
        roi_clss = tf.gather(roi_clss_kep, idxs)
        
        paddings = [[0, self.roi_nms_pst-roi_num], [0, 0]]
        rois = tf.pad(rois, paddings, "CONSTANT")
        paddings = [[0, self.roi_nms_pst-roi_num]]
        roi_clss = tf.pad(roi_clss, paddings, "CONSTANT")
        roi_prbs = tf.pad(roi_prbs, paddings, "CONSTANT")
        return rois, roi_clss, roi_prbs, roi_num
    
    def generate_rois(self, rpn_prbs_pst=None, rpn_prds_pst=None):
        
        elems = [rpn_prbs_pst, rpn_prds_pst]
        rois, roi_clss, roi_prbs, roi_nums = \
            tf.map_fn(self.generate_rois_img, elems, dtype=(tf.float32, tf.int32, tf.float32, tf.int32),
                      parallel_iterations=10, back_prop=False, 
                      swap_memory=False, infer_shape=True)
        return rois, roi_clss, roi_prbs, roi_nums
