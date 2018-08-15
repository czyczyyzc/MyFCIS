import numpy as np
import tensorflow as tf
from .bbox import *

class BBoxesLayer(object):
    
    def __init__(self, mod_tra=True, box_cls_num=None, img_shp=None):
        
        self.img_shp     = img_shp
        self.mod_tra     = mod_tra
        self.box_cls_num = box_cls_num
        self.box_pol_num = 300
        self.box_siz_min = 16
        self.box_prb_min = 0.5
        self.box_nms_pre = None
        self.box_nms_pst = 100   #200
        self.box_nms_max = 0.2   #0.2
        self.box_msk_min = 0.3
        self.box_msk_siz = [21, 21]
        self.meg_ovp_min = 0.4
    
    def generate_boxs_img(self, elems=None):
        #奇葩!和以前一点不一样!!! 这里看成是对ROI的分类、打分、mask预测，而不是对由ROI产生的BOX的，因此不对ROI做剔除!!!
        #按道理说，这么做比之前的做法合理!!!
        #(M, 4) #(M, C) #(M, 4) #(M, C, 21, 21, 2)
        rois, roi_prbs_pst, roi_prds_pst, roi_msks_pst, roi_num = elems
        rois         = rois        [0:roi_num]
        roi_prbs_pst = roi_prbs_pst[0:roi_num]
        roi_prds_pst = roi_prds_pst[0:roi_num]
        roi_msks_pst = roi_msks_pst[0:roi_num]
        
        roi_clss = tf.argmax(roi_prbs_pst, axis=-1)
        roi_clss = tf.cast(roi_clss, tf.int32)
        roi_prbs = tf.reduce_max(roi_prbs_pst, axis=-1)
        
        idxs     = tf.range(roi_num)
        idxs     = tf.stack([idxs, roi_clss], axis=-1)
        roi_msks = tf.gather_nd(roi_msks_pst, idxs) #(M, 21, 21, 2)
        roi_msks = tf.nn.softmax(roi_msks, axis=-1)    #(M, 21, 21, 2)
        roi_msks = roi_msks[..., 1]                 #(M, 21, 21)
        
        #roi_prds_pst = tf.gather_nd(roi_prds_pst, idxs) #如果roi的预测是定类的话要换上这段
        boxs     = bbox_transform_inv(rois, roi_prds_pst)
        boxs     = bbox_clip(boxs, [0.0, 0.0, self.img_shp[0]-1.0, self.img_shp[1]-1.0])
        #剔除过小的box
        idxs     = bbox_filter(boxs, self.box_siz_min)
        boxs     = tf.gather_nd(boxs, idxs)
        box_num  = tf.shape(boxs)[0]
        #padding
        paddings = [[0, self.box_pol_num-box_num], [0, 0]]
        boxs     = tf.pad(boxs,     paddings, "CONSTANT")
        paddings = [[0, self.box_pol_num-roi_num]]
        roi_clss = tf.pad(roi_clss, paddings, "CONSTANT")
        roi_prbs = tf.pad(roi_prbs, paddings, "CONSTANT")
        paddings = [[0, self.box_pol_num-roi_num], [0, 0], [0, 0]]
        roi_msks = tf.pad(roi_msks, paddings, "CONSTANT")
        return boxs, box_num, roi_clss, roi_prbs, roi_msks
        
    def generate_boxs(self, rois=None, roi_prbs_pst=None, roi_prds_pst=None, roi_msks_pst=None, roi_nums=None):
        
        elems = [rois, roi_prbs_pst, roi_prds_pst, roi_msks_pst, roi_nums]
        boxs, box_nums, roi_clss, roi_prbs, roi_msks = \
            tf.map_fn(self.generate_boxs_img, elems, dtype=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return boxs, box_nums, roi_clss, roi_prbs, roi_msks
    
    def concat_boxs_img(self, elems=None):
        
        rois, roi_clss, roi_prbs, roi_msks, roi_num, boxs, box_clss, box_prbs, box_msks, box_num = elems
        rois     = rois    [0:roi_num]                     #(M0, 4)
        roi_clss = roi_clss[0:roi_num]                     #(M0)
        roi_prbs = roi_prbs[0:roi_num]                     #(M0)
        roi_msks = roi_msks[0:roi_num]                     #(M0, 21, 21)
        boxs     = boxs    [0:box_num]                     #(M1, 4)
        box_clss = box_clss[0:box_num]                     #(M1)
        box_prbs = box_prbs[0:box_num]                     #(M1)
        box_msks = box_msks[0:box_num]                     #(M1, 21, 21)
        box_num  = roi_num + box_num
        boxs     = tf.concat([boxs, rois], axis=0)         #(M, 4)
        box_clss = tf.concat([box_clss, roi_clss], axis=0) #(M)
        box_prbs = tf.concat([box_prbs, roi_prbs], axis=0) #(M)
        box_msks = tf.concat([box_msks, roi_msks], axis=0) #(M, 21, 21)
        paddings = [[0, self.box_pol_num*2-box_num], [0, 0]]
        boxs     = tf.pad(boxs,     paddings, "CONSTANT")
        paddings = [[0, self.box_pol_num*2-box_num]]
        box_clss = tf.pad(box_clss, paddings, "CONSTANT")
        box_prbs = tf.pad(box_prbs, paddings, "CONSTANT")
        paddings = [[0, self.box_pol_num*2-box_num], [0, 0], [0, 0]]
        box_msks = tf.pad(box_msks, paddings, "CONSTANT")
        return boxs, box_clss, box_prbs, box_msks, box_num
    
    def concat_boxs(self, rois, roi_clss, roi_prbs, roi_msks, roi_nums, boxs, box_clss, box_prbs, box_msks, box_nums):
        
        elems = [rois, roi_clss, roi_prbs, roi_msks, roi_nums, boxs, box_clss, box_prbs, box_msks, box_nums]
        boxs, box_clss, box_prbs, box_msks, box_nums = \
            tf.map_fn(self.concat_boxs_img, elems, dtype=(tf.float32, tf.int32, tf.float32, tf.float32, tf.int32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return boxs, box_clss, box_prbs, box_msks, box_nums
    
    def merge_boxs_img(self, elems=None):
        
        boxs, box_clss, box_prbs, box_msks, box_num = elems
        boxs     = boxs    [0:box_num] #(M, 4)
        box_clss = box_clss[0:box_num] #(M)
        box_prbs = box_prbs[0:box_num] #(M)
        box_msks = box_msks[0:box_num] #(M, 21, 21)
        
        #设置一个box索引，避免大量的gather操作(prds、msks)，节省内存，提升速度
        kep_idxs = tf.range(box_num)
        #剔除背景box
        idxs     = tf.where(box_clss>0)
        kep_idxs = tf.gather_nd(kep_idxs, idxs)
        box_clss = tf.gather_nd(box_clss, idxs)
        box_prbs = tf.gather_nd(box_prbs, idxs)
        #剔除得分较低的box
        if self.box_prb_min is not None:
            idxs     = tf.where(box_prbs>=self.box_prb_min)
            kep_idxs = tf.gather_nd(kep_idxs, idxs)
            box_clss = tf.gather_nd(box_clss, idxs)
            box_prbs = tf.gather_nd(box_prbs, idxs)
        #进一步剔除过多的box
        if self.box_nms_pre is not None:
            box_nms_pre = tf.minimum(self.box_nms_pre, tf.shape(kep_idxs)[0])
            box_prbs, idxs = tf.nn.top_k(box_prbs, k=box_nms_pre, sorted=True)
            kep_idxs = tf.gather(kep_idxs, idxs)
            box_clss = tf.gather(box_clss, idxs)
        #根据kep_idxs进行剩余的gather操作
        boxs         = tf.gather(boxs,     kep_idxs)
        box_msks     = tf.gather(box_msks, kep_idxs)
        #做逐类的nms
        boxs_tmp     = tf.zeros(dtype=tf.float32, shape=[0, 4])
        box_clss_kep = tf.zeros(dtype=tf.int32,   shape=[0])
        box_prbs_kep = tf.zeros(dtype=tf.float32, shape=[0])
        def cond0(i, boxs, box_clss, box_prbs, boxs_tmp, box_clss_kep, box_prbs_kep):
            c = tf.less(i, self.box_cls_num)
            return c

        def body0(i, boxs, box_clss, box_prbs, boxs_tmp, box_clss_kep, box_prbs_kep):
            #选出对应类的rois
            idxs         = tf.where(tf.equal(box_clss, i))
            boxs_cls     = tf.gather_nd(boxs,     idxs)
            box_clss_cls = tf.gather_nd(box_clss, idxs)
            box_prbs_cls = tf.gather_nd(box_prbs, idxs)
            #进行非极大值抑制操作
            idxs         = tf.image.non_max_suppression(boxs_cls, box_prbs_cls, self.box_nms_pst, self.box_nms_max)
            boxs_cls     = tf.gather(boxs_cls,     idxs)
            box_clss_cls = tf.gather(box_clss_cls, idxs)
            box_prbs_cls = tf.gather(box_prbs_cls, idxs)
            #保存结果
            boxs_tmp     = tf.concat([boxs_tmp,     boxs_cls    ], axis=0)
            box_clss_kep = tf.concat([box_clss_kep, box_clss_cls], axis=0)
            box_prbs_kep = tf.concat([box_prbs_kep, box_prbs_cls], axis=0)
            return [i+1, boxs, box_clss, box_prbs, boxs_tmp, box_clss_kep, box_prbs_kep]
        
        i = tf.constant(1) #要剔除背景类
        [i, boxs, box_clss, box_prbs, boxs_tmp, box_clss_kep, box_prbs_kep] = \
            tf.while_loop(cond0, body0, \
                          loop_vars=[i, boxs, box_clss, box_prbs, boxs_tmp, box_clss_kep, box_prbs_kep], \
                          shape_invariants=[i.get_shape(), boxs.get_shape(), box_clss.get_shape(), box_prbs.get_shape(), \
                                            tf.TensorShape([None, 4]), tf.TensorShape([None]), tf.TensorShape([None])], \
                          parallel_iterations=10, back_prop=False, swap_memory=True)
        
        box_num      = tf.minimum(self.box_nms_pst, tf.shape(boxs_tmp)[0])
        box_prbs_kep, idxs = tf.nn.top_k(box_prbs_kep, k=box_num, sorted=False)
        boxs_tmp     = tf.gather(boxs_tmp,     idxs)
        box_clss_kep = tf.gather(box_clss_kep, idxs)
        
        #融合mask
        boxs_kep     = tf.zeros(dtype=tf.float32, shape=[0, 4])
        box_msks_kep = tf.zeros(dtype=tf.float32, shape=[0]+self.box_msk_siz)
        box_idxs_kep = tf.zeros(dtype=tf.int64,   shape=[0, 1])  #并行计算要保证对应关系
        def cond1(i, boxs, box_clss, box_prbs, box_msks, boxs_tmp, box_clss_kep, boxs_kep, box_msks_kep, box_idxs_kep):
            c = tf.less(i, self.box_cls_num)
            return c

        def body1(i, boxs, box_clss, box_prbs, box_msks, boxs_tmp, box_clss_kep, boxs_kep, box_msks_kep, box_idxs_kep):
            #选出对应类的boxs
            idxs         = tf.where(tf.equal(box_clss, i))
            boxs_cls     = tf.gather_nd(boxs,     idxs)
            box_prbs_cls = tf.gather_nd(box_prbs, idxs)
            box_msks_cls = tf.gather_nd(box_msks, idxs)
            
            box_idxs_cls = tf.where(tf.equal(box_clss_kep, i))   #选出对应类别的box和其相应的idxs
            boxs_run     = tf.gather_nd(boxs_tmp, box_idxs_cls)
            
            boxs_hld     = tf.zeros(dtype=tf.float32, shape=[0, 4])
            box_msks_hld = tf.zeros(dtype=tf.float32, shape=[0]+self.box_msk_siz)
            box_idxs_hld = tf.zeros(dtype=tf.int32,   shape=[0]) #并行计算要保证对应关系
            def cond(j, boxs_run, boxs_cls, box_prbs_cls, box_msks_cls, boxs_hld, box_msks_hld, box_idxs_hld):
                c = tf.less(j, tf.shape(boxs_run)[0])
                return c
            
            def body(j, boxs_run, boxs_cls, box_prbs_cls, box_msks_cls, boxs_hld, box_msks_hld, box_idxs_hld):
                box_run      = boxs_run[j][tf.newaxis, :]
                box_ovps     = bbox_overlaps(boxs_cls, box_run)     #(M0, 1)
                box_ovps     = tf.squeeze(box_ovps, axis=[1])       #(M0)
                idxs         = tf.where(box_ovps>=self.meg_ovp_min)
                boxs_meg     = tf.gather_nd(boxs_cls,     idxs)     #(M1, 4)
                box_prbs_meg = tf.gather_nd(box_prbs_cls, idxs)     #(M1)
                box_msks_meg = tf.gather_nd(box_msks_cls, idxs)     #(M1, 21, 21)
                '''
                assert_op = tf.Assert(tf.size(box_msks_meg)>0, [j, tf.shape(boxs_run)[0], box_ovps, box_run, boxs_cls, \
                                                                idxs, boxs_meg, box_prbs_meg, box_msks_meg], summarize=100)
                with tf.control_dependencies([assert_op]):
                    box_msks_meg = tf.identity(box_msks_meg)
                '''
                box_wgts     = box_prbs_meg / tf.reduce_sum(box_prbs_meg)
                #boxs_meg    = bbox_clip(boxs_meg, [0, 0, self.img_shp[0]-1.0, self.img_shp[1]-1.0]) #之前还原出boxs时已经clip了
                boxs_meg     = tf.cast(tf.round(boxs_meg), dtype=tf.int32)
                box_ymns, box_xmns, box_ymxs, box_xmxs = tf.split(boxs_meg, 4, axis=-1)
                box_hgts     = box_ymxs - box_ymns + 1
                box_wdhs     = box_xmxs - box_xmns + 1
                box_lehs     = tf.concat([box_hgts, box_wdhs], axis=-1)
                box_ymn      = tf.reduce_min(box_ymns)
                box_xmn      = tf.reduce_min(box_xmns)
                box_ymx      = tf.reduce_max(box_ymxs)
                box_xmx      = tf.reduce_max(box_xmxs)
                pads_hgt_fnt = box_ymns - box_ymn
                pads_wdh_fnt = box_xmns - box_xmn
                pads_hgt_bak = box_ymx - box_ymxs
                pads_wdh_bak = box_xmx - box_xmxs
                pads_hgt     = tf.concat([pads_hgt_fnt, pads_hgt_bak], axis=-1) #(M1, 2)
                pads_wdh     = tf.concat([pads_wdh_fnt, pads_wdh_bak], axis=-1) #(M1, 2)
                box_pads     = tf.stack([pads_hgt, pads_wdh], axis=1)           #(M1, 2, 2)
                
                def resize_mask(elems=None):
                    box_msk_meg, box_wgt, box_leh, box_pad = elems
                    box_msk_meg = tf.expand_dims(box_msk_meg, axis=-1)
                    box_msk_meg = tf.image.resize_images(box_msk_meg, box_leh, \
                                                         method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                    box_msk_meg = tf.squeeze(box_msk_meg, axis=[-1])
                    box_msk_meg = tf.cast(box_msk_meg>=self.box_msk_min, dtype=tf.float32)
                    box_msk_meg = box_msk_meg * box_wgt
                    box_msk_meg = tf.pad(box_msk_meg, box_pad, "CONSTANT")
                    return box_msk_meg
                #当没有box时，tf.map_fn不会拆开，所以tf.image.resize_images是安全的
                elems        = [box_msks_meg, box_wgts, box_lehs, box_pads]
                box_msks_meg = tf.map_fn(resize_mask, elems, dtype=tf.float32,
                                         parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
                
                box_msk_meg  = tf.reduce_sum(box_msks_meg, axis=0)
                box_msk_idxs = tf.where(box_msk_meg>=self.box_msk_min)
                box_crd_min  = tf.maximum(tf.cast(tf.reduce_min(box_msk_idxs, axis=0), dtype=tf.int32), 0)
                box_crd_max  = tf.cast(tf.reduce_max(box_msk_idxs, axis=0), dtype=tf.int32)
                box_msk_meg  = box_msk_meg[box_crd_min[0]:box_crd_max[0]+1, box_crd_min[1]:box_crd_max[1]+1]
                box_msk_meg  = tf.expand_dims(box_msk_meg, axis=-1)
                box_msk_meg  = tf.image.resize_images(box_msk_meg, self.box_msk_siz, \
                                                      method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                box_msk_meg  = tf.squeeze(box_msk_meg, axis=[-1])
                box_msk_meg  = tf.expand_dims(box_msk_meg, axis=0)
                box_meg      = tf.concat([box_crd_min, box_crd_max], axis=0) + \
                               tf.stack([box_ymn, box_xmn, box_ymn, box_xmn], axis=0)
                box_meg      = tf.cast(box_meg, dtype=tf.float32)
                box_meg      = tf.expand_dims(box_meg, axis=0)
                box_idx_meg  = tf.expand_dims(j,       axis=0)
                #保存结果
                boxs_hld     = tf.concat([boxs_hld,     box_meg    ], axis=0)
                box_msks_hld = tf.concat([box_msks_hld, box_msk_meg], axis=0)
                box_idxs_hld = tf.concat([box_idxs_hld, box_idx_meg], axis=0)
                return [j+1, boxs_run, boxs_cls, box_prbs_cls, box_msks_cls, boxs_hld, box_msks_hld, box_idxs_hld]
                
            j = tf.constant(0)
            [j, boxs_run, boxs_cls, box_prbs_cls, box_msks_cls, boxs_hld, box_msks_hld, box_idxs_hld] = \
                tf.while_loop(cond, body, \
                              loop_vars=[j, boxs_run, boxs_cls, box_prbs_cls, box_msks_cls, boxs_hld, box_msks_hld, box_idxs_hld], \
                              shape_invariants=[j.get_shape(), boxs_run.get_shape(), boxs_cls.get_shape(), \
                                                box_prbs_cls.get_shape(), box_msks_cls.get_shape(), tf.TensorShape([None, 4]), \
                                                tf.TensorShape([None]+self.box_msk_siz), tf.TensorShape([None])], \
                              parallel_iterations=10, back_prop=False, swap_memory=True)
            box_idxs_hld = tf.gather(box_idxs_cls, box_idxs_hld)
            #保存结果
            boxs_kep     = tf.concat([boxs_kep,     boxs_hld    ], axis=0)
            box_msks_kep = tf.concat([box_msks_kep, box_msks_hld], axis=0)
            box_idxs_kep = tf.concat([box_idxs_kep, box_idxs_hld], axis=0)
            return [i+1, boxs, box_clss, box_prbs, box_msks, boxs_tmp, box_clss_kep, boxs_kep, box_msks_kep, box_idxs_kep]
        
        i = tf.constant(1) #要剔除背景类
        [i, boxs, box_clss, box_prbs, box_msks, boxs_tmp, box_clss_kep, boxs_kep, box_msks_kep, box_idxs_kep] = \
            tf.while_loop(cond1, body1, loop_vars=\
                          [i, boxs, box_clss, box_prbs, box_msks, boxs_tmp, box_clss_kep, boxs_kep, box_msks_kep, box_idxs_kep], \
                          shape_invariants=[i.get_shape(), boxs.get_shape(), box_clss.get_shape(), \
                                            box_prbs.get_shape(), box_msks.get_shape(), boxs_tmp.get_shape(), \
                                            box_clss_kep.get_shape(), tf.TensorShape([None, 4]), \
                                            tf.TensorShape([None]+self.box_msk_siz), tf.TensorShape([None, 1])], \
                          parallel_iterations=10, back_prop=False, swap_memory=True)
        #boxs_kep, box_msks_kep, box_idxs_kep
        box_clss_kep = tf.gather_nd(box_clss_kep, box_idxs_kep)
        box_prbs_kep = tf.gather_nd(box_prbs_kep, box_idxs_kep)
        box_num      = tf.shape(boxs_kep)[0]
        paddings     = [[0, self.box_nms_pst-box_num], [0, 0]]
        boxs_kep     = tf.pad(boxs_kep, paddings,     "CONSTANT")
        paddings     = [[0, self.box_nms_pst-box_num]]
        box_clss_kep = tf.pad(box_clss_kep, paddings, "CONSTANT")
        box_prbs_kep = tf.pad(box_prbs_kep, paddings, "CONSTANT")
        paddings     = [[0, self.box_nms_pst-box_num], [0, 0], [0, 0]]
        box_msks_kep = tf.pad(box_msks_kep, paddings, "CONSTANT")
        return boxs_kep, box_clss_kep, box_prbs_kep, box_msks_kep, box_num
    
    def merge_boxs(self, boxs=None, box_clss=None, box_prbs=None, box_msks=None, box_nums=None):
        elems = [boxs, box_clss, box_prbs, box_msks, box_nums]
        boxs, box_clss, box_prbs, box_msks, box_nums = \
            tf.map_fn(self.merge_boxs_img, elems, dtype=(tf.float32, tf.int32, tf.float32, tf.float32, tf.int32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return boxs, box_clss, box_prbs, box_msks, box_nums
    
