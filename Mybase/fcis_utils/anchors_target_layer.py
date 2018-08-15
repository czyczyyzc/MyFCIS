import cv2
import numpy as np
import tensorflow as tf
from Mybase.comp_utils import *
from Mybase.losses import *
from .bbox import *

#with OHEM(Online Hard Examples Mining)
class AnchorsTargetLayer(object):
    
    def __init__(self, rpns=None, rpn_cls_num=None):
        
        self.rpns        = rpns
        self.rpn_cls_num = rpn_cls_num
        self.rpn_fbg_num = 256
        self.rpn_fgd_num = self.rpn_fbg_num
        self.rpn_fgd_ovp = 0.5     #rpn_fg_overlap
        self.rpn_bgd_ovp = 0.5     #rpn_bg_overlap
        self.rpn_crw_max = 0.001   #crowd_thresh
        '''
        self.rpn_nms_pre = None
        self.rpn_nms_pst = 128
        self.rpn_nms_max = 0.7
        '''
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
        fgd_idxs0 = tf.where(max_ovps>=self.rpn_fgd_ovp)[:, 0]         #无效边框不影响
        fgd_idxs1 = tf.argmax(rpn_ovps, axis=0)                        #最后一个是无效边框
        fgd_idxs1 = fgd_idxs1[:-1]                                     #剔除最后一个无效边框匹配的anchor
        fgd_idxs  = tf.concat([fgd_idxs0, fgd_idxs1], axis=0)
        fgd_idxs, idxs = tf.unique(fgd_idxs)
        fgd_idxs  = tf.expand_dims(fgd_idxs, axis=-1)
        fgd_gtas  = tf.gather_nd(rpn_amxs, fgd_idxs)                   #和fgd_idxs对应
        rpn_gtas  = tensor_update(rpn_gtas, fgd_idxs, fgd_gtas)        #这里的fgd_idxs可以乱序
        
        ######################选背景######################
        bgd_idxs  = tf.where(tf.logical_and(tf.logical_and(max_ovps<self.rpn_bgd_ovp, tf.equal(rpn_gtas, -2)), ncw_msks))
        rpn_gtas  = tensor_update(rpn_gtas, bgd_idxs, -1)              #这里的bgd_idxs可以乱序
        
        ####################整合前背景#####################
        fgd_idxs = tf.where(rpn_gtas>=0)
        bgd_idxs = tf.where(tf.equal(rpn_gtas, -1))
        fbg_idxs = tf.concat([fgd_idxs, bgd_idxs], axis=0)             #确保fgd_idxs/bgd_idxs/fbg_idxs三者的顺序相对应
        fgd_gtas = tf.gather_nd(rpn_gtas, fgd_idxs)                    #和fgd_idxs对应
        fbg_gtas = tf.gather_nd(rpn_gtas, fbg_idxs)                    #和fbg_idxs对应
        fgd_num  = tf.shape(fgd_idxs)[0]
        bgd_num  = tf.shape(bgd_idxs)[0]
        fbg_num  = tf.shape(fbg_idxs)[0]
        fbg_rpns = tf.gather_nd(self.rpns, fbg_idxs)                   #和fbg_idxs对应
        
        ####################Get Target###################
        ###Get类别###
        fgd_prbs_pre = tf.gather(gbxs[:, 4], fgd_gtas)                 #和fgd_idxs对应
        fgd_prbs_pre = tf.cast(fgd_prbs_pre, dtype=tf.int32)
        bgd_prbs_pre = tf.zeros(shape=[bgd_num], dtype=tf.int32)
        fbg_prbs_pre = tf.concat([fgd_prbs_pre, bgd_prbs_pre], axis=0) #和fbg_idxs对应
        ###Get坐标###
        #注意是定类还是不定类
        fgd_prds_tmp = tf.gather(gbxs[:, 0:4], fgd_gtas)               #和fgd_idxs对应
        fgd_rpns     = tf.gather_nd(self.rpns, fgd_idxs)               #和fgd_idxs对应
        fgd_prds_pre = bbox_transform(fgd_rpns, fgd_prds_tmp)          #和fgd_idxs对应 #(M0, 4)
        
        ####################Get Losses###################
        ###Get Class Loss###
        fbg_prbs_pst = tf.gather_nd(rpn_prbs_pst, fbg_idxs)            #和fgd_idxs对应
        fbg_prbs_pst = tf.stop_gradient(fbg_prbs_pst)                  #这里的fgd_prbs_pst为read_only，只为选择hard_examples
        rpn_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fbg_prbs_pre, logits=fbg_prbs_pst) #(M) #和fbg_idxs对应
        ###Get Bbox loss###
        fgd_prds_pst = tf.gather_nd(rpn_prds_pst, fgd_idxs)            #和fgd_idxs对应 #(M0, 4)
        fgd_prds_pst = tf.stop_gradient(fgd_prds_pst)                  #这里的fgd_prds_pst为read_only，只为选择hard_examples
        rpn_prds_los = smooth_l1(3.0, fgd_prds_pst, fgd_prds_pre)
        #rpn_prds_los = tf.reduce_mean(rpn_prds_los, axis=1)           #reduce_mean还是reduce_sum
        rpn_prds_los = tf.reduce_sum(rpn_prds_los, axis=1)             #(M0)
        rpn_loss_tmp = tf.zeros(shape=[bgd_num], dtype=tf.float32)
        rpn_prds_los = tf.concat([rpn_prds_los, rpn_loss_tmp], axis=0) #(M)
        ###Get Total loss###
        rpn_loss = rpn_prbs_los + rpn_prds_los                         #(M) #和fbg_idxs对应
        rpn_loss, fbg_ixxs = tf.nn.top_k(rpn_loss, k=self.rpn_fbg_num, sorted=False) #fbg_idxs中的前背景顺序没有乱
        ###重新定位前背景RPN###
        fbg_idxs = tf.gather(fbg_idxs, fbg_ixxs)
        fbg_num  = tf.shape(fbg_idxs)[0]
        #fbg_rpns = tf.gather(fbg_rpns, fbg_ixxs)
        fbg_prbs_pre = tf.gather(fbg_prbs_pre, fbg_ixxs)
        ###重新定位前景RPN###
        idxs     = tf.where(fbg_ixxs<fgd_num) #0<=&&<=(fgd_num-1)
        fgd_ixxs = tf.gather_nd(fbg_ixxs, idxs)
        fgd_idxs = tf.gather(fgd_idxs, fgd_ixxs)
        fgd_num  = tf.shape(fgd_idxs)[0]
        fgd_prds_pre = tf.gather(fgd_prds_pre, fgd_ixxs)
        '''
        if self.rpn_nms_pre is not None:
            rpn_nms_pre = tf.minimum(self.rpn_nms_pre, tf.shape(fbg_idxs)[0])
            fbg_loss, idxs = tf.nn.top_k(fbg_loss, k=rpn_nms_pre, sorted=True)
            fbg_idxs = tf.gather(fbg_idxs, idxs)
            fbg_clss = tf.gather(fbg_clss, idxs)
            fbg_rpns = tf.gather(fbg_rpns, idxs)
        #做逐类的nms
        fbg_idxs_kep = tf.zeros(shape=[0, 1], dtype=tf.int64)
        fbg_loss_kep = tf.zeros(shape=[0], dtype=tf.float32)
        
        def cond(i, fbg_rpns, fbg_clss, fbg_loss, fbg_idxs, fbg_idxs_kep, fbg_loss_kep):
            c = tf.less(i, self.rpn_cls_num)
            return c

        def body(i, fbg_rpns, fbg_clss, fbg_loss, fbg_idxs, fbg_idxs_kep, fbg_loss_kep):
            #选出对应类的rois
            idxs = tf.where(tf.equal(fbg_clss, i))
            fbg_rpns_cls = tf.gather_nd(fbg_rpns, idxs)
            fbg_loss_cls = tf.gather_nd(fbg_loss, idxs)
            fbg_idxs_cls = tf.gather_nd(fbg_idxs, idxs)
            #进行非极大值抑制操作
            idxs = tf.image.non_max_suppression(fbg_rpns_cls, fbg_loss_cls, self.rpn_nms_pst, self.rpn_nms_max)
            fbg_idxs_cls = tf.gather(fbg_idxs_cls, idxs)
            fbg_loss_cls = tf.gather(fbg_loss_cls, idxs)
            # 保存结果
            fbg_idxs_kep = tf.concat([fbg_idxs_kep, fbg_idxs_cls], axis=0)
            fbg_loss_kep = tf.concat([fbg_loss_kep, fbg_loss_cls], axis=0)
            return [i+1, fbg_rpns, fbg_clss, fbg_loss, fbg_idxs, fbg_idxs_kep, fbg_loss_kep]

        i = tf.constant(0) #要保留背景类
        [i, fbg_rpns, fbg_clss, fbg_loss, fbg_idxs, fbg_idxs_kep, fbg_loss_kep] = \
            tf.while_loop(cond, body, loop_vars=[i, fbg_rpns, fbg_clss, fbg_loss, fbg_idxs, fbg_idxs_kep, fbg_loss_kep], \
                          shape_invariants=[i.get_shape(), fbg_rpns.get_shape(), fbg_clss.get_shape(), fbg_loss.get_shape(), \
                                            fbg_idxs.get_shape(), tf.TensorShape([None, 1]), tf.TensorShape([None])], \
                          parallel_iterations=10, back_prop=False, swap_memory=False)
        fbg_idxs = fbg_idxs_kep
        fbg_loss = fbg_loss_kep
        fbg_num  = tf.minimum(self.rpn_nms_pst, tf.shape(fbg_idxs)[0])
        fbg_loss, idxs = tf.nn.top_k(fbg_loss, k=fbg_num, sorted=True)
        fbg_idxs = tf.gather(fbg_idxs, idxs)
        '''
        ###################计算分类损失####################
        fbg_prbs_pst = tf.gather_nd(rpn_prbs_pst, fbg_idxs)     #和fbg_idxs对应
        rpn_prbs_los = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fbg_prbs_pre, logits=fbg_prbs_pst)
        rpn_prbs_los = tf.reduce_sum(rpn_prbs_los)
        
        ###################计算回归损失####################
        fgd_prds_pst = tf.gather_nd(rpn_prds_pst, fgd_idxs)     #和fgd_idxs对应
        rpn_prds_los = smooth_l1(3.0, fgd_prds_pst, fgd_prds_pre)
        #rpn_prds_los = tf.reduce_mean(rpn_prds_los, axis=1)    #reduce_mean还是reduce_sum
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
