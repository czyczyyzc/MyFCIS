import numpy as np
import tensorflow as tf

from Mybase import layers
from Mybase.layers import *
from Mybase.layers_utils import *
from Mybase.losses import *

from Mybase.fcis_utils.bbox import *
from Mybase.fcis_utils.anchors_layer import *
from Mybase.fcis_utils.anchors_target_layer1 import *
from Mybase.fcis_utils.proposals_layer1 import *
from Mybase.fcis_utils.proposals_target_layer1 import *
from Mybase.fcis_utils.bboxes_layer1 import *
from Mybase.fcis_utils.rois_pooling_layer import *

class Resnet101_FCIS(object):
    
    def __init__(self, cls_num=21, reg=1e-4, drp=0.5, typ=tf.float32):
        
        self.cls_num = cls_num #class number
        self.reg     = reg     #regularization
        self.drp     = drp     #dropout
        self.typ     = typ     #dtype
        
        self.mod_tra = True    #mode training
        self.glb_pol = False   #global pooling
        self.inc_btm = True    #include bottom block
        
        #resnet block setting
        self.res_set = [( 256,  64, [1, 1], [1, 1],  3, True ),  #conv2x 128 /4
                        ( 512, 128, [1, 1], [2, 2],  4, True ),  #conv3x 64  /8  #use
                        (1024, 256, [1, 1], [2, 2], 23, True ),  #conv4x 32  /16 #23--->101 #6--->50
                        (2048, 512, [1, 1], [2, 2],  3, True )]  #conv5x 32  /16 #use
        self.out_srd  = 4 #output stride
        
        self.rpn_rats = [[1.0, 2.0, 0.5]]
        self.rpn_sizs = [[128, 256, 512, 800]]
        self.rpn_nums = [len(self.rpn_rats[i])*len(self.rpn_sizs[i]) for i in range(len(self.rpn_rats))]
        self.rpn_hgts = [[l*np.sqrt(r) for l in sizs for r in rats] for sizs in self.rpn_sizs for rats in self.rpn_rats]
        self.rpn_wdhs = [[l/np.sqrt(r) for l in sizs for r in rats] for sizs in self.rpn_sizs for rats in self.rpn_rats]
        self.rpn_srds = [16]
        #只要不严格对齐，都会造成起始坐标的变化和偏移，想像棋盘上布满珠子，每一个起始点和结束点都是在其中抽取的某一个珠子
        #珠子之间的抽取距离具有累加性，该层累加的距离取决于上一层的步长，原图四个顶点取的珠子不能作为最后一层特征图，这就是shape=[2, 2]的作用
        self.rpn_stas = [   2.5]
        #self.rpn_ends = [ 498.5]
        self.rpn_ends = [1010.5]

        
    def forward(self, imgs=None, lbls=None, gbxs=None, gmks=None, gbx_nums=None, mtra=None, scp=None): 
        #lbl=label #scp=scope #mod_tra=mode train
        
        img_shp = imgs.get_shape().as_list()
        img_num, img_hgt, img_wdh = img_shp[0], img_shp[1], img_shp[2]
        fet_hgt = img_hgt // self.rpn_srds[0]
        fet_wdh = img_wdh // self.rpn_srds[0]
        img_shp = np.stack([img_hgt, img_wdh], axis=0)
        #common parameters
        com_pams = {
            "com":  {"reg": self.reg, "wscale": 0.01, "dtype": self.typ, "reuse": False, "is_train": self.mod_tra, "trainable": True},
            "bn":   {"eps": 1e-5, "decay": 0.9997},
            "relu": {"alpha": -0.1},
            "conv": {"number": 64, "shape": [7, 7], "rate": 1, "stride": [2, 2], "padding": "SAME", "use_bias": True},
            "deconv":   {"number":64,"shape":[2, 2],"rate":1,"stride":[2, 2],"padding":"SAME","out_shape":[28, 28],"use_bias":False},
            "max_pool": {"shape": [3, 3], "stride": [2, 2], "padding": "VALID"},
            "resnet_block": {"block_setting": self.res_set, "output_stride": self.out_srd},
            "resnet_unit":  {"depth_output":1024, "depth_bottle":256, "use_branch":True, "shape":[1, 1], "stride":[1, 1], "rate":1},
            #"pyramid":   {"depth": self.fet_dep},
            "glb_pool":  {"axis": [1, 2]},
            "reshape":   {"shape": []},
            "squeeze":   {"axis": [1, 2]},
            "transpose": {"perm":  [0, 2, 1, 3]},
            #"bilstm":    {"num_h": self.fet_dep//2, "num_o": None, "fbias": 1.0, "tmajr": False},
            #"concat":   {"axis": 0},
            #"split":    {"axis": 0, "number": img_num}
            #"affine":   {"dim": 4096, "use_bias": True},
            #"dropout":  {"keep_p": self.dropout},
        }
        #####################Get the first feature map!####################
        if self.inc_btm:
            print("Get the first feature map!")
            opas = {"op": [{"op": "conv_bn_relu1", "loop": 1, "params":{"com":{"trainable": True }}}, #(None, 512, 512, 64)
                           {"op": "max_pool1", "loop": 1, "params":{}}      #(None, 256, 256, 64) #pool2
                          ], "loop": 1}
            tsr_out = layers_module1(imgs, 0, com_pams, opas, mtra)
            print('')
        #####################Get the resnet blocks!#########################
        print("Get the resnet block!")
        fet_lst = []
        opas = {"op": [{"op": "resnet_block2", "loop": 1, "params":{}}], "loop": 1}
        fet_lst.extend(layers_module1(tsr_out, 1, com_pams, opas, mtra))
        assert len(fet_lst) == 4, "The first resnet block is worng!"
        fet_lst = [fet_lst[2], fet_lst[3]]  #conv4, conv5
        tsr_out = fet_lst[-1]
        
        ################Get the image classification results!###############
        if self.glb_pol: # Global average pooling.
            print("Get the image classification results!")
            com_pams["conv"] = {"number":self.cls_num,"shape":[1, 1],"rate":1,"stride":[1, 1],"padding":"SAME","use_bias":True}
            opas = {"op": [{"op": "global_pool1", "loop": 1, "params": {}},
                           {"op": "conv1",        "loop": 1, "params": {}},
                           {"op": "squeeze1",     "loop": 1, "params": {}},
                          ], "loop": 1}
            scrs = layers_module1(tsr_out, 99, com_pams, opas, mtra) #class scores
            prbs = tf.nn.softmax(scrs) #class probabilities
            print('')
            los_dat  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbls, logits=scrs))
            los_reg  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los      = los_dat + los_reg
            loss     = tf.stack([los, los_dat, los_reg], axis=0)
            return loss, scrs
        
        #####################Get the rpn prediction!########################
        print("Get the rpn prediction!")
        rpn_scrs_pst = []
        rpn_prds_pst = []
        com_pams["com"]["trainable"] = True
        com_pams["conv"] = {"number": 512, "shape": [1, 1], "rate": 1, "stride": [1, 1], "padding": "SAME", "use_bias": True}
        opas0 = {"op": [{"op": "conv_bn_relu1", "loop": 1, "params":{"conv":{"shape": [3, 3]}}}], "loop": 1}
        opas1 = {"op": [{"op": "conv1",    "loop": 1, "params":{"conv":{"number": self.rpn_nums[0]*2}}},
                        {"op": "reshape1", "loop": 1, "params":{"reshape":{"shape": [img_num, -1, 2]}}},
                       ], "loop": 1}
        opas2 = {"op": [{"op": "conv1",    "loop": 1, "params":{"conv":{"number": self.rpn_nums[0]*4}}},
                        {"op": "reshape1", "loop": 1, "params":{"reshape":{"shape": [img_num, -1, 4]}}},
                       ], "loop": 1}
        fet_tmp      = layers_module1(fet_lst[0], 2, com_pams, opas0, mtra)
        rpn_scrs_pst = layers_module1(fet_tmp,    3, com_pams, opas1, mtra)
        rpn_prds_pst = layers_module1(fet_tmp,    4, com_pams, opas2, mtra)
        rpn_prbs_pst = tf.nn.softmax(rpn_scrs_pst, axis=-1)
        print('')
        ################Get the roi and mask prediction!####################
        print("Get the roi and mask prediction!")
        com_pams["com"]["trainable"] = True
        opas0 = {"op": [{"op": "conv_bn_relu1", "loop": 1, "params":{"conv":{"number": 1024}}}], "loop": 1}
        opas1 = {"op": [{"op": "conv1",     "loop": 1, "params":{"conv":{"number": 7*7*self.cls_num*2}}},
                        {"op": "reshape1",  "loop": 1, "params":{"reshape":{"shape": [-1, fet_hgt, fet_wdh, 7*7, self.cls_num*2]}}},
                        {"op": "transpose1","loop": 1, "params":{"transpose":{"perm": [0, 3, 1, 2, 4]}}}, #(N, 7*7, H, W, C)
                       ], "loop": 1}
        opas2 = {"op": [{"op": "conv1",     "loop": 1, "params":{"conv":{"number": 7*7*4}}},
                        {"op": "reshape1",  "loop": 1, "params":{"reshape":{"shape": [-1, fet_hgt, fet_wdh, 7*7, 4]}}},
                        {"op": "transpose1","loop": 1, "params":{"transpose":{"perm": [0, 3, 1, 2, 4]}}}, #(N, 7*7, H, W, 4)
                       ], "loop": 1}
        fet_tmp       = layers_module1(fet_lst[1], 5, com_pams, opas0, mtra)
        roi_scrs_pst_ = layers_module1(fet_tmp,    6, com_pams, opas1, mtra)
        roi_prds_pst_ = layers_module1(fet_tmp,    7, com_pams, opas2, mtra)
        print('')
        ################Generate rpns from anchors_layer!###################
        print("Generate rpns from anchors_layer!")
        rpns = generate_rpns(self.rpn_hgts, self.rpn_wdhs, self.rpn_stas, self.rpn_ends, self.rpn_srds)
        print('')
        ###############Generate rois from proposals_layer!##################
        print("Generate rois from proposals_layer!")
        PL = ProposalsLayer(self.mod_tra, rpns, img_shp)
        rois, roi_prbs, roi_nums = PL.generate_rois(rpn_prbs_pst, rpn_prds_pst)
        print('')
        if self.mod_tra: 
            ##############Generate rois from proposals_target_layer!###############
            print("Generate rois from proposals_target_layer!")
            PT = ProposalsTargetLayer(self.cls_num, img_shp)
            rois, roi_nums = PT.sample_rois(rois, roi_scrs_pst_, roi_prds_pst_, roi_nums, gbxs, gmks, gbx_nums)
            print('')
        ########Generate rois predictions from rois_pooling_layer!##########
        print("Generate roi predictions from rois_pooling_layer!")
        RP = RoisPoolingLayer(self.cls_num, img_shp)
        roi_scrs_pst, roi_prds_pst, roi_msks_pst = RP.rois_pooling(roi_scrs_pst_, roi_prds_pst_, rois, roi_nums)
        roi_prbs_pst = tf.nn.softmax(roi_scrs_pst, axis=-1)
        print("Generate boxs0 from bboxes_layer!")
        BL = BBoxesLayer(self.mod_tra, self.cls_num, img_shp)
        boxs, box_nums, roi_clss, roi_prbs, roi_msks = \
            BL.generate_boxs(rois, roi_prbs_pst, roi_prds_pst, roi_msks_pst, roi_nums)
        print("Generate box predictions from rois_pooling_layer!")
        box_scrs_pst, box_prds_pst, box_msks_pst = RP.rois_pooling(roi_scrs_pst_, roi_prds_pst_, boxs, box_nums)
        box_prbs_pst = tf.nn.softmax(box_scrs_pst, axis=-1)
        print("Generate boxs1 from bboxes_layer!")
        _,    _,        box_clss, box_prbs, box_msks = \
            BL.generate_boxs(boxs, box_prbs_pst, box_prds_pst, box_msks_pst, box_nums)
        print("Generate boxs2 from bboxes_layer!")
        boxs, box_clss, box_prbs, box_msks, box_nums = \
            BL.concat_boxs(rois, roi_clss, roi_prbs, roi_msks, roi_nums, boxs, box_clss, box_prbs, box_msks, box_nums)
        print("Generate boxs3 from bboxes_layer!")
        boxs, box_clss, box_prbs, box_msks, box_nums = \
            BL.merge_boxs(boxs, box_clss, box_prbs, box_msks, box_nums)
        print('')
        '''
        boxs     = rois
        box_clss = tf.zeros(shape=[img_num, tf.shape(rois)[1]], dtype=tf.int32) + 1
        box_prbs = roi_prbs
        box_msks = tf.zeros(shape=[img_num, tf.shape(rois)[1], 21, 21], dtype=tf.float32)
        box_nums = roi_nums
        '''
        if self.mod_tra: 
            ##########################Get the losses!##############################
            print("Get the losses!")
            print("Get the rpn losses!")
            AT = AnchorsTargetLayer(rpns, self.cls_num)
            rpn_prbs_los, rpn_prds_los, rpn_fgd_rat = \
                AT.generate_rpn_loss(rpn_scrs_pst, rpn_prds_pst, gbxs, gbx_nums) #注意用rpn_scrs_pst而不是rpn_prbs_pst
            print("Get the roi losses!")
            roi_prbs_los, roi_prds_los, roi_msks_los, roi_fgd_rat = \
                PT.generate_roi_loss(roi_scrs_pst, roi_prds_pst, roi_msks_pst)   #注意用roi_scrs_pst而不是roi_prbs_pst
            rpn_los  = rpn_prbs_los + rpn_prds_los
            roi_los  = roi_prbs_los + roi_prds_los + roi_msks_los
            los_dat  = rpn_los + roi_los*2.0
            los_reg  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los      = los_dat + los_reg
            loss     = tf.stack([los, rpn_prbs_los, rpn_prds_los, roi_prbs_los, roi_prds_los, roi_msks_los, \
                                 los_reg, rpn_fgd_rat, roi_fgd_rat], axis=0)
            print('')
            return loss, boxs, box_clss, box_prbs, box_msks, box_nums
        else:
            return boxs, box_clss, box_prbs, box_msks, box_nums