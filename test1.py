import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from alexnet import *
#from vgg16net_rcnn import Vgg16Net_RCNN
#from vgg16net_ssd import Vgg16Net_SSD
#from resnet101_mask_rcnn import Resnet101_MaskRCNN
#from vgg16_ctpn_crnn_ctc import Vgg16_CTPN_CRNN_CTC
#from resnet50_dssd_leye import Resnet50_DSSD_LEYE
from resnet101_fcis import Resnet101_FCIS
from Mybase.solver import Solver


def test():
    #model = MyCifar10Net(num_class = 10, reg = 0.001, dropout = 0.5, wscale = 0.01, dtype = tf.float32)
    #model = MyAlexNet(num_class = 10, reg = 0.001, dropout = 0.7, wscale = 0.01, dtype = tf.float32)
    #model = Cifar10_ResNet(num_layers = 44, num_class = 10, reg = 2e-4, dropout = 0.5, dtype = tf.float32)
    #model = Vgg16Net_RCNN(num_classes = 21, reg = 2e-4, dropout = 0.7, dtype = tf.float32)
    #model = Resnet101_MaskRCNN(num_classes=81, reg=1e-4, dropout=0.6, dtype=tf.float32)
    #model = Resnet101_MaskRCNN(num_classes=1000, reg=1e-4, dropout=0.6, dtype=tf.float32)
    #model = Vgg16_CTPN_CRNN_CTC(num_classes=1000, reg=5e-4, dropout=0.5, dtype=tf.float32)
    #mdl = Resnet50_DSSD_LEYE(cls_num=81, reg=1e-4, drp=0.5, typ=tf.float32)
    mdl = Resnet101_FCIS(cls_num=21, reg=1e-4, drp=0.5, typ=tf.float32)
    
    sov = Solver(mdl,
                 opm_cfg={
                     "decay_rule": "fixed",
                     #"optim_rule": "adam",
                     #"lr_base": 0.0002,
                     "optim_rule": "momentum",
                     "lr_base":  0.00005, #0.004
                     "momentum": 0.9,
                     #"optim_rule": "adadelta",
                     #"lr_base": 0.01,
                 },
                 bat_siz     =  2, epc_num = 10000,
                 itr_per_prt = 20, prt_ena = True,
                 mov_ave_dca = 0.99,
                 epc_per_dca = 1)
    #print("TRAINING...")
    #sov.train()
    #print("TESTING...")
    sov.test()
    sov.display_detections(show=False, save=True)
    #solver.show_loss_acc()
    """
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('accuracy')
    plt.xlabel('Epoch')
    
    #plt.subplot(3, 1, 3)
    #plt.title('Validation accuracy')
    #plt.xlabel('Epoch')
    
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label="train_acc")
    plt.plot(solver.val_acc_history, '-o', label="val_acc")

    for i in [1, 2]:
        plt.subplot(2, 1, i)
        plt.legend(loc='upper center', ncol=4)
    
    plt.gcf().set_size_inches(15, 15)
    plt.show()
    """
