import numpy as np
import tensorflow as tf

def generate_rpns(rpn_hgts=None, rpn_wdhs=None, rpn_stas=None, rpn_ends=None, rpn_srds=None):
    
    rpns_lst = []
    for i, rpn_srd in enumerate(rpn_srds):
        hgt_idxs = tf.expand_dims(tf.range(rpn_stas[i], rpn_ends[i]+0.5, rpn_srd), axis=-1)
        wdh_idxs = tf.expand_dims(tf.range(rpn_stas[i], rpn_ends[i]+0.5, rpn_srd), axis=-1)
        hgt_num  = tf.shape(hgt_idxs)[0]
        wdh_num  = tf.shape(wdh_idxs)[0]
        hgt_idxs = tf.reshape(tf.tile(hgt_idxs, [1, wdh_num]), [-1, 1])
        wdh_idxs = tf.reshape(tf.tile(wdh_idxs, [hgt_num, 1]), [-1, 1])
        crd_idxs = tf.concat([hgt_idxs, wdh_idxs], axis=-1)
        crd_idxs = tf.concat([crd_idxs, crd_idxs], axis=-1)
        crd_idxs = tf.expand_dims(crd_idxs, axis=1)
        hgts = np.asarray(rpn_hgts[i], dtype=np.float32)
        wdhs = np.asarray(rpn_wdhs[i], dtype=np.float32)
        dlts = tf.stack([-hgts/2.0, -wdhs/2.0, hgts/2.0, wdhs/2.0], axis=-1)
        rpns = crd_idxs + dlts #(ymin, xmin, ymax, xmax)
        rpns = tf.reshape(rpns, [-1, 4])
        rpns_lst.append(rpns)
    rpns_lst = tf.concat(rpns_lst, axis=0)
    return rpns_lst
