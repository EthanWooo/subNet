# input:[1,24,36,1024]
# output:[1,12,18,2048]

import tensorflow as tf
from network.strategy_network import StrategyNetwork

MODEL_SAVE_PATH = ''


def Strategy(exposure1, exposure2, exposure3, flow1, flow2):
    # return tf.reduce_sum(content, style)
    print("exposure1.shape: ", exposure1.shape)
    print("exposure2.shape: ", exposure2.shape)
    print("exposure3.shape: ", exposure3.shape)
    print("flow1.shape: ", flow1.shape)
    print("flow2.shape: ", flow2.shape)
    con = tf.concat([exposure1, exposure2, exposure3, flow1, flow2], 3)
    print("concat.shape: ", con.shape)
    sn = StrategyNetwork()

    return sn.getConcat(con)
