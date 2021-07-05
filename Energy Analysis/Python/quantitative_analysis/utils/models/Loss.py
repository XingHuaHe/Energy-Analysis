import tensorflow as tf

class MyLoss(tf.keras.losses.Loss):
    def __init__(self, name=None):
        super(MyLoss, self).__init__(name=name)

    def call(self, y_true, y_pre):
        y_true = tf.cast(y_true, tf.float32)
        y_pre = tf.cast(y_pre, tf.float32)
        loss = tf.abs(y_true - y_pre)
        
        # tf.print(loss)
        return loss