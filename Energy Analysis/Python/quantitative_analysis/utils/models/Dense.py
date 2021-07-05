import tensorflow as tf

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        self.units = units
        super(MyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=tf.initializers.he_normal(),
                                 trainable=True,
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer=tf.initializers.he_normal(),
                                 trainable=True,
                                 name='b')
        super(MyDense, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.leaky_relu(tf.matmul(inputs, self.w) + self.b)

    def get_config(self):
        config = super(MyDense, self).get_config()
        config.update({'units':self.units})
        return config