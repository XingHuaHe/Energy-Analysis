import tensorflow as tf
from utils.models.Dense import MyDense

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=1):
        super(MyModel, self).__init__(name='my_model')
        # parameter.
        self.num_classes = num_classes

        # layer 1.
        self.MyDense1 = MyDense(units=20)
        self.MyDense2 = MyDense(units=20)
        self.MyDense3 = MyDense(units=self.num_classes)
        # layer 2.
        self.MyDense4 = MyDense(units=20)
        self.MyDense5 = MyDense(units=20)
        self.MyDense6 = MyDense(units=self.num_classes)
        # layer 3.
        self.MyDense7 = MyDense(units=20)
        self.MyDense8 = MyDense(units=20)
        self.MyDense9 = MyDense(units=self.num_classes)

    @tf.function
    def call(self, inputs):
        # layer 1.
        x = self.MyDense1(inputs)
        x2 = self.MyDense2(x)
        output1 = self.MyDense3(x2)
        # layer 2.
        x4 = self.MyDense4(x2)
        x = tf.keras.layers.add([x2, x4])
        x5 = self.MyDense5(x)
        output2 = self.MyDense6(x5)
        # layer 3.
        x7 = self.MyDense7(x5)
        x = tf.keras.layers.add([x5, x7])
        x8 = self.MyDense8(x)
        output3 = self.MyDense9(x8)

        return (output1, output2, output3)