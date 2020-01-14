from utils import distance
import tensorflow as tf
from tensorflow.python.platform import flags
from model import Model
import  data_generator as dg
FLAGS = flags.FLAGS



label = [[0,0,0,0,1],[0,0,1,0,0]]
x = [[0.1, 0.3, 0.2], [0.1, 0.3, 0.2]]
y = [[0.5, 0.6, 0.7], [0.91, 0.36, 0.2]]
label = tf.convert_to_tensor(label, dtype=tf.float32)
with tf.name_scope('model'):
    for i in range(3):
        x = tf.convert_to_tensor(x, dtype=tf.float32, name="x")
        k = tf.placeholder(dtype=tf.float32)
        print(k.name)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
# with tf.Session() as sess:
#     print(x.eval())
#     print(y.shape)
#     print(y.eval())
#     print(label.shape)
#     print(distance(x,y,label[0]).eval())
#     # print(distance(x,y[1]).eval())
#     print(sess.run(get_dist_category(x,y,label)))




