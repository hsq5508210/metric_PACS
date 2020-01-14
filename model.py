import tensorflow as tf
from tensorflow.python.platform import flags
from utils import conv_block, distance, mse, get_dist_category
FLAGS = flags.FLAGS



class Model:
    def __init__(self, sess, output_dim=10):
        self.input_dim = FLAGS.input_dim
        self.output_dim = output_dim
        self.dim_hidden = FLAGS.filter_num
        self.neighbor_k = FLAGS.k_neighbor
        self.lr = FLAGS.lr
        self.sess = sess
        if FLAGS.data_source == 'PACS':
            self.channels = 3
        if FLAGS.backbone == 'Conv':
            self.construct_weights = self.construct_conv
            self.forward = self.forward_conv
        else:
            self.construct_weights = self.construct_res
            self.forward = self.forward_res
    def get_loss(self, inp, weights, resuse=True):
        support_x, support_y, query_x, query_y = inp
        # print(support_x.shape)
        accurcy = []
        output_s = self.forward(self.support_x, weights, reuse=resuse)
        output_q = self.forward(self.query_x, weights, reuse=resuse)
        # print(output_q.eval())

        predict = self.category_choose(output_s, output_q, self.query_y)
        # print(predict.eval())
        # print(predict.shape,support_y.shape)
        task_losses = mse(predict, support_y)
        return task_losses, accurcy


    def construct_model(self, input_tensor=None):
        if input_tensor is None:
            self.support_x = tf.placeholder(tf.float32, name='support_x')
            self.support_y = tf.placeholder(tf.float32, name='support_y')
            self.query_x = tf.placeholder(tf.float32, name='query_x')
            self.query_y = tf.placeholder(tf.float32, name='query_y')
        else:
            self.support_x = input_tensor['support_set'][0]
            self.support_y = input_tensor['support_set'][1]
            self.query_x = input_tensor['query_set'][0]
            self.query_y = input_tensor['query_set'][1]
        with tf.variable_scope('model', reuse=None) as training_socpe:
            if 'weights' in dir(self):
                training_socpe.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()


        accurcy = []
        losses = []
        # print(weights['conv1'].eval())
    def trainop(self):
        if 'weights' not in dir(self):
            self.weights = weights = self.construct_weights() 
        # else:
        #     self.construct_model()
        print("support y name is:", self.support_y.name)
        losses, acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y), self.weights)
        if FLAGS.train:
            self.loss = loss = tf.reduce_sum(losses) / tf.to_float(FLAGS.query_num * FLAGS.model * FLAGS.way_num)
            # self.train = tf.train.AdamOptimizer(self.lr).minimize(loss)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.gvs = gvs = optimizer.compute_gradients(loss)
            # if FLAGS.data_source == 'PACS':
            #     gvs = [(tf.clip_by_value(grad, -10, 10, var)) for grad, var in gvs]
            self.train = optimizer.apply_gradients(gvs)
        return self.train






    def category_choose(self, output_s, output_q, label_q):
        output_s = self.vectorlize(output_s)
        output_q = self.vectorlize(output_q)
        dist_onehot = tf.map_fn(fn=lambda s:tf.reduce_sum(get_dist_category(s, output_q, label_q), axis=0), elems=output_s)
        return dist_onehot



    def vectorlize(self, x):
        # b, h, w, c = x.shape
        x = tf.reshape(tensor=x, shape=[-1, 14*14*64])
        return x

    def construct_conv(self):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3
        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        return weights


    def forward_conv(self, inp, weights, reuse=False):
        # support_x = self.support_x
        # query_x = self.query_x
        scope = ''
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        return hidden4


    def construct_res(self):
        pass

    def forward_res(self, inp, weights, reuse=False):


        pass



