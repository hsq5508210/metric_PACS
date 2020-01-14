from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow as tf
import numpy as np
import cv2
import os

FLAGS = flags.FLAGS
np.random.seed(0)

def config():
    configs = {}
    if FLAGS.data_source == 'PACS':
        configs['PATH'] = FLAGS.data_PATH
        configs['split_txt_PATH'] = FLAGS.split_txt_PATH
        configs['model'] = ['art_painting', 'cartoon', 'photo', 'sketch']
        configs['split'] = ['train', 'test']
    return configs.values()


def readtxt(filename, image_PATH):
    """
    :param filename: .txt
    :return: context of .txt
    """
    nameList = []
    txt = open(filename, 'r')
    for line in txt:
        item = {'path':"", 'label':0}
        # print(line)
        item['path'], item['label'] = line.split()
        item['path'] = os.path.join(image_PATH, item['path'])
        nameList.append(item)
    txt.close()
    return nameList

def get_split(train, test):
    """

    :param train: txt
    :param test: txt
    :return:data
    """
    data = {'train':{}, 'text':{}}
    data['train'] = readtxt(train)
    data['text'] = readtxt(test)

    return data

def groupByLabel(data, class_num):
    """

    :param data: list producted by readtxt(PATH)
    :param class_num: num of classes
    :return: list for each classes
    """
    group = [[] for i in range(class_num)]
    for item in data:
        group[int(item['label'])-1].append(item['path'])
    return group

def split(model, split_PATH, image_PATH = "/data2/hsq/Project/PACS"):
    """
    :param PATH: root dir of split .txt file.
    :return: file name of data group by label each model e.g. {'art_painting':list[] .....}
    """
    train_or_test = 'train'
    # if not FLAGS.train:
    #     train_or_test = 'test'
    # splits_PATH = PATH
    splits = os.listdir(split_PATH)
    trainsplits = [os.path.join(split_PATH, s) for s in splits if train_or_test in s]
    dic_model_file = {m:s for m in model for s in trainsplits if m in s}
    data_model = {m:groupByLabel(readtxt(dic_model_file[m],image_PATH), 7) for m in model}
    # print(data_model)
    return data_model

def sample_support(support_num=5):
    pass

# def sample_task(data_model=split(), query_num_per_class_per_model=1, class_num=5, support_num_per_class_per_model=1):
def sample_task(query_num_per_class_per_model=1, class_num=5,
                    support_num_per_class_per_model=1):
    """
    :param data_model: list of module name e.g. ['art_painting', 'cartoon', 'photo', 'sketch']
    :param class_num: n-ways.
    :return: list of dict of path and label e.g [{'support_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_072.jpg'], [1]],
                                                'q_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_091.jpg'], [1]]]

    """
    if FLAGS.data_source == 'PACS':
        raw_path, split_txt, model, train_test = config()
        data_model = split(model, split_txt, raw_path)
    classes = []
    task_data = []
    # task_label = []
    while True:
        n = np.random.randint(0, 6)
        if n not in classes:
            classes.append(n)
        if len(classes) == class_num: break
    # print(classes)
    s = {'data':[], 'label':[]}
    q = {'data':[], 'label':[]}
    for m in model:
        for i, c in enumerate(classes):
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            while True:
                idx = np.random.randint(len(data_model[m][c]))
                filename = data_model[m][c][idx]
                if filename not in support_x:
                    support_x.append(filename)
                    support_y.append(i)
                if(len(support_x) == support_num_per_class_per_model): break
            while True:
                idx = np.random.randint(len(data_model[m][c]))
                filename = data_model[m][c][idx]
                if (filename not in support_x) and (filename not in query_x):
                    query_x.append(filename)
                    query_y.append(i)
                if(len(query_x) == query_num_per_class_per_model): break

            s['data'].extend(support_x)
            s['label'].extend(support_y)
            q['data'].extend(query_x)
            q['label'].extend(query_y)
            # task = {'support_xy':[support_x, support_y], 'query_xy':[query_x, query_y] }
            # task_data.append(task)
    return {'support':s, 'query':q}

# def make_set_tensor(dict_set):
#     """
#     :param dic_set: data_dict e.g. support_set={'data': [...], 'label':[...]}
#     :return: image tensors and label-one-hot tensors.
#     """
#     file_name_list = dict_set['data']
#     label_list = dict_set['label']
#     n_ways = max(label_list)+1
#     # print(n_ways)
#     # print(label_list)
#     labels = tf.convert_to_tensor(label_list)
#     labels = tf.one_hot(labels, n_ways)
#     filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(file_name_list, dtype=tf.string), shuffle=False)
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     # if FLAGS.data_source = 'PACS':
#     images = tf.image.decode_jpeg(image_file, channels=3)
#     images.set_shape((227, 227, 3))
#     images = tf.cast(images, tf.float32) / 255.0
#     img_batch = tf.train.batch([images], batch_size=len(file_name_list))
#     # for _ in file_name_list:
#     #     b, h, c = images.shape
#     #     images = tf.reshape(images, [1, b, h, c])
#     #     img_batch = tf.concat([images, img_batch], 0)
#     # print(img_batch.shape)
#     return img_batch, labels

def make_set_tensor(dict_set):
    """
    :param dic_set: data_dict e.g. support_set={'data': [...], 'label':[...]}
    :return: image tensors and label-one-hot tensors.
    """
    file_name_list = dict_set['data']
    label_list = dict_set['label']
    n_ways = max(label_list)+1
    labels = np.eye(n_ways)[label_list]
    img_batch = np.array([cv2.imread(p) for p in file_name_list]).astype(np.float)/255.0
    return img_batch, labels.astype(np.float)

def get_loss(out_put, label):
    pass

# data_model = split()
# x = sample_task(data_model, query_num_per_class_per_model=5, class_num=5, support_num_per_class_per_model=1)
# y = sample_task(data_model, query_num_per_class_per_model=5, class_num=5, support_num_per_class_per_model=1)
# z = sample_task(data_model, query_num_per_class_per_model=5, class_num=5, support_num_per_class_per_model=1)


# images, labels = make_set_tensor(x['support'])
# # print(x)
# with tf.Session() as sess:
#     tf.local_beta2_powers_initializer().run()
#     threads = tf.train.start_queue_runners(sess=sess)
#     for _ in x['support']['data']:
#         img = images.eval().shape
#         # print(img)
#     lab = labels.eval()
#
#     print(lab)


## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.leaky_relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

def inner_product(x, y):
    return tf.reduce_sum(tf.multiply(x, y), axis=1)

def distance(x, y, onehot_label):
    """different distance function."""
    if FLAGS.distance_style == 'euc':
        distance = tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=1))
    elif FLAGS.distance_style == 'cosine':
        distance = inner_product(x, y)/tf.sqrt(inner_product(x,x) * inner_product(y,y))
    elif FLAGS.distance_style == 'inner_product':
        distance = inner_product(x, y)
    # distance = tf.exp(-distance)
    # d = tf.map_fn(fn=lambda s:tf.fill(value=s, dims=onehot_label.shape), elems=distance)
    d = tf.map_fn(fn=lambda s:tf.fill(value=s, dims=(FLAGS.way_num, )), elems=distance)
    return d
def get_dist_category(x, y, onehot_label):
    dist = distance(x, y, onehot_label[0])
    return dist*onehot_label

## Loss functions
def mse(pred, label):
    # pred = tf.reshape(pred, [-1])
    # label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label), axis=1)

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size