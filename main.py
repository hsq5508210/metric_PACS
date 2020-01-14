from tensorflow.python.platform import flags
from data_generator import DataGenerator
from model import Model
import tensorflow as tf
FLAGS = flags.FLAGS
##config dataset
flags.DEFINE_string("data_PATH", default="/home/y192202040/Projects/PACS/", help="The dataset's path.")
flags.DEFINE_string("split_txt_PATH", default="/home/y192202040/Projects/metric_PACS/pacs_filename", help="file with how to split row data.")
flags.DEFINE_string("data_source", default="PACS", help="The dataset's name.")
flags.DEFINE_integer("image_size", default=227, help="input image channels.")
flags.DEFINE_integer("model", default=4, help="The num of data model.")
flags.DEFINE_integer("num_class", default=7, help="The num of category.")

##config model
flags.DEFINE_integer("k_neighbor", default=1, help="the number of k-nearest neighbors.")
flags.DEFINE_integer("input_dim", default=3, help="input image channels.")
flags.DEFINE_string("backbone", default="Conv", help="Model name.")
flags.DEFINE_integer("filter_num", default=64, help="Model name.")
flags.DEFINE_string("distance_style", default="euc", help="how to compute the distance.")
flags.DEFINE_bool("maxpool", default=True, help="use maxpool or not.")
flags.DEFINE_string("norm", default="None", help="choose norm style.")
flags.DEFINE_bool("max_pool", default=True, help="use maxpool or not")



##config train
flags.DEFINE_integer("episode_tr", default=5, help="the total number of training episodes.")
flags.DEFINE_integer("episode_val", default=1000, help="the total number of evaluate episodes.")
flags.DEFINE_integer("episode_ts", default=1000, help="the total number of testing episodes.")
flags.DEFINE_integer("support_num", default=2, help="Num of support per class per model.")
flags.DEFINE_integer("query_num", default=5, help="Num of query per class per model.")
flags.DEFINE_integer("way_num", default=5, help="the number of classify ways.")
flags.DEFINE_integer("iteration", default=60000, help="iterations.")
flags.DEFINE_float("lr", default=0.0001, help="learning rate.")
flags.DEFINE_bool("train", default=True, help="Train or not.")



def train(model, sess):
    # init = tf.global_variables_initializer()
    with sess.as_default():
        data_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num)
        all_tasks = data_generator.make_data_tensor()
        iteration = FLAGS.iteration
        episode = FLAGS.episode_tr
        # model.construct_model(input_tensor=all_tasks[0])
        # model.construct_model()
        trainop, loss = model.trainop()
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iteration):
            for j in range(episode):
                task = all_tasks[j]
                # print(task)
                # task['support_set'][1], task['query_set'][1] = tf.one_hot( task['support_set'][1]), tf.one_hot(task['query_set'][1])
                feed_dic = {model.support_x:task['support_set'][0], model.query_x:task['query_set'][0],
                            model.support_y:task['support_set'][1], model.query_y:task['query_set'][1]}
                # print(task['support_set'][0].shape)
                # model.support_x=task['support_set'][0]
                # model.query_x=task['query_set'][0]
                # model.support_y=task['support_set'][1]
                # model.query_y=task['query_set'][1]

                # predict, mqy = sess.run([model.get_loss((model.support_x, model.support_y, model.query_x, model.query_y), model.weights), model.support_y], feed_dict=feed_dic)
                # loss,_ = sess.run(
                #     [model.get_loss((model.support_x, model.support_y, model.query_x, model.query_y), model.weights),
                #      model.support_y], feed_dict=feed_dic)
                # t_vars = tf.global_variables()
                # print([v.name for v in t_vars])
                _, loss1 = sess.run([trainop, loss], feed_dict=feed_dic)
                print(loss1)

                # sess.run(model.trainop(), feed_dic)


def test():
    pass
def main():


    sess = tf.InteractiveSession()
    model = Model(sess=sess)
    model.construct_model()

    # with sess.as_default():
    #     if FLAGS.train:
    #         train(model, sess)
    # tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()
    train(model,sess)


if __name__ == '__main__':
    main()


