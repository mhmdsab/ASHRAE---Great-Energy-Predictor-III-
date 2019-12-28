import tensorflow as tf

from setup import params_setup
from model_util import create_graph, load_model
from trainer import train


def main():
    para = params_setup()

    graph, model = create_graph(para)

    with tf.Session(graph=graph) as sess:
        load_model(para, sess, model)

        try:
            if para.mode == 'train':
                train(para, sess, model)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


if __name__ == '__main__':
    main()
