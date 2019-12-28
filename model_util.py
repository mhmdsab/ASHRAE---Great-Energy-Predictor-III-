import tensorflow as tf
from Data_prep import generate_dataset
from model import seq2seq
from copy import deepcopy


class Early_Stopping:
    
    def __init__(self, sess, saver, para):
        self.sess = sess
        self.saver = saver
        self.epochs_to_wait = para.epochs_to_wait
        self.test_loss_list = []
        self.counter = 0
        self.para = para
        
    def add_loss(self):
        self.test_loss_list.append(self.test_loss)
        
    def save_best_model(self,test_loss):
        self.test_loss = test_loss
        self.add_loss()
        
        if min(self.test_loss_list) == self.test_loss_list[-1]:
            self.best_loss = self.test_loss
            self.saver.save(self.sess, self.para.best_checkpoint_path)
            self.counter = 0
            
        else:
            print('loss did not improve since test loss was {}'.format(self.best_loss))
            self.counter += 1
            

def create_graph(para):
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.glorot_uniform_initializer()
        data_generator = generate_dataset(para)
        with tf.variable_scope('model', initializer = initializer):
            model = seq2seq(para, data_generator)
            
    return graph, model
    

def create_valid_graph(para):
    valid_para = deepcopy(para)
    valid_para.mode = 'valid'
    valid_para.encoder_dropout_rate = 0
    valid_para.decoder_dropout_rate = 0
    valid_graph, valid_model = create_graph(valid_para)    
    return valid_para, valid_graph, valid_model



def load_model(para, sess, model):
    ckpt = tf.train.get_checkpoint_state(para.every_epoch_checkpoint_path)
    
    if ckpt:
        model.saver.restore(sess, para.every_epoch_checkpoint_path)
        
    else:
        sess.run(tf.global_variables_initializer())
        
        