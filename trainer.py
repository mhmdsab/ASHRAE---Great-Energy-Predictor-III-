import tensorflow as tf
from model_util import Early_Stopping, create_valid_graph, load_model
import numpy as np


def train(para, sess, model):
    valid_para, valid_graph, valid_model = create_valid_graph(para)
    
    with tf.Session(graph = valid_graph) as valid_sess:
        valid_sess.run(tf.global_variables_initializer())
        #training_phase
        early_stopper = Early_Stopping(sess, valid_model.saver, para)
        
        for epoch in range(para.num_epochs):
            sess.run(model.data_generator.iterator.initializer)
            epoch_cost, epoch_kaggle_cost = [], []
            i = 1
            while True:
                try:
                    
                    _, loss, kaggle_loss, global_step = sess.run([model.update,
                                                                  model.loss,
                                                                  model.kaggle_loss,
                                                                  model.global_step])
    
                    epoch_cost.append(loss)
                    epoch_kaggle_cost.append(kaggle_loss)
                    print('{} batches finished, loss: {}, kaggle_loss: {}'\
                          .format(i, loss, kaggle_loss))
                    i+=1
                    
                except tf.errors.OutOfRangeError:
                    
                    model.saver.save(sess, para.every_epoch_checkpoint_path)
                
                    print('{} epochs finished, epoch loss: {} , epoch_kaggle_loss: {}'.format(epoch+1, np.mean(epoch_cost),
                                                                                      np.mean(epoch_kaggle_cost)))
                
                    break
             
            
            #validation_phase
            load_model(valid_para, valid_sess, valid_model)
            valid_sess.run(valid_model.data_generator.iterator.initializer)
            
            assert 'update' not in [x for x in valid_model.__dict__.keys() if x[:1] != '_']
            
            valid_epoch_cost, valid_epoch_kaggle_cost = [], []
            
            while True:
                try:
                    valid_loss, valid_kaggle_loss = valid_sess.run([valid_model.loss, valid_model.kaggle_loss])
                    valid_epoch_cost.append(valid_loss)
                    valid_epoch_kaggle_cost.append(valid_kaggle_loss)
                    
                except tf.errors.OutOfRangeError:
                    
                    print('valid_epoch loss: {} , valid_epoch_kaggle_loss: {}'.format(np.mean(valid_epoch_cost),
                                                                          np.mean(valid_epoch_kaggle_cost)))
                    
                    early_stopper.save_best_model(np.mean(valid_epoch_kaggle_cost))

                    break
                
            if early_stopper.counter > para.epochs_to_wait:
                break
                