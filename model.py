import tensorflow as tf
from tensorflow.python.util import nest

def model(para, encoder_inputs, encoder_meter_reading, decoder_inputs):
    
    '''
    para: hparameters in setup file
    encoder_inputs: inputs for encoder [batch_size, encoder_time_steps, number_of features]
    encoder_meter_reading: meter reading feature [batch_size, encoder_time_steps]
    decoder_inputs: inputs for encoder [batch_size, decoder_time_steps, number_of features-1] (meter_reading feature removed)
    '''
    def encoder(x, num_layers, num_units, involve_dense, num_dense, add_dropout, rate):
        '''
        CuDNNLSTM layer with custom configurations
        x: layer inputs [batch_size, time_steps, num_features]
        num_layers: number of layers to be designed
        involves_dense: (boolean) default is False. if True, add dense layer to output states of each layer.
        num_dense: number of units for involves_dense
        add_dropout: default is False. if True, adds dropout layer to output states of each layer and to added dense layers 
                     if involve_dense is set to True
        rate: dropout rate
        '''
        layers = [tf.keras.layers.CuDNNLSTM(num_units, return_sequences = True, return_state = True) for i in range(num_layers)]
        
        layer_inputs = [x]
        c_list = []
        h_list = []
        
        for i in range(num_layers):
            outputs, h, c = layers[i](layer_inputs[-1])
            
                
            if involve_dense:
                outputs = tf.layers.Dense(num_dense)(outputs)
                    
            if add_dropout:
                outputs = tf.keras.layers.Dropout(rate = rate)(outputs)
            
            layer_inputs.append(outputs)
            h_list.append(h)
            c_list.append(c)
            
        return layer_inputs[1:], tuple(tf.nn.rnn_cell.LSTMStateTuple(c = c, h = h) for c,h in zip(c_list, h_list))  
    


    encoder_states, decoder_initial_state = encoder(encoder_inputs, para.model_num_layers, para.state_num_units, 
                                                    involve_dense = para.involve_dense_in_encoder,  
                                                    num_dense = para.num_dense_in_encoder, 
                                                    add_dropout = para.add_dropout_in_encoder, 
                                                    rate = para.encoder_dropout_rate)
    

    def decoder(num_layers, num_units, dropout_rate):
        '''
        num_layers: number of decoder layers
        num_units: list or integer. if integer, all layers will be with the same num_units
        dropoute_rate: list or float. if float, all gates and states will have same dropout rate.
        '''
        
        if not isinstance(num_units, list):
            num_units = [num_units for i in range(num_layers)]
    
        if not isinstance(dropout_rate, list):
            dropout_rate = [dropout_rate for i in range(3)]
            
            
        layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units[i]),
                                                input_keep_prob = 1-dropout_rate[0],
                                                output_keep_prob = 1-dropout_rate[1],
                                                state_keep_prob = 1-dropout_rate[2]) \
                                                for i in range(num_layers)]
        
        layers = tf.nn.rnn_cell.MultiRNNCell(layers)
        
        return layers

    
    def attention_mechanism(query, meter_reading, encoder_states_, attn_vec_size_):
        '''
        query: LSTMStateTuple from previous step
        meter_reading: meter readings tensor from encoder input
        encoder_states_: all h states from upper layer
        attn_vec_size_: number of units of attention 
        '''
        conv2d = tf.nn.conv2d
        reduce_sum = tf.reduce_sum
        softmax = tf.nn.softmax
        tanh = tf.nn.tanh
        Linear = tf.keras.layers.Dense(attn_vec_size_)
        encoder_units = encoder_states_.get_shape().as_list()[-1]
        time_steps = encoder_states_.get_shape().as_list()[1]
        
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            k = tf.get_variable('k', shape = [1, 1, encoder_units, attn_vec_size_])
            v = tf.get_variable('v', shape = [attn_vec_size_])
            
        states = tf.reshape(encoder_states_,[-1, time_steps, 1, encoder_units])
        hidden = conv2d(states, k, [1,1,1,1], 'SAME')
        
        query = tf.concat(nest.flatten(query), axis = 1)
        y = Linear(tf.concat([query ,meter_reading], 1))
        y = tf.reshape(y, [-1,1,1,attn_vec_size_])
        s = reduce_sum(v * tanh(y + hidden), [2,3])
        a = softmax(s)
        new_attention = tf.reduce_sum(tf.reshape(a,[-1, time_steps, 1, 1]) * hidden, [1,2])
        return new_attention


    decoder_timesteps = decoder_inputs.get_shape().as_list()[1]

    def cond_fn(time, *args):
        return time < decoder_timesteps
        

    def loop_fn(time, initial_prediction, decoder_initial_state_, decoder_outputs):
        
        attention = attention_mechanism(decoder_initial_state_[0], encoder_meter_reading, encoder_states[-1], para.attn_vec_size)
        cell_input = tf.concat([decoder_inputs[:,time,:], initial_prediction, attention], axis = 1)
        new_output, new_state = decoder(para.model_num_layers, para.state_num_units, para.decoder_dropout_rate)\
                                                                                (cell_input, decoder_initial_state_)
        new_prediction = tf.keras.layers.Dense(1)(new_output)
        decoder_outputs = decoder_outputs.write(time, new_prediction)
        time+=1
        return time, new_prediction, new_state, decoder_outputs

    

    init_vars = [tf.constant(0, dtype = tf.int32),
                 encoder_meter_reading[:,-1:],
                 decoder_initial_state,
                 tf.TensorArray(tf.float32, size = decoder_timesteps)]

    
    _, _, _, decoder_predictions = tf.while_loop(cond_fn, loop_fn, init_vars)
    decoder_predictions = decoder_predictions.stack()
    decoder_predictions = tf.squeeze(tf.transpose(decoder_predictions, [1,0,2]), axis = 2)
    
    return decoder_predictions




class seq2seq:
    def __init__(self, para, data_generator):
        self.para = para
        self.data_generator = data_generator
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._build_graph()
        if self.para.mode == "train":
            self._build_optimizer()

        self.saver = tf.train.Saver(max_to_keep=self.para.max_saver)


    def _build_graph(self):
        
        self.encoder_input, self.decoder_input, self.encoder_meter_reading, self.decoder_meter_reading, self.decoder_meter_reading_denorm, self.example_mean, self.example_std = self.data_generator.inputs(self.para.mode, self.para.batch_size)
            
        self.encoder_embedding = tf.keras.layers.Dense(self.para.embedding_len)(self.encoder_input)
        self.decoder_embedding = tf.keras.layers.Dense(self.para.embedding_len)(self.decoder_input)
        
        self.model_preds = model(self.para, self.encoder_embedding, self.encoder_meter_reading, self.decoder_embedding)

        if self.para.mode == "train" or self.para.mode == "valid":
            self.labels = self.decoder_meter_reading
            self.loss = tf.reduce_mean(tf.keras.losses.MSE(self.labels, self.model_preds))
            self.kaggle_loss = tf.reduce_mean(tf.keras.losses.MSLE(self.decoder_meter_reading_denorm, self.model_preds*self.example_std + self.example_mean))
            


    def _build_optimizer(self):

        trainable_variables = tf.trainable_variables()
        
        if self.para.decay_lr:
            lr = tf.train.exponential_decay(self.para.lr, self.global_step,
                                            self.para.lr_decay_rate, 0.995, staircase=True)
        else:
            lr = self.para.lr
            
        self.opt = tf.train.AdamOptimizer(lr)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.para.max_global_norm)
        self.update = self.opt.apply_gradients(zip(clip_gradients, trainable_variables),
                                               global_step=self.global_step )
        

    def _compute_loss(self, preds, labels):
        
        return tf.reduce_mean\
              (tf.losses.mean_squared_error(labels = labels, 
                                            predictions = preds))
              
