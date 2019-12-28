# ASHRAE---Great-Energy-Predictor-III-

a sequence to sequence model with attention network to account for long term sequences.

Model has two main parts: encoder and decoder.
Encoder is cuDNN LSTM. cuDNN works much faster (5x-10x) than native Tensorflow RNNCells.

Decoder is TF LSTMBlockCell, wrapped in tf.while_loop() construct. Code inside the loop gets prediction from previous step and
appends it to the input features for current step.

this model provides a functionality of resampling the hour by hour data which speeds up training 
at the expenses of model accuracy giving an approximate solution.
