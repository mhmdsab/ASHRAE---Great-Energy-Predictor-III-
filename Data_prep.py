import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import os
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import OneHotEncoder as OHE



class DataGenerator(metaclass=ABCMeta):
    def __init__(self, para):
        self.iterator = None
        self.para = para

    def inputs(self, mode, batch_size, num_epochs=None):
        """Reads input data num_epochs times.
        Args:
        mode: String for the corresponding tfrecords ('train', 'validation')
        batch_size: Number of examples per returned batch.
        """
        if mode != "train" and mode != "valid":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'validation'".format(mode))

        filename = self.para.tf_records_url + '/' +mode + "_.tfrecord"

        with tf.name_scope("input"):
            # TFRecordDataset opens a binary file and
            # reads one record at a time.
            # `filename` could also be a list of filenames,
            # which will be read in order.
            dataset = tf.data.TFRecordDataset(filename)

            # The map transformation takes a function and
            # applies it to every element
            # of the dataset.
            dataset = dataset.map(self._decode)


            # The shuffle transformation uses a finite-sized buffer to shuffle
            # elements in memory. The parameter is the number of elements in the
            # buffer. For completely uniform shuffling, set the parameter to be
            # the same as the number of elements in the dataset.
            if mode == "train":
                dataset = dataset.shuffle(2380*16)

            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(batch_size)

            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()


    @abstractmethod
    def _decode(self, serialized_example):
        pass

    
    

class generate_dataset(DataGenerator):
    def __init__(self, para):
        self.para = para
        self.train_data_range = pd.date_range(para.TS_start_date,para.TS_end_date , freq=para.sampling_frequency)
        self.train_written_examples = 0
        self.valid_written_examples = 0
        self.train_excluded_examples = 0
        self.valid_excluded_examples = 0
        self.mean_values = {}
        self.std_values = {}        
        self.three_months_corr = {}
        self.six_months_corr = {}
        self.features = {'x_encoder': tf.FixedLenFeature([self.para.in_sequence_window, 25], dtype=tf.float32),
                         'x_decoder': tf.FixedLenFeature([self.para.out_sequence_window, 24], dtype=tf.float32),
                         'encoder_meter_reading': tf.FixedLenFeature([self.para.in_sequence_window], dtype=tf.float32),
                         'decoder_meter_reading': tf.FixedLenFeature([self.para.out_sequence_window], dtype=tf.float32),
                         'decoder_meter_reading_denorm': tf.FixedLenFeature([self.para.out_sequence_window], dtype=tf.float32),
                         'mean': tf.FixedLenFeature([1], dtype=tf.float32),
                         'std':tf.FixedLenFeature([1], dtype=tf.float32)}
        self.generate_encoders()
        self.generate_dataset()
        self.organize('train', 'valid')
        super().__init__(para)
    
    
    def generate_encoders(self):
        self.encoding_dict = dict([('month_number' , OHE().fit(np.arange(1, 13).reshape(-1,1))),
                                   ('weekday_number' , OHE().fit(np.arange(7).reshape(-1,1))),
                                   ('meter' , OHE().fit(np.arange(4).reshape(-1,1)))])

    
    def generate_dataset(self):
        print('generating raw dataset')
        train = pd.read_csv(self.para.train_url)
        meta = pd.read_csv(self.para.meta_url).loc[:,['building_id', 'primary_use']]
        dataset = pd.merge(train, meta, left_on = 'building_id', right_on = 'building_id')
        dataset['timestamp']= pd.to_datetime(dataset['timestamp'])
        self.dataset = dataset.groupby(['building_id', 'meter'])
        self.groups = list(self.dataset.groups.keys())
    
    
    def autocorrelate(self, df, group):
        three_months_corr = df['meter_reading'].autocorr(lag = int(3*30*(24/int(self.para.sampling_frequency[0]))))
        self.three_months_corr[group] = three_months_corr
        
        six_months_corr = df['meter_reading'].autocorr(lag = int(6*30*(24/int(self.para.sampling_frequency[0]))))
        self.six_months_corr[group] = six_months_corr
        
        df['three_months_lag_autocorr'] = three_months_corr * np.ones(shape = len(df))#shape = (8784)
        df['six_months_lag_autocorr'] = six_months_corr *np.ones(shape = len(df))#shape = (8784)
        return df.astype('float32')
    
    
    
    def One_Hot_Enode(self, df):
        for feature in self.encoding_dict.keys():
            raw_series = df[feature].values.reshape(-1, 1)
            One_Hot_Encoded_array = self.encoding_dict[feature].transform(raw_series).todense()[:,:-1]
            
            for i in range(One_Hot_Encoded_array.shape[1]):
                OHE_feature_name = feature+'_'+str(i)
                df[OHE_feature_name] = One_Hot_Encoded_array[:,i]
            
        return df.astype('float32')


    
    def pad_time_series(self, df):
        padding_df = pd.DataFrame()
        padding_df['any'] = np.ones(shape = (self.para.building_total_len//int(self.para.sampling_frequency[0])))
        padding_df = padding_df.set_index(self.train_data_range)
        merged = pd.merge(padding_df, df, how = 'left', left_index=True, right_index=True).drop('any',1).fillna(0)
        return merged.astype('float32')
    
    
    
    def Normalize_Pad_Split(self, df, group):
        mean = np.mean(df['meter_reading'][df['month_number'] <= self.para.features_extractor_len])
        std = np.std(df['meter_reading'][df['month_number'] <= self.para.features_extractor_len])
        
        if (mean!= mean) or (std!= std):
            mean = np.mean(list(self.mean_values.values()))
            std = np.mean(list(self.std_values.values()))
            
        self.mean_values[group] = mean
        self.std_values[group] = std
        
        df = self.pad_time_series(df)
        
        df['meter_reading_normalized'] = df['meter_reading'].map(lambda x:self.Normalize(x, mean, std))
        df['three_months_lag'] = df['three_months_lag'].map(lambda x:self.Normalize(x, mean, std))
        df['six_months_lag'] = df['six_months_lag'].map(lambda x:self.Normalize(x, mean, std))
        
        train_df = df.loc[:self.para.train_end_date,:].drop(['month_number', 'weekday_number'], 1)
        valid_df = df.loc[self.para.valid_start_date:,:].drop(['month_number', 'weekday_number'], 1)
        return train_df, valid_df
    
        
    
    def organize(self, train_name, valid_name):
        print('organizing raw dataset')
        
        if (os.path.exists(self.para.tf_records_url+'/'+'{}_.tfrecord'.format(train_name))) & \
            (os.path.exists(self.para.tf_records_url+'/'+'{}_.tfrecord'.format(valid_name))):
            self.para.train_kickoff = 'not_first_time'
            train_name, valid_name = '_','_'
        
        with tf.python_io.TFRecordWriter(self.para.tf_records_url+'/'+'{}_.tfrecord'.format(train_name)) as train_writer:
            
            with tf.python_io.TFRecordWriter(self.para.tf_records_url+'/'+'{}_.tfrecord'.format(valid_name)) as valid_writer:
        
                for group in tqdm(self.groups):
                    
                    building_df = self.dataset.get_group(group).set_index('timestamp')
                    
                    building_df = building_df.resample(self.para.sampling_frequency).mean().fillna(method = 'ffill')
        
                    building_df = self.add_time_features(building_df)
                    
                    building_df = self.One_Hot_Enode(building_df)
                    
                    building_df = self.add_timelags(building_df)
                    
                    building_df = self.autocorrelate(building_df, group)
                                
                    train_building_df, valid_building_df = self.Normalize_Pad_Split(building_df, group)
                    
                    
                    if  self.para.train_kickoff == 'first_time':                    
                        self._convert_to_tfrecord(train_building_df, train_writer, train_name, group)
                        self._convert_to_tfrecord(valid_building_df, valid_writer, valid_name, group)

                    
            

    def _convert_to_tfrecord(self, df, writer, mode, group):
        
        df_meter_reading_normalized = df['meter_reading_normalized']
        
        df_meter_reading_unnormalized = df['meter_reading']
        
        df_x = df.drop(['building_id', 'meter_reading', 'meter'],1)
        
        mean = self.mean_values[group]
        
        std = self.std_values[group]
        

            
        for i in range((len(df) - self.para.in_sequence_window)//self.para.out_sequence_window):
            
            start = i * self.para.out_sequence_window
            end = start + self.para.in_sequence_window
            y_end = end + self.para.out_sequence_window
            
            
            if ((df['meter_reading'].iloc[start:end] == 0).sum() < \
                                int(self.para.max_zeros_in_example * self.para.in_sequence_window)) \
                and (std>self.para.min_std):
                
                example = tf.train.Example(features=tf.train.Features(feature={
                        
                    'x_encoder': tf.train.Feature(
                        float_list=tf.train.FloatList(value = df_x.iloc[start:end,:].values.flatten())),
                                                       
                    'x_decoder': tf.train.Feature(
                        float_list=tf.train.FloatList(value = df_x.drop(['meter_reading_normalized'],1) \
                                                      .iloc[end:y_end,:].values.flatten())),
                    
                    'encoder_meter_reading': tf.train.Feature(
                        float_list=tf.train.FloatList(value = df_meter_reading_normalized[start:end]\
                                                                                  .values.flatten())),
                            
                    'decoder_meter_reading': tf.train.Feature(
                        float_list=tf.train.FloatList(value = df_meter_reading_normalized[end:y_end]\
                                                                                  .values.flatten())),
                            
                    'decoder_meter_reading_denorm': tf.train.Feature(
                        float_list=tf.train.FloatList(value = df_meter_reading_unnormalized[end:y_end]\
                                                                                  .values.flatten())),
                            
                    'mean': tf.train.Feature(
                        float_list=tf.train.FloatList(value = [mean])),
                            
                    'std': tf.train.Feature(
                        float_list=tf.train.FloatList(value = [std]))}))
                    
                    
                writer.write(example.SerializeToString())  
                
                if mode == 'train':
                    self.train_written_examples += 1
                    
                elif mode == 'valid':
                    self.valid_written_examples += 1
                    
                    
            else:
                
                if mode == 'train':
                    self.train_excluded_examples+=1
                
                elif mode == 'valid':
                    self.valid_excluded_examples+=1
            
                            


    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features=self.features)
        
        x_encoder = example['x_encoder']
        x_decoder = example['x_decoder']
        encoder_meter_reading = example['encoder_meter_reading']
        decoder_meter_reading = example['decoder_meter_reading']
        decoder_meter_reading_denorm = example['decoder_meter_reading_denorm']
        mean = example['mean']
        std = example['std']
        return x_encoder, x_decoder, encoder_meter_reading, decoder_meter_reading, decoder_meter_reading_denorm, mean, std
    
    
    
    @staticmethod
    def Normalize(value, mean, std):
        return (value-mean)/(std + 1e-100)
    
    
    
    @staticmethod
    def add_timelags(df):
        df['three_months_lag'] = df['meter_reading'].shift(90*6).fillna(0)
        df['six_months_lag'] = df['meter_reading'].shift(90*6*2).fillna(0)
        return df.astype('float32')
    
    
    @staticmethod
    def add_time_features(df):
        df['weekday_number'] = df.index.weekday
        df['month_number'] = df.index.month
        return df
