from werkzeug.wrappers import Request, Response
from werkzeug.utils import secure_filename
import os

import tensorflow as tf


from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from pickle import load
from flask import Flask, request, jsonify, render_template


from sklearn.feature_extraction.text import CountVectorizer
import os
# os.chdir('Flickr8k')



import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re

import numpy as np
import pandas as pd 
# from PIL import Image
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# tf.compat.v1.enable_eager_execution()


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


from flask_caching import Cache
# import predict

app = Flask(__name__) 
i=1
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0.002

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def index():
    return render_template('test.html')



class Rnn_Global_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size,scoring_type):
        super(Rnn_Global_Decoder, self).__init__()
        

        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        
        
        self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        
        self.wc = tf.keras.layers.Dense(units, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

        #For Attention
        self.wa = tf.keras.layers.Dense(units)
        self.wb = tf.keras.layers.Dense(units)
        
        #For Score 3 i.e. Concat score
        self.Vattn = tf.keras.layers.Dense(1)
        self.wd = tf.keras.layers.Dense(units, activation='tanh')

        self.scoring_type = scoring_type

        
    def call(self, sequence, features,hidden):
        
        # features : (64,49,256)
        # hidden : (64,512)
        
        embed = self.embedding(sequence)
        # embed ==> (64,1,256) ==> decoder_input after embedding (embedding dim=256)
       
        output, state = self.gru(embed)       
        #output :(64,1,512)

        score=0
        
        #Dot Score as per paper(Dot score : h_t (dot) h_s') (NB:just need to tweak gru units to 256)
        '''----------------------------------------------------------'''
        if(self.scoring_type=='dot'):
          xt=output #(64,1,512)
          xs=features #(256,49,64)  
          score = tf.matmul(xt, xs, transpose_b=True) 
               
          #score : (64,1,49)

        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''



        # General Score as per Paper ( General score: h_t (dot) Wa (dot) h_s')
        '''----------------------------------------------------------'''
        if(self.scoring_type=='general'):
          score = tf.matmul(output, self.wa(features), transpose_b=True)
          # score :(64,1,49)
        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''


        # Concat score as per paper (score: VT*tanh(W[ht;hs']))    
        '''----------------------------------------------------------'''
        #https://www.tensorflow.org/api_docs/python/tf/tile
        if(self.scoring_type=='concat'):
          tiled_features = tf.tile(features, [1,1,2]) #(64,49,512)
          tiled_output = tf.tile(output, [1,49,1]) #(64,49,512)
          
          concating_ht_hs = tf.concat([tiled_features,tiled_output],2) ##(64,49,1024)
          
          tanh_activated = self.wd(concating_ht_hs)
          score =self.Vattn(tanh_activated)
          #score :(64,49,1), but we want (64,1,49)
          score= tf.squeeze(score, 2)
          #score :(64,49)
          score = tf.expand_dims(score, 1)
          
          #score :(64,1,49)
        '''----------------------------------------------------------'''
        '''----------------------------------------------------------'''



        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # alignment :(64,1,49)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, features)
        # context : (64,1,256)
        
        # Combine the context vector and the LSTM output
        
        output = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)
        # output: concat[(64,1,256):(64,1,512)] = (64,768)

        output = self.wc(output)
        # output :(64,512)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(output)
        # logits/predictions: (64,8239) i.e. (batch_size,vocab_size))

        return logits, state, alignment

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


@app.route('/predict', methods=['POST'])
def predict():
    resp=Response()
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    checkpoint_path = "C:\\Users\\poorn\\Downloads\\image_project\\project\\checkpoint\\train\\ckpt-7"
    train_captions = load(open('./captions.pkl', 'rb'))

    
    def tokenize_caption(top_k,train_captions):
        # Choosing the top k words from vocabulary
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        # oov_token: if given, it will be added to word_index 
        # and used to replace 
        # out-of-vocabulary words during text_to_sequence calls
        
        tokenizer.fit_on_texts(train_captions)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Map '<pad>' to '0'
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # Create the tokenized vectors
        train_seqs = tokenizer.texts_to_sequences(train_captions)
        return train_seqs, tokenizer

    top_k = 5000
    train_seqs , tokenizer = tokenize_caption(top_k ,train_captions)



    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Find the minimum length of any caption in our dataset
    def calc_min_length(tensor):
        return min(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    min_length = calc_min_length(train_seqs)


    #restoring the model
   
    # class Rnn_Local_Decoder(tf.keras.Model):
    #     def __init__(self, embedding_dim, units, vocab_size):
    #         super(Rnn_Local_Decoder, self).__init__()
    #         self.units = units

    #         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    #         self.gru = tf.keras.layers.GRU(self.units,
    #                                     return_sequences=True,
    #                                     return_state=True,
    #                                     recurrent_initializer='glorot_uniform')
            
    #         self.fc1 = tf.keras.layers.Dense(self.units)

    #         self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
    #         self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    #         self.fc2 = tf.keras.layers.Dense(vocab_size)

    #         # Implementing Attention Mechanism 
    #         self.Uattn = tf.keras.layers.Dense(units)
    #         self.Wattn = tf.keras.layers.Dense(units)
    #         self.Vattn = tf.keras.layers.Dense(1)
            


    #     def call(self, x, features, hidden):
            
    #         # features shape ==> (64,49,256) ==> Output from ENCODER
            
    #         # hidden shape == (batch_size, hidden_size) ==>(64,512)
    #         # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)
            
    #         hidden_with_time_axis = tf.expand_dims(hidden, 1)
            
    #         # score shape == (64, 49, 1)
    #         # Attention Function
    #         '''e(ij) = f(s(t-1),h(j))'''
    #         ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
    #         score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
    #         # self.Uattn(features) : (64,49,512)
    #         # self.Wattn(hidden_with_time_axis) : (64,1,512)
    #         # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
    #         # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
    #         # you get 1 at the last axis because you are applying score to self.Vattn
            
            
    #         # Then find Probability using Softmax
    #         '''attention_weights(alpha(ij)) = softmax(e(ij))'''
    #         attention_weights = tf.nn.softmax(score, axis=1)
    #         # attention_weights shape == (64, 49, 1)

            
    #         # Give weights to the different pixels in the image
    #         ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) ''' 
    #         context_vector = attention_weights * features
    #         context_vector = tf.reduce_sum(context_vector, axis=1)
    #         # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
    #         # context_vector shape after sum == (64, 256)
            
            
    #         # x shape after passing through embedding == (64, 1, 256)
    #         x = self.embedding(x)
            
    #         # x shape after concatenation == (64, 1,  512)
    #         x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #         # passing the concatenated vector to the GRU
    #         output, state = self.gru(x)

    #         # shape == (batch_size, max_length, hidden_size)
    #         x = self.fc1(output)

    #         # x shape == (batch_size * max_length, hidden_size)
    #         x = tf.reshape(x, (-1, x.shape[2]))

    #         # Adding Dropout and BatchNorm Layers
    #         x= self.dropout(x)
    #         x= self.batchnormalization(x)
    #         # output shape == (64 * 512)
    #         x = self.fc2(x)
    #         # shape : (64 * 8329(vocab))
    #         return x, state, attention_weights

    #     def reset_state(self, batch_size):
    #         return tf.zeros((batch_size, self.units))
    
    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index) + 1 
    # decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)
    decoder = Rnn_Global_Decoder(embedding_dim,units,vocab_size,"general")

    class VGG16_Encoder(tf.keras.Model):
        # This encoder passes the features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(VGG16_Encoder, self).__init__()
            # shape after fc == (batch_size, 49, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)
            self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

        def call(self, x):
            #x= self.dropout(x)
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x

    encoder = VGG16_Encoder(embedding_dim)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt.restore(checkpoint_path)

    # to_predict_list = request.form.to_dict()
    # Image_path = to_predict_list['pic_url']
    f= request.files['pic']
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    try:
        os.remove(os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))
    except:
        print("No file")
    f.save(os.path.join(THIS_FOLDER,'static','css',secure_filename("temp5.jpg")))
    Image_path =  (os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))

    def load_image(filename):
        image_path=filename
        img = tf.io.read_file(image_path)
        print(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = preprocess_input(img)
        # print('hello',img.numpy())
        return img, image_path
        # print(filename)
        # image = load_img(filename)
        # # # convert the image pixels to a numpy array
        # image = img_to_array(image)
        # # image = (tf.convert_to_tensor(image))
        # # # reshape data for the model
        # image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
        # # # prepare the image for the VGG model
        # image = preprocess_input(image)
        # # print(image)
        # # image = (tf.convert_to_tensor(image))
        # # image = tf.cast(image, tf.float32)
        # # image = tf.image.encode_jpeg(image)
        # img = tf.image.decode_jpeg(image, channels=3)
        # img = tf.image.resize(img, (224, 224))
        # img = preprocess_input(img)
        # return img, filename
        # return image, filename

  
    # if request.method == 'POST':
    #     f= request.files['pic']
    #     print(f)
    #     print(request.files)
    #     # f.save(secure_filename("temp.jpg"))

    #     THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    #     try:
    #         os.remove(os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))
    #     except:
    #         print("No file")
    #     f.save(os.path.join(THIS_FOLDER,'static','css',secure_filename("temp5.jpg")))
    #     load_image(f)


    print("Hi",Image_path)
    attention_features_shape = 49
    image_model = tf.keras.applications.VGG16(include_top=False,weights='imagenet')
    new_input = image_model.input # Any arbitrary shapes with 3 channels
    hidden_layer = image_model.layers[-1].output
    feat_extrac_model = tf.keras.Model(new_input, hidden_layer)
    
    def evaluate(image):
        attn_plt = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = feat_extrac_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attn_plt[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attn_plt

            dec_input = tf.expand_dims([predicted_id], 0)

        attn_plt = attn_plt[:len(result), :]
        return result, attn_plt


    new_img =  Image_path
    result, attention_plot = evaluate(new_img)
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass

    captn =' '
    return render_template('next.html', prediction_text=(captn.join(result).rsplit(' ', 1)[0]),ttt=os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
