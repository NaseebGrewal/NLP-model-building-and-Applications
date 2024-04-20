#!/usr/bin/env python
# coding: utf-8

# 
# # Assignment 2: Transformer Summarizer
# 
# Welcome to the second assignment of course 4. In this assignment you will explore summarization using the transformer model. Yes, you will implement the transformer decoder from scratch, but we will slowly walk you through it. There are many hints in this notebook so feel free to use them as needed. Actually by the end of this notebook you will have implemented the full transformer (both encoder and decoder) but you will only be graded on the implementation of the decoder as the encoder is provided for you.
# 

# ## Table of Contents
# 
# - [Introduction](#0)
# - [1 - Importing the Dataset](#1)
# - [2 - Preprocess the Data](#2)
# - [3 - Positional Encoding](#3)
# - [4 - Masking](#4)
# - [5 - Self-attention](#5)
#     - [Exercise 1 - scaled_dot_product_attention](#ex-1)
# - [6 - Encoder](#6)
#     - [6.1 - Encoder Layer](#6-1)
#     - [6.2 - Full Encoder](#6-2)
# - [7 - Decoder](#7)
#     - [7.1 - Decoder Layer](#7-1)
#     - [Exercise 2 - DecoderLayer](#ex-2)
#     - [7.2 - Full Decoder](#7-2)
#     - [Exercise 3 - Decoder](#ex-3)
# - [8 - Transformer](#8)
#     - [Exercise 4 - Transformer](#ex-4)
# - [9 - Initialize the Model](#9)
# - [10 - Prepare for Training the Model](#10)
# - [11 - Summarization](#11)
#     - [Exercise 5 - next_word](#ex-5)
# - [12 - Train the Model](#12)
# - [13 - Summarize some sentences!](#13)
# 

# <a name='0'></a>
# ## Introduction
# 
# Summarization is an important task in natural language processing and could be useful for a consumer enterprise. For example, bots can be used to scrape articles, summarize them, and then you can use sentiment analysis to identify the sentiment about certain stocks. Who wants to read an article or a long email today anyway, when you can build a transformer to summarize text for you? Let's get started. By completing this assignment you will learn to:  
# 
# - Use built-in functions to preprocess your data
# - Implement DotProductAttention
# - Implement Causal Attention
# - Understand how attention works
# - Build the transformer model
# - Evaluate your model
# - Summarize an article
# 
# As you can tell, this model is slightly different than the ones you have already implemented. This is heavily based on attention and does not rely on sequences, which allows for parallel computing. 

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import utils

import textwrap
wrapper = textwrap.TextWrapper(width=70)

tf.keras.utils.set_random_seed(10)


# In[2]:


import w2_unittest


# <a name='1'></a>
# ## 1 - Import the Dataset
# You have the dataset saved in a .json file, which you can easily open with pandas. The loading function has already been taken care of in `utils.py`.

# In[3]:


data_dir = "data/corpus"

train_data, test_data = utils.get_train_test_data(data_dir)

# Take one example from the dataset and print it
example_summary, example_dialogue = train_data.iloc[10]
print(f"Dialogue:\n{example_dialogue}")
print(f"\nSummary:\n{example_summary}")


# <a name='2'></a>
# ## 2 - Preprocess the data
# 
# First you will do some preprocessing of the data and split it into inputs and outputs. Here you also remove some of the characters that are specific to this dataset and add the `[EOS]` (end of sentence) token to the end, like it was discussed in the lecture videos. You will also add a `[SOS]` (start of sentence) token to the beginning of the sentences.

# In[4]:


document, summary = utils.preprocess(train_data)
document_test, summary_test = utils.preprocess(test_data)


# Now perform the standard preprocessing with the tensorflow library. You will need to modify the filters, because you dont want the `[EOS]` tokens to be removed.
# 
# Then create the vocabulary by combining the data in the documents and the summaries and using `.fit_on_texts()`:

# In[5]:


# The [ and ] from default tokens cannot be removed, because they mark the SOS and EOS token.
filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
oov_token = '[UNK]'

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token, lower=False)

documents_and_summary = pd.concat([document, summary], ignore_index=True)

tokenizer.fit_on_texts(documents_and_summary)

inputs = tokenizer.texts_to_sequences(document)
targets = tokenizer.texts_to_sequences(summary)

vocab_size = len(tokenizer.word_index) + 1

print(f'Size of vocabulary: {vocab_size}')


# Now you can pad the tokenized sequences for the training data.
# 
# For the purpose of this notebook you need to limit the length of the sequences, as transformers are really big models and are not meant to be trained in such small environments.

# In[6]:


# Limit the size of the input and output data for being able to run it in this environment.
encoder_maxlen = 150
decoder_maxlen = 50

# Pad the sequences.
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

# Create the final training dataset.
BUFFER_SIZE = 10000
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# <a name='3'></a>
# ## 3 - Positional Encoding
# 
# In sequence to sequence tasks, the relative order of your data is extremely important to its meaning. When you were training sequential neural networks such as RNNs, you fed your inputs into the network in order. Information about the order of your data was automatically fed into your model. However, when you train a Transformer network using multi-head attention, you feed your data into the model all at once. While this dramatically reduces training time, there is no information about the order of your data. This is where positional encoding is useful.
# 
# You have learned how to implement the positional encoding in one of this week's labs. Here you will use the `positional_encoding` function to create positional encodings for your transformer. The function is already implemented for you.

# In[7]:


def positional_encoding(positions, d_model):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int): Maximum number of positions to be encoded 
        d_model (int): Encoding size 
    
    Returns:
        pos_encoding (tf.Tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """
    
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    # initialize a matrix angle_rads of all the angles 
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


# <a name='4'></a>
# ## 4 - Masking
# 
# There are two types of masks that are useful when building your Transformer network: the *padding mask* and the *look-ahead mask*. Both help the softmax computation give the appropriate weights to the words in your input sentence. 
# 
# You have already learned how to implement and use them in one of this week's labs. Here they are implemented for you.

# In[8]:


def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)
    
    Returns:
        mask (tf.Tensor): binary tensor of size (n, 1, m)
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
  
    # add extra dimensions to add the padding to the attention logits. 
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :] 


def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length (int): matrix size
    
    Returns:
        mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 


# <a name='5'></a>
# ## 5 - Self-Attention
# 
# As the authors of the Transformers paper state, "Attention is All You Need". 
# 
# <img src="images/self-attention.png" alt="Encoder" width="600"/>
# <caption><center><font color='purple'><b>Figure 1: Self-Attention calculation visualization</font></center></caption>
#     
# The use of self-attention paired with traditional convolutional networks allows for parallelization which speeds up training. You will implement **scaled dot product attention** which takes in a query, key, value, and a mask as inputs to return rich, attention-based vector representations of the words in your sequence. This type of self-attention can be mathematically expressed as:
# $$
# \text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}+{M}\right) V\tag{4}\
# $$
# 
# * $Q$ is the matrix of queries 
# * $K$ is the matrix of keys
# * $V$ is the matrix of values
# * $M$ is the optional mask you choose to apply 
# * ${d_k}$ is the dimension of the keys, which is used to scale everything down so the softmax doesn't explode
# 
# <a name='ex-1'></a>
# ### Exercise 1 - scaled_dot_product_attention 
# 
# Implement the function `scaled_dot_product_attention()` to create attention-based representations.
# 
# **Reminder**: The boolean mask parameter can be passed in as `none` or as either padding or look-ahead. 
#     
# * Multiply (1. - mask) by -1e9 before adding it to the scaled attention logits. 
# 
# **Additional Hints**
# * You may find [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul) useful for matrix multiplication (check how you can use the parameter transpose_b).
# * You can use [tf.keras.activations.softmax](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax) for softmax.

# In[40]:


# GRADED FUNCTION: scaled_dot_product_attention
def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

    Arguments:
        q (tf.Tensor): query of shape (..., seq_len_q, depth)
        k (tf.Tensor): key of shape (..., seq_len_k, depth)
        v (tf.Tensor): value of shape (..., seq_len_v, depth_v)
        mask (tf.Tensor): mask with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    ### START CODE HERE ### 
    # Multiply q and k transposed.  
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    
    # scale matmul_qk with the square root of dk  
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  
    
    # add the mask to the scaled tensor.  
    if mask is not None:  # Don't replace this None  
        scaled_attention_logits += (1. - mask) * -1e9  
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
    
    # Multiply the attention weights by v  
    output = tf.matmul(attention_weights, v)
    
    
#     # Multiply q and k transposed.  
#     matmul_qk = tf.matmul(q, k, transpose_b=True)  
  
#     # scale matmul_qk with the square root of dk  
#     dk = tf.cast(tf.shape(k)[-1], tf.float32)  
#     scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  
  
#     # add the mask to the scaled tensor.  
#     if mask is not None:  # Don't replace this None  
#         scaled_attention_logits += (mask * -1e9)  # change this line to multiply the mask  
  
#     # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.  
#     attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
  
#     # Multiply the attention weights by v  
#     output = tf.matmul(attention_weights, v)
 
    
#     # Multiply q and k transposed.
#     matmul_qk = None

#     # scale matmul_qk with the square root of dk
#     dk = tf.cast(None, tf.float32)
#     scaled_attention_logits = None

#     # add the mask to the scaled tensor.
#     if mask is not None:  # Don't replace this None
#         scaled_attention_logits += None

#     # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
#     attention_weights = None

#     # Multiply the attention weights by v
#     output = None
    
    ### END CODE HERE ###

    return output, attention_weights


# In[41]:


# Test your function!
q = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]).astype(np.float32)
k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [1, 1, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]]).astype(np.float32)
v = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [1, 1]]).astype(np.float32)
mask = np.array([[[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 0, 1, 1]]])

ou, atw = scaled_dot_product_attention(q, k, v, mask)
ou = np.around(ou, decimals=2)
atw = np.around(atw, decimals=2)

print(f"Output:\n {ou}")
print(f"\nAttention weigths:\n {atw}")


# ##### __Expected Output__
# 
# ```
# Output:
#  [[[1.   0.62]
#   [0.62 0.62]
#   [0.74 0.31]]]
# 
# Attention weigths:
#  [[[0.   0.38 0.   0.23 0.38]
#   [0.38 0.   0.   0.23 0.38]
#   [0.26 0.43 0.   0.16 0.16]]]
# ```

# In[42]:


# UNIT TEST
w2_unittest.test_scaled_dot_product_attention(scaled_dot_product_attention)


# Excellent work! You can now implement self-attention. With that, you can start building the encoder block! 

# <a name='6'></a>
# ## 6 - Encoder
# 
# The Transformer Encoder layer pairs self-attention and convolutional neural network style of processing to improve the speed of training and passes K and V matrices to the Decoder, which you'll build later in the assignment. In this section of the assignment, you will implement the Encoder by pairing multi-head attention and a feed forward neural network (Figure 2a). 
# <img src="images/encoder_layer.png" alt="Encoder" width="400"/>
# <caption><center><font color='purple'><b>Figure 2a: Transformer encoder layer</font></center></caption>
# 
# * `MultiHeadAttention` you can think of as computing the self-attention several times to detect different features. 
# * Feed forward neural network contains two Dense layers which we'll implement as the function `FullyConnected`
# 
# Your input sentence first passes through a *multi-head attention layer*, where the encoder looks at other words in the input sentence as it encodes a specific word. The outputs of the multi-head attention layer are then fed to a *feed forward neural network*. The exact same feed forward network is independently applied to each position.
#    
# * For the `MultiHeadAttention` layer, you will use the [MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention) implemented in Keras. If you're curious about how to split the query matrix Q, key matrix K, and value matrix V into different heads, you can look through the implementation. 
# * You will also use the [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) with two dense layers to built the feed forward neural network layers.

# In[44]:


def FullyConnected(embedding_dim, fully_connected_dim):
    """
    Returns a sequential model consisting of two dense layers. The first dense layer has
    fully_connected_dim neurons and is activated by relu. The second dense layer has
    embedding_dim and no activation.

    Arguments:
        embedding_dim (int): output dimension
        fully_connected_dim (int): dimension of the hidden layer

    Returns:
        _ (tf.keras.Model): sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])


# <a name='6-1'></a>
# ### 6.1 Encoder Layer
# 
# Now you can pair multi-head attention and feed forward neural network together in an encoder layer! You will also use residual connections and layer normalization to help speed up training (Figure 2a).
# 
# The encoder block (Figure 2) is is already implemented for you. Take a very close look at its implementation, as you will later have to create the decoder yourself, and a lot of the code is very similar. The encoder block performs the following steps: 
# 1. It takes the Q, V, K matrices and a boolean mask to a multi-head attention layer. Remember that to compute *self*-attention Q, V and K are the same. You will also perform Dropout in this multi-head attention layer during training. 
# 2. There is a skip connection to add your original input `x` and the output of the multi-head attention layer. 
# 3. After adding the skip connection, the output passes through the first normalization layer.
# 4. Finally, steps 1-3 are repeated but with the feed forward neural network with a dropout layer instead of the multi-head attention layer. 
# 
# <details>
#   <summary><font size="2" color="darkgreen"><b>Additional Information (Click to expand)</b></font></summary>
#     
# * The `__init__` method creates all the layers that will be accesed by the the `call` method. Wherever you want to use a layer defined inside  the `__init__`  method you will have to use the syntax `self.[insert layer name]`. 
# * You will find the documentation of [MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention) helpful. *Note that if query, key and value are the same, then this function performs self-attention.*
# * The call arguments for `self.mha` are (Where B is for batch_size, T is for target sequence shapes, and S is output_shape):
#  - `query`: Query Tensor of shape (B, T, dim).
#  - `value`: Value Tensor of shape (B, S, dim).
#  - `key`: Optional key Tensor of shape (B, S, dim). If not given, will use the same value for both key and value, which is the most common case.
#  - `attention_mask`: a boolean mask of shape (B, T, S), that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.
#  - `return_attention_scores`: A boolean to indicate whether the output should be attention output if True, or (attention_output, attention_scores) if False. Defaults to False.
#  - `training`: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout). Defaults to either using the training mode of the parent layer/model, or False (inference) if there is no parent layer. Take a look at [tf.keras.layers.Dropout](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/layers/Dropout) for more details (Additional reading in [Keras FAQ](https://keras.io/getting_started/faq/#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute))

# In[45]:


class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            mask (tf.Tensor): Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out (tf.Tensor): Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        # calculate self-attention using mha(~1 line).
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training
        self_mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, fully_connected_dim)
        
        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the  
        # output of the multi-head attention layer
        skip_x_attention = self.layernorm1(x + self_mha_output)  # (batch_size, input_seq_len, fully_connected_dim)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)  # (batch_size, input_seq_len, fully_connected_dim)
        
        # apply dropout layer to ffn output during training
        # use `training=training`
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        
        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output
        # to get the output of the encoder layer
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)  # (batch_size, input_seq_len, embedding_dim)
        
        return encoder_layer_out
    


# <a name='6-2'></a>
# ### 6.2 - Full Encoder
# 
# Now you're ready to build the full Transformer Encoder (Figure 2b), where you will embed your input and add the positional encodings you calculated. You will then feed your encoded embeddings to a stack of Encoder layers. 
# 
# <img src="images/encoder.png" alt="Encoder" width="330"/>
# <caption><center><font color='purple'><b>Figure 2b: Transformer Encoder</font></center></caption>
# 
# The Encoder class is implemented for you. It performs the following steps: 
# 1. Pass the input through the Embedding layer.
# 2. Scale the embedding by multiplying it by the square root of the embedding dimension. 
# 3. Add the position encoding: self.pos_encoding `[:, :seq_len, :]` to the embedding.
# 4. Pass the encoded embedding through a dropout layer
# 5. Pass the output of the dropout layer through the stack of encoding layers using a for loop.

# In[46]:


class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, embedding_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            mask (tf.Tensor): Boolean mask to ensure that the padding is not 
                    treated as part of the input

        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = tf.shape(x)[1]
        
        # Pass input through the Embedding layer
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        # use `training=training`
        x = self.dropout(x, training=training)
        # Pass the output through the stack of encoding layers 
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, embedding_dim)


# <a name='7'></a>
# ## 7 - Decoder
# 
# Now it is time to implement the decoder. You have seen it in the videos and you can use some help by looking at the encoder implementation above. The Decoder layer takes the K and V matrices generated by the Encoder and computes the second multi-head attention layer with the Q matrix from the output (Figure 3a).
# 
# <img src="images/decoder_layer.png" alt="Decoder" width="250"/>
# <caption><center><font color='purple'><b>Figure 3a: Transformer Decoder layer</font></center></caption>
# 
# <a name='7-1'></a>    
# ### 7.1 - Decoder Layer
# Again, you'll pair multi-head attention with a feed forward neural network, but this time you'll implement two multi-head attention layers. You will also use residual connections and layer normalization to help speed up training (Figure 3a).
# 
# <a name='ex-2'></a>    
# ### Exercise 2 - DecoderLayer
#     
# Implement `DecoderLayer()` using the `call()` method
#     
# 1. Block 1 is a multi-head attention layer with a residual connection, and look-ahead mask. Like in the `EncoderLayer`, Dropout is defined within the multi-head attention layer.
# 2. Block 2 will take into account the output of the Encoder, so the multi-head attention layer will receive K and V from the encoder, and Q from the Block 1. You will then apply a normalization layer and a residual connection, just like you did before with the `EncoderLayer`.
# 3. Finally, Block 3 is a feed forward neural network with dropout and normalization layers and a residual connection.
#     
# **Additional Hints:**
# * The first two blocks are fairly similar to the EncoderLayer except you will return `attention_scores` when computing self-attention

# In[ ]:


# GRADED FUNCTION: DecoderLayer  
class DecoderLayer(tf.keras.layers.Layer):  
    """  
    The decoder layer is composed by two multi-head attention blocks,   
    one that takes the new input and uses self-attention, and the other   
    one that combines it with the output of the encoder, followed by a  
    fully connected block.   
    """  
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):  
        super(DecoderLayer, self).__init__()  
  
        self.mha1 = tf.keras.layers.MultiHeadAttention(  
            num_heads=num_heads,  
            key_dim=embedding_dim,  
            dropout=dropout_rate  
        )  
  
        self.mha2 = tf.keras.layers.MultiHeadAttention(  
            num_heads=num_heads,  
            key_dim=embedding_dim,  
            dropout=dropout_rate  
        )  
  
        self.ffn = FullyConnected(  
            embedding_dim=embedding_dim,  
            fully_connected_dim=fully_connected_dim  
        )  
  
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)  
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)  
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)  
  
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)  
      
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):  
        """  
        Forward pass for the Decoder Layer  
          
        Arguments:  
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)  
            enc_output (tf.Tensor): Tensor of shape(batch_size, input_seq_len, fully_connected_dim)  
            training (bool): Boolean, set to true to activate  
                        the training mode for dropout layers  
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input  
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer  
        Returns:  
            out3 (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)  
            attn_weights_block1 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)  
            attn_weights_block2 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)  
        """  
          
        ### START CODE HERE ###  
        # BLOCK 1  
        attn_out1, attn_weights_block1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)  
        attn_out1 = self.dropout_ffn(attn_out1, training=training)  
        out1 = self.layernorm1(attn_out1 + x)  
          
        # BLOCK 2  
        # Ensure that the encoder output 'enc_output' is being used as 'v' and 'k', and the output of the first block 'out1' is used as 'q'.  
        attn_out2, attn_weights_block2 = self.mha2(  
            out1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True  
        )  
        attn_out2 = self.dropout_ffn(attn_out2, training=training)  
        out2 = self.layernorm2(attn_out2 + out1)  
          
        # BLOCK 3  
        ffn_output = self.ffn(out2)  
        ffn_output = self.dropout_ffn(ffn_output, training=training)  
        out3 = self.layernorm3(ffn_output + out2)  
          
        return out3, attn_weights_block1, attn_weights_block2 
        
        
        
        # BLOCK 1  
        # calculate self-attention and return attention scores as attn_weights_block1.  
        attn_out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)  
        attn_out1 = self.dropout_ffn(attn_out1, training=training)  
        out1 = self.layernorm1(attn_out1 + x)  
          
        # BLOCK 2  
        attn_out2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # (batch_size, target_seq_len, d_model)  
        attn_out2 = self.dropout_ffn(attn_out2, training=training)  
        out2 = self.layernorm2(attn_out2 + out1)  
          
        # BLOCK 3  
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)  
        ffn_output = self.dropout_ffn(ffn_output, training=training)  
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)  
        ### END CODE HERE ###  
          
        return out3, attn_weights_block1, attn_weights_block2  


# In[52]:


# GRADED FUNCTION: DecoderLayer
class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output (tf.Tensor): Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            out3 (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attn_weights_block1 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)
            attn_weights_block2 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        
        ### START CODE HERE ###
        # BLOCK 1  
        attn_out1, attn_weights_block1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)  
        attn_out1 = self.dropout_ffn(attn_out1, training=training)  
        out1 = self.layernorm1(attn_out1 + x)  
          
        # BLOCK 2  
        # Ensure that the encoder output 'enc_output' is being used as 'v' and 'k', and the output of the first block 'out1' is used as 'q'.  
        attn_out2, attn_weights_block2 = self.mha2(  
            out1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True  
        )  
        attn_out2 = self.dropout_ffn(attn_out2, training=training)  
        out2 = self.layernorm2(attn_out2 + out1)  
          
        # BLOCK 3  
        ffn_output = self.ffn(out2)  
        ffn_output = self.dropout_ffn(ffn_output, training=training)  
        out3 = self.layernorm3(ffn_output + out2)  
          
#         return out3, attn_weights_block1, attn_weights_block2 
        
#         # BLOCK 1  
#         # calculate self-attention and return attention scores as attn_weights_block1.  
#         attn_out1, attn_weights_block1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)  
#         attn_out1 = self.dropout_ffn(attn_out1, training=training)  
#         out1 = self.layernorm1(attn_out1 + x)  
  
#         # BLOCK 2  
#         # calculate self-attention using the Q from the first block and K and V from the encoder output.   
#         attn_out2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True)  
#         attn_out2 = self.dropout_ffn(attn_out2, training=training)  
#         out2 = self.layernorm2(attn_out2 + out1)  
          
#         # BLOCK 3  
#         ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)  
#         ffn_output = self.dropout_ffn(ffn_output, training=training)  
#         out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)  
          
#         return out3, attn_weights_block1, attn_weights_block2 
        
        
        
#         # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim)
        
#         # BLOCK 1
#         # calculate self-attention and return attention scores as attn_weights_block1.
#         # Dropout will be applied during training (~1 line).
#         mult_attn_out1, attn_weights_block1 = None
        
#         # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
#         Q1 = None

#         # BLOCK 2
#         # calculate self-attention using the Q from the first block and K and V from the encoder output. 
#         # Dropout will be applied during training
#         # Return attention scores as attn_weights_block2 (~1 line) 
#         mult_attn_out2, attn_weights_block2 = None
        
#         # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block (~1 line)
#         mult_attn_out2 = None
                
#         #BLOCK 3
#         # pass the output of the second block through a ffn
#         ffn_output = None
        
#         # apply a dropout layer to the ffn output
#         # use `training=training`
#         ffn_output = None
        
#         # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
#         out3 = None
        ### END CODE HERE ###

        return out3, attn_weights_block1, attn_weights_block2
    


# In[53]:


# Test your function!
key_dim = 12
n_heads = 16

decoderLayer_test = DecoderLayer(embedding_dim=key_dim, num_heads=n_heads, fully_connected_dim=32)

q = np.ones((1, 15, key_dim))
encoder_test_output = tf.convert_to_tensor(np.random.rand(1, 7, 8))
look_ahead_mask = create_look_ahead_mask(q.shape[1])

out, attn_w_b1, attn_w_b2 = decoderLayer_test(q, encoder_test_output, False, look_ahead_mask, None)

print(f"Using embedding_dim={key_dim} and num_heads={n_heads}:\n")
print(f"q has shape:{q.shape}")
print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

print(f"Output of decoder layer has shape:{out.shape}")
print(f"Att Weights Block 1 has shape:{attn_w_b1.shape}")
print(f"Att Weights Block 2 has shape:{attn_w_b2.shape}")


# ##### __Expected Output__
# 
# ```
# Output:
# Using embedding_dim=12 and num_heads=16:
# 
# q has shape:(1, 15, 12)
# Output of encoder has shape:(1, 7, 8)
# 
# Output of decoder layer has shape:(1, 15, 12)
# Att Weights Block 1 has shape:(1, 16, 15, 15)
# Att Weights Block 2 has shape:(1, 16, 15, 7)
# ```

# In[54]:


# UNIT TEST
w2_unittest.test_decoderlayer(DecoderLayer, create_look_ahead_mask)


# <a name='7-2'></a> 
# ### 7.2 - Full Decoder
# You're almost there! Time to use your Decoder layer to build a full Transformer Decoder (Figure 3b). You will embed your output and add positional encodings. You will then feed your encoded embeddings to a stack of Decoder layers. 
# 
# 
# <img src="images/decoder.png" alt="Decoder" width="300"/>
# <caption><center><font color='purple'><b>Figure 3b: Transformer Decoder</font></center></caption>
# 
# <a name='ex-3'></a>     
# ### Exercise 3 - Decoder
# 
# Implement `Decoder()` using the `call()` method to embed your output, add positional encoding, and implement multiple decoder layers.
#  
# In this exercise, you will initialize your Decoder with an Embedding layer, positional encoding, and multiple DecoderLayers. Your `call()` method will perform the following steps: 
# 1. Pass your generated output through the Embedding layer.
# 2. Scale your embedding by multiplying it by the square root of your embedding dimension. Remember to cast the embedding dimension to data type `tf.float32` before computing the square root.
# 3. Add the position encoding: self.pos_encoding `[:, :seq_len, :]` to your embedding.
# 4. Pass the encoded embedding through a dropout layer, remembering to use the `training` parameter to set the model training mode. 
# 5. Pass the output of the dropout layer through the stack of Decoding layers using a for loop.

# In[ ]:


# GRADED FUNCTION: Decoder  
import tensorflow as tf  
  
class DecoderLayer(tf.keras.layers.Layer):  
    # Assume DecoderLayer is already implemented as per the exercise requirements.  
    pass  
  
def positional_encoding(position, d_model):  
    # Assume positional_encoding is already implemented as per the exercise requirements.  
    pass  
  
class Decoder(tf.keras.layers.Layer):  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,  
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):  
        super(Decoder, self).__init__()  
  
        self.embedding_dim = embedding_dim  
        self.num_layers = num_layers  
  
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.embedding_dim)  
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)  
  
        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,  
                                        num_heads=num_heads,  
                                        fully_connected_dim=fully_connected_dim,  
                                        dropout_rate=dropout_rate,  
                                        layernorm_eps=layernorm_eps)   
                           for _ in range(self.num_layers)]  
        self.dropout = tf.keras.layers.Dropout(dropout_rate)  
      
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):  
        seq_len = tf.shape(x)[1]  
        attention_weights = {}  
  
        ### START CODE HERE ###  
        # create word embeddings   
        x = self.embedding(x)  
          
        # scale embeddings by multiplying by the square root of their dimension  
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))  
          
        # add positional encodings to word embedding  
        x += self.pos_encoding[:, :seq_len, :]  
      
        # apply a dropout layer to x  
        x = self.dropout(x, training=training)  
      
        # use a for loop to pass x through a stack of decoder layers and update attention_weights  
        for i in range(self.num_layers):  
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)  
      
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1  
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2  
        ### END CODE HERE ###  
          
        # x.shape == (batch_size, target_seq_len, fully_connected_dim)  
        return x, attention_weights  


# In[55]:


# GRADED FUNCTION: Decoder
class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """ 
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output (tf.Tensor):  Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        ### START CODE HERE ###
        # create word embeddings   
        x = self.embedding(x)  
          
        # scale embeddings by multiplying by the square root of their dimension  
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))  
          
        # add positional encodings to word embedding  
        x += self.pos_encoding[:, :seq_len, :]  
      
        # apply a dropout layer to x  
        x = self.dropout(x, training=training)  
      
        # use a for loop to pass x through a stack of decoder layers and update attention_weights  
        for i in range(self.num_layers):  
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)  
      
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1  
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        
#         # create word embeddings 
#         x = None
        
#         # scale embeddings by multiplying by the square root of their dimension
#         x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        
#         # add positional encodings to word embedding
#         x += None

#         # apply a dropout layer to x
#         # use `training=training`
#         x = None

#         # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
#         for i in range(self.num_layers):
#             # pass x and the encoder output through a stack of decoder layers and save the attention weights
#             # of block 1 and 2 (~1 line)
#             x, block1, block2 = None

#             #update attention_weights dictionary with the attention weights of block 1 and block 2
#             attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = None
#             attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = None
        ### END CODE HERE ###
        
        # x.shape == (batch_size, target_seq_len, fully_connected_dim)
        return x, attention_weights


# In[56]:


# Test your function!
n_layers = 5
emb_d = 13
n_heads = 17
fully_connected_dim = 16
target_vocab_size = 300
maximum_position_encoding = 6

x = np.array([[3, 2, 1, 1], [2, 1, 1, 0], [2, 1, 1, 0]])

encoder_test_output = tf.convert_to_tensor(np.random.rand(3, 7, 9))

look_ahead_mask = create_look_ahead_mask(x.shape[1])

decoder_test = Decoder(n_layers, emb_d, n_heads, fully_connected_dim, target_vocab_size,maximum_position_encoding)
                   
outd, att_weights = decoder_test(x, encoder_test_output, False, look_ahead_mask, None)

print(f"Using num_layers={n_layers}, embedding_dim={emb_d} and num_heads={n_heads}:\n")
print(f"x has shape:{x.shape}")
print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

print(f"Output of decoder has shape:{outd.shape}\n")
print("Attention weights:")
for name, tensor in att_weights.items():
    print(f"{name} has shape:{tensor.shape}")


# ##### __Expected Output__
# 
# ```
# Using num_layers=5, embedding_dim=13 and num_heads=17:
# 
# x has shape:(3, 4)
# Output of encoder has shape:(3, 7, 9)
# 
# Output of decoder has shape:(3, 4, 13)
# 
# Attention weights:
# decoder_layer1_block1_self_att has shape:(3, 17, 4, 4)
# decoder_layer1_block2_decenc_att has shape:(3, 17, 4, 7)
# decoder_layer2_block1_self_att has shape:(3, 17, 4, 4)
# decoder_layer2_block2_decenc_att has shape:(3, 17, 4, 7)
# decoder_layer3_block1_self_att has shape:(3, 17, 4, 4)
# decoder_layer3_block2_decenc_att has shape:(3, 17, 4, 7)
# decoder_layer4_block1_self_att has shape:(3, 17, 4, 4)
# decoder_layer4_block2_decenc_att has shape:(3, 17, 4, 7)
# decoder_layer5_block1_self_att has shape:(3, 17, 4, 4)
# decoder_layer5_block2_decenc_att has shape:(3, 17, 4, 7)
# ```

# In[57]:


# UNIT TEST
w2_unittest.test_decoder(Decoder, create_look_ahead_mask, create_padding_mask)


# <a name='8'></a> 
# ## 8 - Transformer
# 
# Phew! This has been quite the assignment! Congratulations! You've done all the hard work, now it's time to put it all together.  
# 
# <img src="images/transformer.png" alt="Transformer" width="550"/>
# <caption><center><font color='purple'><b>Figure 4: Transformer</font></center></caption>
#     
# The flow of data through the Transformer Architecture is as follows:
# * First your input passes through an Encoder, which is just repeated Encoder layers that you implemented:
#     - embedding and positional encoding of your input
#     - multi-head attention on your input
#     - feed forward neural network to help detect features
# * Then the predicted output passes through a Decoder, consisting of the decoder layers that you implemented:
#     - embedding and positional encoding of the output
#     - multi-head attention on your generated output
#     - multi-head attention with the Q from the first multi-head attention layer and the K and V from the Encoder
#     - a feed forward neural network to help detect features
# * Finally, after the Nth Decoder layer, one dense layer and a softmax are applied to generate prediction for the next output in your sequence.
# 
# <a name='ex-4'></a> 
# ### Exercise 4 - Transformer
# 
# Implement `Transformer()` using the `call()` method
# 1. Pass the input through the Encoder with the appropiate mask.
# 2. Pass the encoder output and the target through the Decoder with the appropiate mask.
# 3. Apply a linear transformation and a softmax to get a prediction.

# In[ ]:


# GRADED FUNCTION: Transformer  
class Transformer(tf.keras.Model):  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,   
                 target_vocab_size, max_positional_encoding_input,  
                 max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):  
        super(Transformer, self).__init__()  
  
        self.encoder = Encoder(num_layers=num_layers,  
                               embedding_dim=embedding_dim,  
                               num_heads=num_heads,  
                               fully_connected_dim=fully_connected_dim,  
                               input_vocab_size=input_vocab_size,  
                               maximum_position_encoding=max_positional_encoding_input,  
                               dropout_rate=dropout_rate,  
                               layernorm_eps=layernorm_eps)  
  
        self.decoder = Decoder(num_layers=num_layers,   
                               embedding_dim=embedding_dim,  
                               num_heads=num_heads,  
                               fully_connected_dim=fully_connected_dim,  
                               target_vocab_size=target_vocab_size,   
                               maximum_position_encoding=max_positional_encoding_target,  
                               dropout_rate=dropout_rate,  
                               layernorm_eps=layernorm_eps)  
  
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  
      
    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):  
        # call self.encoder with the appropriate arguments to get the encoder output  
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)  
          
        # call self.decoder with the appropriate arguments to get the decoder output  
        dec_output, attention_weights = self.decoder(  
            output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask  
        )  
          
        # pass decoder output through a linear layer  
        final_output = self.final_layer(dec_output)  
  
        return final_output, attention_weights  


# In[58]:


# GRADED FUNCTION: Transformer
class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
    
    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            input_sentence (tf.Tensor): Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the input sentence
            output_sentence (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the output sentence
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask (tf.Tensor): Boolean mask to ensure that the padding is not 
                    treated as part of the input
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            dec_padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            final_output (tf.Tensor): The final output of the model
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """
        ### START CODE HERE ###
        # call self.encoder with the appropriate arguments to get the encoder output  
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)  
          
        # call self.decoder with the appropriate arguments to get the decoder output  
        dec_output, attention_weights = self.decoder(  
            output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask  
        )  
          
        # pass decoder output through a linear layer  
        final_output = self.final_layer(dec_output)
#         # call self.encoder with the appropriate arguments to get the encoder output
#         enc_output = None
        
#         # call self.decoder with the appropriate arguments to get the decoder output
#         # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
#         dec_output, attention_weights = None
        
#         # pass decoder output through a linear layer and softmax (~1 line)
#         final_output = None
        ### END CODE HERE ###

        return final_output, attention_weights


# In[59]:


# Test your function!
n_layers = 3
emb_d = 13
n_heads = 17
fully_connected_dim = 8
input_vocab_size = 300
target_vocab_size = 350
max_positional_encoding_input = 12
max_positional_encoding_target = 12

transformer = Transformer(n_layers, 
    emb_d, 
    n_heads, 
    fully_connected_dim, 
    input_vocab_size, 
    target_vocab_size, 
    max_positional_encoding_input,
    max_positional_encoding_target)

# 0 is the padding value
sentence_a = np.array([[2, 3, 1, 3, 0, 0, 0]])
sentence_b = np.array([[1, 3, 4, 0, 0, 0, 0]])

enc_padding_mask = create_padding_mask(sentence_a)
dec_padding_mask = create_padding_mask(sentence_a)

look_ahead_mask = create_look_ahead_mask(sentence_a.shape[1])

test_summary, att_weights = transformer(
    sentence_a,
    sentence_b,
    False,
    enc_padding_mask,
    look_ahead_mask,
    dec_padding_mask
)

print(f"Using num_layers={n_layers}, target_vocab_size={target_vocab_size} and num_heads={n_heads}:\n")
print(f"sentence_a has shape:{sentence_a.shape}")
print(f"sentence_b has shape:{sentence_b.shape}")

print(f"\nOutput of transformer (summary) has shape:{test_summary.shape}\n")
print("Attention weights:")
for name, tensor in att_weights.items():
    print(f"{name} has shape:{tensor.shape}")


# ##### __Expected Output__
# 
# ```
# Using num_layers=3, target_vocab_size=350 and num_heads=17:
# 
# sentence_a has shape:(1, 7)
# sentence_b has shape:(1, 7)
# 
# Output of transformer (summary) has shape:(1, 7, 350)
# 
# Attention weights:
# decoder_layer1_block1_self_att has shape:(1, 17, 7, 7)
# decoder_layer1_block2_decenc_att has shape:(1, 17, 7, 7)
# decoder_layer2_block1_self_att has shape:(1, 17, 7, 7)
# decoder_layer2_block2_decenc_att has shape:(1, 17, 7, 7)
# decoder_layer3_block1_self_att has shape:(1, 17, 7, 7)
# decoder_layer3_block2_decenc_att has shape:(1, 17, 7, 7)
# ```

# In[60]:


# UNIT TEST
w2_unittest.test_transformer(Transformer, create_look_ahead_mask, create_padding_mask)


# <a name='9'></a>
# ## 9 - Initialize the Model
# Now that you have defined the model, you can initialize and train it. First you can initialize the model with the parameters below. Note that generally these models are much larger and you are using a smaller version to fit this environment and to be able to train it in just a few minutes.
# 
# The base model described in the original Transformer paper used `num_layers=6`, `embedding_dim=512`, and `fully_connected_dim=2048`.

# In[61]:


# Define the model parameters
num_layers = 2
embedding_dim = 128
fully_connected_dim = 128
num_heads = 2
positional_encoding_length = 256

# Initialize the model
transformer = Transformer(
    num_layers, 
    embedding_dim, 
    num_heads, 
    fully_connected_dim,
    vocab_size, 
    vocab_size, 
    positional_encoding_length, 
    positional_encoding_length,
)


# <a name='10'></a>
# ## 10 - Prepare for Training the Model
# 
# The original transformer paper uses Adam optimizer with custom learning rate scheduling, which we define in the cell below. This was empirically shown to produce faster convergence.

# In[62]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embedding_dim)

optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# Below you can plot, how the custom learning rate looks like.

# In[63]:


plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')


# Next, you set up the loss. Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
# 
# You will use the sparse categorical cross-entropy loss function (`tf.keras.losses.SparseCategoricalCrossentropy`) and set the parameter `from_logits` to False since the Transformer does not output raw logits since the last layer has a softmax activation:

# In[64]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def masked_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')

# Here you will store the losses, so you can later plot them
losses = []


# Now you can define your custom training function. If you are not very advanced with tensorflow, you can understand this function as an alternative to using `model.compile()` and `model.fit()`, but with added extra flexibility.

# In[65]:


@tf.function
def train_step(model, inp, tar):
    """
    One training step for the transformer
    Arguments:
        inp (tf.Tensor): Input data to summarize
        tar (tf.Tensor): Target (summary)
    Returns:
        None
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # Create masks
    enc_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    dec_padding_mask = create_padding_mask(inp) # Notice that both encoder and decoder padding masks are equal

    with tf.GradientTape() as tape:
        predictions, _ = model(
            inp,
            tar_inp, 
            True, 
            enc_padding_mask, 
            look_ahead_mask, 
            dec_padding_mask
        )
        loss = masked_loss(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)


# Now you are ready for training the model. But before starting the training, you can also define one more set of functions to perform the inference. Because you are using a custom training loop, you can do whatever you want between the training steps. And wouldnt't it be fun to see after each epoch some examples of how the model performs?

# <a name='11'></a>
# ## 11 - Summarization
# 
# The last thing you will implement is inference. With this, you will be able to produce actual summaries of the documents. You will use a simple method called greedy decoding, which means you will predict one word at a time and append it to the output. You will start with an `[SOS]` token and repeat the word by word inference until the model returns you the `[EOS]` token or until you reach the maximum length of the sentence (you need to add this limit, otherwise a poorly trained model could give you infinite sentences without ever producing the `[EOS]` token.
# 
# <a name='ex-5'></a> 
# ### Exercise 5 - next_word
# Write a helper function that predicts the next word, so you can use it to write the whole sentences. Hint: this is very similar to what happens in the train_step, but you have to set the training of the model to False.

# In[ ]:


# GRADED FUNCTION: next_word  
def next_word(model, encoder_input, output):  
    """  
    Helper function for summarization that uses the model to predict just the next word.  
    Arguments:  
        model (tf.keras.Model): The Transformer model  
        encoder_input (tf.Tensor): Input data to summarize  
        output (tf.Tensor): (incomplete) target (summary)  
    Returns:  
        predicted_id (tf.Tensor): The id of the predicted word  
    """  
    ### START CODE HERE ###  
    # Create a padding mask for the input (encoder)  
    enc_padding_mask = create_padding_mask(encoder_input)  
    # Create a look-ahead mask for the output  
    look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])  
    # Combine it with the padding mask for the decoder input  
    dec_target_padding_mask = create_padding_mask(output)  
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  
    # Create a padding mask for the input (decoder)  
    dec_padding_mask = enc_padding_mask  
      
    # Run the prediction of the next word with the transformer model  
    predictions, attention_weights = model(  
        encoder_input,  
        output,  
        False,  # We are doing inference, set training to False  
        enc_padding_mask,  
        look_ahead_mask,  
        dec_padding_mask  
    )  
    ### END CODE HERE ###  
  
    # Select the last word from the sequence  
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)  
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)  
      
    return predicted_id  
  
# Implement the create_padding_mask and create_look_ahead_mask helper functions  
def create_padding_mask(seq):  
    # Create padding mask from input sequences (1 for padding, 0 for real token)  
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)  
  
def create_look_ahead_mask(size):  
    # Create look-ahead mask (upper triangular matrix of 1s)  
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  
    return mask  # (seq_len, seq_len)  
  
# Assuming tokenizer and transformer model are already defined and properly trained.  
# Assuming encoder_maxlen is defined as the maximum length of the encoder input.  
# Assuming the transformer model has been instantiated and trained appropriately.  
  
# Code to check if the function works (provided by the user)  
# Commented out as it cannot be executed in this environment  
# since the tokenizer and transformer model are not defined here.  
  
# Take a random sentence as an input  
# input_document = tokenizer.texts_to_sequences(["a random sentence"])  
# input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')  
# encoder_input = tf.expand_dims(input_document[0], 0)  
  
# Take the start of sentence token as the only token in the output to predict the next word  
# output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)  
  
# predict the next word with your function  
# predicted_token = next_word(transformer, encoder_input, output)  
# print(f"Predicted token: {predicted_token}")  
  
# predicted_word = tokenizer.sequences_to_texts(predicted_token.numpy())[0]  
# print(f"Predicted word: {predicted_word}")  


# In[66]:


# GRADED FUNCTION: next_word
def next_word(model, encoder_input, output):
    """
    Helper function for summarization that uses the model to predict just the next word.
    Arguments:
        encoder_input (tf.Tensor): Input data to summarize
        output (tf.Tensor): (incomplete) target (summary)
    Returns:
        predicted_id (tf.Tensor): The id of the predicted word
    """
    ### START CODE HERE ###
    # Implement the create_padding_mask and create_look_ahead_mask helper functions  
    def create_padding_mask(seq):  
        # Create padding mask from input sequences (1 for padding, 0 for real token)  
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)  

    def create_look_ahead_mask(size):  
        # Create look-ahead mask (upper triangular matrix of 1s)  
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  
        return mask  # (seq_len, seq_len) 

    # Create a padding mask for the input (encoder)  
    enc_padding_mask = create_padding_mask(encoder_input)  
    # Create a look-ahead mask for the output  
    look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])  
    # Combine it with the padding mask for the decoder input  
    dec_target_padding_mask = create_padding_mask(output)  
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  
    # Create a padding mask for the input (decoder)  
    dec_padding_mask = enc_padding_mask  
      
    # Run the prediction of the next word with the transformer model  
    predictions, attention_weights = model(  
        encoder_input,  
        output,  
        False,  # We are doing inference, set training to False  
        enc_padding_mask,  
        look_ahead_mask,  
        dec_padding_mask  
    )  
    
    
#     # Create a padding mask for the input (encoder)
#     enc_padding_mask = None
#     # Create a look-ahead mask for the output
#     look_ahead_mask = None
#     # Create a padding mask for the input (decoder)
#     dec_padding_mask = None

#     # Run the prediction of the next word with the transformer model
#     predictions, attention_weights = None(
#         None,
#         None,
#         None,
#         None,
#         None,
#         None
#     )
    ### END CODE HERE ###

    predictions = predictions[: ,-1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    return predicted_id


# Check if your function works.

# In[67]:


# Take a random sentence as an input
input_document = tokenizer.texts_to_sequences(["a random sentence"])
input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
encoder_input = tf.expand_dims(input_document[0], 0)

# Take the start of sentence token as the only token in the output to predict the next word
output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)

# predict the next word with your function
predicted_token = next_word(transformer, encoder_input, output)
print(f"Predicted token: {predicted_token}")

predicted_word = tokenizer.sequences_to_texts(predicted_token.numpy())[0]
print(f"Predicted word: {predicted_word}")


# ##### __Expected Output__
# 
# ```
# Predicted token: [[14859]]
# Predicted word: masses
# ```

# In[68]:


# UNIT TEST
w2_unittest.test_next_word(next_word, transformer, encoder_input, output)


# In[69]:


def summarize(model, input_document):
    """
    A function for summarization using the transformer model
    Arguments:
        input_document (tf.Tensor): Input data to summarize
    Returns:
        _ (str): The summary of the input_document
    """    
    input_document = tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    encoder_input = tf.expand_dims(input_document[0], 0)
    
    output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)
    
    for i in range(decoder_maxlen):
        predicted_id = next_word(model, encoder_input, output)
        output = tf.concat([output, predicted_id], axis=-1)
        
        if predicted_id == tokenizer.word_index["[EOS]"]:
            break

    return tokenizer.sequences_to_texts(output.numpy())[0]  # since there is just one translated document


# Now you can already summarize a sentence! But beware, since the model was not yet trained at all, it will just produce nonsense.

# In[70]:


training_set_example = 0

# Check a summary of a document from the training set
print('Training set example:')
print(document[training_set_example])
print('\nHuman written summary:')
print(summary[training_set_example])
print('\nModel written summary:')
summarize(transformer, document[training_set_example])


# <a name='12'></a>
# # 12 - Train the model
# 
# Now you can finally train the model. Below is a loop that will train your model for 20 epochs. note that it should take about 30 seconds per epoch (with the exception of the first few epochs which can take a few minutes each).
# 
# Note that after each epoch you perform the summarization on one of the sentences in the test set and print it out, so you can see how your model is improving.

# In[71]:


# Take an example from the test set, to monitor it during training
test_example = 0
true_summary = summary_test[test_example]
true_document = document_test[test_example]

# Define the number of epochs
epochs = 20

# Training loop
for epoch in range(epochs):
    
    start = time.time()
    train_loss.reset_states()
    number_of_batches=len(list(enumerate(dataset)))

    for (batch, (inp, tar)) in enumerate(dataset):
        print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
        train_step(transformer, inp, tar)
    
    print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
    losses.append(train_loss.result())
    
    print (f'Time taken for one epoch: {time.time() - start} sec')
    print('Example summarization on the test set:')
    print('  True summarization:')
    print(f'    {true_summary}')
    print('  Predicted summarization:')
    print(f'    {summarize(transformer, true_document)}\n')


# Plot the loss funtion.

# In[ ]:


plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')


# <a name='13'></a>
# # 13 - Summarize some Sentences!
# 
# Below you can see an example of summarization of a sentence from the training set and a sentence from the test set. See if you notice anything interesting about them!

# In[ ]:


training_set_example = 0

# Check a summary of a document from the training set
print('Training set example:')
print(document[training_set_example])
print('\nHuman written summary:')
print(summary[training_set_example])
print('\nModel written summary:')
print(summarize(transformer, document[training_set_example]))


# In[ ]:


test_set_example = 3

# Check a summary of a document from the test set
print('Test set example:')
print(document_test[test_set_example])
print('\nHuman written summary:')
print(summary_test[test_set_example])
print('\nModel written summary:')
print(summarize(transformer, document_test[test_set_example]))


# If you critically examine the output of the model, you can notice a few things:
#  - In the training set the model output is (almost) identical to the real output (already after 20 epochs and even more so with more epochs). This might be because the training set is relatively small and the model is relatively big and has thus learned the sentences in the training set by heart (overfitting).
#  - While the performance on the training set looks amazing, it is not so good on the test set. The model overfits, but fails to generalize. Again an easy candidate to blame is the small training set and a comparatively large model, but there might be a variety of other factors.
#  - Look at the test set example 3 and its summarization. Would you summarize it the same way as it is written here? Sometimes the data may be ambiguous. And the training of **your model can only be as good as your data**.
# 
# Here you only use a small dataset, to show that something can be learned in a reasonable amount of time in a relatively small environment. Generally, large transformers are trained on more than one task and on very large quantities of data to achieve superb performance. You will learn more about this in the rest of this course.

# **Congratulations on finishing this week's assignment!** You did a lot of work and now you should have a better understanding of the Transformers and their building blocks (encoder and decoder) and how they can be used for text summarization. And remember: you dont need to change much to use the same model for a translator, just change the dataset and it should work!
# 
# **Keep it up!**
