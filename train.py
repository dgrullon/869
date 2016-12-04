import vgg
import tensorflow as tf
import numpy as np

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def train_network(style_image, content_image, alpha, beta, iterations, vgg_path, use_avg_pool=False):
    # game plan:
    # precompute gram matrices for each content and style layer
    # make loss function with squared differences
    # optimize across


    style_shape = (1,) + style_image.shape
    style_image_placeholder = vgg.preprocess(tf.placeholder(tf.float32, shape=style_shape, name='style_image'))
    style_net = vgg(vgg_path, style_image_placeholder, use_avg_pool)
    style_grams = {}
    for style_layer in style_layers:
        features = style_net[style_layer].eval(feed_dict={style_image_placeholder:np.array([style_image])})
        features = np.reshape(features, (-1, features.shape[3]))
        style_grams[style_layer] = np.matrix_multiply(features.transpose(), features)
    
    content_shape = (1,) + content_image.shape
    content_image_placeholder = vgg.preprocess(tf.placeholder(tf.float32, shape=content_shape, name='content_image'))
    content_net = vgg(vgg_path, content_image_placeholder, use_avg_pool)
    content_grams = {}  
    features = content_net[CONTENT_LAYER].eval(feed_dict={content_image_placeholder:np.array([content_image])})
    features = np.reshape(features, (-1, features.shape[3]))
    content_grams[CONTENT_LAYER] = np.matrix_multiply(features.transpose(), features)                                                    
    
    
