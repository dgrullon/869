import vgg
import tensorflow as tf
import numpy as np

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def stylize(style_image, content_image, alpha, beta, iterations, vgg_path, use_avg_pool=False):
    # game plan:
    # precompute gram matrices for each content and style layer
    # make loss function with squared differences
    # optimize across
    style_shape = (1,) + style_image.shape
    with tf.Graph().as_default(), tf.Session() as sess:
        print("precomputing style grams")
        style_image_placeholder = vgg.preprocess(tf.placeholder(tf.float32, shape=style_shape, name='style_image'))
        style_net = vgg.net(vgg_path, style_image_placeholder, use_avg_pool)
        style_grams = {}
        style_pre = np.array([vgg.preprocess(style_image)])
        for style_layer in STYLE_LAYERS:
            features = style_net[style_layer].eval(feed_dict={style_image_placeholder:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            style_grams[style_layer] = np.matmul(features.transpose(), features)
        
        print("precomputing content grams")
        content_shape = (1,) + content_image.shape
        content_image_placeholder = tf.placeholder(tf.float32, shape=content_shape, name='content_image')
        content_net = vgg.net(vgg_path, content_image_placeholder, use_avg_pool)
        content_grams = {}
        content_pre = np.array([vgg.preprocess(content_image)])
        content_grams[CONTENT_LAYER] = content_net[CONTENT_LAYER].eval(feed_dict={content_image_placeholder:content_pre})                                              
    

    with tf.Graph().as_default():
        # White noise image. 0.256 is taken from online
        initial_image = tf.random_normal(content_shape) * 0.256
        image = tf.Variable(initial_image)
        net = vgg.net(vgg_path, image, use_avg_pool)

        # Content Loss
        # FROM ONLINE:
        # content_weight * (2 * tf.nn.l2_loss(
        #         net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
        #         content_features[CONTENT_LAYER].size)

        # Change this later.

        loss_content =  tf.nn.l2_loss(net[CONTENT_LAYER] - content_grams[CONTENT_LAYER]) /content_grams[CONTENT_LAYER].size

        
        # Style Loss
        
        losses_style = []
        style_net = vgg.net(vgg_path, image, use_avg_pool)

        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            features = tf.reshape(layer, (-1, number))
            size = height * width * number
            gram = tf.matmul(tf.transpose(features), features) / size

            losses_style.append(tf.nn.l2_loss(gram - style_grams[style_layer]) / style_grams[style_layer].size )

        loss_style = np.sum(losses_style)

        loss = alpha * loss_content + beta * loss_style

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("starting training")
            for i in range(iterations):
                last_step = (i == iterations - 1)
                train_step.run()

                if last_step:
                    print("finished")
                    return vgg.unprocess(image.eval().reshape(style_shape[1:]))
                    
