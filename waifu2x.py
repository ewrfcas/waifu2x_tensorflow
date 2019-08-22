import json, numpy as np
import cv2
import tensorflow as tf


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


class upconv7(object):
    def __init__(self, model_path, slice=True):
        model = json.load(open(model_path, 'r'))
        self.n_layers = len(model)

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='input')
        x = self.input
        for i in range(self.n_layers):
            conv_weights = tf.constant(np.array(model[i]['weight']), dtype=tf.float32)
            conv_weights = tf.transpose(conv_weights, [2, 3, 1, 0])
            bias = tf.constant(np.array(model[i]['bias']), dtype=tf.float32)
            if i < self.n_layers - 1:
                x = tf.nn.conv2d(x, filter=conv_weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                x = tf.nn.leaky_relu(x, 0.1)
            else:
                x_shape = shape_list(x)
                output_shape = [x_shape[0], x_shape[1] * 2, x_shape[2] * 2, 3]
                x = tf.nn.conv2d_transpose(x, filter=conv_weights, output_shape=output_shape, strides=[1, 2, 2, 1],
                                           padding='SAME') + bias
                x = tf.clip_by_value(x, 0.0, 1.0)

        self.output_img = x
        if slice:  # slice for abandon the padding black rectangle
            self.output_img = self.output_img[:, 2:-2, 2:-2, :]
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

    def up_scale(self, img_path, output_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        output_img = self.sess.run(self.output_img, feed_dict={self.input: np.expand_dims(img, 0)})
        output_img = np.squeeze(output_img, 0)
        output_img *= 255.0
        output_img = np.clip(output_img, 0, 255)
        output_img = output_img.astype(np.int32)[:, :, ::-1]
        cv2.imwrite(output_path, output_img)


class vgg7(object):
    def __init__(self, model_path, slice=True):
        model = json.load(open(model_path, 'r'))
        self.n_layers = len(model)

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='input')
        x = self.input
        for i in range(self.n_layers):
            conv_weights = tf.constant(np.array(model[i]['weight']), dtype=tf.float32)
            conv_weights = tf.transpose(conv_weights, [2, 3, 1, 0])
            bias = tf.constant(np.array(model[i]['bias']), dtype=tf.float32)
            x = tf.nn.conv2d(x, filter=conv_weights, strides=[1, 1, 1, 1], padding='SAME') + bias
            if i < self.n_layers - 1:
                x = tf.nn.leaky_relu(x, 0.1)
            else:
                x = tf.clip_by_value(x, 0.0, 1.0)

        self.output_img = x
        if slice:
            self.output_img = self.output_img[:, 1:-1, 1:-1, :]
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

    def up_scale(self, img_path, output_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img.shape[0] * 2, img.shape[1] * 2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        output_img = self.sess.run(self.output_img, feed_dict={self.input: np.expand_dims(img, 0)})
        output_img = np.squeeze(output_img, 0)
        output_img *= 255.0
        output_img = np.clip(output_img, 0, 255)
        output_img = output_img.astype(np.int32)[:, :, ::-1]
        cv2.imwrite(output_path, output_img)
