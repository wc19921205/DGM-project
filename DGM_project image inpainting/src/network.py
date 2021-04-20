import tensorflow as tf

def conv_layer(x, filter_shape, stride):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding='SAME')


def dilated_conv_layer(x, filter_shape, dilation):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')


def deconv_layer(x, filter_shape, output_shape, stride):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1])


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)


class Network:
    def __init__(self, x, mask, local_x, global_completion, local_completion, is_training, batch_size):
        self.batch_size = batch_size
        # keep all other pixels and set the hole to be completed
        # imitation: results after completion, but other pixels exceept for the whole are affected
        self.imitation = self.generator(x * (1 - mask), is_training)
        # take only the mask part and get non-mask part from original x
        self.completion = self.imitation * mask + x * (1 - mask)
        self.real = self.discriminator(x, local_x, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, reuse=True)
        # l2 loss between input x(128x128) and the whole output(128x128)
        self.g_loss = self.calc_g_loss(x, self.completion)
        # sigmoid
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # network structure according to paper
    # all parameters are from paper
    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)

        return x


    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = full_connection_layer(output, 1)

        # output: possibility that the input is real natural pic
        return output

    # computes the mean of elements across dimensions of a tensor
    def calc_g_loss(self, x, completion):
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)


    def calc_d_loss(self, real, fake):
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha

