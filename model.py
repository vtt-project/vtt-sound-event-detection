import tensorflow as tf
slim = tf.contrib.slim


def vggish_rnn(features, num_classes=10, batch_size=64, state_size=64, batch_norm=False):
    kernel_size = 3
    stride = 1
    with tf.variable_scope('vggish'):
        net = features
        with tf.variable_scope('conv1'):
            net = tf.layers.conv2d(
                net, 64, kernel_size, stride, activation=tf.nn.relu, padding='same')
            net = tf.layers.average_pooling2d(net, [1, 2], [1, 2], 'same')
        with tf.variable_scope('conv2'):
            net = tf.layers.conv2d(
                net, 128, kernel_size, stride, activation=tf.nn.relu, padding='same')
            net = tf.layers.average_pooling2d(net, [1, 2], [1, 2], 'same')
            if batch_norm:
                net = tf.layers.batch_normalization(net, momentum=0.9)
        with tf.variable_scope('conv3'):
            net = tf.layers.conv2d(
                net, 256, kernel_size, stride, activation=tf.nn.relu, padding='same')
            net = tf.layers.average_pooling2d(net, [1, 2], [1, 2], 'same')
        with tf.variable_scope('conv4'):
            net = tf.layers.conv2d(
                net, 512, kernel_size, stride, activation=tf.nn.relu, padding='same')
            net = tf.layers.average_pooling2d(net, [1, 2], [1, 2], 'same')
        with tf.variable_scope('conv5'):
            net = tf.layers.conv2d(
                net, 1024, kernel_size, stride, activation=tf.nn.relu, padding='same')
            net = tf.layers.average_pooling2d(net, [1, 2], [1, 2], 'same')
            if batch_norm:
                net = tf.layers.batch_normalization(net, momentum=0.99)
        with tf.variable_scope('conv6'):
            net = tf.layers.conv2d(
                net, 128, kernel_size, stride, activation=tf.nn.relu, padding='same')
        with tf.variable_scope('conv'):
            net = tf.layers.conv2d(
                net, 1, 1, 1, activation=tf.nn.relu, padding='same')

    with tf.variable_scope('rnn'):
        gru = tf.contrib.rnn.GRUCell(state_size, activation=tf.nn.relu)
        h = tf.random_normal((batch_size, state_size))
        outputs, _ = tf.nn.dynamic_rnn(gru, net[:, :, :, 0], initial_state=h)
        predict = tf.layers.dense(
            outputs[:, -1, :], num_classes, activation=tf.nn.sigmoid, name='logits')
    return tf.identity(predict)
