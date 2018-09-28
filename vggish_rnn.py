from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa
import os
from model import  vggish_rnn

# set CUDA visible devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# set Flags
flags = tf.app.flags
# slim = tf.contrib.slim
flags.DEFINE_integer('num_batches', 100000, 'Number of batches of examples.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_string('checkpoint', './checkpoint/vggish_rnn', 'Path to the VGGish checkpoint file.')
flags.DEFINE_string('audio_path', './audio_clips/','Path to audio path.')
flags.DEFINE_boolean('resume', False,'Resume from latest saved state.')
flags.DEFINE_string('audio_train_list', './lists/audio_train.txt', 'List of files for training')
flags.DEFINE_string('audio_val_list', './lists/audio_val.txt', 'List of files for training')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_string('gpu', '0', 'GPU to use')

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

classes = {'background music': 0, 'speaking': 1, 'background laughing':2,
           'crashing': 3, 'knocking': 4, 'opening/closing doors': 5,
           'clapping': 6, 'shouting': 7, 'laughing': 8, 'door bell': 9}
num_classes = len(classes) 
# Training Parameters
batch_size = FLAGS.batch_size
display_step = 20

# Optimizer Parameters
LEARNING_RATE = FLAGS.learning_rate # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.
BETA1 = 0.9 # Beta 1

# Parameters of STFT 
sr = 44100
hop_length = 1024
num_mel_bands = 96
n_fft = 2048
win_length = 1536

train_dicts = []
val_dicts = []

with open(FLAGS.audio_train_list, 'r') as f:
    lines = f.readlines()
    for l in lines:
        [fname, _, _, cls] = l.strip().split('\t')
        d = {}
        d['file_name'] = fname
        d['class'] = cls
        d['labels'] = []
        d['raw'], _ = librosa.load(os.path.join(FLAGS.audio_path, fname), sr=sr)
        for c in cls.strip().split('|'):
            d['labels'].append(classes[c])
        train_dicts.append(d)

print('The train data has been loaded!')

with open(FLAGS.audio_val_list, 'r') as f:
    lines = f.readlines()
    for l in lines:
        d = {}
        [fname, _, _, cls] = l.strip().split('\t')
        d['file_name'] = fname
        d['class'] = cls
        d['labels'] = []
        d['raw'], _ = librosa.load(os.path.join(FLAGS.audio_path, fname), sr=sr)
        for c in cls.strip().split('|'):
            d['labels'].append(classes[c])
        val_dicts.append(d)
    
print('The validation data has been loaded')

def _load_stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length):
    """
    Form a batch of training data to form x and y
    Randomly sample a 1-sec clips from the audio clips and compute the stft
    """
    num_sample_wav = 1
    num_files = batch_size/num_sample_wav
    choice = np.random.randint(0, len(data), (num_files))
    data_list = [data[i] for i in choice]
    nf = int(np.ceil(np.float(sr)/hop_length))
    feat_example = np.zeros((batch_size, nf, n_fft/2+1, 1))
    label_example = np.zeros((batch_size, len(classes)))
    k = 0
    for d in data_list:
        raw = d['raw']
        cls = d['labels']
        if len(raw) - sr > 0: 
            sf = np.random.randint(0, len(raw)-sr, (num_sample_wav))
        else:
            sf = np.zeros((num_sample_wav), dtype=np.int)
            
        for s in sf:
            stft = librosa.core.stft(raw[s:s+sr], n_fft, hop_length, win_length)
            feat_example[k,:,:,0] = np.abs(stft).T
            label_example[k,cls] = 1
            k += 1
    return feat_example, label_example

# The data loader
loader = _load_stft
with tf.Graph().as_default(), tf.Session() as sess:
    # tf Graph input
    X = tf.placeholder("float", [batch_size, int(np.ceil(np.float(sr)/hop_length)), n_fft/2+1, 1], name='input_features')
    Y = tf.placeholder("float", [batch_size, num_classes])
    logits = vggish_rnn(X, num_classes=num_classes, state_size=128, batch_size=batch_size, batch_norm=True)
    # Write the graph def:
    tf.train.write_graph(sess.graph_def, FLAGS.checkpoint+'/model', 'graph.pbtxt', as_text=False)
    print('The graph_def has been written!')
    xent = tf.losses.log_loss(Y, logits)
    loss_op = tf.reduce_mean(xent, name='loss_op')
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, name='global_step', trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
        tf.summary.scalar('loss', loss_op)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=BETA1,epsilon=ADAM_EPSILON)
        optimizer.minimize(loss_op, global_step=global_step,name='train_op')
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        global_step_tensor = sess.graph.get_tensor_by_name(
            'train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        # saver
        saver = tf.train.Saver(tf.global_variables())
        # merge all the summaries and write them out to FLAGS.checkpoint
        merged = tf.summary.merge_all()
        summary_tr_writer = tf.summary.FileWriter(FLAGS.checkpoint + '/train', sess.graph)
        summary_vd_writer = tf.summary.FileWriter(FLAGS.checkpoint + '/valid', sess.graph)
        # resume the training
        if FLAGS.resume:
            latest = tf.train.latest_checkpoint(FLAGS.checkpoint + '/model')
            if not latest:
                print, "No checkpoint to continue from in", FLAGS.checkpoint
                try:
                    os.stat(FLAGS.checkpoint + '/model')
                except:                        os.mkdir(FLAGS.checkpoint + '/model')
                print, "make model directory in " + FLAGS.checkpoint + '/model'
            else:
                print,"resume", latest
                saver.restore(sess, latest)
        # The training loop.
        print('Ready to train sound detection algorithm using rnn....')
        for step in range(FLAGS.num_batches):
            if step % 50 == 0 or step == 1:
                #batch_x, batch_y
                batch_x, batch_y = loader(val_dicts)
                # Run optimization op (backprop)
                [summary, num_step, loss] = sess.run([merged,global_step_tensor, loss_op],
                                                     feed_dict={X: batch_x, Y: batch_y})
                # write summary for validation
                summary_vd_writer.add_summary(summary, num_step)
                # save the trained model
                checkpoint_path = os.path.join(FLAGS.checkpoint + '/model', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step_tensor)
            else:
                #batch_x, batch_y
                batch_x, batch_y = loader(train_dicts)
                # Run optimization op (backprop)
                [summary, num_step, loss, _] = sess.run([merged,global_step_tensor, loss_op, train_op],
                                                                feed_dict={X: batch_x, Y: batch_y})
                # write summary for validation
                summary_tr_writer.add_summary(summary, num_step)
                if step % display_step == 0 :
                    print('Step %d: loss %g' % (num_step, loss))
