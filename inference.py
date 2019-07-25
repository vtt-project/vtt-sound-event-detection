from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa
import os
import json
from tqdm import tqdm
from model import vggish_rnn

# from tensorflow.python.platform import gfile
# set CUDA visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# set Flags
flags = tf.app.flags
flags.DEFINE_string('checkpoint', './checkpoint/vggish_rnn',
                    'Path to the checkpoint file')
flags.DEFINE_float('threshold', 0.5, 'Threshold for the final output')
flags.DEFINE_string('gpu', "1", 'GPU to use')
flags.DEFINE_string("input_file", None,
                    "Input file (wav format, 44100Hz, 16bit)")
flags.DEFINE_string('output_file', None,
                    'Output prediction json file')
flags.DEFINE_float(
    'time_step', 1.0, 'Time per step, please choose 0.1, 0.2, 0.5 or 1.0')
flags.DEFINE_string('gt_file', None,
                    'Ground truth')
flags.DEFINE_boolean('bn', True, 'BN')

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
"""
A dictionary for mapping the class name to numbers
"""
classes = {'background music': 0, 'speaking': 1, 'background laughing': 2,
           'crashing': 3, 'knocking': 4, 'opening/closing doors': 5,
           'clapping': 6, 'shouting': 7, 'laughing': 8, 'door bell': 9}
classes_list = ['background music', 'speaking', 'background laughing',
                'crashing', 'knocking', 'opening/closing doors',
                'clapping', 'shouting', 'laughing', 'door bell']


def _time_from_string(start_time, end_time):
    """
    Get the time from  start and stop time in string form, times in seconds
    Auto swap start time and end time if start time is before end time
    """
    if len(start_time) > 5:
        start_time = start_time[-5::]
        end_time = end_time[-5::]
    sm, ss = start_time.split(':')
    em, es = end_time.split(':')
    st = int(sm)*60 + int(ss)
    et = int(em)*60 + int(es)
    if et < st:
        return et, st
    return st, et


def _get_events(event_results_list):
    """
    Get 3 lists of start time, end time and event label from result list
    """
    startTime = []
    endTime = []
    label = []
    for ev in event_results_list:
        st, et = _time_from_string(ev['start_time'], ev['end_time'])
        if(st == et):
            continue
        startTime.append(st)
        endTime.append(et)
        label.append(ev['sound_type'])
    return np.array(startTime), np.array(endTime), np.asarray(label)


def _get_label(event_list, classes):
    label = np.zeros(len(classes))
    for ev in event_list:
        if ev in classes.keys():
            label[classes[ev]] = 1
    return label


def _get_event_from_scroll(scroll, predictions, times, classes=None):
    """
    Extract instance of event start and stop from event scroll
    """
    # pdb.set_trace()
    results = []
    new_scroll = dict(scroll)
    for n, pre in enumerate(predictions):
        if scroll[classes[n]]['startTime'] == -1:
            if pre:
                '''
ti                If there is no event of 'class[n]' in the previous scroll, set the startTime, endTime
                '''
                new_scroll[classes[n]]['startTime'] = times[0]
                new_scroll[classes[n]]['endTime'] = times[1]
        else:
            if pre:
                '''
                If the event continues, change the stopTime
                '''
                # pdb.set_trace()
                new_scroll[classes[n]]['endTime'] = times[1]
            else:
                '''
                Extract the result instance from the scroll
                '''
                results.append({'start_time': scroll[classes[n]]['startTime'],
                                'end_time': scroll[classes[n]]['endTime'],
                                'sound_type': classes[n]})
                new_scroll[classes[n]]['startTime'] = -1
                new_scroll[classes[n]]['endTime'] = -1

    return new_scroll, results


"""
Parameters of mel feature extraction
"""
sr = 44100.0                    # Sampling rate of the audio
hop_length = 1024.0             # Hope size of the STFT transformation
n_fft = 2048                    # FFT length
win_length = 1536               # The window length of STFT

"""
List files to evaluate
"""
input_wav = FLAGS.input_file
output_json = FLAGS.output_file
gt_json = FLAGS.gt_file
assert input_wav is not None, "Please specify a input wav file"

# Training Parameters
batch_size = 1
num_classes = len(classes)  # TUT-synthetic SED total classes (0-15)

samples_per_step = int(sr)
time_per_step = samples_per_step/sr
samples_per_slide = int(sr*FLAGS.time_step)
time_per_slide = FLAGS.time_step

with tf.Session() as sess:
    X = tf.placeholder("float", [None, 44, 1025, 1], name='new_features')
    logits = vggish_rnn(X, num_classes=num_classes,
                        batch_size=1, state_size=128, batch_norm=FLAGS.bn)
    saver = tf.train.Saver(tf.global_variables())
    latest = tf.train.latest_checkpoint(FLAGS.checkpoint + '/model')
    saver = tf.train.Saver()
    saver.restore(sess, latest)

    # The training loop.
    print('Ready to test sound detection algorithm using rnn....')
    if output_json is not None:
        print('Output will be written to {}'.format(output_json))
    else:
        print("No output file is specificy, evaluation only!")
    # Variable to save the results
    true_num = 0
    predict_num = 0
    tp, fn, tn, fp = 0, 0, 0, 0
    I = np.zeros(num_classes)
    U = np.zeros(num_classes)

    audio_file = input_wav
    """
    Load the audio files
    """
    au, _ = librosa.load(
        audio_file, sr=sr)  # Load the audio samples to a numpy array
    # Duration of the loaded file in seconds
    T = len(au)/sr

    """
    Load the json ground truth file
    """
    if gt_json is not None:
        truth_json = gt_json
        with open(truth_json, 'r') as ffp:
            ann_dict = json.load(ffp)
            event_results_list = ann_dict['sound_results']
        startTime, endTime, label = _get_events(event_results_list)
        sT = 0                          # The first moment of the audio
        eT = 1
        label_secs = []

        while not (eT > (T-time_per_slide)):
            label_id = (startTime <= (sT+0.1)) & (endTime >= eT)
            labels = label[label_id]

            if len(labels) > 0:
                label_secs.append(list(labels))
            else:
                label_secs.append(['None'])
            sT += time_per_slide
            eT += time_per_slide

        num_batches = len(label_secs)
    else:
        num_batches = T-1

    """
    Output dictionary for each file
    """
    out_dict = {'file_name': os.path.splitext(os.path.basename(audio_file))[0]+'.json',
                'input_file': os.path.basename(audio_file),
                'sound_results': []}

    """
    Event scroll for the event detection
    """
    event_scroll = {}
    for key in classes.keys():
        d = {}
        d['startTime'] = -1
        d['stopTime'] = -1
        event_scroll[key] = d

    for step in tqdm(np.arange(num_batches, dtype=int), desc=audio_file):
        #batch_x, batch_y
        x = au[step *
               samples_per_slide:(step*samples_per_slide+samples_per_step)]
        stft = librosa.core.stft(x, hop_length=int(
            hop_length), n_fft=int(n_fft), win_length=int(win_length))
        batch_x = np.zeros((1, 44, 1025, 1))
        batch_x[0, :, :, 0] = np.abs(stft).T
        # Run the prediction only
        [pred] = sess.run([logits], feed_dict={X: batch_x})

        if gt_json:
            batch_y = _get_label(label_secs[step], classes)
            batch_y = batch_y[np.newaxis, :]
            # Calculate the metrics: True Positive, False Negative, True Negative, False Positive of this step
            TP = (batch_y > 0.5) * (pred > FLAGS.threshold)
            FN = (batch_y > 0.5) > (pred > FLAGS.threshold)
            TN = (batch_y < 0.5) * (pred < FLAGS.threshold)
            FP = (batch_y < 0.5) * (pred > FLAGS.threshold)
            I = I + TP.astype(np.int)
            U = U + (FP + batch_y.astype(np.bool)).astype(np.int)
            # Cummulative metrics
            tp += np.sum(TP, axis=0)
            fn += np.sum(FN, axis=0)
            tn += np.sum(TN, axis=0)
            fp += np.sum(FP, axis=0)
        # Push to event scroll:
        event_scroll, results = _get_event_from_scroll(
            event_scroll, pred[0] > FLAGS.threshold, (step, step + samples_per_step/sr), classes=classes_list)

        out_dict['sound_results'] = out_dict['sound_results'] + results

    # Dump file to json
    if output_json is not None:
        with open(os.path.join(output_json), 'w') as f:
            json.dump(out_dict, f, indent=4)

if gt_json is not None:
    acc = (tp + tn + 0.01) / (tp + fn + tn + fp + 0.01).astype(np.float)
    pre = (tp + 1.) / (tp + fp + 1.).astype(np.float)
    rec = (tp+0.01) / (tp + fn + 0.01).astype(np.float)
    f1 = 2*pre*rec/(pre+rec+0.01)
    print('TP:', tp)
    print('FN:', fn)
    print('TN:', tn)
    print('FP:', fp)
    print('acc:', acc)
    print('pre:', pre)
    print('rec:', rec)
    print('f1_score:', f1)
    print('IoU:', I.astype(np.float)/U)
    print('Overall IoU:', np.sum(I)/np.sum(U).astype(np.float))
