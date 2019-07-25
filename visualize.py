import sed_vis
import dcase_util
import argparse
import json
import os
# import pdb
parser = argparse.ArgumentParser()

parser.add_argument('--audio-file', type=str,
                    default='friends/audio_wav/s01_ep08.wav')
parser.add_argument('--ground-truth', type=str, default=None)
parser.add_argument('--prediction', type=str, default=None)

args = parser.parse_args()


def _time_from_string(start_time, end_time):
    # Get time by seconds from "mm:ss" or "hh:mm:ss"
    if isinstance(start_time, unicode):
        if len(start_time) > 5:
            start_time = start_time[-5::]
            end_time = end_time[-5::]

        sm, ss = start_time.split(':')
        em, es = end_time.split(':')
        st = int(sm) * 60 + int(ss)
        et = int(em) * 60 + int(es)
        if et < st:
            return et, st
        return st, et
    else:
        return start_time, end_time


def _convert_to_dcase_format(ann_dict, file_name=None, string_time=True):
    ev_list = []
    sound_results = ann_dict['sound_results']

    get_time = _time_from_string

    for ev in sound_results:
        d = {}
        d['onset'], d['offset'] = get_time(ev['start_time'], ev['end_time'])
        d['file_name'] = file_name
        d['event_label'] = ev['sound_type']
        ev_list.append(d)

    return ev_list


if args.ground_truth is not None:
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    truth = _convert_to_dcase_format(
        ground_truth, file_name=os.path.basename(args.audio_file))

if args.prediction is not None:
    with open(args.prediction, 'r') as f:
        prediction = json.load(f)
    pred = _convert_to_dcase_format(
        prediction, file_name=os.path.basename(args.audio_file))

audio_container = dcase_util.containers.AudioContainer().load(
    args.audio_file, mono=True)
truth_list = dcase_util.containers.MetaDataContainer(truth)
pred_list = dcase_util.containers.MetaDataContainer(pred)
event_lists = {'reference': truth_list,
               'estimated': pred_list}
"""
Visualize the predicted and ground_truth sound events
"""
vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                audio_signal=audio_container.data,
                                                sampling_rate=audio_container.fs)
vis.show()
