import json
import math
import os

from argparse import ArgumentParser
from datetime import datetime

import numpy as np

from sklearn import tree
from sklearn.externals import joblib

import sc2reader as sc2
from sc2reader.events.game import AddToControlGroupEvent
from sc2reader.events.game import GetControlGroupEvent
from sc2reader.events.game import SetControlGroupEvent
from sc2reader.events.game import SelectionEvent

MODELS_DIRECTORY = 'models'
MODELS_METADATA_FILE = 'models.json'

sc2.configure(debug=True)

def safe_read(dict, key, default_value=None):
    try:
        return dict[key]
    except:
        return default_value

class Player:
    def __init__(self, player, bucket_size_minutes = 5, max_game_length=45):  # player object from sc2reader
        self.name = player.name
        self.url = player.url
        # Actual race (only difference from pick_race if pick_race is Random)
        self.play_race = player.play_race
        # Race chosen (only difference from play_race if pick_race is Random)
        self.pick_race = player.pick_race
        self.hotkeys = [Hotkey(x, bucket_size_minutes=bucket_size_minutes, max_game_length=max_game_length) for x in range(0, 10)]
        self.first_hotkey = -1

    def __str__(self):
        race = self.play_race[:1]
        if self.play_race != self.pick_race:
            race += '{}'.format(self.pick_race[:1])
        return '{}-{}'.format(self.name, race)

    def handle_event(self, event):
        key = event.control_group
        if self.first_hotkey < 0:
            self.first_hotkey = key
        self.hotkeys[key].handle_event(event)

    def verbose(self):
        race = self.play_race[:1]
        if self.play_race != self.pick_race:
            race += '/{}'.format(self.pick_race[:1])
        hotkeys = ''
        for hk in self.hotkeys:
            hotkeys += '{}\n'.format(hk)
        return '{} ({}):\n{}'.format(self.name, race, hotkeys)

    def get_dataset(self):
        race = -1
        if self.play_race is 'Terran':
            race = 0
        elif self.play_race is 'Protoss':
            race = 1
        elif self.pick_race is 'Zerg':
            race = 2
        dataset = [race, self.first_hotkey]
        for hk in self.hotkeys:
            dataset += hk.get_dataset()
        return dataset

class Hotkey:
    def __init__(self, key, bucket_size_minutes, max_game_length):
        self.key = key
        self.buckets = self._init_buckets(
            bucket_size_minutes, max_game_length)
        self.bucket_size_minutes = bucket_size_minutes


    def __str__(self):
        buckets = 'Buckets:\n'
        for b in self.buckets:
            buckets += '{}'.format(b)
        return str('{}:\n{}\n\n'.format(self.key, buckets))

    def _init_buckets(self, bucket_size_minutes, max_game_length):
        buckets = []
        for n in range(0, max_game_length, bucket_size_minutes):
            buckets.append(Bucket(n, n + bucket_size_minutes))

        return buckets

    def handle_event(self, event):
        minute = math.floor(event.second / 60)
        bucket = math.floor((minute - minute % self.bucket_size_minutes) / self.bucket_size_minutes)

        ev = event.name
        try:
            if ev is 'SetControlGroupEvent':
                self.buckets[bucket].set_event()
            elif ev is 'GetControlGroupEvent':
                self.buckets[bucket].get_event()
            elif ev is 'AddToControlGroupEvent':
                self.buckets[bucket].add_event()
        except Exception as e:
            print('Cannot handle event - game is probably too long: min={} bucket={} error={}'.format(minute, bucket, e))

    def get_bucket_labels(self):
        return [b.__str__() for b in self.buckets]

    def get_dataset(self):
        dataset = []
        for b in self.buckets:
            dataset += b.get_dataset()
        return dataset


class Bucket:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.set_events = 0
        self.get_events = 0
        self.add_events = 0

    def __str__(self):
        return '{} -> {}'.format(self.lower_bound, self.upper_bound)

    def verbose(self):
        return '    [{} -> {}]: {} {} {}\n'.format(
            self.lower_bound, self.upper_bound,
            self.set_events,self.get_events, self.add_events)

    def set_event(self):
        self.set_events += 1

    def get_event(self):
        self.get_events += 1

    def add_event(self):
        self.add_events += 1

    def get_dataset(self):
        return [self.set_events, self.get_events, self.add_events]

class ModelInfo:
    def __init__(self, name, args=None):
        self.name = name
        self.training_data_size = 0
        self.args = {} if args is None else vars(args)
        self.bucket_size_minutes = -1 if args is None else args.bucket_size
        self.labels = {}
        self.classifications = []

        if args is None:
            self.load()

    def save(self, filename=os.path.join(MODELS_DIRECTORY, MODELS_METADATA_FILE)):
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('{}')

        j = {}
        with open(filename, 'r') as f:
            j = json.load(f)
            info = {
                'name': self.name,
                'args': self.args,
                'labels': self.labels,
                'training_data_size': self.training_data_size,
                'bucket_size_minutes': self.bucket_size_minutes,
                'classifications': self.classifications
            }
            j[self.name] = info

        with open(filename, 'w') as f:
            json.dump(j, f, sort_keys=True, indent=4)

    def load(self, filename=os.path.join(MODELS_DIRECTORY, MODELS_METADATA_FILE)):
        with open(filename, 'r') as f:
            j = json.load(f)
            info = safe_read(j, self.name)
        
            self.args = safe_read(info, 'args', {})
            self.training_data_size = safe_read(info, 'training_data_size', 0)
            self.labels = safe_read(info, 'labels', [])
            self.bucket_size_minutes = safe_read(info, 'bucket_size_minutes', 5)
            self.classifications = safe_read(info, 'classifications', [])

    def add_label(self, label):
        value = safe_read(self.labels, label, 0)
        self.labels[label] = value + 1

    def add_classification(self, accuracy, mismatches):
        self.classifications.append({
            'accuracy': accuracy,
            'mismatches': mismatches
        })

def get_players(replay):
    players = []
    for p in replay.players:
        players.append(Player(p))

    player_dict = dict()
    for p in players:
        player_dict[p.name] = p

    return player_dict

# Train a classifier using all replay files in the given directory
def train_classifier(replay_directory, bucket_size_minutes, model_info):
    # Recursively find all .SC2Replay files in the given directory
    training_files = [os.path.join(root, fname) for root, _, files in os.walk(
            replay_directory) for fname in files if '.SC2Replay' in fname]

    X = []  # Data
    y = []  # Labels

    file_count = len(training_files)
    model_info.training_data_size = file_count
    for n, filename in enumerate(training_files):
        replay = sc2.load_replay(filename, load_level=4)

        players = get_players(replay)

        print('Processing {} of {}\n{} vs {} on {}\n'.format(n+1, file_count, replay.players[0], replay.players[1], replay.map_name))

        for event in replay.game_events:
            player_name = event.player.name
            if player_name not in players:
                continue

            if event.name in ['AddToControlGroupEvent', 'SetControlGroupEvent', 'GetControlGroupEvent']:
                players[player_name].handle_event(event)

        for name, player in players.items():
            X.append(player.get_dataset())
            y.append(player.__str__())
            model_info.add_label(player.__str__())

    X = np.asarray(X)
    y = np.asarray(y)

    print('shape(X)={}, shape(y)={}'.format(X.shape, y.shape))
    print('Constructing classifier...')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print('Classifier built successfully!')

    return clf, model_info

# Load a classifier from a saved .pkl file
def load_classifier(filename):
    return joblib.load(filename)

# Try to identify the players in the given replay using the given model
def classify(clf, filename):
    replay = sc2.load_replay(filename, load_level=4)
    log_info = 'Analysing file \'{}\'\n'.format(filename)
    log_info += 'Map \'{}\' played in region \'{}\' on date {}. Length={}\n'.format(replay.map_name, replay.region, replay.date, replay.length)
    players = get_players(replay)

    for event in replay.game_events:
        player_name = event.player.name
        if player_name not in players:
            continue

        if event.name in ['AddToControlGroupEvent', 'SetControlGroupEvent', 'GetControlGroupEvent']:
            players[player_name].handle_event(event)

    mismatches = []
    n = 1
    for name, player in players.items():
        mismatch = None
        data = [player.get_dataset()]
        prediction = clf.predict(data)

        player_name = '{}'.format(player)
        prediction_name = '{}'.format(prediction[0])

        log_info += 'Player {}: replay=\'{}\', prediction=\'{}\'\n'.format(n, player_name, prediction_name)

        if player_name != prediction_name:
            mismatch = {
                'name': player_name,
                'prediction': prediction_name,
                'replay': filename,
                'game_length': '{}'.format(replay.game_length),
                'opponent': '{}'.format([p for p in players if p != name][0])
            }

            log_info += '# MISMATCH #\n'
            mismatches.append(mismatch)
        n += 1

    log_info += '\n\n'

    return log_info, mismatches
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        help='Directory containing replay files for training the model (recursive)')
    parser.add_argument('--load', type=str, help='Filename of a saved model')
    parser.add_argument('--save', type=str, help='Filename to save the trained model',
                        default='{}.pkl'.format(datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M')))
    parser.add_argument('--classify', type=str, nargs='+', help="A directory, or one or more replay filenames to categorise using our model")
    parser.add_argument('--bucket_size', type=int, help='Length in minutes of sampling buckets. default=5', default=5)
    args = parser.parse_args()

    if args.train:
        if not os.path.exists(MODELS_DIRECTORY):
            os.mkdir(MODELS_DIRECTORY)
        model_info = ModelInfo(args.save, args)
        clf, model_info = train_classifier(args.train, args.bucket_size, model_info)

        # Save classifier for later use
        print('Saving classifier to file {}'.format(args.save))
        joblib.dump(clf, os.path.join('models', args.save))
        model_info.save()

    elif args.load:
        clf = load_classifier(os.path.join(MODELS_DIRECTORY, args.load))
    else:
        clf = None

    if args.classify:
        if not clf:
            raise ValueError('No classifier - please provide a replay directory or a previously saved model')

        # Write output to a timestamped directory
        output_dir = 'logs'
        os.mkdir(output_dir)

        model_name = args.load if args.load else args.save
        model_info = ModelInfo(model_name)
        with open(os.path.join(output_dir, '{}.txt'.format(datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S'))), 'w') as f:
            f.write('{}\n\n'.format(args))

            files = []
            for file_or_dir in args.classify:
                if os.path.isfile(file_or_dir) and '.SC2Replay' in file_or_dir:
                    files.append(file_or_dir)
                if os.path.isdir(file_or_dir):
                    files += [os.path.join(root, fname) for root, _, files in os.walk(file_or_dir) for fname in files if '.SC2Replay' in fname]

            print('Classifying {} files: {}'.format(len(files), files))

            mismatches = []
            for replay in files:
                info, mismatch = classify(clf, replay)
                print(info)
                f.write(info)

                if mismatch:
                    mismatches += mismatch

            accuracy = (len(replay) - len(mismatches)) / len(replay) * 100
            mismatches_info = ''
            for m in mismatches:
                mismatches_info += '  {}\n'.format(m)
            accuracy_info = 'Accuracy: {:.2f}%\nMismatches: [\n{}]'.format(accuracy, mismatches_info)

            print(accuracy_info)
            f.write(accuracy_info)

            model_info.add_classification(accuracy, mismatches)
            model_info.save()

        print('Done!')