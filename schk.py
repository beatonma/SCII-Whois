import os

from argparse import ArgumentParser

# from sklearn import tree
# from sklearn.externals import joblib

import sc2reader as sc2
from sc2reader.events.game import ControlGroupEvent, GetControlGroupEvent

sc2.configure(debug=True)

class Game:
    def __init__(self, replay):
        self.date = replay.date
        self.map_name = replay.map_name
        self.region = replay.region
        self.length = replay.game_length
        # self.sc2_version = replay.release

    def __str__(self):
        return '{} {} {} {}'.format(self.map_name, self.region, self.date, self.length)

class Player:
    def __init__(self, player):
        self.play_race = player.play_race
        self.pick_race = player.pick_race
        self.url = player.url
        self.name = player.name
        self.result = player.result
        self.control_group_events = []

    def __str__(self):
        events = ''
        for event in self.control_group_events:
            events += '({}) Group: {}, Type: {}\n'.format(type(event), event.control_group, event.update_type)
        events += '({} events)'.format(len(self.control_group_events))
        return '{} ({}) {} ({}) {}\nControl group events:\n{}'.format(self.name, self.result, self.play_race, self.pick_race, self.url, events)

    def add_event(self, control_group_event):
        self.control_group_events.append(control_group_event)


def get_players(replay):
    players = []
    for p in replay.players:
        players.append(Player(p))

    player_dict = dict()
    for p in players:
        player_dict[p.name] = p

    return player_dict

def train(replays):
    X = list()  # Data
    y = list()  # Labels

    for file in replays:
        replay = sc2.load_replay(file, load_level=4)

        game = Game(replay)
        players = get_players(replay)

        for event in replay.game_events:
            if issubclass(type(event), ControlGroupEvent):
                players[event.player.name].add_event(event)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('training_directory', type=str, help='Directory containing replay files for trainging the model')
    args = parser.parse_args()

    training_files = [os.path.join(root, fname) for root, _, fname in os.walk(args.training_directory)]
    print(training_files)




# if __name__ == '__main__':
#     replay = sc2.load_replay(os.path.join('replays', 'sample.SC2Replay'), load_level=4)

#     game = Game(replay)
#     players = get_players(replay)

#     print('{}'.format(replay))

#     for event in replay.game_events:
#         if issubclass(type(event), ControlGroupEvent):
#             players[event.player.name].add_event(event)
#             # print('ControlGroupEvent: {} {} {}'.format(event.player.name, event.control_group, event.update_type))
#     print('players:')
#     for n, p in players.items():
#         print(p)