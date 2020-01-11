# coding: utf-8

from interface import AbstractHmmParams
import os
import json


DATA    = 'data'
DEFAULT = 'default'

class DefaultHmmParams(AbstractHmmParams):

    def __init__(self,):
        self.py2hz_dict      = self.readjson('data/hmm_py2hz.json')
        self.start_dict      = self.readjson('data/hmm_start.json')
        self.emission_dict   = self.readjson('data/hmm_emission.json')
        self.transition_dict = self.readjson('data/hmm_transition.json')

    def readjson(self, filename):
        return json.load(open(filename, 'r'))

    def pwd(self):
        return os.path.dirname(os.path.abspath(__file__))

    def start(self, state):
        ''' get start prob of state(hanzi) '''

        data = self.start_dict[DATA]
        default = self.start_dict[DEFAULT]

        if state in data:
            prob = data[state]
        else:
            prob = default
        return float(prob)


    def emission(self, hanzi, pinyin):
        ''' state (hanzi) -> observation (pinyin) '''

        data = self.emission_dict[DATA]
        default = self.emission_dict[DEFAULT] 

        if hanzi not in data:
            return float( default )
        
        prob_dict = data[hanzi]

        if pinyin not in prob_dict:
            return float( default )
        else:
            return float( prob_dict[pinyin] )

    def transition(self, from_state, to_state):
        ''' state -> state '''

        data = self.transition_dict[DATA]
        default = self.transition_dict[DEFAULT]

        if from_state not in data:
            return float( default )
        
        prob_dict = data[from_state]

        if to_state in prob_dict:
            return float( prob_dict[to_state] )
        
        if DEFAULT in prob_dict:
            return float( prob_dict[DEFAULT] )

        return float( default )

    def get_states(self, observation):
        ''' get states which produce the given obs '''
        return [hanzi for hanzi in self.py2hz_dict[observation]]

        

