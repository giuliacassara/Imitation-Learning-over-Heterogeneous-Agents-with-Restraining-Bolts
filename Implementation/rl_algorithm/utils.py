import os
from copy import copy, deepcopy
from typing import cast

import inferrer
import pythomata
from flloat.semantics import PLInterpretation
from inferrer.automaton.dfa import DFAWrapper
from pythomata.dfa import DFA


class mydefaultdict(dict):
    def __init__(self, x):
        super().__init__()
        self._default = x

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            self[key] = copy(self._default)
            return self[key]


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def from_inferrer_to_pythomata(dfa: inferrer.automaton.dfa.DFAWrapper) -> pythomata.dfa.DFA:
    """Transfom an inferrer.automaton.dfa.DFAWrapper into pythomata.DFA."""

    char2sym = dfa.char2sym
    unwrapped_dfa = dfa.unwrapped

    transitions = {
        state.name: {
            PLInterpretation({char2sym[char]}): state2.name
            for char, state2 in unwrapped_dfa._transitions[state].items()
        } for state in unwrapped_dfa.states
    }
    accepting_states = {s.name for s in unwrapped_dfa.accept_states}
    initial_state = unwrapped_dfa._start_state.name
    new_dfa = pythomata.dfa.DFA.from_transitions(initial_state, accepting_states, transitions)
    return new_dfa


def post_process_dfa(dfa):
    """Add a loop transition to every state."""
    alphabet = dfa.alphabet.union({PLInterpretation(set())})
    states = dfa.states
    initial_state = dfa.initial_state
    accepting_states = dfa.accepting_states
    transitions = deepcopy(dfa.transition_function)

    for state in states:
        if PLInterpretation(set()) not in transitions[state]:
            transitions[state][PLInterpretation(set())] = state

    return DFA(states, alphabet, initial_state, accepting_states, transitions)


def learn_dfa(arguments) -> pythomata.dfa.DFA:
    positive_traces_filepath = os.path.join(arguments.output_dir, "positive_traces.txt")
    negative_traces_filepath = os.path.join(arguments.output_dir, "negative_traces.txt")
    dfa = inferrer.learn.learn(positive_traces_filepath, negative_traces_filepath, algorithm_id="lstar", separator=";")
    dfa = cast(DFAWrapper, dfa)
    new_dfa = from_inferrer_to_pythomata(dfa)
    new_dfa = new_dfa.complete()
    new_dfa = post_process_dfa(new_dfa)
    return new_dfa
