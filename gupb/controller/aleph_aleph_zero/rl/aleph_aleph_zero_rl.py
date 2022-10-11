from random import random

import numpy as np
import torch
from torch import optim, nn

from gupb import controller
from gupb.controller.aleph_aleph_zero.rl.net import NeuralNetwork
from gupb.controller.aleph_aleph_zero.utils import taxicab_distance
from gupb.model import characters, arenas, coordinates
from gupb.model.characters import Facing
from gupb.model.coordinates import sub_coords

TOTAL_CHAMPS = 4
CENTER = (9,9)

ACTIONS = {
    0: characters.Action.STEP_FORWARD,
    1: characters.Action.TURN_LEFT,
    2: characters.Action.TURN_RIGHT,
    3: characters.Action.ATTACK
}

class AlephAlephZeroBotRL(controller.Controller):
    def __init__(self, first_name):
        self.first_name = first_name

        self.position = None

        self.menhir_position = None
        self.menhir_seen = False
        self.menhir_pos_updated = False

        self.known_champion_positions = dict()  # champ: position
        self.champions_last_seen = dict()  # champ: int

        self.mists = []

        self.round_count = 0

        self.model = NeuralNetwork(in_size=3*TOTAL_CHAMPS+5, out_size=len(ACTIONS.keys()))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()

        self.replay_memory = []

        self.epsilon = self.model.initial_epsilon

    def _update_knowledge(self, new_knowledge: characters.ChampionKnowledge):
        self.position = new_knowledge.position

        for champion in self.champions_last_seen.keys():
            self.champions_last_seen[champion]+=1

        for tile, tile_desc in new_knowledge.visible_tiles.items():
            if tile_desc.character is not None and tile!=new_knowledge.position:
                self.champions_last_seen[tile_desc.character]=0
                self.known_champion_positions[tile_desc.character]=tile

        self.remaining_champions = new_knowledge.no_of_champions_alive

        self.facing = self._calculate_facing(new_knowledge)

    def _calculate_facing(self, new_knowledge):  # afaik, we have to calculate this
        facing_vals = {f.value: f for f in Facing}
        for coord in new_knowledge.visible_tiles:
            if sub_coords(coord, new_knowledge.position) in facing_vals:
                return facing_vals[sub_coords(coord, new_knowledge.position)]


    def vectorize_knowledge(self):
        info_as_list = []

        info_as_list.append(self.position[0])
        info_as_list.append(self.position[1])
        info_as_list.append(self.facing.value[0])
        info_as_list.append(self.facing.value[1])

        champs_info = []
        for champ_name in self.champions_last_seen:
            champs_info.append((
                self.champions_last_seen[champ_name], self.known_champion_positions[champ_name]
            ))

        while len(champs_info)<TOTAL_CHAMPS-1:
            champs_info.append((1000, (9,9)))

        champs_info.sort(key= lambda x: (x[0],taxicab_distance(self.position,x[1])))

        info_as_list+=champs_info

        info_as_list+=[self.remaining_champions, self.round_count]

        return np.array(info_as_list)

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:

        self._update_knowledge(knowledge)

        self.round_count+=1

        v_k = self.vectorize_knowledge()

        output = self.model(v_k)
        action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)

        random_action = random.random() <= self.epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(self.model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        action[action_index] = 1

        self.last_state = v_k
        self.last_action = action
        return ACTIONS[action_index]

    def praise(self, score: int) -> None:

        q_value = torch.sum(self.model(self.last_state) * self.last_action, dim=1)
        self.model.zero_grad()


    def reset(self, arena_description: arenas.ArenaDescription) -> None:
        pass

    @property
    def name(self) -> str:
        return self.first_name

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.LIME
