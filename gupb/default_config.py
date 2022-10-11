from gupb.controller import keyboard, aleph_aleph_zero
from gupb.controller import random
from gupb.controller.aleph_aleph_zero import aleph_aleph_zero_rl

CONFIGURATION = {
    'arenas': [
        'lone_sanctum',
    ],
    'controllers': [
        aleph_aleph_zero_rl.AlephAlephZeroBotRL("A"),
        aleph_aleph_zero_rl.AlephAlephZeroBotRL("B"),
        aleph_aleph_zero_rl.AlephAlephZeroBotRL("C"),
        aleph_aleph_zero_rl.AlephAlephZeroBotRL("D"),
    ],
    'start_balancing': False,
    'visualise': True,
    'show_sight': None,
    'runs_no': 5,
    'profiling_metrics': [],
}


