from gupb.controller.aleph_aleph_zero.aleph_aleph_zero import AlephAlephZeroBot

__all__ = [
    'aleph_aleph_zero',
    'aleph_aleph_zero_rl',
    'POTENTIAL_CONTROLLERS'
]

POTENTIAL_CONTROLLERS = [
    AlephAlephZeroBot("AA0")
]