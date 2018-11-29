from .env import AvenueEnv
from .wrappers import TimeLimit


class Circuit(AvenueEnv):
    host_ids = {'linux': '1t0Uy29qUteBUzot2pfEyXKWWnrBAJRN-'}
    visual = False
    asset_name = 'circuit'


class CircuitSegmentation(AvenueEnv):
    host_ids = {'linux': '1hjB1OimKrkEzOHBAQTOI6syzi-UT_GH9'}
    visual = True
    asset_name = 'circuit_segmentation'


class DatasetCollection(AvenueEnv):
    host_ids = {'linux': '1WE--vDGYKYMBYPsuCqJJehTKHCIx8zgl'}
    visual = True
    asset_name = 'dataset_collection'


class CircuitVisual(AvenueEnv):
    host_ids = {'linux': '175-NVmuqQawlubyMd_1eT6qigIa0RNBi'}
    visual = True
    asset_name = 'circuit_visual'


class RaceAgainstTime(AvenueEnv):
    host_ids = {'linux': '1dmPPK4mFTnYPnatpSWIme0QmYSvFR9l-', 'darwin':'15Z21R9RlaQGN1jv-ipZSoN5PJYjNS3DB'}
    visual = True
    asset_name = 'race_against_time'


class RaceAgainstTimeSolo(AvenueEnv):
    host_ids = {'linux': '1imEoe9CWyij9fIQwQwHEVdspDWDEHjNH'}
    visual = True
    asset_name = 'race_against_time_solo'


class CircuitRgb(AvenueEnv):
    host_ids = {'linux': '1UZ-Wv-yFBhjlr-mNEECDUXwrnwOEEjt0'}
    visual = True
    asset_name = 'circuit_rgb'


def CircuitRgb_v1():
    env = CircuitRgb()
    env = TimeLimit(env, max_episode_steps=10000)
    return env