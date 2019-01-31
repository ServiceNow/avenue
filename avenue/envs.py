from .env import AvenueEnv, VisualAvenueEnv, UnityEnv, AllStatesAvenueEnv
from .wrappers import DifferentialActions, DifferentialActionsVisual
from gym.wrappers import TimeLimit

class Circuit(AvenueEnv):
    host_ids = {'linux': '1t0Uy29qUteBUzot2pfEyXKWWnrBAJRN-'}
    asset_name = 'circuit'

class CircuitSegmentation(VisualAvenueEnv):
    host_ids = {'linux': '1hjB1OimKrkEzOHBAQTOI6syzi-UT_GH9'}
    asset_name = 'circuit_segmentation'


class DatasetCollection(VisualAvenueEnv):
    host_ids = {'linux': '1WE--vDGYKYMBYPsuCqJJehTKHCIx8zgl'}
    asset_name = 'dataset_collection'


class CircuitVisual(VisualAvenueEnv):
    host_ids = {'linux': '175-NVmuqQawlubyMd_1eT6qigIa0RNBi'}
    asset_name = 'circuit_visual'


class BirdView(VisualAvenueEnv):
    host_ids = {'linux': '1dmPPK4mFTnYPnatpSWIme0QmYSvFR9l-', 'darwin':'15Z21R9RlaQGN1jv-ipZSoN5PJYjNS3DB'}
    asset_name = 'race_against_time'


class BirdViewSolo(VisualAvenueEnv):
    host_ids = {'linux': '1imEoe9CWyij9fIQwQwHEVdspDWDEHjNH'}
    asset_name = 'race_against_time_solo'


class CircuitRgb(VisualAvenueEnv):
    host_ids = {'linux': '16ppvjY8xT7p5R-bVGe6OtNle6rLW8PuN'}
    visual = True
    asset_name = 'circuit_rgb'


class ScenarioZoom(AllStatesAvenueEnv):
    host_ids = {'linux': '16ppvjY8xT7p5R-bVGe6OtNle6rLW8PuN'}
    visual = True
    asset_name = 'scenario_zoom'
    vector_state_class = "AvenueStateZoom"

def Circuit_v1():
    env = Circuit()
    env = TimeLimit(env, max_episode_steps=10000)
    env = DifferentialActions(env)
    return env

def CircuitRgb_v1():
    env = CircuitRgb()
    env = DifferentialActionsVisual(env)
    return env