from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui
import asyncio
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage, FeedbackStage
from nicewebrl import get_logger
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from xminigrid.rendering.text_render import _text_encode_rule, _encode_tile
from rendering import render

import os
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = int(os.getenv("MAX_STEPS", 200))
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


class PlaygroundTimestepWrapper(TimestepWrapper):
  def reset(self, key: jax.random.PRNGKey, params=None):
    timestep = self._env.reset(key=key, params=params)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)

  def step(self, key, state, action, params=None):
    if isinstance(state, TimeStep):
      state = state.state
    timestep = self._env.step(params=params, timestep=state, action=action)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)


########################################
# Define actions and corresponding keys
########################################
class Actions(IntEnum):
  forward = 0
  clockwise = 1
  counter_clockwise = 2
  pick_up = 3
  put_down = 4
  toggle = 5


# Only first 3 actions are actually used
actions = jnp.array([0, 1, 2, 3, 4, 5])
action_keys = ["ArrowUp", "ArrowRight", "ArrowLeft", "p", "d", "t"]  # Mapping to keys
action_to_name = [Actions(int(i)).name for i in actions]


########################################
# Create multiple environments with different rulesets
########################################
def text_encode_goal(goal: list[int]) -> str:
  # copied and edited from: https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/rendering/text_render.py#L140
  goal_id = goal[0]
  if goal_id == 1:
    return f"Agent_Hold({_encode_tile(goal[1:3])})"
  elif goal_id == 3:
    return f"Agent_Near({_encode_tile(goal[1:3])})"
  elif goal_id == 4:
    return f"Tile_Near({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 7:
    return f"Tile_Near_Up_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 8:
    return f"Tile_Near_Right_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 9:
    return f"Tile_Near_Down_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 10:
    return f"Tile_Near_Left_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id == 11:
    return f"Agent_Near_Up_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 12:
    return f"Agent_Near_Right_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 13:
    return f"Agent_Near_Down_Goal({_encode_tile(goal[1:3])})"
  elif goal_id == 14:
    return f"Agent_Near_Left_Goal({_encode_tile(goal[1:3])})"
  else:
    raise RuntimeError(f"Rendering: Unknown goal id: {goal_id}")


def describe_ruleset(ruleset) -> str:
  str = "GOAL:" + "\n"
  goal = text_encode_goal(ruleset.goal.tolist())
  goal.split()
  str += text_encode_goal(ruleset.goal.tolist()) + "\n"
  str += "\n"
  str += "RULES:" + "\n"
  for rule in ruleset.rules.tolist():
    if rule[0] != 0:
      str += _text_encode_rule(rule) + "\n"
  str += "\n"
  str += "INIT TILES:" + "\n"
  for tile in ruleset.init_tiles.tolist():
    if tile[0] != 0:
      str += _encode_tile(tile) + "\n"

  return str


num_envs = 3
def render_fn(timestep: nicewebrl.TimeStep):
  return render(timestep.state.grid, timestep.state.agent).astype(jnp.uint8)


########################################
# Define Instruction Stage
########################################

all_stages = []


async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """Press the arrows keys to move the agent 
      
      Press:
      - p to pick up an object
      - d to drop an object
      - t to transform an object"""
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)

########################################
# Define Environment Stages
########################################


def make_image_html(src):
  html = f"""
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="width: 400px; height: 400px; object-fit: contain;">
    </div>
    """
  return html


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
  rendered_img = stage.render_fn(timestep)
  new_obs_base64 = base64_npimage(rendered_img)
  stage_state = stage.get_user_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )
    ui.markdown("""
    You have 200 steps to figure out and complete the task. You can ask the AI for help.
    
    You control the red triangle. The direction the point is facing is "forward".
    """)
    # the direction the agent will move when you press the up arrow key. Right rotates the agent clockwise, left rotates the agent counter-clockwise.
    ui.html(make_image_html(src=new_obs_base64))
    ui.markdown("""
    actions: <br>
      - ArrowUp: move forward <br>
      - ArrowRight: rotate clockwise <br>
      - ArrowLeft: rotate counter-clockwise <br>
      - p: pick up an object <br>
      - d: drop an object <br>
      - t: transform an object
    """)


def evaluate_success_fn(timestep: TimeStep, params: Optional[object] = None):
  return timestep.last() and timestep.reward > 0



def make_environment_stages(
    name: str = 'small-1m',
    rng: jax.random.PRNGKey = jax.random.key(42)):
  # Create 5 different stages
  env_stages = []

  # create env + env_params. a bit redundant but OK
  env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
  jax_env = PlaygroundTimestepWrapper(env, autoreset=False, use_params=True)
  jax_web_env = JaxWebEnv(
    env=jax_env,
    actions=actions,
    render_fn=render_fn,
  )
  # NOTE: we need different compiled functions per setting, because the diffrent benchmarks use pytrees with different shapes
  benchmark = xminigrid.load_benchmark(name=name)
  rule = benchmark.sample_ruleset(rng)
  env_params = env_params.replace(
    ruleset=rule, # needed so that env compiles with right shapes
    max_steps=MAX_EPISODE_TIMESTEPS, view_size=11)
  jax_web_env.precompile(dummy_env_params=env_params)
  vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, env_params)

  for i in range(num_envs):
    # Each env has different ruleset
    rng, rng_ = jax.random.split(rng)
    rule = benchmark.sample_ruleset(rng_)
    env_params = env_params.replace(ruleset=rule)

    environment_stage = EnvStage(
      name=f"Environment {i + 1}",
      web_env=jax_web_env,
      action_keys=action_keys,
      action_to_name=action_to_name,
      env_params=env_params,
      render_fn=render_fn,
      vmap_render_fn=vmap_render_fn,
      display_fn=env_stage_display_fn,
      evaluate_success_fn=evaluate_success_fn,
      min_success=MIN_SUCCESS_EPISODES,
      max_episodes=MAX_STAGE_EPISODES,
      verbosity=VERBOSITY,
      msg_display_time=2,
      metadata=dict(
        desc=f"XLand environment {i + 1}",
        stage_number=i + 1,
      ),
    )
    env_stages.append(environment_stage)
  return env_stages

########################################
# Define Feedback Stage
########################################


async def feedback_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown("Please answer the following questions:")

    questions = [
      "How helpful was the AI?",
      "How human-like was the AI?",
    ]

    answers = {}
    completed_all = asyncio.Event()

    # record answer and see if finished
    def recoder_answer(question, answer):
      answers[question] = answer
      if all((i is not None for i in answers.values())):
        completed_all.set()

    # make handler factory for each question
    def make_handler(question):
      return lambda e: recoder_answer(question, e.value)

    # make radio buttons for each question
    for i, q in enumerate(questions):
      with ui.row():
        ui.label(q)
        ui.radio([1, 2, 3, 4, 5], on_change=make_handler(q)).props("inline")
        answers[q] = None
    await completed_all.wait()
    return answers


feedback_stage = FeedbackStage(
  name="Feedback",
  display_fn=feedback_display_fn,
)


########################################
# Define Experiment
########################################
def make_experiment(name: str, rng: jax.random.PRNGKey):
  env_stages = make_environment_stages(name=name, rng=rng)
  all_stages = [instruction_stage, *env_stages, feedback_stage]
  randomize = [False] + [True] * len(env_stages) + [False]
  experiment = nicewebrl.Experiment(
    blocks=all_stages,
    randomize=randomize
  )
  return experiment

experiment_set = nicewebrl.ExperimentSet(
  experiments=dict(
      small=make_experiment(name="small-1m", rng=jax.random.key(42)),
      high=make_experiment(name="high-1m", rng=jax.random.key(43))
  )
) 

