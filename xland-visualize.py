import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
  import os
  import json
  import glob
  import config
  import re
  import jax.numpy as jnp
  import nicewebrl
  from typing import List
  from collections import defaultdict
  from flax import serialization
  import polars as pl
  from glob import glob

  import jax

  import config
  from experiment_structure import jax_web_env, env_params, describe_ruleset

  return describe_ruleset, jax


@app.cell
def _(describe_ruleset, jax):
  from xminigrid.experimental.img_obs import RGBImgObservationWrapper
  import matplotlib.pyplot as plt
  import xminigrid

  def create_env_with_ruleset(ruleset_key):
    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    benchmark = xminigrid.load_benchmark(name="trivial-1m")
    rule = benchmark.sample_ruleset(jax.random.key(ruleset_key))
    rule_text = describe_ruleset(rule)

    env_params = env_params.replace(
      ruleset=rule,
      max_steps=50,
      view_size=11,
    )
    env = RGBImgObservationWrapper(env)
    return env, benchmark, env_params, rule_text

  env, benchmark, env_params2, rule_text = create_env_with_ruleset(0)
  rng = jax.random.PRNGKey(0)
  example_timestep = env.reset(env_params2, rng)

  from render import render

  render = jax.jit(render)
  plt.imshow(render(example_timestep.state.grid, example_timestep.state.agent))
  plt.axis("off")
  plt.show()
  return


@app.cell
def _():
  return


if __name__ == "__main__":
  app.run()
