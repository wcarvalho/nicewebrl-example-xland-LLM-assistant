# TODO: this is rendering mostly ported or adapted from the original Minigrid. A bit dirty right now...
import functools
import math

import jax
import jax.numpy as jnp

from xminigrid.core.constants import Colors, Tiles
from xminigrid.types import AgentState, IntOrArray


# JAX-compatible geometric functions
def jax_point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float):
  """JAX-compatible version of point_in_rect."""

  def fn(y, x):
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

  return fn


def jax_point_in_circle(cx: float, cy: float, r: float):
  """JAX-compatible version of point_in_circle."""

  def fn(y, x):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2

  return fn


def jax_point_in_triangle(p1, p2, p3):
  """JAX-compatible version of point_in_triangle."""
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3

  def fn(y, x):
    # Use barycentric coordinates to check if point is in triangle
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)

  return fn


def jax_rotate_fn(fn, cx: float, cy: float, theta: float):
  """JAX-compatible version of rotate_fn."""
  cos_theta = jnp.cos(theta)
  sin_theta = jnp.sin(theta)

  def rotated_fn(y, x):
    # Translate to origin
    x_centered = x - cx
    y_centered = y - cy

    # Rotate
    x_rotated = x_centered * cos_theta - y_centered * sin_theta
    y_rotated = x_centered * sin_theta + y_centered * cos_theta

    # Translate back
    x_final = x_rotated + cx
    y_final = y_rotated + cy

    return fn(y_final, x_final)

  return rotated_fn


# JAX-compatible rendering utilities
def jax_fill_coords(img: jnp.ndarray, mask_fn, color) -> jnp.ndarray:
  """JAX-compatible version of fill_coords that returns a new array."""
  color_array = jnp.array(color, dtype=img.dtype)
  if color_array.shape == ():
    # Single value, broadcast to RGB
    color_array = jnp.array([color_array, color_array, color_array])
  elif len(color_array.shape) == 1 and color_array.shape[0] == 3:
    # RGB color
    color_array = color_array
  else:
    # Expand to match image channels if needed
    color_array = jnp.broadcast_to(color_array, (3,))

  # Generate coordinate grid and apply mask function
  h, w, c = img.shape
  y_coords, x_coords = jnp.mgrid[0:h, 0:w]

  # Convert to normalized coordinates [0, 1]
  y_norm = y_coords / h
  x_norm = x_coords / w

  # Apply the mask function to get boolean mask
  mask = mask_fn(y_norm, x_norm)

  # Apply color where mask is True
  return jnp.where(mask[..., None], color_array, img)


def jax_downsample(img: jnp.ndarray, factor: int) -> jnp.ndarray:
  """JAX-compatible downsampling."""
  h, w, c = img.shape
  new_h, new_w = h // factor, w // factor
  # Reshape and mean over the downsampling blocks
  reshaped = img[: new_h * factor, : new_w * factor].reshape(
    new_h, factor, new_w, factor, c
  )
  return jnp.mean(reshaped, axis=(1, 3)).astype(img.dtype)


def jax_highlight_img(img: jnp.ndarray, alpha: float = 0.2) -> jnp.ndarray:
  """JAX-compatible image highlighting."""
  highlight_color = jnp.array([255, 255, 0], dtype=img.dtype)  # Yellow highlight
  return img * (1 - alpha) + highlight_color * alpha


# JAX-compatible detailed rendering functions
def render_wall_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a wall tile."""
  color_map = get_color_for_id(color)
  return jax_fill_coords(img, jax_point_in_rect(0, 1, 0, 1), color_map)


def render_floor_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a floor tile with grid lines."""
  color_map = get_color_for_id(color)
  # Draw grid lines (top and left edges)
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  # Draw tile floor
  img = jax_fill_coords(img, jax_point_in_rect(0.031, 1, 0.031, 1), color_map // 2)
  return img


def render_goal_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a goal tile."""
  color_map = get_color_for_id(color)
  # Draw grid lines (top and left edges)
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  # Draw tile
  img = jax_fill_coords(img, jax_point_in_rect(0.031, 1, 0.031, 1), color_map)
  return img


def render_ball_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a ball tile."""
  color_map = get_color_for_id(color)
  img = render_floor_jax(img, Colors.BLACK)
  img = jax_fill_coords(img, jax_point_in_circle(0.5, 0.5, 0.31), color_map)
  return img


def render_key_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a key tile."""
  color_map = get_color_for_id(color)
  img = render_floor_jax(img, Colors.BLACK)
  # Vertical quad
  img = jax_fill_coords(img, jax_point_in_rect(0.50, 0.63, 0.31, 0.88), color_map)
  # Teeth
  img = jax_fill_coords(img, jax_point_in_rect(0.38, 0.50, 0.59, 0.66), color_map)
  img = jax_fill_coords(img, jax_point_in_rect(0.38, 0.50, 0.81, 0.88), color_map)
  # Ring
  img = jax_fill_coords(img, jax_point_in_circle(cx=0.56, cy=0.28, r=0.190), color_map)
  img = jax_fill_coords(
    img,
    jax_point_in_circle(cx=0.56, cy=0.28, r=0.064),
    jnp.array([0, 0, 0], dtype=jnp.uint8),
  )
  return img


def render_pyramid_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a pyramid tile."""
  color_map = get_color_for_id(color)
  img = render_floor_jax(img, Colors.BLACK)
  tri_fn = jax_point_in_triangle((0.15, 0.8), (0.85, 0.8), (0.5, 0.2))
  img = jax_fill_coords(img, tri_fn, color_map)
  return img


def render_player_jax(img: jnp.ndarray, direction: int) -> jnp.ndarray:
  """Render a player/agent."""
  tri_fn = jax_point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
  # Rotate the agent based on its direction
  tri_fn = jax_rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (direction - 1))
  img = jax_fill_coords(img, tri_fn, COLORS_MAP[Colors.RED])
  return img


def get_color_for_id(color_id: int) -> jnp.ndarray:
  """Get color array for a given color ID using conditionals."""
  return jnp.where(
    color_id == Colors.RED,
    COLORS_MAP[Colors.RED],
    jnp.where(
      color_id == Colors.GREEN,
      COLORS_MAP[Colors.GREEN],
      jnp.where(
        color_id == Colors.BLUE,
        COLORS_MAP[Colors.BLUE],
        jnp.where(
          color_id == Colors.PURPLE,
          COLORS_MAP[Colors.PURPLE],
          jnp.where(
            color_id == Colors.YELLOW,
            COLORS_MAP[Colors.YELLOW],
            jnp.where(
              color_id == Colors.GREY,
              COLORS_MAP[Colors.GREY],
              jnp.where(
                color_id == Colors.BLACK,
                COLORS_MAP[Colors.BLACK],
                jnp.where(
                  color_id == Colors.ORANGE,
                  COLORS_MAP[Colors.ORANGE],
                  jnp.where(
                    color_id == Colors.BROWN,
                    COLORS_MAP[Colors.BROWN],
                    COLORS_MAP[Colors.WHITE],  # Default
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    ),
  )


COLORS_MAP = {
  Colors.EMPTY: jnp.array((255, 255, 255), dtype=jnp.uint8),  # just a placeholder
  Colors.RED: jnp.array((255, 0, 0), dtype=jnp.uint8),
  Colors.GREEN: jnp.array((0, 255, 0), dtype=jnp.uint8),
  Colors.BLUE: jnp.array((0, 0, 255), dtype=jnp.uint8),
  Colors.PURPLE: jnp.array((112, 39, 195), dtype=jnp.uint8),
  Colors.YELLOW: jnp.array((255, 255, 0), dtype=jnp.uint8),
  Colors.GREY: jnp.array((100, 100, 100), dtype=jnp.uint8),
  Colors.BLACK: jnp.array((0, 0, 0), dtype=jnp.uint8),
  Colors.ORANGE: jnp.array((255, 140, 0), dtype=jnp.uint8),
  Colors.WHITE: jnp.array((255, 255, 255), dtype=jnp.uint8),
  Colors.BROWN: jnp.array((160, 82, 45), dtype=jnp.uint8),
  Colors.PINK: jnp.array((225, 20, 147), dtype=jnp.uint8),
}


def _render_empty(img: jnp.ndarray, color: int):
  img = jax_fill_coords(
    img, point_in_rect(0.45, 0.55, 0.2, 0.65), COLORS_MAP[Colors.RED]
  )
  img = jax_fill_coords(
    img, point_in_rect(0.45, 0.55, 0.7, 0.85), COLORS_MAP[Colors.RED]
  )

  img = jax_fill_coords(img, point_in_rect(0, 0.031, 0, 1), COLORS_MAP[Colors.RED])
  img = jax_fill_coords(img, point_in_rect(0, 1, 0, 0.031), COLORS_MAP[Colors.RED])
  img = jax_fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), COLORS_MAP[Colors.RED])
  img = jax_fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), COLORS_MAP[Colors.RED])
  return img


def _render_floor(img: jnp.ndarray, color: int):
  # draw the grid lines (top and left edges)
  img = jax_fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
  img = jax_fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
  # draw tile
  img = jax_fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color] / 2)
  return img

  # # other grid lines (was used for paper visualizations)
  # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
  # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))
  #


def _render_wall(img: jnp.ndarray, color: int):
  img = jax_fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS_MAP[color])
  return img


def _render_ball(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  img = jax_fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS_MAP[color])
  return img


def _render_square(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  img = jax_fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS_MAP[color])
  return img


def _render_pyramid(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  tri_fn = point_in_triangle(
    (0.15, 0.8),
    (0.85, 0.8),
    (0.5, 0.2),
  )
  img = jax_fill_coords(img, tri_fn, COLORS_MAP[color])
  return img


def _render_hex(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  img = jax_fill_coords(img, point_in_hexagon(0.35), COLORS_MAP[color])
  return img


def _render_star(img: jnp.ndarray, color: int):
  # yes, this is a hexagram not a star, but who cares...
  img = _render_floor(img, Colors.BLACK)
  tri_fn2 = point_in_triangle(
    (0.15, 0.75),
    (0.85, 0.75),
    (0.5, 0.15),
  )
  tri_fn1 = point_in_triangle(
    (0.15, 0.3),
    (0.85, 0.3),
    (0.5, 0.9),
  )
  img = jax_fill_coords(img, tri_fn1, COLORS_MAP[color])
  img = jax_fill_coords(img, tri_fn2, COLORS_MAP[color])
  return img


def _render_goal(img: jnp.ndarray, color: int):
  # draw the grid lines (top and left edges)
  img = jax_fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
  img = jax_fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
  # draw tile
  img = jax_fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color])
  return img

  # # other grid lines (was used for paper visualizations)
  # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
  # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))


def _render_key(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  # Vertical quad
  img = jax_fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), COLORS_MAP[color])
  # Teeth
  img = jax_fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), COLORS_MAP[color])
  img = jax_fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), COLORS_MAP[color])
  # Ring
  img = jax_fill_coords(
    img, point_in_circle(cx=0.56, cy=0.28, r=0.190), COLORS_MAP[color]
  )
  img = jax_fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))
  return img


def _render_door_locked(img: jnp.ndarray, color: int):
  img = jax_fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
  img = jax_fill_coords(
    img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * COLORS_MAP[color]
  )
  # Draw key slot
  img = jax_fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), COLORS_MAP[color])
  return img


def _render_door_closed(img: jnp.ndarray, color: int):
  img = jax_fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
  img = jax_fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
  img = jax_fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), COLORS_MAP[color])
  img = jax_fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
  # Draw door handle
  img = jax_fill_coords(
    img, point_in_circle(cx=0.75, cy=0.50, r=0.08), COLORS_MAP[color]
  )
  return img


def _render_door_open(img: jnp.ndarray, color: int):
  img = _render_floor(img, Colors.BLACK)
  # draw the grid lines (top and left edges)
  img = jax_fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
  img = jax_fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
  # draw door
  img = jax_fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), COLORS_MAP[color])
  img = jax_fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
  return img


def _render_player(img: jnp.ndarray, direction: int):
  tri_fn = point_in_triangle(
    (0.12, 0.19),
    (0.87, 0.50),
    (0.12, 0.81),
  )
  # Rotate the agent based on its direction
  tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (direction - 1))
  img = jax_fill_coords(img, tri_fn, COLORS_MAP[Colors.RED])
  return img


TILES_FN_MAP = {
  Tiles.FLOOR: _render_floor,
  Tiles.WALL: _render_wall,
  Tiles.BALL: _render_ball,
  Tiles.SQUARE: _render_square,
  Tiles.PYRAMID: _render_pyramid,
  Tiles.HEX: _render_hex,
  Tiles.STAR: _render_star,
  Tiles.GOAL: _render_goal,
  Tiles.KEY: _render_key,
  Tiles.DOOR_LOCKED: _render_door_locked,
  Tiles.DOOR_CLOSED: _render_door_closed,
  Tiles.DOOR_OPEN: _render_door_open,
  Tiles.EMPTY: _render_empty,
}


# TODO: add highlight for can_see_through_walls=Fasle
def get_highlight_mask(
  grid: jnp.ndarray, agent: AgentState | None, view_size: int
) -> jnp.ndarray:
  mask = jnp.zeros(
    (grid.shape[0] + 2 * view_size, grid.shape[1] + 2 * view_size), dtype=jnp.bool_
  )
  return mask
  # if agent is None:
  #    return mask

  # agent_y, agent_x = agent.position + view_size
  # if agent.direction == 0:
  #    y, x = agent_y - view_size + 1, agent_x - (view_size // 2)
  # elif agent.direction == 1:
  #    y, x = agent_y - (view_size // 2), agent_x
  # elif agent.direction == 2:
  #    y, x = agent_y, agent_x - (view_size // 2)
  # elif agent.direction == 3:
  #    y, x = agent_y - (view_size // 2), agent_x - view_size + 1
  # else:
  #    raise RuntimeError("Unknown direction")

  # mask[y: y + view_size, x: x + view_size] = True
  # mask = mask[view_size:-view_size, view_size:-view_size]
  # assert mask.shape == (grid.shape[0], grid.shape[1])

  # return mask


def render_tile(
  tile: tuple,
  agent_direction: int | None = None,
  highlight: bool = False,
  tile_size: int = 32,
  subdivs: int = 3,
) -> jnp.ndarray:
  img = jnp.full(
    (tile_size * subdivs, tile_size * subdivs, 3), dtype=jnp.uint8, fill_value=255
  )
  # draw tile
  img = TILES_FN_MAP[tile[0]](img, tile[1])
  # draw agent if on this tile
  if agent_direction is not None:
    img = _render_player(img, agent_direction)

  if highlight:
    img = jax_highlight_img(img, alpha=0.2)

  # downsample the image to perform supersampling/anti-aliasing
  img = jax_downsample(img, subdivs)

  return img


def render(
  grid: jnp.ndarray,
  agent: AgentState | None = None,
  view_size: IntOrArray = 7,
  tile_size: IntOrArray = 32,
) -> jnp.ndarray:
  tile_size_static = int(tile_size)  # Convert to concrete value

  # compute the total grid size
  height_px = grid.shape[0] * tile_size_static
  width_px = grid.shape[1] * tile_size_static

  img = jnp.full((height_px, width_px, 3), dtype=jnp.uint8, fill_value=255)

  # compute agent fov highlighting
  highlight_mask = get_highlight_mask(grid, agent, int(view_size))

  def render_single_tile(img, tile_pos):
    y, x = tile_pos
    tile_data = grid[y, x]

    # Check if agent is on this tile
    agent_here = jnp.where(
      agent is not None, jnp.array_equal(jnp.array([y, x]), agent.position), False
    )

    agent_direction = jnp.where(
      agent_here, agent.direction if agent is not None else -1, -1
    )

    # Create tile with supersampling (subdivs=3 like original)
    subdivs = 3
    tile_size_subdivided = tile_size_static * subdivs
    tile_img = jnp.full(
      (tile_size_subdivided, tile_size_subdivided, 3), dtype=jnp.uint8, fill_value=255
    )

    # Render the tile based on its type - call the appropriate render function
    tile_img = jnp.where(
      tile_data[0] == Tiles.WALL,
      render_wall_jax(tile_img, tile_data[1]),
      jnp.where(
        tile_data[0] == Tiles.FLOOR,
        render_floor_jax(tile_img, tile_data[1]),
        jnp.where(
          tile_data[0] == Tiles.GOAL,
          render_goal_jax(tile_img, tile_data[1]),
          jnp.where(
            tile_data[0] == Tiles.BALL,
            render_ball_jax(tile_img, tile_data[1]),
            jnp.where(
              tile_data[0] == Tiles.KEY,
              render_key_jax(tile_img, tile_data[1]),
              jnp.where(
                tile_data[0] == Tiles.PYRAMID,
                render_pyramid_jax(tile_img, tile_data[1]),
                tile_img,  # Default case
              ),
            ),
          ),
        ),
      ),
    )

    # Add agent rendering if needed
    tile_img = jnp.where(
      agent_here, render_player_jax(tile_img, agent_direction), tile_img
    )

    # Downsample the tile (anti-aliasing)
    tile_img = jax_downsample(tile_img, subdivs)

    # Update the main image
    img = jax.lax.dynamic_update_slice(
      img, tile_img, (y * tile_size_static, x * tile_size_static, 0)
    )

    return img

  # Create grid of tile positions
  tile_positions = jnp.stack(jnp.mgrid[0 : grid.shape[0], 0 : grid.shape[1]], axis=-1)
  tile_positions = tile_positions.reshape(-1, 2)

  # Use scan to iterate over tiles
  img = jax.lax.fori_loop(
    0,
    len(tile_positions),
    lambda i, img: render_single_tile(img, tile_positions[i]),
    img,
  )

  return img
