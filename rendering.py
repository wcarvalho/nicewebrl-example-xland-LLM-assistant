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
  return jax_fill_coords(img, jax_point_in_rect(0, 1, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8))


def render_floor_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a floor tile."""
  color_map = get_color_for_id(color)
  # Fill with black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  return img


def render_goal_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a goal tile."""
  color_map = get_color_for_id(color)
  # Fill with black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  # Draw tile (slightly inset from edges to leave room for grid lines)
  img = jax_fill_coords(img, jax_point_in_rect(0.031, 1, 0.031, 1), color_map)
  return img


def render_ball_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a ball tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  # Draw grid lines (top and left edges)
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  img = jax_fill_coords(img, jax_point_in_circle(0.5, 0.5, 0.31), color_map)
  return img


def render_key_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a key tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
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
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  tri_fn = jax_point_in_triangle((0.15, 0.8), (0.85, 0.8), (0.5, 0.2))
  img = jax_fill_coords(img, tri_fn, color_map)
  return img


def render_player_jax(img: jnp.ndarray, direction: int) -> jnp.ndarray:
  """Render a player/agent."""
  tri_fn = jax_point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
  # Rotate the agent based on its direction
  coeff = -0.5 * (direction - 1)
  # print("coeff", coeff)
  tri_fn = jax_rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=coeff * math.pi)
  img = jax_fill_coords(img, tri_fn, COLORS_MAP[Colors.RED])
  return img


def render_square_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a square tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  img = jax_fill_coords(img, jax_point_in_rect(0.25, 0.75, 0.25, 0.75), color_map)
  return img


def jax_point_in_hexagon(radius: float):
  """JAX-compatible hexagon check function."""

  def fn(y, x):
    # Convert to hexagon coordinates (center at 0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5

    # Regular hexagon defined by 6 half-planes
    # The hexagon vertices are at angles: 0°, 60°, 120°, 180°, 240°, 300°
    # For a regular hexagon, we check if point is inside all 6 half-planes

    # Convert to angle and distance
    angle = jnp.arctan2(dy, dx)
    dist = jnp.sqrt(dx**2 + dy**2)

    # For a regular hexagon, the distance from center to edge varies with angle
    # The formula is: r_edge = radius / cos(angle_mod_60)
    angle_mod_60 = jnp.abs((angle + jnp.pi / 6) % (jnp.pi / 3) - jnp.pi / 6)
    max_dist_at_angle = radius / jnp.cos(angle_mod_60)

    return dist <= max_dist_at_angle

  return fn


def render_hex_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a hexagon tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  img = jax_fill_coords(img, jax_point_in_hexagon(0.35), color_map)
  return img


def render_star_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a star (hexagram) tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  # Two triangles to form a hexagram
  tri_fn1 = jax_point_in_triangle((0.15, 0.3), (0.85, 0.3), (0.5, 0.9))
  tri_fn2 = jax_point_in_triangle((0.15, 0.75), (0.85, 0.75), (0.5, 0.15))
  img = jax_fill_coords(img, tri_fn1, color_map)
  img = jax_fill_coords(img, tri_fn2, color_map)
  return img


def render_door_locked_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a locked door tile."""
  color_map = get_color_for_id(color)
  img = jax_fill_coords(img, jax_point_in_rect(0.00, 1.00, 0.00, 1.00), color_map)
  img = jax_fill_coords(
    img, jax_point_in_rect(0.06, 0.94, 0.06, 0.94), color_map * 0.45
  )
  # Draw key slot
  img = jax_fill_coords(img, jax_point_in_rect(0.52, 0.75, 0.50, 0.56), color_map)
  return img


def render_door_closed_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render a closed door tile."""
  color_map = get_color_for_id(color)
  img = jax_fill_coords(img, jax_point_in_rect(0.00, 1.00, 0.00, 1.00), color_map)
  img = jax_fill_coords(
    img,
    jax_point_in_rect(0.04, 0.96, 0.04, 0.96),
    jnp.array([0, 0, 0], dtype=jnp.uint8),
  )
  img = jax_fill_coords(img, jax_point_in_rect(0.08, 0.92, 0.08, 0.92), color_map)
  img = jax_fill_coords(
    img,
    jax_point_in_rect(0.12, 0.88, 0.12, 0.88),
    jnp.array([0, 0, 0], dtype=jnp.uint8),
  )
  # Draw door handle
  img = jax_fill_coords(img, jax_point_in_circle(cx=0.75, cy=0.50, r=0.08), color_map)
  return img


def render_door_open_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render an open door tile."""
  color_map = get_color_for_id(color)
  # Use black background
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 1), jnp.array([0, 0, 0], dtype=jnp.uint8)
  )
  # Draw grid lines (top and left edges)
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100], dtype=jnp.uint8)
  )
  # Draw door
  img = jax_fill_coords(img, jax_point_in_rect(0.88, 1.00, 0.00, 1.00), color_map)
  img = jax_fill_coords(
    img,
    jax_point_in_rect(0.92, 0.96, 0.04, 0.96),
    jnp.array([0, 0, 0], dtype=jnp.uint8),
  )
  return img


def render_empty_jax(img: jnp.ndarray, color: int) -> jnp.ndarray:
  """Render an empty tile."""
  img = jax_fill_coords(
    img, jax_point_in_rect(0.45, 0.55, 0.2, 0.65), COLORS_MAP[Colors.RED]
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0.45, 0.55, 0.7, 0.85), COLORS_MAP[Colors.RED]
  )
  img = jax_fill_coords(img, jax_point_in_rect(0, 0.031, 0, 1), COLORS_MAP[Colors.RED])
  img = jax_fill_coords(img, jax_point_in_rect(0, 1, 0, 0.031), COLORS_MAP[Colors.RED])
  img = jax_fill_coords(
    img, jax_point_in_rect(1 - 0.031, 1, 0, 1), COLORS_MAP[Colors.RED]
  )
  img = jax_fill_coords(
    img, jax_point_in_rect(0, 1, 1 - 0.031, 1), COLORS_MAP[Colors.RED]
  )
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
                    jnp.where(
                      color_id == Colors.WHITE,
                      COLORS_MAP[Colors.WHITE],
                      jnp.where(
                        color_id == Colors.PINK,
                        COLORS_MAP[Colors.PINK],
                        jnp.where(
                          color_id == Colors.EMPTY,
                          COLORS_MAP[Colors.EMPTY],
                          COLORS_MAP[Colors.WHITE],  # Default
                        ),
                      ),
                    ),
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


# TODO: add highlight for can_see_through_walls=Fasle
def get_highlight_mask(
  grid: jnp.ndarray, agent: AgentState | None, view_size: int
) -> jnp.ndarray:
  mask = jnp.zeros(
    (grid.shape[0] + 2 * view_size, grid.shape[1] + 2 * view_size), dtype=jnp.bool_
  )
  return mask


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
    if agent is not None:
      agent_here = jnp.array_equal(jnp.array([y, x]), agent.position)
      agent_direction = jnp.where(agent_here, agent.direction, -1)
    else:
      agent_here = False
      agent_direction = -1

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
                jnp.where(
                  tile_data[0] == Tiles.SQUARE,
                  render_square_jax(tile_img, tile_data[1]),
                  jnp.where(
                    tile_data[0] == Tiles.HEX,
                    render_hex_jax(tile_img, tile_data[1]),
                    jnp.where(
                      tile_data[0] == Tiles.STAR,
                      render_star_jax(tile_img, tile_data[1]),
                      jnp.where(
                        tile_data[0] == Tiles.DOOR_LOCKED,
                        render_door_locked_jax(tile_img, tile_data[1]),
                        jnp.where(
                          tile_data[0] == Tiles.DOOR_CLOSED,
                          render_door_closed_jax(tile_img, tile_data[1]),
                          jnp.where(
                            tile_data[0] == Tiles.DOOR_OPEN,
                            render_door_open_jax(tile_img, tile_data[1]),
                            jnp.where(
                              tile_data[0] == Tiles.EMPTY,
                              render_empty_jax(tile_img, tile_data[1]),
                              tile_img,  # Default case
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
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

    # Add grid lines (top and left edges) for all tiles
    tile_img = jax_fill_coords(
      tile_img, jax_point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100], dtype=jnp.uint8)
    )
    tile_img = jax_fill_coords(
      tile_img, jax_point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100], dtype=jnp.uint8)
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


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from xminigrid.core.constants import Colors, Tiles

  # Define all tile types and their names
  tile_types = [
    (Tiles.EMPTY, "EMPTY"),
    (Tiles.FLOOR, "FLOOR"),
    (Tiles.WALL, "WALL"),
    (Tiles.BALL, "BALL"),
    (Tiles.SQUARE, "SQUARE"),
    (Tiles.PYRAMID, "PYRAMID"),
    (Tiles.GOAL, "GOAL"),
    (Tiles.KEY, "KEY"),
    (Tiles.DOOR_LOCKED, "DOOR_LOCKED"),
    (Tiles.DOOR_CLOSED, "DOOR_CLOSED"),
    (Tiles.DOOR_OPEN, "DOOR_OPEN"),
    (Tiles.HEX, "HEX"),
    (Tiles.STAR, "STAR"),
  ]

  # Colors to cycle through
  colors = [Colors.RED, Colors.GREEN, Colors.BLUE, Colors.YELLOW, Colors.PURPLE]

  # Calculate grid dimensions
  n_tiles = len(tile_types)
  cols = 3
  rows = (n_tiles + cols - 1) // cols  # Ceiling division

  width = 3
  fig, axes = plt.subplots(rows, cols, figsize=(width, width * rows))
  axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

  plot_idx = 0
  render = jax.jit(render)
  for i, (tile_type, tile_name) in enumerate(tile_types):
    if plot_idx >= len(axes):
      break

    # Cycle through colors for each tile type
    color = colors[i % len(colors)]
    color_name = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE"][colors.index(color)]

    # Create a single tile grid
    grid = jnp.array([[[tile_type, color]]], dtype=jnp.int32)
    # Render the tile
    img = render(grid)

    # Plot
    ax = axes[plot_idx]
    ax.imshow(img)
    ax.set_title(f"{tile_name}\n{color_name}")
    ax.axis("off")

    plot_idx += 1

  # Hide unused subplots
  for i in range(plot_idx, len(axes)):
    axes[i].axis("off")

  plt.tight_layout()
  plt.savefig("tile_rendering_test.png", dpi=150, bbox_inches="tight")
  plt.show()

  print(f"Generated visualization with {plot_idx} tile/color combinations")
