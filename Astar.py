"""
Heading-aware A* route planner on a 2D cost raster.

State space: (row, col, heading_index)
  - heading_index: 0-7, representing 8 compass directions (45° increments)
    0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE

Curvature constraint: from heading h, only transitions to headings within
  `max_turn_steps` (circular distance) are allowed per step.
  max_turn_steps=1 → max 45° turn per step
  max_turn_steps=2 → max 90° turn per step

No-go cells: cost <= 0 or cost == inf → impassable.
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (row_delta, col_delta) for each heading index 0-7
MOVE_DELTAS: List[Tuple[int, int]] = [
    (0, 1),    # 0: East
    (-1, 1),   # 1: North-East
    (-1, 0),   # 2: North
    (-1, -1),  # 3: North-West
    (0, -1),   # 4: West
    (1, -1),   # 5: South-West
    (1, 0),    # 6: South
    (1, 1),    # 7: South-East
]

NUM_HEADINGS = len(MOVE_DELTAS)

# Movement distance multiplier per direction (diagonal costs sqrt(2))
MOVE_DIST: List[float] = [
    1.0, math.sqrt(2), 1.0, math.sqrt(2),
    1.0, math.sqrt(2), 1.0, math.sqrt(2),
]

# Human-readable heading names for debugging
HEADING_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    """A position + heading in the grid."""

    __slots__ = ("row", "col", "heading")

    def __init__(self, row: int, col: int, heading: int) -> None:
        self.row = row
        self.col = col
        self.heading = heading  # 0-7

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.row, self.col, self.heading)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.row == other.row and self.col == other.col and self.heading == other.heading

    def __hash__(self) -> int:
        return hash((self.row, self.col, self.heading))

    def __repr__(self) -> str:
        return (
            f"State(row={self.row}, col={self.col}, "
            f"heading={self.heading}={HEADING_NAMES[self.heading]})"
        )


# ---------------------------------------------------------------------------
# Path result
# ---------------------------------------------------------------------------

class PathResult:
    """Container for a solved path."""

    def __init__(
        self,
        states: List[State],
        total_cost: float,
        cost_grid: np.ndarray,
    ) -> None:
        self.states = states                              # full (row, col, heading) path
        self.coords = [(s.row, s.col) for s in states]   # (row, col) only
        self.headings = [s.heading for s in states]
        self.total_cost = total_cost
        self._cost_grid = cost_grid
        
    def get_numpy_path(self) -> np.ndarray:
        """Return path as a numpy array of shape cost matrix rows x cols, with 1s on the path and 0s elsewhere."""
        path_array = np.zeros_like(self._cost_grid, dtype=int)
        for row, col in self.coords:
            path_array[row, col] = 1
        return path_array
    
    def plot_path(self, save_path: Optional[str] = None, show=False) -> None:
        """Plot the path on the cost grid."""
        array = self.get_numpy_path()
        plt.imshow(self._cost_grid, cmap='gray', origin='lower')
        plt.imshow(array, cmap='Reds', alpha=0.6, origin='lower')
        plt.colorbar(label='Cost')
        plt.title(f"Path (total cost: {self.total_cost:.2f})")
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        
    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return (
            f"PathResult(length={len(self)}, total_cost={self.total_cost:.4f}, "
            f"start={self.states[0]}, goal={self.states[-1]})"
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class AStarPlanner:
    """
    Heading-aware A* planner on a 2D cost raster.

    Parameters
    ----------
    cost_grid : np.ndarray, shape (rows, cols)
        Traversal cost per cell. Higher = more expensive to cross.
        Cells with cost <= 0 or cost == inf are treated as impassable (no-go).
    max_turn_steps : int
        Maximum change in heading index per move step (circular distance).
        Each heading index step = 45°.  Default 2 → max 90° turn per step.
    heuristic_weight : float
        Weight applied to the heuristic (w=1 → standard A*, w>1 → faster but
        suboptimal weighted A*).  Default 1.0.
    """

    def __init__(
        self,
        cost_grid: np.ndarray,
        max_turn_steps: int = 2,
        heuristic_weight: float = 1.0,
    ) -> None:
        if cost_grid.ndim != 2:
            raise ValueError("cost_grid must be a 2-D array.")
        self.cost_grid = cost_grid.astype(float)
        self.rows, self.cols = cost_grid.shape
        self.max_turn_steps = max_turn_steps
        self.heuristic_weight = heuristic_weight

        # Minimum finite cell cost — used to keep the heuristic admissible
        finite_mask = np.isfinite(self.cost_grid) & (self.cost_grid > 0)
        self._min_cost: float = (
            float(self.cost_grid[finite_mask].min()) if finite_mask.any() else 1.0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        start_heading: Optional[int] = None,
        goal_heading: Optional[int] = None,
    ) -> Optional[PathResult]:
        """
        Find the lowest-cost path from start to goal.

        Parameters
        ----------
        start, goal : (row, col)
        start_heading : int or None
            Initial heading (0-7).  None = all headings tried simultaneously,
            i.e. the agent can depart in any direction.
        goal_heading : int or None
            Required arrival heading.  None = any heading accepted at goal.

        Returns
        -------
        PathResult, or None if no path exists.
        """
        self._validate_cell("start", start)
        self._validate_cell("goal", goal)

        g_score: Dict[Tuple[int, int, int], float] = {}
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}
        # heap entries: (f, g, state_tuple)
        heap: List[Tuple[float, float, Tuple[int, int, int]]] = []

        init_headings = range(NUM_HEADINGS) if start_heading is None else [start_heading]
        for h in init_headings:
            key = (start[0], start[1], h)
            g_score[key] = 0.0
            came_from[key] = None
            f = self._heuristic(start[0], start[1], goal)
            heapq.heappush(heap, (f, 0.0, key))

        goal_key: Optional[Tuple[int, int, int]] = None

        while heap:
            f, g, current = heapq.heappop(heap)

            # Discard stale entries
            if g > g_score.get(current, math.inf):
                continue

            row, col, heading = current

            # Goal check
            if (row, col) == goal:
                if goal_heading is None or heading == goal_heading:
                    goal_key = current
                    break

            for next_key, move_cost in self._neighbors(row, col, heading):
                tentative_g = g + move_cost
                if tentative_g < g_score.get(next_key, math.inf):
                    g_score[next_key] = tentative_g
                    came_from[next_key] = current
                    h_val = self._heuristic(next_key[0], next_key[1], goal)
                    heapq.heappush(heap, (tentative_g + h_val, tentative_g, next_key))

        if goal_key is None:
            return None

        states = self._reconstruct(came_from, goal_key)
        return PathResult(states, g_score[goal_key], self.cost_grid)
    
    def plot_cost_grid(self, save_path: Optional[str] = None, show=False) -> None:
        """Plot the cost grid."""
        plt.imshow(self.cost_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Cost')
        plt.title("Cost Grid")
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_passable(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        c = self.cost_grid[row, col]
        return c > 0 and np.isfinite(c)

    def _validate_cell(self, name: str, cell: Tuple[int, int]) -> None:
        if not self._is_passable(*cell):
            raise ValueError(
                f"{name} cell {cell} is out of bounds, a no-go cell (cost <= 0), "
                "or has infinite cost."
            )

    def _heuristic(self, row: int, col: int, goal: Tuple[int, int]) -> float:
        """
        Admissible heuristic: Euclidean distance scaled by minimum cell cost.
        Multiplied by heuristic_weight for weighted A*.
        """
        dist = math.hypot(goal[0] - row, goal[1] - col)
        return self.heuristic_weight * dist * self._min_cost

    def _heading_diff(self, h1: int, h2: int) -> int:
        diff = abs(h1 - h2) % NUM_HEADINGS
        return min(diff, NUM_HEADINGS - diff)

    def _neighbors(
        self, row: int, col: int, heading: int
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """Return (next_state_key, move_cost) for all valid transitions."""
        results = []
        for next_h in range(NUM_HEADINGS):
            if self._heading_diff(heading, next_h) > self.max_turn_steps:
                continue
            dr, dc = MOVE_DELTAS[next_h]
            nr, nc = row + dr, col + dc
            if not self._is_passable(nr, nc):
                continue
            cell_cost = self.cost_grid[nr, nc]
            move_cost = MOVE_DIST[next_h] * cell_cost
            results.append(((nr, nc, next_h), move_cost))
        return results

    def _reconstruct(
        self,
        came_from: Dict[Tuple, Optional[Tuple]],
        goal_key: Tuple[int, int, int],
    ) -> List[State]:
        path: List[State] = []
        key: Optional[Tuple[int, int, int]] = goal_key
        while key is not None:
            path.append(State(*key))
            key = came_from[key]
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Quick usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a small cost raster (higher = more expensive, 0/inf = no-go)
    size = 200
    grid = np.ones((size, size), dtype=float)
    
    grid += np.random.rand(size, size) * 5  # add some noise to costs

    # Vertical wall with a single gap at row 10
    grid[5:15, 10] = 0.0
    # grid[10, 10] = 1.0

    # High-cost region (e.g. shallow water / swamp)
    grid[0:8, 12:18] = 5.0

    # Hard no-go island
    grid[12:16, 3:7] = np.inf

    
    planner = AStarPlanner(
        cost_grid=grid,
        max_turn_steps=1,      # max 45° per step
        heuristic_weight=1.0,  # admissible → optimal path
    )
    
    planner.plot_cost_grid(show=True)

    result = planner.solve(
        start=(0, 0),
        goal=(size - 1, size - 1),
        start_heading=None,   # free departure heading
        goal_heading=None,    # free arrival heading
    )

    if result is None:
        print("No path found.")
    else:
        print(result)
        print("Path coords:", result.coords)
        print("Headings:   ", [HEADING_NAMES[h] for h in result.headings])
        
        # Optional: plot the path
        result.plot_path(show=True)
