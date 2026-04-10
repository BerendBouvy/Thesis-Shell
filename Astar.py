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

import math
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.widgets import Button, Slider

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
# Numba acceleration (optional — falls back to pure Python if not installed)
# ---------------------------------------------------------------------------
try:
    import numba
    from numba import njit
    from numba.typed import List as NbList
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Numpy move tables — compatible with both Numba njit and pure Python
_NB_MOVE_DR = np.array([ 0, -1, -1, -1,  0,  1,  1,  1], dtype=np.int64)
_NB_MOVE_DC = np.array([ 1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int64)
_NB_MOVE_DIST_ARR = np.array(
    [1.0, math.sqrt(2), 1.0, math.sqrt(2), 1.0, math.sqrt(2), 1.0, math.sqrt(2)],
    dtype=np.float64,
)

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _nb_push(hf, hg, hs, hn, f, g, s):
        """Push (f, g, s) onto a binary min-heap keyed on f."""
        n = hn[0]
        if n < len(hf):
            hf[n] = f
            hg[n] = g
            hs[n] = s
        else:
            hf.append(f)
            hg.append(g)
            hs.append(s)
        hn[0] = n + 1
        i = n
        while i > 0:
            p = (i - 1) >> 1
            if hf[p] > hf[i]:
                tf = hf[p]; tg = hg[p]; ts = hs[p]
                hf[p] = hf[i]; hg[p] = hg[i]; hs[p] = hs[i]
                hf[i] = tf;    hg[i] = tg;    hs[i] = ts
                i = p
            else:
                break

    @njit(cache=True)
    def _nb_pop(hf, hg, hs, hn):
        """Pop and return the (f, g, s) with minimum f."""
        f0 = hf[0]; g0 = hg[0]; s0 = hs[0]
        n = hn[0] - 1
        hn[0] = n
        hf[0] = hf[n]; hg[0] = hg[n]; hs[0] = hs[n]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            sm = i
            if l < n and hf[l] < hf[sm]:
                sm = l
            if r < n and hf[r] < hf[sm]:
                sm = r
            if sm != i:
                tf = hf[sm]; tg = hg[sm]; ts = hs[sm]
                hf[sm] = hf[i]; hg[sm] = hg[i]; hs[sm] = hs[i]
                hf[i] = tf;     hg[i] = tg;     hs[i] = ts
                i = sm
            else:
                break
        return f0, g0, s0

    @njit(cache=True)
    def _nb_astar(
        cost_grid, rows, cols,
        start_r, start_c,
        goal_r, goal_c, goal_h,
        init_headings,
        max_turn_steps, min_cost, hw,
        move_dr, move_dc, move_dist,
        hf, hg, hs, hn,
    ):
        """
        Core A* in nopython mode.
        State encoding:  sid = row * cols * 8 + col * 8 + heading
        goal_h = -1 → accept any arrival heading.
        Returns: came_from (int64 flat array), g_score (float64 flat array), goal_sid.
        First call triggers JIT compilation (~10-30 s); subsequent calls are fast.
        """
        NUM_H = np.int64(8)
        N = rows * cols * NUM_H

        g_score   = np.full(N, np.inf,       dtype=np.float64)
        came_from = np.full(N, np.int64(-1), dtype=np.int64)

        dr0 = np.float64(goal_r - start_r)
        dc0 = np.float64(goal_c - start_c)
        f0  = hw * math.sqrt(dr0 * dr0 + dc0 * dc0) * min_cost
        for h in init_headings:
            sid = np.int64(start_r * cols * NUM_H + start_c * NUM_H + h)
            g_score[sid] = 0.0
            _nb_push(hf, hg, hs, hn, f0, 0.0, sid)

        goal_sid = np.int64(-1)

        while hn[0] > 0:
            f, g, sid = _nb_pop(hf, hg, hs, hn)

            if g > g_score[sid]:
                continue

            row     = sid // (cols * NUM_H)
            col     = (sid // NUM_H) % cols
            heading = sid % NUM_H

            if row == goal_r and col == goal_c:
                if goal_h < 0 or heading == goal_h:
                    goal_sid = sid
                    break

            for next_h in range(NUM_H):
                diff = (heading - next_h) % NUM_H
                if diff > NUM_H - diff:
                    diff = NUM_H - diff
                if diff > max_turn_steps:
                    continue
                nr = row + move_dr[next_h]
                nc = col + move_dc[next_h]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                cell_cost = cost_grid[nr, nc]
                if cell_cost <= 0.0 or not np.isfinite(cell_cost):
                    continue
                tentative_g = g + move_dist[next_h] * cell_cost
                nsid = np.int64(nr * cols * NUM_H + nc * NUM_H + next_h)
                if tentative_g < g_score[nsid]:
                    g_score[nsid]   = tentative_g
                    came_from[nsid] = sid
                    drn = np.float64(goal_r - nr)
                    dcn = np.float64(goal_c - nc)
                    h_val = hw * math.sqrt(drn * drn + dcn * dcn) * min_cost
                    _nb_push(hf, hg, hs, hn, tentative_g + h_val, tentative_g, nsid)

        return came_from, g_score, goal_sid

    @njit(cache=True)
    def _nb_astar_record(
        cost_grid, rows, cols,
        start_r, start_c,
        goal_r, goal_c, goal_h,
        init_headings,
        max_turn_steps, min_cost, hw,
        move_dr, move_dc, move_dist,
        hf, hg, hs, hn,
        exp_sids, exp_gs,
    ):
        """Like _nb_astar but appends each expanded (sid, g) to exp_sids / exp_gs."""
        NUM_H = np.int64(8)
        N = rows * cols * NUM_H

        g_score   = np.full(N, np.inf,       dtype=np.float64)
        came_from = np.full(N, np.int64(-1), dtype=np.int64)

        dr0 = np.float64(goal_r - start_r)
        dc0 = np.float64(goal_c - start_c)
        f0  = hw * math.sqrt(dr0 * dr0 + dc0 * dc0) * min_cost
        for h in init_headings:
            sid = np.int64(start_r * cols * NUM_H + start_c * NUM_H + h)
            g_score[sid] = 0.0
            _nb_push(hf, hg, hs, hn, f0, 0.0, sid)

        goal_sid = np.int64(-1)

        while hn[0] > 0:
            f, g, sid = _nb_pop(hf, hg, hs, hn)

            if g > g_score[sid]:
                continue

            row     = sid // (cols * NUM_H)
            col     = (sid // NUM_H) % cols
            heading = sid % NUM_H

            exp_sids.append(sid)
            exp_gs.append(g)

            if row == goal_r and col == goal_c:
                if goal_h < 0 or heading == goal_h:
                    goal_sid = sid
                    break

            for next_h in range(NUM_H):
                diff = (heading - next_h) % NUM_H
                if diff > NUM_H - diff:
                    diff = NUM_H - diff
                if diff > max_turn_steps:
                    continue
                nr = row + move_dr[next_h]
                nc = col + move_dc[next_h]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                cell_cost = cost_grid[nr, nc]
                if cell_cost <= 0.0 or not np.isfinite(cell_cost):
                    continue
                tentative_g = g + move_dist[next_h] * cell_cost
                nsid = np.int64(nr * cols * NUM_H + nc * NUM_H + next_h)
                if tentative_g < g_score[nsid]:
                    g_score[nsid]   = tentative_g
                    came_from[nsid] = sid
                    drn = np.float64(goal_r - nr)
                    dcn = np.float64(goal_c - nc)
                    h_val = hw * math.sqrt(drn * drn + dcn * dcn) * min_cost
                    _nb_push(hf, hg, hs, hn, tentative_g + h_val, tentative_g, nsid)

        return came_from, g_score, goal_sid


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
# Exploration record (for animation)
# ---------------------------------------------------------------------------

class ExplorationRecord:
    """
    Records every node expansion during A* for post-hoc animation.

    Attributes
    ----------
    expansions : list of (row, col, heading, g)
        Every node popped from the heap, in order.
    came_from : dict
        Final came_from map from the search (used to reconstruct paths).
    cost_grid : np.ndarray
    start, goal : (row, col)
    final_result : PathResult or None
    """

    def __init__(
        self,
        expansions: List[Tuple[int, int, int, float]],
        came_from: Dict,
        cost_grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        final_result: Optional[PathResult],
    ) -> None:
        self.expansions = expansions
        self.came_from = came_from
        self.cost_grid = cost_grid
        self.start = start
        self.goal = goal
        self.final_result = final_result

        rows, cols = cost_grid.shape
        # For each (row, col), record the first expansion index that visited it.
        # Used for O(rows*cols) frame rendering instead of O(frame*step).
        self._first_visited = np.full((rows, cols), -1, dtype=int)
        for i, (r, c, _h, _g) in enumerate(expansions):
            if self._first_visited[r, c] == -1:
                self._first_visited[r, c] = i

    # ------------------------------------------------------------------

    def _path_to(self, key: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """Reconstruct (row, col) path from start to key via came_from."""
        path = []
        cur: Optional[Tuple] = key
        while cur is not None:
            path.append((cur[0], cur[1]))
            cur = self.came_from.get(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------

    def animate(
        self,
        step: int = 1,
        interval: int = 30,
        max_scatter_pts: int = 5_000,
        show_path_trace: Optional[bool] = None,
    ) -> mpl_animation.FuncAnimation:
        """
        Open an interactive matplotlib window to replay the A* search.

        The last frame always shows the final solution path (if one was found).

        Parameters
        ----------
        step : int
            Number of node expansions to advance per animation frame.
            Higher = faster playback (fewer frames total).
        interval : int
            Starting milliseconds between animation ticks.
        max_scatter_pts : int
            Maximum number of points drawn in the cost scatter plot.
            If the search expanded more nodes than this, the scatter is
            uniformly subsampled so each frame update stays fast.
            Default 5 000.  Set to 0 to disable the scatter entirely.
        show_path_trace : bool or None
            Whether to draw the path from start to the current expanding node
            each frame.  Tracing requires a came_from traversal per frame and
            can be slow for large grids.
            None (default) = enabled only when n_exp <= 200 000.
        """
        rows, cols = self.cost_grid.shape
        n_exp = len(self.expansions)
        # Exploration frames + 1 extra solution frame at the end
        n_search_frames = max(1, math.ceil(n_exp / step))
        SOLUTION_FRAME = n_search_frames
        total_frames = n_search_frames + 1

        if show_path_trace is None:
            show_path_trace = True

        # ---- scatter subsampling ------------------------------------------
        # Compute a fixed stride so we never plot more than max_scatter_pts.
        # scatter_idx[i] = original expansion index for the i-th scatter point.
        if max_scatter_pts > 0 and n_exp > max_scatter_pts:
            sc_stride = max(1, n_exp // max_scatter_pts)
            scatter_idx = np.arange(0, n_exp, sc_stride, dtype=int)
        else:
            scatter_idx = np.arange(n_exp, dtype=int)
        n_sc = len(scatter_idx)

        # ---- figure layout ------------------------------------------------
        fig = plt.figure(figsize=(14, 6.5))
        ax_map      = fig.add_axes([0.04, 0.22, 0.52, 0.72])
        ax_cost     = fig.add_axes([0.62, 0.54, 0.35, 0.40])
        ax_path     = fig.add_axes([0.62, 0.22, 0.35, 0.26])
        ax_frame_sl = fig.add_axes([0.12, 0.07, 0.76, 0.025])
        ax_btn      = fig.add_axes([0.44, 0.01, 0.12, 0.05])

        # ---- static background --------------------------------------------
        ax_map.imshow(self.cost_grid, cmap="gray", origin="lower", alpha=0.75)
        ax_map.plot(self.start[1], self.start[0], "go", ms=7, label="Start", zorder=5)
        ax_map.plot(self.goal[1],  self.goal[0],  "r*", ms=11, label="Goal",  zorder=5)
        ax_map.legend(loc="upper left", fontsize=8)
        ax_map.set_title("A* Exploration")

        # ---- dynamic layers -----------------------------------------------
        blank = np.zeros((rows, cols), dtype=np.float32)

        explored_im = ax_map.imshow(
            blank.copy(), cmap="Blues", origin="lower", alpha=0.55, vmin=0, vmax=1,
        )
        path_im = ax_map.imshow(
            blank.copy(), cmap="Reds", origin="lower", alpha=0.75, vmin=0, vmax=1,
        )
        solution_im = ax_map.imshow(
            blank.copy(), cmap="Greens", origin="lower", alpha=0.0, vmin=0, vmax=1,
        )
        (cur_dot,) = ax_map.plot([], [], "o", color="orange", ms=5, zorder=6)

        frame_text = ax_map.text(
            0.02, 0.97, "", transform=ax_map.transAxes, va="top",
            color="white", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
        )

        # ---- precompute scatter data (subsampled) -------------------------
        g_values = np.array([e[3] for e in self.expansions], dtype=np.float32)
        dist_from_start = np.array([
            math.hypot(e[0] - self.start[0], e[1] - self.start[1])
            for e in self.expansions
        ], dtype=np.float32)
        max_dist = float(dist_from_start.max()) if n_exp > 0 else 1.0

        sc_xs    = scatter_idx.astype(np.float32)
        sc_ys    = g_values[scatter_idx]
        sc_color = dist_from_start[scatter_idx]

        ax_cost.set_xlabel("Expansion #", fontsize=8)
        ax_cost.set_ylabel("g of expanded node", fontsize=8)
        ax_cost.set_title(
            f"g per expanded node (colour = dist from start)"
            + (f"\n[subsampled 1:{sc_stride}]" if n_exp > max_scatter_pts > 0 else ""),
            fontsize=9,
        )
        ax_cost.set_xlim(0, n_exp)
        ax_cost.set_ylim(0, float(g_values.max()) * 1.05 if n_exp > 0 else 1)
        ax_cost.tick_params(labelsize=7)

        if max_scatter_pts > 0:
            cost_scatter = ax_cost.scatter(
                [], [], c=[], cmap="coolwarm", vmin=0, vmax=max_dist,
                s=2, alpha=0.7, zorder=3,
            )
            fig.colorbar(cost_scatter, ax=ax_cost, label="dist from start", shrink=0.8, pad=0.02)
        else:
            cost_scatter = None

        (cur_pt,) = ax_cost.plot([], [], "ko", ms=5, zorder=6)

        if self.final_result is not None:
            ax_cost.axhline(
                self.final_result.total_cost, color="green",
                lw=1, ls="--", label=f"solution cost {self.final_result.total_cost:.2f}",
            )
            ax_cost.legend(fontsize=7)

        # ---- solution path cost plot (bottom-right) -----------------------
        ax_path.set_xlabel("Step along solution path", fontsize=8)
        ax_path.set_ylabel("Cumulative cost", fontsize=8)
        ax_path.set_title("Optimal path — cumulative cost", fontsize=9)
        ax_path.tick_params(labelsize=7)

        if self.final_result is not None:
            sol_keys: List[Tuple[int, int, int]] = [s.as_tuple() for s in self.final_result.states]
            g_lookup: Dict[Tuple[int, int, int], float] = {
                e[:3]: float(e[3]) for e in self.expansions
            }
            sol_gs = [g_lookup.get(k, 0.0) for k in sol_keys]
            ax_path.plot(range(len(sol_gs)), sol_gs, "g-", lw=1.5, label="optimal path")
            ax_path.set_xlim(0, len(sol_gs) - 1)
            ax_path.set_ylim(0, sol_gs[-1] * 1.05)
            ax_path.legend(fontsize=7)
        else:
            ax_path.text(0.5, 0.5, "No solution found", ha="center", va="center",
                         transform=ax_path.transAxes, fontsize=9, color="red")

        # ---- precompute solution path overlay -----------------------------
        solution_data = blank.copy()
        if self.final_result is not None:
            for r, c in self.final_result.coords:
                solution_data[r, c] = 1.0

        # Reusable path array to avoid alloc every frame
        path_data = blank.copy()

        # ---- slider & button ----------------------------------------------
        frame_slider = Slider(
            ax_frame_sl, "Frame", 0, total_frames - 1, valinit=0, valstep=1,
        )
        btn = Button(ax_btn, "Play")

        # ---- render -------------------------------------------------------
        def render(frame_idx: int) -> None:
            is_solution = frame_idx >= SOLUTION_FRAME
            exp_idx = n_exp - 1 if is_solution else min(frame_idx * step, n_exp - 1)

            # Explored overlay — fast boolean comparison on precomputed array
            explored = (self._first_visited >= 0) & (self._first_visited <= exp_idx)
            explored_im.set_data(explored)

            if is_solution:
                path_im.set_data(blank)
                cur_dot.set_data([], [])
                solution_im.set_data(solution_data)
                solution_im.set_alpha(0.8)
                label = "Solution"
                if self.final_result is not None:
                    label += f"  cost = {self.final_result.total_cost:.2f}"
            else:
                solution_im.set_alpha(0.0)

                if show_path_trace:
                    # Reuse path_data buffer — clear only previously set cells
                    path_data[:] = 0.0
                    cur_key = self.expansions[exp_idx][:3]
                    for r, c in self._path_to(cur_key):
                        path_data[r, c] = 1.0
                    path_im.set_data(path_data)
                else:
                    path_im.set_data(blank)

                r, c, _h, g = self.expansions[exp_idx]
                cur_dot.set_data([c], [r])
                label = f"step {exp_idx + 1}/{n_exp}  g = {g:.2f}"

            # Cost scatter — only update up to the current subsampled point
            if cost_scatter is not None:
                n_visible = int(np.searchsorted(scatter_idx, exp_idx, side="right"))
                if n_visible > 0:
                    cost_scatter.set_offsets(
                        np.column_stack([sc_xs[:n_visible], sc_ys[:n_visible]])
                    )
                    cost_scatter.set_array(sc_color[:n_visible])
            cur_pt.set_data([exp_idx], [float(g_values[exp_idx])])

            frame_text.set_text(label)
            fig.canvas.draw_idle()

        render(0)

        # ---- playback state -----------------------------------------------
        state = {"frame": 0, "playing": False}

        def tick(_i: int) -> None:
            if not state["playing"]:
                return
            nf = min(state["frame"] + 1, total_frames - 1)
            state["frame"] = nf
            frame_slider.set_val(nf)        # triggers on_frame_slider → render
            if nf >= total_frames - 1:
                state["playing"] = False
                btn.label.set_text("Play")
                fig.canvas.draw_idle()

        def on_frame_slider(val: float) -> None:
            state["frame"] = int(val)
            render(state["frame"])

        def on_btn(_event) -> None:
            if state["frame"] >= total_frames - 1:
                state["frame"] = 0          # restart from beginning
            state["playing"] = not state["playing"]
            btn.label.set_text("Pause" if state["playing"] else "Play")
            fig.canvas.draw_idle()

        frame_slider.on_changed(on_frame_slider)
        btn.on_clicked(on_btn)

        anim = mpl_animation.FuncAnimation(
            fig, tick, interval=interval, cache_frame_data=False,
        )
        plt.show()
        return anim


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
        Uses Numba JIT acceleration when available, otherwise pure Python.

        Parameters
        ----------
        start, goal : (row, col)
        start_heading : int or None
            Initial heading (0-7).  None = all headings tried simultaneously.
        goal_heading : int or None
            Required arrival heading.  None = any heading accepted at goal.

        Returns
        -------
        PathResult, or None if no path exists.
        """
        self._validate_cell("start", start)
        self._validate_cell("goal", goal)

        if _NUMBA_AVAILABLE:
            return self._solve_numba(start, goal, start_heading, goal_heading)
        return self._solve_python(start, goal, start_heading, goal_heading)

    def solve_with_recording(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        start_heading: Optional[int] = None,
        goal_heading: Optional[int] = None,
    ) -> Tuple[Optional[PathResult], "ExplorationRecord"]:
        """
        Like solve(), but also returns an ExplorationRecord for animation.
        Uses Numba JIT acceleration when available, otherwise pure Python.

        Returns
        -------
        (PathResult or None, ExplorationRecord)
        """
        self._validate_cell("start", start)
        self._validate_cell("goal", goal)

        if _NUMBA_AVAILABLE:
            return self._solve_numba_record(start, goal, start_heading, goal_heading)
        return self._solve_python_record(start, goal, start_heading, goal_heading)

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
    # Pure-Python implementations (fallback)
    # ------------------------------------------------------------------

    def _solve_python(self, start, goal, start_heading, goal_heading):
        import heapq as _hq
        g_score = {}
        came_from = {}
        heap = []
        init_headings = range(NUM_HEADINGS) if start_heading is None else [start_heading]
        for h in init_headings:
            key = (start[0], start[1], h)
            g_score[key] = 0.0
            came_from[key] = None
            _hq.heappush(heap, (self._heuristic(start[0], start[1], goal), 0.0, key))
        goal_key = None
        while heap:
            _, g, current = _hq.heappop(heap)
            if g > g_score.get(current, math.inf):
                continue
            row, col, heading = current
            if (row, col) == goal and (goal_heading is None or heading == goal_heading):
                goal_key = current
                break
            for next_key, move_cost in self._neighbors(row, col, heading):
                tg = g + move_cost
                if tg < g_score.get(next_key, math.inf):
                    g_score[next_key] = tg
                    came_from[next_key] = current
                    _hq.heappush(heap, (tg + self._heuristic(next_key[0], next_key[1], goal), tg, next_key))
        if goal_key is None:
            return None
        return PathResult(self._reconstruct(came_from, goal_key), g_score[goal_key], self.cost_grid)

    def _solve_python_record(self, start, goal, start_heading, goal_heading):
        import heapq as _hq
        g_score = {}
        came_from = {}
        heap = []
        expansions = []
        init_headings = range(NUM_HEADINGS) if start_heading is None else [start_heading]
        for h in init_headings:
            key = (start[0], start[1], h)
            g_score[key] = 0.0
            came_from[key] = None
            _hq.heappush(heap, (self._heuristic(start[0], start[1], goal), 0.0, key))
        goal_key = None
        while heap:
            _, g, current = _hq.heappop(heap)
            if g > g_score.get(current, math.inf):
                continue
            row, col, heading = current
            expansions.append((row, col, heading, g))
            if (row, col) == goal and (goal_heading is None or heading == goal_heading):
                goal_key = current
                break
            for next_key, move_cost in self._neighbors(row, col, heading):
                tg = g + move_cost
                if tg < g_score.get(next_key, math.inf):
                    g_score[next_key] = tg
                    came_from[next_key] = current
                    _hq.heappush(heap, (tg + self._heuristic(next_key[0], next_key[1], goal), tg, next_key))
        result = None
        if goal_key is not None:
            result = PathResult(self._reconstruct(came_from, goal_key), g_score[goal_key], self.cost_grid)
        return result, ExplorationRecord(expansions, came_from, self.cost_grid, start, goal, result)

    # ------------------------------------------------------------------
    # Numba implementations
    # ------------------------------------------------------------------

    def _nb_heap_arrays(self) -> tuple:
        """Allocate empty typed lists for the Numba heap."""
        hf = NbList.empty_list(numba.float64)
        hg = NbList.empty_list(numba.float64)
        hs = NbList.empty_list(numba.int64)
        hn = np.zeros(1, dtype=np.int64)
        return hf, hg, hs, hn

    def _nb_decode_sid(self, sid: int) -> Tuple[int, int, int]:
        """Decode a flat state id back to (row, col, heading)."""
        heading = int(sid % NUM_HEADINGS)
        col     = int((sid // NUM_HEADINGS) % self.cols)
        row     = int(sid // (self.cols * NUM_HEADINGS))
        return row, col, heading

    def _nb_reconstruct(self, came_from_arr: np.ndarray, goal_sid: int) -> List[State]:
        path = []
        sid = int(goal_sid)
        while sid >= 0:
            path.append(State(*self._nb_decode_sid(sid)))
            sid = int(came_from_arr[sid])
        path.reverse()
        return path

    def _solve_numba(self, start, goal, start_heading, goal_heading):
        init_h = np.arange(NUM_HEADINGS, dtype=np.int64) if start_heading is None \
                 else np.array([start_heading], dtype=np.int64)
        goal_h = np.int64(-1) if goal_heading is None else np.int64(goal_heading)
        hf, hg, hs, hn = self._nb_heap_arrays()
        came_from_arr, g_score_arr, goal_sid = _nb_astar(
            self.cost_grid, np.int64(self.rows), np.int64(self.cols),
            np.int64(start[0]), np.int64(start[1]),
            np.int64(goal[0]),  np.int64(goal[1]), goal_h,
            init_h,
            np.int64(self.max_turn_steps),
            np.float64(self._min_cost), np.float64(self.heuristic_weight),
            _NB_MOVE_DR, _NB_MOVE_DC, _NB_MOVE_DIST_ARR,
            hf, hg, hs, hn,
        )
        if goal_sid < 0:
            return None
        states = self._nb_reconstruct(came_from_arr, goal_sid)
        return PathResult(states, float(g_score_arr[goal_sid]), self.cost_grid)

    def _solve_numba_record(self, start, goal, start_heading, goal_heading):
        init_h = np.arange(NUM_HEADINGS, dtype=np.int64) if start_heading is None \
                 else np.array([start_heading], dtype=np.int64)
        goal_h = np.int64(-1) if goal_heading is None else np.int64(goal_heading)
        hf, hg, hs, hn = self._nb_heap_arrays()
        exp_sids: NbList = NbList.empty_list(numba.int64)
        exp_gs:   NbList = NbList.empty_list(numba.float64)
        came_from_arr, g_score_arr, goal_sid = _nb_astar_record(
            self.cost_grid, np.int64(self.rows), np.int64(self.cols),
            np.int64(start[0]), np.int64(start[1]),
            np.int64(goal[0]),  np.int64(goal[1]), goal_h,
            init_h,
            np.int64(self.max_turn_steps),
            np.float64(self._min_cost), np.float64(self.heuristic_weight),
            _NB_MOVE_DR, _NB_MOVE_DC, _NB_MOVE_DIST_ARR,
            hf, hg, hs, hn,
            exp_sids, exp_gs,
        )
        # Decode expansions: (row, col, heading, g)
        expansions = []
        for sid, g in zip(exp_sids, exp_gs):
            expansions.append((*self._nb_decode_sid(int(sid)), float(g)))

        # Build came_from dict only for visited states (came_from_arr != -1 or start)
        came_from_dict = {}
        for sid, g in zip(exp_sids, exp_gs):
            key = self._nb_decode_sid(int(sid))
            parent_sid = int(came_from_arr[int(sid)])
            came_from_dict[key] = self._nb_decode_sid(parent_sid) if parent_sid >= 0 else None

        result = None
        if goal_sid >= 0:
            states = self._nb_reconstruct(came_from_arr, goal_sid)
            result = PathResult(states, float(g_score_arr[goal_sid]), self.cost_grid)

        record = ExplorationRecord(expansions, came_from_dict, self.cost_grid, start, goal, result)
        return result, record

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
    import time
    start_time = time.time()
    # Build a small cost raster (higher = more expensive, 0/inf = no-go)
    size = (30, 70)
    grid = np.ones(size, dtype=float)
    
    grid += np.random.rand(*size) * 5  # add some noise to costs

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
        heuristic_weight=2  # admissible → optimal path
    )
    
    # planner.plot_cost_grid(show=True)

    result, record = planner.solve_with_recording(
        start=(0, 0),
        goal=(size[0] - 1, size[1] - 1),
        start_heading=None,   # free departure heading
        goal_heading=None,    # free arrival heading
    )
    

    end_time = time.time()
    print(f"Planning took {end_time - start_time:.2f} seconds.")
    if result is None:
        print("No path found.")
    else:
        print(result)
        print(f"Nodes expanded: {len(record.expansions)}")

        # Animate the search — press Play to watch it unfold
        # step=5 advances 5 expansions per frame for faster playback
        record.animate(step=10, interval=1)
