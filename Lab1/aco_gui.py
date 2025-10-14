# aco_tk_animated.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import sys

# Use TkAgg backend
matplotlib.use("TkAgg")


# ---------- Tooltip helper ----------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if "insert" in dir(self.widget) else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=4, ipady=2)

    def hide(self, _=None):
        tw = self.tip_window
        if tw:
            tw.destroy()
        self.tip_window = None


# ---------- ACO Implementation ----------
class ACO:
    def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, Q=100.0):
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        self.points = None
        self.distances = None
        self.pheromones = None

        self.all_best_paths = []
        self.best_len_history = []
        self.best_iter_found = None

    def init_map(self, n_points, map_seed=None, x_range=(0, 10), y_range=(0, 10)):
        if map_seed is not None:
            np.random.seed(int(map_seed))
        else:
            np.random.seed(None)

        xs = np.random.uniform(x_range[0], x_range[1], n_points)
        ys = np.random.uniform(y_range[0], y_range[1], n_points)
        self.points = np.column_stack((xs, ys))
        self.distances = self._compute_distance_matrix(self.points)
        self.pheromones = np.ones((n_points, n_points))

        # reset history
        self.all_best_paths = []
        self.best_len_history = []
        self.best_iter_found = None
        np.random.seed(None)

    @staticmethod
    def _compute_distance_matrix(points):
        n = len(points)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                mat[i, j] = mat[j, i] = d
        return mat

    def _move_agent(self, visited, current):
        unvisited = np.where(~visited)[0]
        taus = self.pheromones[current, unvisited] ** self.alpha
        etas = 1.0 / (self.distances[current, unvisited] + 1e-12)
        probs = taus * (etas ** self.beta)
        s = probs.sum()
        if s <= 0 or np.isnan(s):
            return np.random.choice(unvisited)
        probs = probs / s
        return np.random.choice(unvisited, p=probs)

    def iteration(self, n_ants=10, start_point=None):
        n = len(self.points)
        paths = []
        lengths = []

        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            current = np.random.randint(n) if start_point is None else int(start_point)
            visited[current] = True
            path = [current]
            total = 0.0

            while not np.all(visited):
                nxt = self._move_agent(visited, current)
                total += self.distances[current, nxt]
                path.append(nxt)
                visited[nxt] = True
                current = nxt

            total += self.distances[path[-1], path[0]]  # close tour
            paths.append(path)
            lengths.append(total)

        # evaporate
        self.pheromones *= self.evaporation

        # deposit
        for path, L in zip(paths, lengths):
            deposit = self.Q / (L + 1e-12)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromones[a, b] += deposit
                self.pheromones[b, a] += deposit
            a, b = path[-1], path[0]
            self.pheromones[a, b] += deposit
            self.pheromones[b, a] += deposit

        # get iteration's best
        idx_best = int(np.argmin(lengths))
        iter_best_len = lengths[idx_best]
        iter_best_path = paths[idx_best]

        # determine global best
        global_best_len = self.best_len_history[-1] if self.best_len_history else np.inf
        if iter_best_len < global_best_len:
            # new global best
            self.all_best_paths.append(iter_best_path)
            self.best_len_history.append(iter_best_len)
            self.best_iter_found = len(self.best_len_history)  # 1-based
        else:
            # keep previous best
            if self.all_best_paths:
                self.all_best_paths.append(self.all_best_paths[-1])
            else:
                self.all_best_paths.append(iter_best_path)
            self.best_len_history.append(global_best_len)

        return {
            "paths": paths,
            "lengths": lengths,
            "iter_best_path": iter_best_path,
            "iter_best_len": iter_best_len,
            "global_best_len": self.best_len_history[-1],
            "best_iter_found": self.best_iter_found,
            "pheromones": self.pheromones.copy(),
        }


# ---------- helpers ----------
def wrap_list(values, max_chars=90):
    """Wrap a comma-separated list into multiple lines so it fits the info box."""
    s = ", ".join(values)
    lines, cur = [], ""
    for token in s.split(", "):
        add = (", " if cur else "") + token
        if len(cur) + len(add) > max_chars:
            lines.append(cur)
            cur = token
        else:
            cur += add
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def format_route(route, max_chars=70, sep=" "):
    """Return a wrapped 'a b c ... a' string (space-separated indices) for the cycle."""
    if not route:
        return ""
    seq = route + [route[0]]  # close tour
    tokens = [str(x) for x in seq]

    lines, cur = [], ""
    for t in tokens:
        add = (sep if cur else "") + t
        if len(cur) + len(add) > max_chars:
            lines.append(cur)
            cur = t
        else:
            cur += add
    if cur:
        lines.append(cur)
    return "\n".join(lines)

# ---------- Animation + Plotting ----------
def run_visualization(params):
    # unpack params
    n_points = params["n_points"]
    n_ants = params["n_ants"]
    n_iterations = params["n_iterations"]
    alpha = params["alpha"]
    beta = params["beta"]
    evap = params["evap"]
    Q = params["Q"]
    map_seed = params["map_seed"]
    start_point = params["start_point"]
    animate_live = params["animate_live"]
    pheromone_top_pct = params["pheromone_top_pct"]

    aco = ACO(alpha=alpha, beta=beta, evaporation=evap, Q=Q)
    aco.init_map(n_points, map_seed=map_seed)
    pts = aco.points
    n = n_points

    # precompute reference edges & distances
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_distances = {e: aco.distances[e] for e in edges}

    # create figure with GridSpec (bottom row for info + convergence; padding fixed)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 1.0],   # <- taller bottom row
        hspace=0.34,
        wspace=0.36
    )

    ax_live = fig.add_subplot(gs[0:2, 0:2])     # left big panel (live)
    ax_ref = fig.add_subplot(gs[0:2, 2:4])      # right big panel (reference)
    ax_info = fig.add_subplot(gs[2, 0:2])       # bottom-left info
    ax_conv = fig.add_subplot(gs[2, 2:4])       # bottom-right convergence

    # configure axes limits
    margin = 0.8
    x_min, x_max = np.min(pts[:, 0]) - margin, np.max(pts[:, 0]) + margin
    y_min, y_max = np.min(pts[:, 1]) - margin, np.max(pts[:, 1]) + margin
    for ax in (ax_live, ax_ref):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

    ax_live.set_title("Live pheromone + current iteration best path")
    ax_ref.set_title("Reference: all potential paths + distances")
    ax_conv.set_title("Convergence (global best distance per iteration)", fontsize=10)

    # reference graph (light gray)
    ref_lines = [[pts[i], pts[j]] for (i, j) in edges]
    lc_ref = LineCollection(ref_lines, colors=(0.7, 0.7, 0.7, 0.25), linewidths=0.7)
    ax_ref.add_collection(lc_ref)

    # ---- draw nodes: orange fill, black stroke, centered labels ----
    node_size = 150
    # live panel
    ax_live.scatter(pts[:, 0], pts[:, 1], s=node_size, c="#f5a623", edgecolor="black", linewidth=1.6, zorder=5)
    for idx, (x, y) in enumerate(pts):
        ax_live.text(x, y, str(idx), fontsize=9, color="black", ha="center", va="center", zorder=6)
    # reference panel
    ax_ref.scatter(pts[:, 0], pts[:, 1], s=node_size * 0.85, c="black", edgecolor="black", linewidth=1.2, zorder=5)
    for idx, (x, y) in enumerate(pts):
        ax_ref.text(x, y, str(idx), fontsize=8, color="white", ha="center", va="center", zorder=6)

    # distance labels on reference for small graphs
    if n_points <= 30:
        for (i, j), d in edge_distances.items():
            mid = (pts[i] + pts[j]) / 2.0
            ax_ref.text(mid[0], mid[1], f"{d:.1f}", fontsize=6, color="gray")

    # dynamic artists on live panel
    pheromone_collection = LineCollection([], linewidths=1.0, cmap=cm.Greens, alpha=0.7)
    ax_live.add_collection(pheromone_collection)
    ants_lines_collection = LineCollection([], colors=(0.2, 0.2, 1.0, 0.18), linewidths=1.0)
    ax_live.add_collection(ants_lines_collection)
    best_path_collection = LineCollection([], colors=(0, 0, 0, 1.0), linewidths=3.0)
    ax_live.add_collection(best_path_collection)

    # bottom-left info panel
    ax_info.axis("off")
    info_box = ax_info.text(
        0.02,
        0.95,  # inside panel; leaves headroom
        "",
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.95, edgecolor="#c69c6d"),
        clip_on=True
    )

    # convergence plot
    conv_x = []
    conv_y = []
    conv_line, = ax_conv.plot([], [], marker="o", markersize=3)
    ax_conv.grid(True, alpha=0.35)
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best distance")

    # helper: build pheromone segments (top percentile)
    def build_pheromone_segments(pher, top_pct=pheromone_top_pct):
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(pher[i, j])
        vals = np.array(vals)
        if vals.size == 0:
            threshold = 0.0
        else:
            threshold = np.percentile(vals, top_pct)
        strengths = []
        segments = []
        for i in range(n):
            for j in range(i + 1, n):
                val = pher[i, j]
                if val >= threshold:
                    segments.append([pts[i], pts[j]])
                    strengths.append(val)
        strengths = np.array(strengths) if strengths else np.array([0.0])
        return segments, strengths

    # draw a best path on the live panel
    def draw_best_path(path):
        if not path:
            best_path_collection.set_segments([])
            return
        coords = [pts[idx] for idx in path] + [pts[path[0]]]
        best_path_collection.set_segments([coords])

    # non-animated (fast) mode
    if not animate_live:
        for it in range(n_iterations):
            aco.iteration(n_ants=n_ants, start_point=start_point)

        segs, strengths = build_pheromone_segments(aco.pheromones)
        if segs:
            pheromone_collection.set_segments(segs)
            norm = (strengths - strengths.min()) / (strengths.ptp() + 1e-12)
            widths = 0.5 + 5.0 * norm
            pheromone_collection.set_linewidths(widths)
            cmap = matplotlib.colormaps.get_cmap("Blues")
            colors = cmap(0.08 + 0.92 * norm)
            pheromone_collection.set_color(colors)

        global_best = aco.all_best_paths[-1] if aco.all_best_paths else None
        draw_best_path(global_best)
        conv_x[:] = list(range(1, len(aco.best_len_history) + 1))
        conv_y[:] = list(aco.best_len_history)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()

        route_str = format_route(global_best or [], max_chars=60)
        info = (
            f"Finished (no animation)\n"
            f"Iterations: {n_iterations}\n"
            f"Global best distance: {aco.best_len_history[-1]:.4f}\n"
            f"Best found at iteration: {aco.best_iter_found}\n"
            f"Best route (indices): \n{route_str}"
        )
        info_box.set_text(info)

        title = f"ACO (seed={map_seed}, N={n_points}, ants={n_ants}"
        if start_point is not None:
            title += f", start={start_point}"
        title += ")"
        fig.suptitle(title, fontsize=14)
        plt.show()
        return aco

    # animated update function
    def update(frame):
        res = aco.iteration(n_ants=n_ants, start_point=start_point)
        pher = res["pheromones"]
        ants_paths = res["paths"]
        ants_lengths = res["lengths"]
        iter_best_len = res["iter_best_len"]
        global_best_len = res["global_best_len"]
        best_iter_found = res["best_iter_found"]

        segs, strengths = build_pheromone_segments(pher)
        if segs:
            pheromone_collection.set_segments(segs)
            norm = (strengths - strengths.min()) / (strengths.ptp() + 1e-12)
            widths = 0.5 + 5.0 * norm
            pheromone_collection.set_linewidths(widths)
            cmap = matplotlib.colormaps.get_cmap("Blues")
            colors = cmap(0.08 + 0.92 * norm)
            pheromone_collection.set_color(colors)
        else:
            pheromone_collection.set_segments([])

        ants_lines = []
        for p in ants_paths:
            coords = [pts[idx] for idx in p] + [pts[p[0]]]
            ants_lines.append(coords)
        ants_lines_collection.set_segments(ants_lines)

        global_best_path = aco.all_best_paths[-1] if aco.all_best_paths else []
        draw_best_path(global_best_path)

        sorted_lengths = sorted(ants_lengths)
        shown = [f"{v:.1f}" for v in (sorted_lengths[:12] if len(sorted_lengths) > 12 else sorted_lengths)]
        route_str = format_route(global_best_path, max_chars=60)
        wrapped_distances = wrap_list(shown, max_chars=90)

        info_text = (
            f"Iter: {frame + 1}/{n_iterations}\n"
            f"Current iter best: {iter_best_len:.4f}\n"
            f"Global best: {global_best_len:.4f}\n"
            f"Best found at iter: {best_iter_found}\n"
            f"Best route (indices): \n{route_str}\n"
            f"Ant distances (smallest shown):\n{wrapped_distances}"
        )
        info_box.set_text(info_text)

        conv_x.append(frame + 1)
        conv_y.append(global_best_len)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()

        title = f"ACO animation (seed={map_seed}, N={n_points}, ants={n_ants}"
        if start_point is not None:
            title += f", start={start_point}"
        title += ")"
        fig.suptitle(title, fontsize=14)
        return pheromone_collection, ants_lines_collection, best_path_collection, info_box, conv_line

    anim = FuncAnimation(fig, update, frames=n_iterations, interval=200, blit=False, repeat=False)
    plt.show()
    return aco


# ---------- Tkinter Input Dialog ----------
class ParamDialog:
    def __init__(self, root):
        self.root = root
        self.root.title("ACO parameters")
        self.result = None

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # define vars with defaults
        self.vars = {
            "n_points": tk.StringVar(value="100"),
            "n_ants": tk.StringVar(value="100"),
            "n_iterations": tk.StringVar(value="200"),
            "alpha": tk.StringVar(value="1.0"),
            "beta": tk.StringVar(value="2.5"),
            "evaporation": tk.StringVar(value="0.6"),
            "Q": tk.StringVar(value="100.0"),
            "map_seed": tk.StringVar(value="77"),
            "start_point": tk.StringVar(value="50"),
            "animate_live": tk.BooleanVar(value=False),
            "pheromone_top_pct": tk.StringVar(value="92"),
        }

        # layout inputs with labels and tooltips
        rows = [
            ("n_points", "Nodes (5..100)"),
            ("n_ants", "Ants count"),
            ("n_iterations", "Iterations"),
            ("alpha", "Alpha (pheromone importance)"),
            ("beta", "Beta (distance importance)"),
            ("evaporation", "Evaporation rate (0..1)"),
            ("Q", "Q (pheromone deposit multiplier)"),
            ("map_seed", "Map seed (int, reproducible layout)"),
            ("start_point", "Start point (None or integer index)"),
            ("pheromone_top_pct", "Pheromone top % (to show top edges)"),
        ]
        for i, (k, label) in enumerate(rows):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4)
            ent = ttk.Entry(frm, textvariable=self.vars[k], width=18)
            ent.grid(row=i, column=1, sticky="w", pady=4)
            hint = {
                "n_points": "Number of nodes (5..100). Affects performance.",
                "n_ants": "Number of ants per iteration (more ants -> slower).",
                "n_iterations": "Number of ACO iterations to run.",
                "alpha": "Influence of pheromone (higher -> follow pheromone more).",
                "beta": "Influence of inverse distance (higher -> favor shorter edges).",
                "evaporation": "Pheromone evaporation factor per iteration (0..1).",
                "Q": "Amount of pheromone deposited; scales deposits.",
                "map_seed": "Integer seed for reproducible node placement.",
                "start_point": "If None -> random start per ant; else integer index 0..n_points-1.",
                "pheromone_top_pct": "Show only top-percent edges by pheromone to reduce clutter.",
            }[k]
            ToolTip(ent, hint)

        ttk.Checkbutton(frm, text="Show live animation", variable=self.vars["animate_live"]).grid(
            row=len(rows), column=0, columnspan=2, sticky="w", pady=6
        )
        ToolTip(frm, "When unchecked, the algorithm runs quickly and final results are shown.")

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=len(rows) + 1, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=6)

    def on_start(self):
        try:
            n_points = int(self.vars["n_points"].get())
            if not (5 <= n_points <= 100):
                raise ValueError("n_points must be between 5 and 100")
            n_ants = int(self.vars["n_ants"].get())
            n_iterations = int(self.vars["n_iterations"].get())
            alpha = float(self.vars["alpha"].get())
            beta = float(self.vars["beta"].get())
            evap = float(self.vars["evaporation"].get())
            Q = float(self.vars["Q"].get())
            map_seed = int(self.vars["map_seed"].get())
            sp_val = self.vars["start_point"].get().strip()
            if sp_val.lower() in ("none", ""):
                start_point = None
            else:
                start_point = int(sp_val)
                if not (0 <= start_point < n_points):
                    raise ValueError(f"start_point must be in [0, {n_points-1}] or None")
            animate_live = bool(self.vars["animate_live"].get())
            pheromone_top_pct = float(self.vars["pheromone_top_pct"].get())
            if not (1 <= pheromone_top_pct <= 100):
                raise ValueError("pheromone_top_pct must be between 1 and 100")

            self.result = {
                "n_points": n_points,
                "n_ants": n_ants,
                "n_iterations": n_iterations,
                "alpha": alpha,
                "beta": beta,
                "evap": evap,
                "Q": Q,
                "map_seed": map_seed,
                "start_point": start_point,
                "animate_live": animate_live,
                "pheromone_top_pct": pheromone_top_pct,
            }
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e))

    def on_cancel(self):
        self.result = None
        self.root.destroy()


# ---------- Main ----------
def main():
    root = tk.Tk()
    dialog = ParamDialog(root)
    root.mainloop()

    if not dialog.result:
        print("Cancelled.")
        sys.exit(0)

    params = dialog.result
    run_visualization(params)


if __name__ == "__main__":
    main()
