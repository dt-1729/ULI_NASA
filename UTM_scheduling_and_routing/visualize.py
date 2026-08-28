import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from matplotlib.cm import get_cmap
from matplotlib import cm  # for colormap
from tabulate import tabulate
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib as mpl
from MIRS import MIRS
import os
import pickle
from pathlib import Path
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotNetwork(
    figuresize,
    wp_xy,
    mask,
    dist_mat,
    sd_mat,
    routes,
    schedules,
    agent_colors,
    showEdgeLength=True,
    save_path=None,
    dpi=300,
    show_plot=True,
    show_segment_times=False,
    segment_time_decimals=1,
    segment_label_size=12,
    segment_label_offset=0.018,
):
    """
    Plot the waypoint network and directed agent traversals.

    For every directed edge traversal:

    - Agents are ordered by departure time.
    - The earliest-departing agent is shown farthest along the edge.
    - The latest-departing agent is shown closest to the source.
    - Each arrow is labeled with its departure and end time.
    - Labels are aligned with the corresponding arrow.
    - Labels remain upright while preserving the physical start/end order.

    Expected schedule convention
    ----------------------------
    For path edge path[k] -> path[k + 1]:

        schedule[k + 1] = departure time from path[k]
        schedule[k + 2] = arrival/end time at path[k + 1]
    """

    wp_xy = np.asarray(wp_xy)
    mask = np.asarray(mask)
    dist_mat = np.asarray(dist_mat)
    sd_mat = np.asarray(sd_mat)

    n_waypoints = wp_xy.shape[0]

    fig, ax = plt.subplots(figsize=figuresize)

    # Ensures that the data-coordinate direction of an edge is
    # represented correctly on screen.
    ax.set_aspect("equal", adjustable="box")

    # ============================================================
    # STEP 1: Plot waypoints
    # ============================================================
    for waypoint_id, (x, y) in enumerate(wp_xy):
        ax.scatter(
            x,
            y,
            color="skyblue",
            s=500,
            alpha=0.3,
            zorder=2,
        )

        ax.text(
            x,
            y,
            rf"{waypoint_id}",
            fontsize=16,
            color="grey",
            ha="center",
            va="center",
            zorder=3,
        )

    # ============================================================
    # STEP 2: Plot network edges in the background
    # ============================================================
    for i in range(n_waypoints):
        for j in range(i + 1, n_waypoints):
            if mask[i, j] != 1:
                continue

            x1, y1 = wp_xy[i]
            x2, y2 = wp_xy[j]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="skyblue",
                alpha=0.2,
                linewidth=6,
                zorder=1,
            )

            if showEdgeLength:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                ax.text(
                    mid_x,
                    mid_y,
                    rf"${dist_mat[i, j]:.1f}$",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                )

    # ============================================================
    # STEP 3: Collect directed edge traversals
    # ============================================================
    directed_edge_traversals = defaultdict(list)

    for agent_id, path in enumerate(routes):
        schedule = schedules[agent_id]

        # The final edge uses schedule[len(path)], so the schedule
        # must contain at least len(path) + 1 entries.
        minimum_schedule_length = len(path) + 1

        if len(schedule) < minimum_schedule_length:
            raise ValueError(
                f"Schedule for agent {agent_id} has length "
                f"{len(schedule)}, but route length {len(path)} "
                f"requires at least {minimum_schedule_length} entries."
            )

        for k in range(len(path) - 1):
            source = path[k]
            destination = path[k + 1]

            departure_time = schedule[k + 1]
            end_time = schedule[k + 2]

            directed_edge_traversals[(source, destination)].append(
                (agent_id, departure_time, end_time)
            )

    # ============================================================
    # STEP 4: Draw ordered directed arrows and time labels
    # ============================================================
    for (
        source,
        destination,
    ), traversals in directed_edge_traversals.items():

        source_x, source_y = wp_xy[source]
        destination_x, destination_y = wp_xy[destination]

        edge_dx = destination_x - source_x
        edge_dy = destination_y - source_y
        edge_length = np.hypot(edge_dx, edge_dy)

        if edge_length <= 1e-12:
            continue

        # Every directed traversal uses the half-edge extending from
        # its source waypoint toward the midpoint of the full edge.
        half_dx = edge_dx / 2
        half_dy = edge_dy / 2

        # Unit vector perpendicular to the directed edge.
        #
        # When the edge direction reverses, this vector also reverses.
        # Therefore, labels for opposite travel directions naturally
        # appear on opposite sides of the edge.
        perpendicular_x = -edge_dy / edge_length
        perpendicular_y = edge_dx / edge_length

        # Earliest departure first.
        traversals = sorted(
            traversals,
            key=lambda traversal: traversal[1],
        )

        n_traversals = len(traversals)

        if n_traversals == 0:
            continue

        segment_dx = half_dx / n_traversals
        segment_dy = half_dy / n_traversals

        # Reduce label size when many agents traverse the same
        # directed edge.
        adaptive_label_size = max(
            4,
            segment_label_size
            - 0.35 * max(0, n_traversals - 3),
        )

        for rank, (
            agent_id,
            departure_time,
            end_time,
        ) in enumerate(traversals):

            # The earliest agent has rank 0 and should be visually
            # farthest from the source waypoint.
            position_index = n_traversals - 1 - rank

            segment_start_x = (
                source_x + position_index * segment_dx
            )
            segment_start_y = (
                source_y + position_index * segment_dy
            )

            # Leave a small visual gap between neighboring arrows.
            arrow_fraction = 0.92

            arrow_dx = arrow_fraction * segment_dx
            arrow_dy = arrow_fraction * segment_dy

            ax.quiver(
                segment_start_x,
                segment_start_y,
                arrow_dx,
                arrow_dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=agent_colors[agent_id],
                width=0.004,
                headwidth=5,
                headlength=6,
                alpha=0.9,
                zorder=10 + position_index,
            )

            if not show_segment_times:
                continue

            # ----------------------------------------------------
            # Position the label at the center of the arrow
            # ----------------------------------------------------
            label_x = segment_start_x + 0.5 * arrow_dx
            label_y = segment_start_y + 0.5 * arrow_dy

            offset = segment_label_offset * edge_length

            label_x += offset * perpendicular_x
            label_y += offset * perpendicular_y

            # ----------------------------------------------------
            # Calculate the actual directed arrow angle
            # ----------------------------------------------------
            raw_angle = np.degrees(
                np.arctan2(arrow_dy, arrow_dx)
            )

            # Matplotlib text is easier to read when its rotation lies
            # between -90 and 90 degrees.
            #
            # If we rotate by 180 degrees to keep the text upright,
            # the readable text direction becomes opposite the actual
            # arrow direction. In that case, the time order and arrow
            # symbol must also be reversed.
            reverse_label = False

            if raw_angle > 90:
                text_angle = raw_angle - 180
                reverse_label = True

            elif raw_angle < -90:
                text_angle = raw_angle + 180
                reverse_label = True

            else:
                text_angle = raw_angle

            departure_string = (
                f"{departure_time:.{segment_time_decimals}f}"
            )
            end_string = (
                f"{end_time:.{segment_time_decimals}f}"
            )

            if reverse_label:
                # Reading direction is opposite the physical arrow:
                #
                # arrow head <- end time ... departure time <- tail
                time_label = (
                    end_string
                    + r"$\leftarrow$"
                    + departure_string
                )
            else:
                # Reading direction matches the physical arrow:
                #
                # tail -> departure time ... end time -> arrow head
                time_label = (
                    departure_string
                    + r"$\rightarrow$"
                    + end_string
                )

            ax.text(
                label_x,
                label_y,
                time_label,
                fontsize=adaptive_label_size,
                color="black",
                ha="center",
                va="center",
                rotation=text_angle,
                rotation_mode="anchor",

                # Critical: rotate text according to the transformed
                # data-coordinate direction of the arrow.
                transform_rotates_text=True,

                clip_on=True,
                zorder=30,
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

    # ============================================================
    # STEP 5: Plot start and destination labels
    # ============================================================
    start_groups = defaultdict(list)
    destination_groups = defaultdict(list)

    for agent_id, start_waypoint in enumerate(sd_mat[:, 0]):
        start_groups[int(start_waypoint)].append(agent_id)

    for agent_id, destination_waypoint in enumerate(sd_mat[:, 1]):
        destination_groups[int(destination_waypoint)].append(agent_id)

    for start_waypoint, agents in start_groups.items():
        x, y = wp_xy[start_waypoint]

        label = ", ".join(
            rf"$a_{{{agent_id}}}$"
            for agent_id in agents
        )

        ax.text(
            x,
            y,
            label,
            color="darkgreen",
            fontsize=16,
            ha="left",
            va="top",
            fontweight="bold",
            zorder=40,
        )

    for destination_waypoint, agents in destination_groups.items():
        x, y = wp_xy[destination_waypoint]

        label = ", ".join(
            rf"$a_{{{agent_id}}}$"
            for agent_id in agents
        )

        ax.text(
            x,
            y,
            label,
            color="red",
            fontsize=16,
            ha="left",
            va="bottom",
            fontweight="bold",
            zorder=40,
        )

    # ============================================================
    # STEP 6: Agent-color strip
    # ============================================================
    if routes is not None and len(routes) > 0:
        n_agents = len(routes)

        strip_ax = fig.add_axes(
            [0.1, 0.02, 0.8, 0.06]
        )

        strip_ax.set_xlim(0, n_agents)
        strip_ax.set_ylim(0, 1)

        for agent_id in range(n_agents):
            agent_color = agent_colors[agent_id]

            strip_ax.add_patch(
                plt.Rectangle(
                    (agent_id, 0),
                    1,
                    1,
                    color=agent_color,
                )
            )

            rgb = np.asarray(agent_color[:3])

            text_color = (
                "white"
                if np.mean(rgb) < 0.5
                else "black"
            )

            strip_ax.text(
                agent_id + 0.5,
                0.5,
                rf"$a_{{{agent_id}}}$",
                ha="center",
                va="center",
                fontsize=16,
                color=text_color,
            )

        strip_ax.axis("off")

        # Reserve space for the color strip.
        fig.subplots_adjust(bottom=0.1)

    # ============================================================
    # Save/show/close
    # ============================================================
    # Allow the plot to use the requested figure dimensions.
    # transform_rotates_text=True keeps time labels aligned
    # with the visually transformed arrows.
    ax.set_aspect("auto")

    # Add a small amount of padding around the network.
    ax.margins(x=0.05, y=0.05)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    plt.close(fig)


def plot_waypoint_agent_schedules(
    routes,
    schedules,
    schedule_matrix,
    association_matrix,
    process_T,
    tolArray,
    agent_colors,
    figuresize,

    # Timeline appearance
    bar_thickness=0.12,
    marker_size=200,
    index_size=480,

    # Axis text
    x_tick_size=50,
    y_tick_size=50,
    x_label_size=48,
    y_label_size=48 ,

    # Colorbar text
    colorbar_tick_size=14,
    colorbar_label_size=16,

    # Safety-color interpretation
    safe_gap_multiplier=2.0,

    # Output
    save_path=None,
    dpi=300,
    show_plot=True,
):
    """
    Plot waypoint arrival timelines and inter-agent safety gaps.

    Safety-color interpretation
    ---------------------------
    gap < tolerance
        Red-to-yellow region: unsafe.

    gap = tolerance
        Yellow: boundary of the safety constraint.

    tolerance < gap < safe_gap_multiplier * tolerance
        Yellow-to-green region.

    gap >= safe_gap_multiplier * tolerance
        Green: comfortably safe.

    Tick behavior
    -------------
    Grid lines are shown at every generated x-axis tick and every active
    waypoint row. Tick labels are displayed only when sufficient space is
    available, based on the figure size and requested font sizes.

    Notes
    -----
    The following arguments are retained for compatibility but are not used
    by this single-panel version:

        routes, schedules, process_T
    """

    schedule_matrix = np.asarray(schedule_matrix)
    association_matrix = np.asarray(association_matrix)
    tolArray = np.asarray(tolArray)

    if association_matrix.ndim != 2:
        raise ValueError("association_matrix must be a two-dimensional array.")

    if schedule_matrix.shape != association_matrix.shape:
        raise ValueError(
            "schedule_matrix and association_matrix must have the same shape. "
            f"Received {schedule_matrix.shape} and "
            f"{association_matrix.shape}."
        )

    n_agents, n_waypoints = association_matrix.shape

    if len(tolArray) < n_waypoints:
        raise ValueError(
            f"tolArray has length {len(tolArray)}, but the problem has "
            f"{n_waypoints} waypoints."
        )

    # ============================================================
    # Collect active waypoint information before plotting
    # ============================================================
    active_waypoint_data = []

    for waypoint_idx in range(n_waypoints):
        # Use a threshold instead of exact floating-point equality.
        agent_indices = np.flatnonzero(
            association_matrix[:, waypoint_idx] > 0.5
        )

        if len(agent_indices) == 0:
            continue

        arrival_times = schedule_matrix[
            agent_indices,
            waypoint_idx,
        ]

        # Sort agents by arrival time.
        order = np.argsort(arrival_times)

        sorted_agents = agent_indices[order]
        sorted_times = arrival_times[order]

        active_waypoint_data.append(
            {
                "waypoint": waypoint_idx,
                "agents": sorted_agents,
                "times": sorted_times,
                "tolerance": float(tolArray[waypoint_idx]),
            }
        )

    n_active_waypoints = len(active_waypoint_data)

    if n_active_waypoints == 0:
        raise ValueError(
            "No active waypoints were found in association_matrix."
        )

    # ============================================================
    # Create figure
    # ============================================================
    fig, ax = plt.subplots(figsize=figuresize)

    # Explicit red -> yellow -> green map.
    safety_cmap = LinearSegmentedColormap.from_list(
        "safety_cmap",
        [
            (0.0, "red"),
            (0.5, "gold"),
            (1.0, "green"),
        ],
    )

    safety_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # ============================================================
    # Adapt marker and marker-label sizes to the available height
    # ============================================================
    figure_height_points = float(figuresize[1]) * 72.0

    available_points_per_row = (
        figure_height_points / max(n_active_waypoints, 1)
    )

    effective_marker_size = min(
        marker_size,
        max(10.0, 0.55 * available_points_per_row),
    )

    effective_agent_font_size = min(
        index_size,
        max(5.0, 0.25 * available_points_per_row),
    )

    # ============================================================
    # Draw waypoint rows
    # ============================================================
    for row_idx, waypoint_data in enumerate(active_waypoint_data):
        waypoint_idx = waypoint_data["waypoint"]
        sorted_agents = waypoint_data["agents"]
        sorted_times = waypoint_data["times"]
        tolerance = waypoint_data["tolerance"]

        # --------------------------------------------------------
        # Draw safety-colored gaps
        # --------------------------------------------------------
        for i in range(len(sorted_times) - 1):
            time_1 = float(sorted_times[i])
            time_2 = float(sorted_times[i + 1])

            gap = time_2 - time_1

            if tolerance > 0:
                # gap = tolerance maps exactly to 0.5: yellow.
                safety_score = gap / (
                    safe_gap_multiplier * tolerance
                )
            else:
                safety_score = 1.0 if gap > 0 else 0.0

            safety_score = float(
                np.clip(safety_score, 0.0, 1.0)
            )

            segment_color = safety_cmap(safety_score)

            ax.fill_betweenx(
                [
                    row_idx - bar_thickness,
                    row_idx + bar_thickness,
                ],
                time_1,
                time_2,
                color=segment_color,
                alpha=0.50,
                zorder=2,
            )

        # --------------------------------------------------------
        # Draw agent markers and labels
        # --------------------------------------------------------
        for arrival_time, agent_id in zip(
            sorted_times,
            sorted_agents,
        ):
            agent_id = int(agent_id)
            agent_color = agent_colors[agent_id]

            ax.plot(
                arrival_time,
                row_idx,
                marker="s",
                linestyle="none",
                markersize=effective_marker_size,
                markerfacecolor=agent_color,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=5,
            )

            # Select contrasting text color.
            rgb = np.asarray(agent_color[:3], dtype=float)

            luminance = (
                0.2126 * rgb[0]
                + 0.7152 * rgb[1]
                + 0.0722 * rgb[2]
            )

            marker_text_color = (
                "black" if luminance > 0.55 else "white"
            )

            # Use only the numeric index inside the marker.
            # This is more compact than a_i for large scenarios.
            # ax.text(
            #     arrival_time,
            #     row_idx,
            #     str(agent_id),
            #     fontsize=effective_agent_font_size,
            #     color=marker_text_color,
            #     ha="center",
            #     va="center",
            #     fontweight="bold",
            #     zorder=6,
            #     clip_on=True,
            # )

    # ============================================================
    # Basic axis formatting
    # ============================================================
    ax.set_ylim(
        -0.5,
        n_active_waypoints - 0.5,
    )

    ax.set_xlabel(
        "Time (s)",
        fontsize=x_label_size,
    )

    ax.set_ylabel(
        r"$w_i$",
        fontsize=y_label_size,
    )

    ax.margins(x=0.03)

    # ============================================================
    # Colorbar in a separate axis
    # ============================================================
    scalar_mappable = mpl.cm.ScalarMappable(
        cmap=safety_cmap,
        norm=safety_norm,
    )
    scalar_mappable.set_array([])

    # This creates dedicated space to the right rather than placing
    # the colorbar inside the timeline axes.
    divider = make_axes_locatable(ax)

    colorbar_ax = divider.append_axes(
        "right",
        size="3.5%",
        pad=0.45,
    )

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
        orientation="vertical",
    )

    colorbar.set_ticks(
        [0.0, 0.5, 1.0]
    )

    colorbar.set_ticklabels(
        [
            "Unsafe",
            "Boundary",
            "Safe",
        ]
    )

    colorbar.ax.tick_params(
        labelsize=colorbar_tick_size,
    )

    colorbar.set_label(
        (
            "Inter-agent gap safety\n"
            rf"(yellow: gap = tolerance; "
            rf"green: gap $\geq$ "
            rf"{safe_gap_multiplier:g}$\times$tolerance)"
        ),
        fontsize=colorbar_label_size,
        labelpad=12,
    )

    for spine in colorbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.4)

    # Draw once so that the final axis dimensions are available.
    fig.canvas.draw()

    # ============================================================
    # Adaptive x-axis ticks
    # ============================================================
    axis_bbox_inches = (
        ax.get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
    )

    axis_width_inches = axis_bbox_inches.width
    axis_height_inches = axis_bbox_inches.height

    x_min, x_max = ax.get_xlim()

    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
        ax.set_xlim(x_min, x_max)

    # Number of x-grid lines depends on physical axis width.
    x_grid_bins = int(
        np.clip(
            axis_width_inches * 1.4,
            5,
            40,
        )
    )

    x_locator = MaxNLocator(
        nbins=x_grid_bins,
        min_n_ticks=4,
    )

    x_grid_ticks = np.asarray(
        x_locator.tick_values(x_min, x_max)
    )

    # Remove locator ticks lying outside the displayed range.
    x_grid_ticks = x_grid_ticks[
        (x_grid_ticks >= x_min)
        & (x_grid_ticks <= x_max)
    ]

    if len(x_grid_ticks) < 2:
        x_grid_ticks = np.linspace(
            x_min,
            x_max,
            5,
        )

    # Estimate the number of labels that fit without overlap.
    axis_width_points = axis_width_inches * 72.0

    max_visible_x_labels = max(
        2,
        int(
            axis_width_points
            / max(4.5 * x_tick_size, 1)
        ),
    )

    x_label_step = max(
        1,
        int(
            np.ceil(
                len(x_grid_ticks)
                / max_visible_x_labels
            )
        ),
    )

    x_span = abs(x_max - x_min)

    def format_x_value(value):
        if x_span >= 100:
            return f"{value:,.0f}"

        if x_span >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")

        return f"{value:.2f}".rstrip("0").rstrip(".")

    x_tick_labels = []

    for tick_index, tick_value in enumerate(x_grid_ticks):
        show_label = (
            tick_index % x_label_step == 0
            or tick_index == len(x_grid_ticks) - 1
        )

        if show_label:
            x_tick_labels.append(
                format_x_value(tick_value)
            )
        else:
            x_tick_labels.append("")

    ax.set_xticks(x_grid_ticks)

    ax.set_xticklabels(
        x_tick_labels,
        fontsize=x_tick_size,
    )

    # ============================================================
    # Adaptive y-axis labels
    # ============================================================
    # All active waypoint rows receive ticks and grid lines.
    y_grid_ticks = np.arange(n_active_waypoints)

    axis_height_points = axis_height_inches * 72.0

    max_visible_y_labels = max(
        2,
        int(
            axis_height_points
            / max(1.8 * y_tick_size, 1)
        ),
    )

    y_label_step = max(
        1,
        int(
            np.ceil(
                n_active_waypoints
                / max_visible_y_labels
            )
        ),
    )

    y_tick_labels = []

    for row_idx, waypoint_data in enumerate(active_waypoint_data):
        waypoint_idx = waypoint_data["waypoint"]

        show_label = (
            row_idx % y_label_step == 0
            or row_idx == n_active_waypoints - 1
        )

        if show_label:
            y_tick_labels.append(
                rf"${waypoint_idx}$"
            )
        else:
            y_tick_labels.append("")

    ax.set_yticks(y_grid_ticks)

    ax.set_yticklabels(
        y_tick_labels,
        fontsize=y_tick_size,
    )

    # ============================================================
    # Grid lines
    # ============================================================
    # Every generated x tick has a vertical grid line, even when
    # its numeric label is hidden.
    ax.grid(
        axis="x",
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
        zorder=0,
    )

    # Every active waypoint row has a horizontal grid line, even
    # when its waypoint number is hidden.
    ax.grid(
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # ============================================================
    # Save/show/close
    # ============================================================
    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    plt.close(fig)