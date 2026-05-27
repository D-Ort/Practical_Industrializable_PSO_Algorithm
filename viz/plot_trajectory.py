#----------------------------------------------------------------------------------
# Trajectory plot function for PSO results.
# @author: David Ortega Lozano
# @date: 2026-05-03
# @version: 0.2
# @description: Function for plotting the convergence of trajectory of PSO results.
#----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import plotly.graph_objects as go
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

def plot_trajectory_2D(data: dict, file_name: str) -> None:

    fig = go.Figure()

    methods = data["name"]

    all_frames = []

    # ==========================================
    # Build frames for every method + iteration
    # ==========================================

    for method_idx, method_name in enumerate(methods):

        particle_positions = data["particle_positions"][method_idx]
        best_positions = data["best_positions"][method_idx]

        num_particles = len(particle_positions)
        num_iterations = len(particle_positions[0])

        for iteration in range(num_iterations):

            particle_x = []
            particle_y = []

            for particle_id in range(num_particles):

                pos = particle_positions[particle_id][iteration]

                particle_x.append(pos[0])
                particle_y.append(pos[1])

            best_pos = best_positions[iteration]

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=particle_x,
                        y=particle_y,
                        mode="markers",
                        marker=dict(size=10),
                        name="Particles"
                    ),

                    go.Scatter(
                        x=[best_pos[0]],
                        y=[best_pos[1]],
                        mode="markers",
                        marker=dict(size=16, symbol="star"),
                        name="Global Best"
                    )
                ],

                name=f"{method_idx}_{iteration}"
            )

            all_frames.append(frame)

    fig.frames = all_frames

    # ==========================================
    # Initial frame
    # ==========================================

    initial_particles = data["particle_positions"][0]
    initial_best = data["best_positions"][0]

    x0 = [p[0][0] for p in initial_particles]
    y0 = [p[0][1] for p in initial_particles]

    best0 = initial_best[0]

    fig.add_trace(
        go.Scatter(
            x=x0,
            y=y0,
            mode="markers",
            marker=dict(size=10),
            name="Particles"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[best0[0]],
            y=[best0[1]],
            mode="markers",
            marker=dict(size=16, symbol="star"),
            name="Global Best"
        )
    )

    # ==========================================
    # Slider (iterations)
    # ==========================================

    num_iterations = len(initial_particles[0])

    steps = []

    for iteration in range(num_iterations):

        step = dict(
            method="animate",
            args=[
                [f"0_{iteration}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}
                }
            ],
            label=str(iteration)
        )

        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=steps
    )]

    # ==========================================
    # Dropdown (methods)
    # ==========================================

    dropdown_buttons = []

    for method_idx, method_name in enumerate(methods):

        button = dict(
            label=method_name,
            method="animate",
            args=[
                [f"{method_idx}_0"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}
                }
            ]
        )

        dropdown_buttons.append(button)

    # ==========================================
    # Layout
    # ==========================================

    fig.update_layout(

        title="PSO Particle Trajectories",

        xaxis_title="X",
        yaxis_title="Y",

        sliders=sliders,

        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.15
            )
        ],

        width=900,
        height=700
    )

    # ==========================================
    # Save HTML
    # ==========================================

    fig.write_html(file_name)