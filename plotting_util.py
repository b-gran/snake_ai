from typing import Any
from training_util import average_reward


def get_plotter():
    plot_all_time = None
    from lrcurve import PlotLearningCurve
    plot_all_time = PlotLearningCurve(
        mappings={
            'loss': { 'facet': 'loss', 'line': 'train' },
            'reward': { 'facet': 'reward', 'line': 'train' },
        },
        facet_config={
            'loss': { 'name': 'avg loss', 'scale': 'linear', 'limit': [0, None]  },
            'reward': { 'name': 'avg reward', 'scale': 'linear', 'limit': [-10.0, None] }
        },
        xaxis_config={ 'name': 'count', 'limit': [0, None] }
    )

    def plotter(action_count: int, solver: Any):
        if action_count % 200 == 0:
            plot_all_time.append(action_count, {
                'loss': sum(solver.loss_buffer) / max(1, len(solver.loss_buffer)),
                'reward': average_reward(solver.memory, 100)
            })

            plot_all_time.draw()

    return plotter