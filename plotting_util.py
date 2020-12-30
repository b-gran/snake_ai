from typing import Any, Deque
from training_util import average_reward


def get_plotter():
    plot_all_time = None
    from lrcurve import PlotLearningCurve
    plot_all_time = PlotLearningCurve(
        mappings={
            'loss': { 'facet': 'loss', 'line': 'train' },
            'reward': { 'facet': 'reward', 'line': 'train' },
            'score': { 'facet': 'score', 'line': 'train' },
            'prediction': { 'facet': 'prediction', 'line': 'train' },
        },
        facet_config={
            'loss': { 'name': 'avg loss', 'scale': 'linear', 'limit': [0, None] },
            'reward': { 'name': 'avg reward', 'scale': 'linear', 'limit': [-10.0, None] },
            'score': { 'name': 'avg score', 'scale': 'linear', 'limit': [-1.0, None] },
            'prediction': { 'name': 'avg prediction', 'scale': 'linear', 'limit': [-1.0, None] },
        },
        xaxis_config={ 'name': 'count', 'limit': [0, None] }
    )

    def plotter(action_count: int, solver: Any, scores: Deque):
        plot_all_time.append(action_count, {
            'loss': sum(solver.loss_buffer) / max(1, len(solver.loss_buffer)),
            'reward': average_reward(solver.memory, 5000),
            'score': sum(scores) / max(1, len(scores)),
            'prediction': sum(solver.prediction_buffer) / max(1, len(solver.prediction_buffer)),
        })

        plot_all_time.draw()

    return plotter


def get_pyplot_plotter():
    import numpy as np
    import matplotlib.pyplot as plt

    x = []
    loss = []
    reward = []
    score = []
    prediction = []

    def plotter(action_count: int, solver: Any, scores: Deque):
        x.append(action_count)
        loss.append(sum(solver.loss_buffer) / max(1, len(solver.loss_buffer)))
        reward.append(average_reward(solver.memory, 5000))
        score.append(sum(scores) / max(1, len(scores)))
        prediction.append(sum(solver.prediction_buffer) / max(1, len(solver.prediction_buffer)))

        plt.figure(figsize=(4, 16))

        plt.subplot(4, 1, 1)
        plt.title('loss')
        plt.plot(x, loss)

        plt.subplot(4, 1, 2)
        plt.title('reward')
        plt.plot(x, reward)

        plt.subplot(4, 1, 3)
        plt.title('score')
        plt.plot(x, score)

        plt.subplot(4, 1, 4)
        plt.title('prediction')
        plt.plot(x, prediction)

        plt.show()

    return plotter