from qsprpred.plotting.classification import ROCPlot
from qsprpred.plotting.classification import MetricsPlot


def roc_plot(model, show=False, save=False):
    plot = ROCPlot([model])
    plot.make("cv", save=save, show=show)
    plot.make("ind", save=save, show=show)


def metrics_plot(model, show=False, save=False):
    plot = MetricsPlot([model])
    figs, summary = plot.make(show=show, save=save)
    return summary
