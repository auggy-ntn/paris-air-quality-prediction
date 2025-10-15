# Plot the forecasted values
import plotly.graph_objects as go

import constants.constants as cst


def plot_predictions(train, forecast, target):
    """Plots the predictions for a model for a given target variable.

    Args:
        train (pd.DataFrame): Training data
        forecast (pd.DataFrame): Forecasted data
        target (str): Target variable name

    Returns:
        fig: plotly figure
    """
    fig = go.Figure()

    # Plot training data
    fig.add_trace(
        go.Scatter(
            x=train[cst.DATE],
            y=train[target],
            name="Training Data",
            line=dict(color=cst.COLOR_TRAIN),
        )
    )

    # Plot forecasted data
    fig.add_trace(
        go.Scatter(
            x=forecast[cst.DATE],
            y=forecast[target],
            name="Forecasted Data",
            line=dict(color=cst.COLOR_PREDICTION),
        )
    )

    fig.update_layout(
        title="Time Series Forecast",
        xaxis_title="Date",
        yaxis_title=target,
        legend_title="Legend",
        template="plotly_white",
    )

    return fig
