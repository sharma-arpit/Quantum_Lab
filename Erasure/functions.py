"""
These functions implement some building blocks for the data analysis of the quantum eraser experiment.
They show how to read file, fit functions and plot the interferograms.
It is up to the student to combine them and feed them the right data.
"""

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import scipy.special as sspec


def read_interferogram(file_name):
    """
    Read an interferogram file into a pandas dataframe.

    Params
    ------
    file_name: str
        Path of the file containing the interferogram as string.

    Returns
    -------
    dataframe: pandas.DataFrame
        dataframe containing the data of the interferogram.
        Relevant series are "Coincidence rate (Hz) - T&B" and "Stage position (um)".
    """
    file = open(file_name, mode="r", encoding='ISO-8859-1')
    content = file.read()
    file.close()

    content = content.replace("\u00b5", "u")
    content = content.replace(",", ".")
    content = content.replace(";", ",")

    dataframe = pd.read_csv(StringIO(content))
    return dataframe


def read_scan(file_name):
    return read_interferogram(file_name)


def test_sin(x, amplitude, period, phase, offset):
    return offset + amplitude * np.sin(2 * np.pi / period * x + phase)


def test_sincabs(x, amplitude, width, h_offset, v_offset):
    sinc = lambda x: sspec.sinc(x / np.pi)
    return v_offset + amplitude * np.abs(sinc(np.pi * (x - h_offset) / width))


def get_oscillation_parameters_from_dataframe(
    dataframe, subtract_accidental_coincidences=False
):
    """
    Get parameters of interferogram by fitting a sine wave.

    Params
    ------
    dataframe: pandas.DataFrame
        dataframe containing the data of the interferogram.
        Relevant series are "Coincidence rate (Hz) - T&B" and "Stage position (um)".

    Returns
    -------
    res: dict
        Python dictionary containing relevant parameters like visibility, period, etc.
    """
    dataframe["x"] = dataframe["Stage position (um)"]
    if subtract_accidental_coincidences:
        t_coinc = 4e-9
        dataframe["Accidental coincidence rate (Hz) - T&B"] = (
            (dataframe["Counts - T"] / (dataframe["Capture duration (ms)"] / 1000))
            * (dataframe["Counts - B"] / (dataframe["Capture duration (ms)"] / 1000))
            * t_coinc
        )
        dataframe["y"] = np.maximum(
            0,
            dataframe["Coincidence rate (Hz) - T&B"]
            - dataframe["Accidental coincidence rate (Hz) - T&B"],
        )
        dataframe["sd_y"] = np.sqrt(
            dataframe["Coincidence rate (Hz) - T&B"]
            + t_coinc**2
            * (dataframe["Counts - T"] / (dataframe["Capture duration (ms)"] / 1000))
            * (dataframe["Counts - B"] / (dataframe["Capture duration (ms)"] / 1000))
            * (
                (dataframe["Counts - T"] / (dataframe["Capture duration (ms)"] / 1000))
                + (
                    dataframe["Counts - B"]
                    / (dataframe["Capture duration (ms)"] / 1000)
                )
            )
        )
    else:
        dataframe["y"] = dataframe["Coincidence rate (Hz) - T&B"]
        dataframe["sd_y"] = np.sqrt(dataframe["Coincidence rate (Hz) - T&B"])
    offset = dataframe["Coincidence rate (Hz) - T&B"].mean()
    amplitude = dataframe["Coincidence rate (Hz) - T&B"].max() - offset
    period = 0.404
    popt, pcov = sopt.curve_fit(
        test_sin,
        dataframe["x"],
        dataframe["y"],
        sigma=dataframe["sd_y"],
        absolute_sigma=True,
        p0=[amplitude, period, 0, offset],
    )
    perr = np.sqrt(np.diag(pcov))
    amplitude, period, phase, offset = popt
    sd_amplitude, sd_period, sd_phase, sd_offset = perr
    visibility = amplitude / offset
    sd_visibility = visibility * np.sqrt(
        sd_amplitude**2 / amplitude**2 + sd_offset**2 / offset**2
    )
    res = {
        "start_position": dataframe["x"].min(),
        "stop_position": dataframe["x"].max(),
        "center_position": dataframe["x"].mean(),
        "raw_maximum": dataframe["y"].max(),
        "raw_minimum": dataframe["y"].min(),
        "raw_visibility": (dataframe["y"].max() - dataframe["y"].min())
        / (dataframe["y"].max() + dataframe["y"].min()),
        "sd_raw_visibility": 2
        * np.sqrt(
            dataframe["y"].max() ** 2 * dataframe["y"].min()
            + dataframe["y"].min() ** 2 * dataframe["y"].max()
        )
        / (dataframe["y"].max() + dataframe["y"].min()) ** 2,
        "amplitude": amplitude,
        "period": period,
        "phase": phase,
        "offset": offset,
        "visibility": visibility,
        "sd_amplitude": sd_amplitude,
        "sd_period": sd_period,
        "sd_phase": sd_phase,
        "sd_offset": sd_offset,
        "sd_visibility": sd_visibility,
    }
    return res


def get_oscillation_parameters_from_interferogram(
    file_name, subtract_accidental_coincidences=False
):
    df = read_interferogram(file_name)
    res = get_oscillation_parameters_from_dataframe(
        df, subtract_accidental_coincidences
    )
    return res


def fit_and_plot_interferogram(file_name, label, title):
    """
    Fit and plot an interferogram.

    Params
    ------
    file_name: str
        Path of the file containing the interferogram as string.

    Returns
    -------
    fig, ax: matplotlib figure and axis
        matplotlib objects containing the plot.
    """
    dataframe = read_interferogram(file_name)
    res = get_oscillation_parameters_from_dataframe(dataframe)
    size = 1000
    xrange = np.linspace(
        dataframe["Stage position (um)"].min(),
        dataframe["Stage position (um)"].max(),
        size,
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        dataframe["Stage position (um)"],
        dataframe["Coincidence rate (Hz) - T&B"],
        np.sqrt(dataframe["Coincidence rate (Hz) - T&B"]),
        capsize=2,
        marker=".",
        linestyle="none",
        label="Data"
    )
    ax.plot(
        xrange,
        test_sin(xrange, res["amplitude"], res["period"], res["phase"], res["offset"]), label=label
    )
    ax.set_xlabel("Stage position (um)")
    ax.set_ylabel("Coincidence rate (Hz) - T&B")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def read_gra(file_name):
    """
    Read a file produced by the GRA tab of the software into a dictionary.

    Params
    ------
    file_name: str
        Path of the file containing the GRA data as string.

    Returns
    -------
    res: dict
        Python dictionary containing relevant parameters like coincidence rate.
    """
    file = open(file_name, mode="r")
    content = file.read()
    file.close()

    content = content.replace("\u00b5", "u")
    content = content.replace(",", ".")
    content = content.replace(";", ",")

    dataframe = pd.read_csv(StringIO(content), header=None, names=["Key", "Value"])
    row = dataframe.loc[dataframe["Key"] == "Rates (Hz): T"]
    singles_t = row["Value"].values[0]
    row = dataframe.loc[dataframe["Key"] == "Rates (Hz): B"]
    singles_b = row["Value"].values[0]
    row = dataframe.loc[dataframe["Key"] == "Rates (Hz): T&B"]
    coincidences = row["Value"].values[0]
    row = dataframe.loc[dataframe["Key"] == "Capture Duration (s)"]
    duration = row["Value"].values[0]
    res = {
        "singles_t": singles_t,
        "singles_b": singles_b,
        "coincidences": coincidences,
        "duration": duration,
    }
    return res


def analyze_gra(data, coincidence_window=4e-9):
    sd_singles_t = np.sqrt(data["singles_t"] / data["duration"])
    sd_singles_b = np.sqrt(data["singles_b"] / data["duration"])
    sd_coincidences = np.sqrt(data["coincidences"] / data["duration"])
    acc_coincidences = data["singles_t"] * data["singles_b"] * coincidence_window
    sd_acc_coincidences = np.sqrt(
        (data["singles_b"] * coincidence_window * sd_singles_t) ** 2
        + (data["singles_t"] * coincidence_window * sd_singles_b) ** 2
    )
    sub_coincidences = data["coincidences"] - acc_coincidences
    sd_sub_coincidences = np.sqrt(sd_coincidences**2 + sd_acc_coincidences**2)
    res = dict()
    res.update(data)
    res.update(
        {
            "sd_singles_t": sd_singles_t,
            "sd_singles_b": sd_singles_b,
            "sd_coincidences": sd_coincidences,
            "acc_coincidences": acc_coincidences,
            "sd_acc_coincidences": sd_acc_coincidences,
            "sub_coincidences": sub_coincidences,
            "sd_sub_coincidences": sd_sub_coincidences,
        }
    )
    return res


def get_rate_from_gra(
    file_name, subtract_accidental_coincidences=False, coincidence_window=4e-9
):
    res = analyze_gra(read_gra(file_name), coincidence_window)
    if subtract_accidental_coincidences:
        return res["sub_coincidences"], res["sd_sub_coincidences"]
    else:
        return res["coincidences"], res["sd_coincidences"]
