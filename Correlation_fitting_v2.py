import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
import csv

plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.size"] = 9

## Load csv file
file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")])
path = os.path.abspath(os.path.join(file_path, ".."))
df = pd.read_csv(file_path, sep=";")

## Read variables from csv file
n0 = np.array(df["1st step stagnation freezing fraction"].tolist())
mass_caught = np.array(df["Mass caught [kg]"].tolist())
mass_caught_per_area = mass_caught / 0.1
beta_0 = np.array(df["1st step stagnation beta"].tolist())
iced_cd = np.array(df["iced cd"].tolist())
clean_cd = np.array(df["clean cd"].tolist())
delta_cd = iced_cd - clean_cd
cd_ratio = iced_cd / clean_cd


## Function that generates the plots
def plotting(predicted, simulation, x_axis, y_axis, plot_range, n_stagnation):
    plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
    colors = n_stagnation
    plt.fill_between([0, 5], [0, 5 * 0.9], [0, 5 * 1.1], color="green", alpha=0.2, label="10%")
    plt.fill_between([0, 5], [0, 5 * 0.8], [0, 5 * 0.9], color="blue", alpha=0.2, label="20%")
    plt.fill_between([0, 5], [0, 5 * 1.1], [0, 5 * 1.2], color="blue", alpha=0.2)
    plt.fill_between([0, 5], [0, 5 * 0.7], [0, 5 * 0.8], color="red", alpha=0.2, label="30%")
    plt.fill_between([0, 5], [0, 5 * 1.2], [0, 5 * 1.3], color="red", alpha=0.2)
    plt.scatter(predicted, simulation, c=colors, cmap="coolwarm_r", edgecolors="black")
    plt.colorbar(label="Freezing fraction")
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.plot([0, 5], [0, 5], c="r", label="Perfect prediction")
    plt.legend()
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.tight_layout()


## Function that is responsible for the analysis of the correlation
def analysis(simulation, predicted, plotting_bool, n_stagnation):
    rmsre_value = np.sqrt(np.sum(((simulation - predicted) / simulation) ** 2, axis=0) / len(simulation))
    rmse_value = np.sqrt(np.mean((simulation - predicted) ** 2))
    within_10 = np.mean(np.abs(simulation - predicted) <= 0.1 * predicted) * 100
    within_20 = np.mean(np.abs(simulation - predicted) <= 0.2 * predicted) * 100
    within_30 = np.mean(np.abs(simulation - predicted) <= 0.3 * predicted) * 100
    outside_30 = np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    between_20_30 = np.sum(0.2 * predicted < np.abs(simulation - predicted)) - np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    max_offset = np.max(np.abs(simulation - predicted) / predicted) * 100
    print(f"RMSRE: {rmsre_value}")
    print(f"Within 10%: {within_10}%, within 20%: {within_20}%, within 30%: {within_30}%")
    print(f"Outside 30%: {outside_30}, maximal offset: {max_offset}%, between 20&30%: {between_20_30}\n")
    if plotting_bool:
        plotting(
            predicted=predicted,
            simulation=simulation,
            x_axis="Predicted drag coefficient increase",
            y_axis="Simulated drag coefficient increase",
            plot_range=[0.005, 0.018],
            n_stagnation=n_stagnation,
        )


## Correlations that are investigated
def one_part(variable_input, z21, z22, z23, z24, z25):
    mass, beta, n = variable_input
    result = z21 * mass**z22 * beta**z23 * n**z24 + z25
    return result


def one_part_without_beta(variable_input, z21, z22, z24, z25):
    mass, n = variable_input
    result = z21 * mass**z22 * n**z24 + z25
    return result


def one_part_without_n(variable_input, z21, z22, z24, z25):
    mass, beta = variable_input
    result = z21 * mass**z22 * beta**z24 + z25
    return result


def one_part_simple(variable_input, z21, z22, z25):
    mass, beta, n = variable_input
    result = z21 * mass**z22 + z25
    return result


def two_part(variable_input, glaze1, glaze2, glaze3, glaze4, glaze5, rime1, rime2, rime3, rime4, rime5):
    mass, beta, n, change = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * beta[mask1] ** glaze3 * n[mask1] ** glaze4 + glaze5

    mask2 = ~mask1
    result[mask2] = rime1 * mass[mask2] ** rime2 * beta[mask2] ** rime3 * n[mask2] ** rime4 + rime5
    return result


def two_part_without_beta(variable_input, glaze1, glaze2, glaze4, glaze5, rime1, rime2, rime4, rime5):
    mass, n, change = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * n[mask1] ** glaze4 + glaze5

    mask2 = ~mask1
    result[mask2] = rime1 * mass[mask2] ** rime2 * n[mask2] ** rime4 + rime5
    return result


def two_part_without_n(variable_input, glaze1, glaze2, glaze4, glaze5, rime1, rime2, rime4, rime5):
    mass, n, beta, change = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * beta[mask1] ** glaze4 + glaze5

    mask2 = ~mask1
    result[mask2] = rime1 * mass[mask2] ** rime2 * beta[mask2] ** rime4 + rime5
    return result


def two_part_without_beta_and_n(variable_input, glaze1, glaze2, glaze5, rime1, rime2, rime5):
    mass, n, change = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change
    result[mask1] = glaze1 * mass[mask1] ** glaze2 + glaze5

    mask2 = ~mask1
    result[mask2] = rime1 * mass[mask2] ** rime2 + rime5
    return result


def three_part(
    variable_input, rime1, rime2, rime3, rime4, rime5, glaze1, glaze2, glaze3, glaze4, glaze5, mixed1, mixed2, mixed3, mixed4, mixed5
):
    mass, beta, n, change1, change2 = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change1
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * beta[mask1] ** glaze3 * n[mask1] ** glaze4 + glaze5

    mask2 = n >= change2
    result[mask2] = rime1 * mass[mask2] ** rime2 * beta[mask2] ** rime3 * n[mask2] ** rime4 + rime5

    mask3 = ~(mask1 | mask2)
    result[mask3] = mixed1 * mass[mask3] ** mixed2 * beta[mask3] ** mixed3 * n[mask3] ** mixed4 + mixed5
    return result


def three_part_without_beta(variable_input, rime1, rime2, rime4, rime5, glaze1, glaze2, glaze4, glaze5, mixed1, mixed2, mixed4, mixed5):
    mass, n, change1, change2 = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change1
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * n[mask1] ** glaze4 + glaze5

    mask2 = n >= change2
    result[mask2] = rime1 * mass[mask2] ** rime2 * n[mask2] ** rime4 + rime5

    mask3 = ~(mask1 | mask2)
    result[mask3] = mixed1 * mass[mask3] ** mixed2 * n[mask3] ** mixed4 + mixed5
    return result


def three_part_without_n(variable_input, rime1, rime2, rime4, rime5, glaze1, glaze2, glaze4, glaze5, mixed1, mixed2, mixed4, mixed5):
    mass, n, beta, change1, change2 = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change1
    result[mask1] = glaze1 * mass[mask1] ** glaze2 * beta[mask1] ** glaze4 + glaze5

    mask2 = n >= change2
    result[mask2] = rime1 * mass[mask2] ** rime2 * beta[mask2] ** rime4 + rime5

    mask3 = ~(mask1 | mask2)
    result[mask3] = mixed1 * mass[mask3] ** mixed2 * beta[mask3] ** mixed4 + mixed5
    return result


def three_part_simple(variable_input, rime1, rime2, rime5, glaze1, glaze2, glaze5, mixed1, mixed2, mixed5):
    mass, n, change1, change2 = variable_input

    result = np.zeros_like(n)
    mask1 = n <= change1
    result[mask1] = glaze1 * mass[mask1] ** glaze2 + glaze5

    mask2 = n >= change2
    result[mask2] = rime1 * mass[mask2] ** rime2 + rime5

    mask3 = (n > change1) & (n < change2)
    result[mask3] = mixed1 * mass[mask3] ** mixed2 + mixed5
    return result


def masking(drag, variables_set, mask_set):
    vars_filtered = [x[mask_set] for x in variables_set]
    drag_filtered = drag[mask_set]
    vars_filtered_other_half = [x[~mask_set] for x in variables_set]
    drag_filtered_other_half = drag[~mask_set]
    return vars_filtered, drag_filtered, vars_filtered_other_half, drag_filtered_other_half


def error(model, truth):
    rmse = np.sqrt(np.mean((model - truth) ** 2))
    rmsre = np.sqrt(np.sum(((truth - model) / truth) ** 2, axis=0) / len(truth))
    return rmse, rmsre


def calculation(variable_set, cd_change, model, masking_set):
    vars_1st_half, delta_cd_1st_half, vars_2nd_half, delta_cd_2nd_half = masking(cd_change, variable_set, masking_set)
    [parameters, _] = curve_fit(f=model, xdata=vars_1st_half, ydata=delta_cd_1st_half, p0=initial_guess, maxfev=5000)
    prediction_2nd_half = model(vars_2nd_half, *parameters)
    full_prediction = model(variable_set, *parameters)
    rmse_2nd_half, rmsre_2nd_half = error(prediction_2nd_half, delta_cd_2nd_half)
    rmse_full, rmsre_full = error(full_prediction, delta_cd)
    return rmsre_full, rmsre_2nd_half, parameters, full_prediction


##Analysis of correlations
initial_guess = [0.02, 0.5, -1, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "one_part.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["rmse_entire"] + ["rmsre_test"] + ["constant"] + ["mass"] + ["beta"] + ["n"] + ["offset"])
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    variables = [mass_caught_per_area, beta_0, n0]
    rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, one_part, mask)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([rmsre_entire] + [rmsre_test] + params.tolist())
    if rmsre_test < rmsre_best:
        rmsre_best, prediction_best, params_best = rmsre_test, prediction_entire, params
print(f"One-part: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} * n^{params_best[3]} + {params_best[4]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "one_part_simple.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["rmse_entire"] + ["rmsre_test"] + ["constant"] + ["mass"] + ["offset"])
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    variables = [mass_caught_per_area, beta_0, n0]
    rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, one_part_simple, mask)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([rmsre_entire] + [rmsre_test] + params.tolist())
    if rmsre_test < rmsre_best:
        rmsre_best, prediction_best, params_best = rmsre_test, prediction_entire, params
print(f"One-part simple: {params_best[0]} * mass^{params_best[1]} + {params_best[2]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "one_part_without_beta.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["rmsre_entire"] + ["rmsre_test"] + ["constant"] + ["mass"] + ["n"] + ["offset"])
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    variables = [mass_caught_per_area, n0]
    rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, one_part_without_beta, mask)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([rmsre_entire] + [rmsre_test] + params.tolist())
    if rmsre_test < rmsre_best:
        rmsre_best, prediction_best, params_best = rmsre_test, prediction_entire, params
print(f"One-part without beta: {params_best[0]} * mass^{params_best[1]} * n^{params_best[2]} + {params_best[3]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "one_part_without_n.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["rmsre_entire"] + ["rmsre_test"] + ["constant"] + ["mass"] + ["beta"] + ["offset"])
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    variables = [mass_caught_per_area, beta_0]
    rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, one_part_without_n, mask)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([rmsre_entire] + [rmsre_test] + params.tolist())
    if rmsre_test < rmsre_best:
        rmsre_best, prediction_best, params_best = rmsre_test, prediction_entire, params
print(f"One-part without n: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} + {params_best[3]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -1, -0.2, 0.006, 0.02, 0.5, -1, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "two_part.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmsre_entire"]
        + ["rmsre_test"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_beta"]
        + ["glaze_n"]
        + ["glaze_offset"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_beta"]
        + ["rime_n"]
        + ["rime_offset"]
        + ["switch"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch in np.arange(0.2, 0.96, 0.01):
        switch = np.full_like(n0, round(switch, 2))
        variables = [mass_caught_per_area, beta_0, n0, switch]
        rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, two_part, mask)
        if rmsre_test < min_rmsre_test:
            min_rmsre_entire = rmsre_entire
            min_switch = switch[0]
            min_params = params
            min_rmsre_test = rmsre_test
            min_prediction = prediction_entire
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch_best = min_switch
print(f"Two-part switch: {switch_best}")
print(f"Two-part: Glaze: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} * n^{params_best[3]} + {params_best[4]}")
print(f"Two-part: Rime: {params_best[5]} * mass^{params_best[6]} * beta^{params_best[7]} * n^{params_best[8]} + {params_best[9]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "two_part_without_beta.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_n"]
        + ["glaze_offset"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_n"]
        + ["rime_offset"]
        + ["switch"]
    )
plt.figure()
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch in np.arange(0.2, 0.96, 0.01):
        switch = np.full_like(n0, round(switch, 2))
        variables = [mass_caught_per_area, n0, switch]
        try:
            rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, two_part_without_beta, mask)
            if rmsre_test < min_rmsre_test:
                min_rmsre_entire = rmsre_entire
                min_switch = switch[0]
                min_params = params
                min_rmsre_test = rmsre_test
                min_prediction = prediction_entire
        except:
            pass
    # plt.scatter(mass_caught_per_area, (mass_caught_per_area ** min_params[1]) * min_params[0])
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch_best = min_switch
plt.xlabel("Mass caught per meter span in kg/m")
plt.ylabel("Mass-dependent part of the correlation")
print(f"Two-part without beta: switch: {switch_best}")
print(f"Two-part without beta: Glaze: {params_best[0]} * mass^{params_best[1]} * n^{params_best[2]} + {params_best[3]}")
print(f"Two-part without beta: Rime: {params_best[4]} * mass^{params_best[5]} * n^{params_best[6]} + {params_best[7]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "two_part_without_n.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_beta"]
        + ["glaze_offset"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_beta"]
        + ["rime_offset"]
        + ["switch"]
    )
plt.figure()
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 8) + [False] * (len(delta_cd) - len(delta_cd) // 8))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch in np.arange(0.2, 0.96, 0.01):
        switch = np.full_like(n0, round(switch, 2))
        variables = [mass_caught_per_area, n0, beta_0, switch]
        try:
            rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, two_part_without_n, mask)
            if rmsre_test < min_rmsre_test:
                min_rmsre_entire = rmsre_entire
                min_switch = switch[0]
                min_params = params
                min_rmsre_test = rmsre_test
                min_prediction = prediction_entire
        except:
            pass
    # plt.scatter(mass_caught_per_area, (mass_caught_per_area ** min_params[1]) * min_params[0])
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch_best = min_switch
plt.xlabel("Mass caught per meter span in kg/m")
plt.ylabel("Mass-dependent part of the correlation")
print(f"Two-part without n: switch: {switch_best}")
print(f"Two-part without n: Glaze: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} + {params_best[3]}")
print(f"Two-part without n: Rime: {params_best[4]} * mass^{params_best[5]} * beta^{params_best[6]} + {params_best[7]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, 0.006, 0.02, 0.5, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "two_part_without_beta_and_n.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_offset"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_offset"]
        + ["switch"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch in np.arange(0.2, 0.96, 0.01):
        switch = np.full_like(n0, round(switch, 2))
        variables = [mass_caught_per_area, n0, switch]
        rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, two_part_without_beta_and_n, mask)
        if rmsre_test < min_rmsre_test:
            min_rmsre_entire = rmsre_entire
            min_switch = switch[0]
            min_params = params
            min_rmsre_test = rmsre_test
            min_prediction = prediction_entire
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch_best = min_switch
plt.xlabel("Mass caught per meter span in kg/m")
plt.ylabel("Mass-dependent part of the correlation")
print(f"Two-part without beta and n: switch: {switch_best}")
print(f"Two-part without beta and n: Glaze: {params_best[0]} * mass^{params_best[1]} + {params_best[2]}")
print(f"Two-part without beta and n: Rime: {params_best[3]} * mass^{params_best[4]} + {params_best[5]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -1, -0.2, 0.006, 0.02, 0.5, -1, -0.2, 0.006, 0.02, 0.5, -1, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "three_part.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_beta"]
        + ["rime_n"]
        + ["rime_offset"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_beta"]
        + ["glaze_n"]
        + ["glaze_offset"]
        + ["mixed_constant"]
        + ["mixed_mass"]
        + ["mixed_beta"]
        + ["mixed_n"]
        + ["mixed_offset"]
        + ["switch1"]
        + ["switch2"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch1 in np.arange(0.2, 0.8, 0.025):
        switch1 = np.full_like(n0, round(switch1, 2))
        for switch2 in np.arange(switch1[0] + 0.1, 0.96, 0.025):
            switch2 = np.full_like(n0, round(switch2, 2))
            variables = [mass_caught_per_area, beta_0, n0, switch1, switch2]
            try:
                rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, three_part, mask)
                if rmsre_test < min_rmsre_test:
                    min_rmsre_entire = rmsre_entire
                    min_switch1 = switch1[0]
                    min_switch2 = switch2[0]
                    min_params = params
                    min_rmsre_test = rmsre_test
                    min_prediction = prediction_entire
            except:
                pass
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch1] + [min_switch2])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch1_best = min_switch1
        switch2_best = min_switch2
print(f"Three-part switch1: {switch1_best}, switch2: {switch2_best}")
print(f"Three-part: Rime: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} * n^{params_best[3]} + {params_best[4]}")
print(f"Three-part: Glaze: {params_best[5]} * mass^{params_best[6]} * beta^{params_best[7]} * n^{params_best[8]} + {params_best[9]}")
print(f"Three-part: Mixed: {params_best[10]} * mass^{params_best[11]} * beta^{params_best[12]} * n^{params_best[13]} + {params_best[14]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "three_part_without_beta.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_n"]
        + ["rime_offset"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_n"]
        + ["glaze_offset"]
        + ["mixed_constant"]
        + ["mixed_mass"]
        + ["mixed_n"]
        + ["mixed_offset"]
        + ["switch1"]
        + ["switch2"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch1 in np.arange(0.2, 0.8, 0.025):
        switch1 = np.full_like(n0, round(switch1, 2))
        for switch2 in np.arange(switch1[0] + 0.1, 0.96, 0.025):
            switch2 = np.full_like(n0, round(switch2, 2))
            variables = [mass_caught_per_area, n0, switch1, switch2]
            try:
                rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, three_part_without_beta, mask)
                if rmsre_test < min_rmsre_test:
                    min_rmsre_entire = rmsre_entire
                    min_switch1 = switch1[0]
                    min_switch2 = switch2[0]
                    min_params = params
                    min_rmsre_test = rmsre_test
                    min_prediction = prediction_entire
            except:
                pass
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch1] + [min_switch2])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch1_best = min_switch1
        switch2_best = min_switch2
print(f"Three-part without beta: switch1: {switch1_best}, switch2: {switch2_best}")
print(f"Three-part without beta: Rime: {params_best[0]} * mass^{params_best[1]} * n^{params_best[2]}+ {params_best[3]}")
print(f"Three-part without beta: Glaze: {params_best[4]} * mass^{params_best[5]} * n^{params_best[6]} + {params_best[7]}")
print(f"Three-part without beta: Mixed: {params_best[8]} * mass^{params_best[9]}* n^{params_best[10]} + {params_best[11]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006, 0.02, 0.5, -0.2, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "three_part_without_n.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_beta"]
        + ["rime_offset"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_beta"]
        + ["glaze_offset"]
        + ["mixed_constant"]
        + ["mixed_mass"]
        + ["mixed_beta"]
        + ["mixed_offset"]
        + ["switch1"]
        + ["switch2"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch1 in np.arange(0.2, 0.8, 0.025):
        switch1 = np.full_like(n0, round(switch1, 2))
        for switch2 in np.arange(switch1[0] + 0.1, 0.96, 0.025):
            switch2 = np.full_like(n0, round(switch2, 2))
            variables = [mass_caught_per_area, n0, beta_0, switch1, switch2]
            try:
                rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, three_part_without_n, mask)
                if rmsre_test < min_rmsre_test:
                    min_rmsre_entire = rmsre_entire
                    min_switch1 = switch1[0]
                    min_switch2 = switch2[0]
                    min_params = params
                    min_rmsre_test = rmsre_test
                    min_prediction = prediction_entire
            except:
                pass
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch1] + [min_switch2])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch1_best = min_switch1
        switch2_best = min_switch2
print(f"Three-part without n: switch1: {switch1_best}, switch2: {switch2_best}")
print(f"Three-part without n: Rime: {params_best[0]} * mass^{params_best[1]} * beta^{params_best[2]} + {params_best[3]}")
print(f"Three-part without n: Glaze: {params_best[4]} * mass^{params_best[5]} * beta^{params_best[6]} + {params_best[7]}")
print(f"Three-part without n: Mixed: {params_best[8]} * mass^{params_best[9]} * beta^{params_best[10]} + {params_best[11]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

initial_guess = [0.02, 0.5, 0.006, 0.02, 0.5, 0.006, 0.02, 0.5, 0.006]
rmsre_best = np.inf
csv_path = os.path.join(path, "three_part_simple.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["rmse_entire"]
        + ["rmsre_test"]
        + ["rime_constant"]
        + ["rime_mass"]
        + ["rime_offset"]
        + ["glaze_constant"]
        + ["glaze_mass"]
        + ["glaze_offset"]
        + ["mixed_constant"]
        + ["mixed_mass"]
        + ["mixed_offset"]
        + ["switch1"]
        + ["switch2"]
    )
for i in range(25):
    mask = np.array([True] * (len(delta_cd) // 2) + [False] * (len(delta_cd) - len(delta_cd) // 2))
    np.random.shuffle(mask)
    min_switch, min_params, min_rmse, min_rmsre_test, min_prediction = [], [], [], float("inf"), []
    for switch1 in np.arange(0.2, 0.8, 0.025):
        switch1 = np.full_like(n0, round(switch1, 2))
        for switch2 in np.arange(switch1[0] + 0.1, 0.96, 0.025):
            switch2 = np.full_like(n0, round(switch2, 2))
            variables = [mass_caught_per_area, n0, switch1, switch2]
            try:
                rmsre_entire, rmsre_test, params, prediction_entire = calculation(variables, delta_cd, three_part_simple, mask)
                if rmsre_test < min_rmsre_test:
                    min_rmsre_entire = rmsre_entire
                    min_switch1 = switch1[0]
                    min_switch2 = switch2[0]
                    min_params = params
                    min_rmsre_test = rmsre_test
                    min_prediction = prediction_entire
            except:
                pass
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([min_rmsre_entire] + [min_rmsre_test] + min_params.tolist() + [min_switch1] + [min_switch2])
    if min_rmsre_test < rmsre_best:
        rmsre_best = min_rmsre_test
        prediction_best = min_prediction
        params_best = min_params
        switch1_best = min_switch1
        switch2_best = min_switch2
print(f"Three-part without beta and n: switch1: {switch1_best}, switch2: {switch2_best}")
print(f"Three-part without beta and n: Rime: {params_best[0]} * mass^{params_best[1]} + {params_best[2]}")
print(f"Three-part without beta and n: Glaze: {params_best[3]} * mass^{params_best[4]} + {params_best[5]}")
print(f"Three-part without beta and n: Mixed: {params_best[6]} * mass^{params_best[7]} + {params_best[8]}")
print(f"RMSRE test set: {rmsre_best}")
analysis(simulation=delta_cd, predicted=prediction_best, plotting_bool=True, n_stagnation=n0)

plt.show()
