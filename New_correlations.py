from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib


def one_part(variables):
    mass_eq, n = variables
    result = 0.036363 * mass_eq**0.9868 * n**-0.44813 + 0.00620471
    return result


def two_part(variables):
    mass_eq, n = variables

    result = np.zeros_like(n)
    mask1 = n <= 0.58
    result[mask1] = 0.0779956 * mass_eq[mask1] ** 0.995121 * n[mask1] ** 0.0615398 + 0.006110

    mask2 = ~mask1
    result[mask2] = 0.0097733 * mass_eq[mask2] ** 0.49834 * n[mask2] ** -0.10122 + 0.005706431
    return result


def three_part(variables):
    mass_eq, n = variables

    result = np.zeros_like(n)
    mask1 = n <= 0.32
    result[mask1] = 0.02046 * mass_eq[mask1] ** 0.571231 * n[mask1] ** -0.104278 + 0.00528544

    mask2 = n >= 0.6
    result[mask2] = 0.0090994 * mass_eq[mask2] ** 0.456513 * n[mask2] ** -0.189551 + 0.00558426

    mask3 = ~(mask1 | mask2)
    result[mask3] = 0.0350924 * mass_eq[mask3] ** 0.91197 * n[mask3] ** -0.69441 + 0.00589743
    return result


def plotting(x, y, file_name, value, masking):
    plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
    colors = n_stagnation
    plt.fill_between([0, 5], [0, 5 * 0.9], [0, 5 * 1.1], color="green", alpha=0.2, label="10%")
    plt.fill_between([0, 5], [0, 5 * 0.8], [0, 5 * 0.9], color="blue", alpha=0.2, label="20%")
    plt.fill_between([0, 5], [0, 5 * 1.1], [0, 5 * 1.2], color="blue", alpha=0.2)
    plt.fill_between([0, 5], [0, 5 * 0.7], [0, 5 * 0.8], color="red", alpha=0.2, label="30%")
    plt.fill_between([0, 5], [0, 5 * 1.2], [0, 5 * 1.3], color="red", alpha=0.2)
    plt.scatter(x[~masking], y[~masking], c=colors[~masking], cmap="coolwarm_r", edgecolors="black")
    #plt.scatter(x[masking], y[masking], c=colors[masking], marker="*", cmap="coolwarm_r", edgecolors="black")
    plt.scatter(x[masking], y[masking], c=colors[masking], cmap="coolwarm_r", edgecolors="black")
    plt.plot([0, 10], [0, 10], color="red", label="Perfect agreement line")
    plt.colorbar(label="Freezing fraction")
    axes_min = np.minimum(x, y).min() * 0.9 if np.minimum(x, y).min() > 0 else np.minimum(x, y).min() * 1.5
    axes_max = np.maximum(x, y).max() * 1.1
    plt.xlim([axes_min, axes_max]), plt.ylim([axes_min, axes_max])
    if value == "delta_cd":
        plt.ylabel("Simulated drag coefficient increase")
        plt.xlabel("Predicted drag coefficient increase")
    elif value == "cd":
        plt.ylabel("Simulated drag coefficient")
        plt.xlabel("Predicted drag coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.savefig(f"{file_name[:-4]}.tiff")
    plt.close()


def analysis(simulation, predicted):
    rmse_value = np.sqrt(np.mean((simulation - predicted) ** 2))
    rmse_check = np.sqrt(np.sum((simulation - predicted) ** 2, axis=0) / len(simulation))
    rmsre = np.sqrt(np.sum(((simulation - predicted) / simulation) ** 2, axis=0) / len(simulation))
    within_10 = np.mean(np.abs(simulation - predicted) <= 0.1 * predicted) * 100
    within_20 = np.mean(np.abs(simulation - predicted) <= 0.2 * predicted) * 100
    within_30 = np.mean(np.abs(simulation - predicted) <= 0.3 * predicted) * 100
    outside_30 = np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    between_20_30 = np.sum(0.2 * predicted < np.abs(simulation - predicted)) - np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    max_offset = np.max(np.abs(simulation - predicted) / predicted) * 100
    print(f"RMSE: {rmse_value}, RMSE Check: {rmse_check}, RMSRE: {rmsre}")
    print(f"Within 10%: {within_10}%, within 20%: {within_20}%, within 30%: {within_30}%")
    print(f"Outside 30%: {outside_30}, maximal offset: {max_offset}%, between 20&30%: {between_20_30}\n")


plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.size"] = 9

file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")])
path = os.path.abspath(os.path.join(file_path, ".."))
with open(os.path.join(path, "output.txt"), "w") as f:
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        df = pd.read_csv(file_path, sep=";")

        mass = np.array(df["Mass caught [kg]"].tolist())
        n_stagnation = np.array(df["1st step stagnation freezing fraction"].tolist())
        chord = np.array(df["Chord [m]"].tolist())
        aoa = np.array(df["AOA [deg]"].tolist())
        iced_cd = np.array(df["iced cd"].tolist())
        clean_cd = np.array(df["clean cd"].tolist())
        temperature = np.array(df["Temperature [K]"].tolist())
        lwc = np.array(df["LWC [g/m3]"].tolist())
        mvd = np.array(df["MVD [um]"].tolist())

        valid_combinations = [(271.15, 0.76, 15), (263.15, 0.6, 15), (271.15, 2.469, 20), (267.15, 2.347, 20)]
        mask = np.zeros_like(temperature, dtype=bool)
        for t, l, m in valid_combinations:
            mask |= (temperature == t) & (lwc == l) & (mvd == m)

        delta_cd = iced_cd - clean_cd

        mass_span = mass / 0.1
        chord_adjusted_mass = mass / 0.1 * 0.27 / chord

        conditions = [aoa == 0, aoa == 2, aoa == 4]
        base_values = [0.0055, 0.006, 0.0075]
        base_cd = np.select(conditions, base_values, default=np.nan)

        delta_cd_one = one_part([mass_span, n_stagnation])
        print("One-part correlation:")
        analysis(delta_cd, delta_cd_one)
        plotting(delta_cd_one, delta_cd, os.path.join(path, "one_part_delta.png"), "delta_cd", mask)
        cd_one = delta_cd_one + 0.008
        print("One-part correlation + base clean cd:")
        analysis(iced_cd, cd_one)
        plotting(cd_one, iced_cd, os.path.join(path, "one_part_cd.png"), "cd", mask)

        delta_cd_one_adj = one_part([chord_adjusted_mass, n_stagnation])
        print("One-part correlation, AOA & chord adjusted:")
        analysis(delta_cd, delta_cd_one_adj)
        plotting(delta_cd_one_adj, delta_cd, os.path.join(path, "one_part_delta_adjusted.png"), "delta_cd", mask)
        cd_one_adj = delta_cd_one_adj + base_cd
        print("One-part correlation + AOA-adjusted clean cd, chord adjusted:")
        analysis(iced_cd, cd_one_adj)
        plotting(cd_one_adj, iced_cd, os.path.join(path, "one_part_cd_adjusted.png"), "cd", mask)

        delta_cd_two = two_part([mass_span, n_stagnation])
        print("Two-part correlation:")
        analysis(delta_cd, delta_cd_two)
        plotting(delta_cd_two, delta_cd, os.path.join(path, "two_part_delta.png"), "delta_cd", mask)
        cd_two = delta_cd_two + 0.008
        print("Two-part correlation + 0.008 clean cd:")
        analysis(iced_cd, cd_two)
        plotting(cd_two, iced_cd, os.path.join(path, "two_part_cd.png"), "cd", mask)

        delta_cd_two_adj = two_part([chord_adjusted_mass, n_stagnation])
        print("Two-part correlation, AOA&chord adjusted:")
        analysis(delta_cd, delta_cd_two_adj)
        plotting(delta_cd_two_adj, delta_cd, os.path.join(path, "two_part_delta_adj.png"), "delta_cd", mask)
        cd_two_adj = delta_cd_two_adj + base_cd
        print("Two-part correlation + AOA-based clean cd, chord adjusted:")
        analysis(iced_cd, cd_two_adj)
        plotting(cd_two_adj, iced_cd, os.path.join(path, "two_part_cd_adj.png"), "cd", mask)

        delta_cd_three = three_part([mass_span, n_stagnation])
        print("Three-part correlation:")
        analysis(delta_cd, delta_cd_three)
        plotting(delta_cd_three, delta_cd, os.path.join(path, "three_part_delta.png"), "delta_cd", mask)
        cd_three = delta_cd_three + 0.008
        print("Three-part correlation + 0.008 clean cd:")
        analysis(iced_cd, cd_three)
        plotting(cd_three, iced_cd, os.path.join(path, "three_part_cd.png"), "cd", mask)

        delta_cd_three_adj = three_part([chord_adjusted_mass, n_stagnation])
        print("Three-part correlation, chord adjusted:")
        analysis(delta_cd, delta_cd_three_adj)
        plotting(delta_cd_three_adj, delta_cd, os.path.join(path, "three_part_delta_adj.png"), "delta_cd", mask)
        cd_three_adj = delta_cd_three_adj + base_cd
        print("Three-part correlation + AOA-adjusted clean cd, chord adjusted:")
        analysis(iced_cd, cd_three_adj)
        plotting(cd_three_adj, iced_cd, os.path.join(path, "three_part_cd_adj.png"), "cd", mask)
