from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import contextlib


def bragg_optimizing(variables, z1, z2):
    rough, chord_eq, ac_eq, e_eq = variables
    result = 0.01 * (15.8 * np.log(rough / chord_eq) + z1 * ac_eq * e_eq + z2)
    return result


def fleming_optimizing(variables, z1, z2):
    rough, chord_l, v, em, lwc_eq, time_eq, rho_i, aoa_eq, cd_cl = variables
    result = (
        (0.158 * np.log(rough / chord_l) + z1 * v * em * lwc_eq / 1000 * time_eq / (rho_i * chord_l) + z2) * ((aoa_eq + 6) / 10) * cd_cl
    )
    return result


plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.size"] = 9

file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")])
path = os.path.abspath(os.path.join(file_path, ".."))

with open(os.path.join(path, "output.txt"), "w") as f:
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        df = pd.read_csv(file_path, sep=";")
        beta_0 = np.array(df["1st step stagnation beta"].tolist())
        lwc = np.array(df["LWC [g/m3]"].tolist())
        u = np.array(df["Speed [m/s]"].tolist())
        time = np.array(df["Time [s]"].tolist())
        rho_ice = np.array(df["Ice density [kg/m3]"].tolist())
        chord = np.array(df["Chord [m]"].tolist())
        mvd = np.array(df["MVD [um]"].tolist())
        T = np.array(df["Temperature [K]"].tolist())
        aoa = np.array(df["AOA [deg]"].tolist())
        roughness = np.array(df["Max. roughness height [m]"].tolist())
        ac = np.array(df["Accumulation parameter"].tolist())
        mass = np.array(df["Mass caught [kg]"].tolist())
        n_stagnation = np.array(df["1st step stagnation freezing fraction"].tolist())
        iced_cd = np.array(df["iced cd"].tolist())
        clean_cd = np.array(df["clean cd"].tolist())

        delta_cd = iced_cd - clean_cd
        cd_ratio = iced_cd / clean_cd
        rel_cd_increase = delta_cd / clean_cd

        ## Han-Palacios correlation
        area = 0.1 * 0.0304103  # CFD span * projected height for 0.27m chord @ 4° AOA
        e = mass / (area * lwc / 1000 * u * time)
        le_dia = 2 * 0.0048 * chord
        rho_air = 101325 / (287 * T)  # Pressure is 101325 Pa and universal gas constant is 287 J/(kg*K)
        mu_air = 1.458 * 10**-6 * T ** (3 / 2) / (T + 110.4)
        ac_hpc = lwc / 1000 * u * time / (rho_ice * le_dia)
        re_mvd = u * mvd * 10**-6 * rho_air / mu_air
        cd_hpc = (2.69 * beta_0 * ac_hpc * re_mvd + 3800 * T / 273.15 + 9.65 * (aoa - 3.352) ** 2 - 3663) * 10**-4
        plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
        plt.scatter(cd_hpc, iced_cd, s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        min_limit = min(x_limits[0], y_limits[0])
        max_limit = max(x_limits[1], y_limits[1])
        plt.plot([-10, 10], [-10, 10], color="red", label="Perfect agreement line")
        plt.xlim([min_limit, max_limit]), plt.ylim([min_limit, max_limit])
        plt.ylabel("Simulated iced drag coefficient")
        plt.xlabel("Predicted iced drag coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "hpc.png"))
        plt.savefig(os.path.join(path, "hpc.tiff"))
        plt.close()

        ## Bragg Rime correlation
        rime_filter = n_stagnation == 1
        perfect_I = 100 * rel_cd_increase - 15.8 * np.log(roughness / chord) - 28000 * ac * e
        average_I = np.mean(perfect_I[rime_filter])
        rel_cd_increase_Bragg = 0.01 * (15.8 * np.log(roughness / chord) + 28000 * ac * e + average_I)
        perfect_I_1000 = 100 * rel_cd_increase - 15.8 * np.log(roughness / chord) - 28 * ac * e
        average_I_1000 = np.mean(perfect_I_1000[rime_filter])
        rel_cd_increase_Bragg_1000 = 0.01 * (15.8 * np.log(roughness / chord) + 28 * ac * e + average_I_1000)
        perfect_I_60 = 100 * rel_cd_increase - 15.8 * np.log(roughness / chord) - 28000 / 60 * ac * e
        average_I_60 = np.mean(perfect_I_60[rime_filter])
        rel_cd_increase_Bragg_60 = 0.01 * (15.8 * np.log(roughness / chord) + 28000 / 60 * ac * e + average_I_60)
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        sns.histplot(perfect_I, kde=True, stat="density", bins=40, color="black", edgecolor="black")
        plt.xlabel("Required value for I for a perfect fit")
        plt.ylabel("Density")
        plt.xticks(np.arange(-200, 300, 100))
        plt.tight_layout()
        plt.savefig(os.path.join(path, "perfect_Bragg_I_histogram.png"))
        plt.savefig(os.path.join(path, "perfect_Bragg_I_histogram.tiff"))
        plt.close()
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.plot(perfect_I[rime_filter], "o", linestyle="none", markersize=3, color="black")
        plt.gca().get_xaxis().set_visible(False)
        plt.ylabel("I for a Perfect agreement line")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "perfect_Bragg_I.png"))
        plt.savefig(os.path.join(path, "perfect_Bragg_I.tiff"))
        plt.close()
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.scatter(rel_cd_increase_Bragg[rime_filter], rel_cd_increase[rime_filter], s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        min_limit = min(x_limits[0], y_limits[0])
        max_limit = max(x_limits[1], y_limits[1])
        plt.plot([-10, 10], [-10, 10], color="red", label="Perfect agreement line")
        plt.xlim([min_limit, max_limit]), plt.ylim([min_limit, max_limit])
        plt.ylabel("Simulated relative drag change")
        plt.xlabel("Predicted relative drag change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Bragg_original.png"))
        plt.savefig(os.path.join(path, "Bragg_original.tiff"))
        plt.close()

        variable_input = [roughness, chord, ac, e]
        variable_input_rime = [x[rime_filter] for x in variable_input]
        initial_guess = [16000, 200]
        [params, _] = curve_fit(f=bragg_optimizing, xdata=variable_input_rime, ydata=rel_cd_increase[rime_filter], p0=initial_guess)
        prediction = bragg_optimizing(variable_input_rime, *params)
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.scatter(prediction, rel_cd_increase[rime_filter], s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        min_limit = min(x_limits[0], y_limits[0])
        max_limit = max(x_limits[1], y_limits[1])
        plt.plot([-10, 10], [-10, 10], color="red", label="Perfect agreement line")
        plt.xlim([min_limit, max_limit]), plt.ylim([min_limit, max_limit])
        plt.ylabel("Simulated relative drag change")
        plt.xlabel("Predicted relative drag change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Bragg_optimized.png"))
        plt.savefig(os.path.join(path, "Bragg_optimized.tiff"))
        plt.close()
        print(f"Bragg: z1 = {params[0]}, z2 = {params[1]}")
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.scatter(rel_cd_increase_Bragg_1000[rime_filter], rel_cd_increase[rime_filter], s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        plt.plot([-10, 10], [-10, 10], color="red", label="Perfect agreement line")
        plt.xlim([0, x_limits[-1]]), plt.ylim([0, y_limits[-1]])
        plt.ylabel("Simulated relative drag change")
        plt.xlabel("Predicted relative drag change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Bragg_1000.png"))
        plt.savefig(os.path.join(path, "Bragg_1000.tiff"))
        plt.close()

        ## Fleming and Lednicer rime correlation
        delta_cd_fleming = (
            (0.158 * np.log(roughness / chord) + 175 * u * e * lwc / 1000 * time / (rho_ice * chord) + 1.7) * ((aoa + 6) / 10) * clean_cd
        )
        delta_cd_fleming_60 = (
            (0.158 * np.log(roughness / chord) + 175 * u * e * lwc / 60000 * time / (rho_ice * chord) + 1.7) * ((aoa + 6) / 10) * clean_cd
        )
        delta_cd_fleming_1000 = (
            (0.158 * np.log(roughness / chord) + 175 * u * e * lwc / 1000 / 1000 * time / (rho_ice * chord) + 1.7)
            * ((aoa + 6) / 10)
            * clean_cd
        )
        plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
        plt.scatter(delta_cd_fleming[rime_filter], delta_cd[rime_filter], s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        min_limit = min(x_limits[0], y_limits[0])
        max_limit = max(x_limits[1], y_limits[1])
        plt.plot([0, 10], [0, 10], color="red", label="Perfect agreement line")
        plt.xlim([min_limit, max_limit]), plt.ylim([min_limit, max_limit])
        plt.ylabel("Simulated change in drag coefficient")
        plt.xlabel("Predicted change in drag coefficient")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Fleming_original.png"))
        plt.savefig(os.path.join(path, "Fleming_original.tiff"))
        plt.close()

        variable_input = [roughness, chord, u, e, lwc, time, rho_ice, aoa, clean_cd]
        variable_input_rime = [x[rime_filter] for x in variable_input]
        initial_guess = [175, 1.7]
        [params, _] = curve_fit(f=fleming_optimizing, xdata=variable_input_rime, ydata=delta_cd[rime_filter], p0=initial_guess)
        prediction = fleming_optimizing(variable_input_rime, *params)
        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.scatter(prediction, delta_cd[rime_filter], s=3, color="black", label="Data points")
        x_limits, y_limits = plt.gca().get_xlim(), plt.gca().get_ylim()
        min_limit = min(x_limits[0], y_limits[0])
        max_limit = max(x_limits[1], y_limits[1])
        plt.plot([-10, 10], [-10, 10], color="red", label="Perfect agreement line")
        plt.xlim([min_limit, max_limit]), plt.ylim([min_limit, max_limit])
        plt.ylabel("Simulated change in drag coefficient")
        plt.xlabel("Predicted change in drag coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Fleming_optimized.png"))
        plt.savefig(os.path.join(path, "Fleming_optimized.tiff"))
        plt.close()
        print(f"Fleming: z1 = {params[0]}, z2 = {params[1]}")

        plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54))
        plt.scatter(delta_cd_fleming_1000[rime_filter], delta_cd[rime_filter], s=3, color="black", label="Data points")
        y_limits = plt.gca().get_ylim()
        plt.plot([0, 10], [0, 10], color="red", label="Perfect agreement line")
        plt.xlim([0, 0.012]), plt.ylim([0, y_limits[-1]])
        plt.ylabel("Simulated change in drag coefficient")
        plt.xlabel("Predicted change in drag coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Fleming_1000.png"))
        plt.savefig(os.path.join(path, "Fleming_1000.tiff"))
        plt.close()
