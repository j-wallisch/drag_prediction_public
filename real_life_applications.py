import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
import csv
from scipy.optimize import curve_fit


def analysis(simulation, predicted):
    rmse_value = np.sqrt(np.mean((simulation - predicted) ** 2))
    rmsre_value = np.sqrt(np.sum(((simulation - predicted) / simulation) ** 2, axis=0) / len(simulation))
    within_10 = np.mean(np.abs(simulation - predicted) <= 0.1 * predicted) * 100
    within_20 = np.mean(np.abs(simulation - predicted) <= 0.2 * predicted) * 100
    within_30 = np.mean(np.abs(simulation - predicted) <= 0.3 * predicted) * 100
    outside_30 = np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    between_20_30 = np.sum(0.2 * predicted < np.abs(simulation - predicted)) - np.sum(np.abs(simulation - predicted) > 0.3 * predicted)
    max_offset = np.max(np.abs(simulation - predicted) / predicted) * 100
    print(f"RMSRE: {rmsre_value}")
    print(f"Within 10%: {within_10}%, within 20%: {within_20}%, within 30%: {within_30}%")
    print(f"Outside 30%: {outside_30}, maximal offset: {max_offset}%, between 20&30%: {between_20_30}\n")


def plotting(x, y, n_stagnation, filename):
    plt.subplots(figsize=(15 / 2.54, 7.5 / 2.54))
    colors = n_stagnation
    plt.fill_between([0, 5], [0, 5 * 0.9], [0, 5 * 1.1], color="green", alpha=0.2, label="10%")
    plt.fill_between([0, 5], [0, 5 * 0.8], [0, 5 * 0.9], color="blue", alpha=0.2, label="20%")
    plt.fill_between([0, 5], [0, 5 * 1.1], [0, 5 * 1.2], color="blue", alpha=0.2)
    plt.fill_between([0, 5], [0, 5 * 0.7], [0, 5 * 0.8], color="red", alpha=0.2, label="30%")
    plt.fill_between([0, 5], [0, 5 * 1.2], [0, 5 * 1.3], color="red", alpha=0.2)
    plt.scatter(x, y, c=colors, cmap="coolwarm_r", edgecolors="black")
    plt.plot([0, 10], [0, 10], color="red", label="Perfect agreement line")
    plt.colorbar(label="Freezing fraction")
    plt.xlim([0.005, 0.018]), plt.ylim([0.005, 0.018])
    plt.ylabel("Simulated drag coefficient increase")
    plt.xlabel("Predicted drag coefficient increase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(f"{filename[:-4]}.tiff")
    plt.close()


plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.size"] = 9

file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")])
path = os.path.abspath(os.path.join(file_path, ".."))
df = pd.read_csv(file_path, sep=";")

lwc = np.array(df["LWC [g/m3]"].tolist())
#lwc = 0.6
speed = np.array(df["Speed [m/s]"].tolist())
#speed = 50
time = np.array(df["Time [s]"].tolist())
#time = 60
mass = np.array(df["Mass caught [kg]"].tolist())
t_a = np.array(df["Temperature [K]"].tolist())
#t_a = 263.15
n0 = np.array(df["1st step stagnation freezing fraction"].tolist())
beta_0 = np.array(df["1st step stagnation beta"].tolist())
#beta_0 = 0.9
iced_cd = np.array(df["iced cd"].tolist())
clean_cd = np.array(df["clean cd"].tolist())
chord = np.array(df["Chord [m]"].tolist())
#chord = 0.4
delta_cd = iced_cd - clean_cd
# cd_ratio = iced_cd / clean_cd

t_film = 0.5 * (t_a + 273.15)
k_air = 0.0236  # conductivity of air
d = 2 * 0.005 * chord  # twice the airfoil leading edge radius (0.5% chord for the RG-15)
pr = 0.706  # Prandtl number of air
mu_air = 1.716 * 10**-5 * (t_film / 273.15) ** 1.5 * (273.15 + 110.4) / (t_film + 110.4)  # Dynamic viscosity of air
t_s = 273.15  # surface temperature 0° except when freezing fraction is 1
c_p_a = 1005  # Specific heat of air at constant pressure
lambda_v = 2501000  # latent heat of vaporization
p_ww = 616  # water vapor pressure over surface with 0°C
p_st = 101325  # static ambient pressure
t_f = 273.15  # freezing temperature 0°C except when freezing fraction is 1
c_p_ws = (1.0074 + 8.29 * 10**-5 * (t_a - 273.15) ** 2) * 4187  # Specific heat of water at constant pressure, incl. unit conversion
lambda_f = 334000  # latent heat of freezing
rho_air = p_st / (287 * t_film)  # air density, universal gas constant is 287 J/(kg*K)
d_v = 0.211 * (t_film / 273.15) ** 1.94 * 1 / 10000  # diffusivity

mass_per_span = mass / 0.1
h_proj = 0.030410317  # projected height at 4° AOA with 0.27m chord
e = mass_per_span / (lwc / 1000 * h_proj * speed * time)  # total collection efficiency
m = lwc / 1000 * speed * h_proj * time * e  # mass per span as used in the correlation

re_r = (rho_air * speed * d / 2) / mu_air
nu = 0.81 * re_r**0.5 * pr**0.4
h_c = nu * k_air / d
t_bl = t_a + speed**2 / (2 * c_p_a)
q_c = h_c * (t_s - t_bl)

p_w = np.exp(
    -100.7938
    - 3256.721 / t_a
    + 24.1521 * np.log(t_a)
    - 0.059768 * t_a
    + np.tanh(t_a - 228.9) * (-21.48 + 691.844 / t_a + 3.54003 * np.log(t_a) - 0.00344392 * t_a)
)
sc = mu_air / (rho_air * d_v)
h_g = h_c / c_p_a * (pr / sc) ** 0.67
m_e = h_g * (p_ww - p_w) / p_st
q_e = m_e * lambda_v

m_dot = beta_0 * lwc / 1000 * speed
q_w = m_dot * c_p_ws * (t_f - t_a)

q_k = m_dot * speed**2 / 2

q_f = q_c + q_e + q_w - q_k
n = np.minimum(1, q_f / (m_dot * lambda_f))
mask = t_a == 273.15
# plt.scatter(n0[~mask], n[~mask])
# plt.xlabel("Simulated n")
# plt.ylabel("Calculated n")
# plt.plot([0, 5], [0, 5], c="r", label="Perfect prediction")
# plt.xlim([0, 1]), plt.ylim([0, 1])


def three_part_fully_calculated(variables):
    mass_eq, n_eq = variables

    result = np.zeros_like(n_eq)
    mask1 = n_eq <= 0.32
    result[mask1] = 0.0205 * mass_eq[mask1] ** 0.571 * n_eq[mask1] ** -0.104 + 0.005285

    mask2 = n_eq >= 0.6
    result[mask2] = 0.0091 * mass_eq[mask2] ** 0.456 * n_eq[mask2] ** -0.190 + 0.005584

    mask3 = ~(mask1 | mask2)
    result[mask3] = 0.0351 * mass_eq[mask3] ** 0.912 * n_eq[mask3] ** -0.694 + 0.005897
    return result


independent_vars_filtered = [x[~mask] for x in [m, n]]
prediction = three_part_fully_calculated(independent_vars_filtered)
delta_cd_filtered = delta_cd[~mask]
analysis(delta_cd_filtered, prediction)
plotting(prediction, delta_cd_filtered, n0[~mask], os.path.join(path, "three_part.png"))

m_dot_avg = 0.65 * lwc / 1000 * speed
q_w_avg = m_dot_avg * c_p_ws * (t_f - t_a)

q_k_avg = m_dot_avg * speed**2 / 2

q_f_avg = q_c + q_e + q_w_avg - q_k_avg
n_avg = np.minimum(1, q_f_avg / (m_dot_avg * lambda_f))
mask = t_a == 273.15
plt.figure()
plt.scatter(n0[~mask], n_avg[~mask])
plt.xlabel("Simulated n")
plt.ylabel("Calculated n")
plt.plot([0, 5], [0, 5], c="r", label="Perfect prediction")
plt.xlim([0, 1]), plt.ylim([0, 1])
independent_vars_filtered = [x[~mask] for x in [m, n_avg]]
prediction = three_part_fully_calculated(independent_vars_filtered)
delta_cd_filtered = delta_cd[~mask]
analysis(delta_cd_filtered, prediction)
plotting(prediction, delta_cd_filtered, n0[~mask], os.path.join(path, "three_part_averaged.png"))


def one_part_limited(variables, z1, z_lwc, z_temp, z_time, z2):
    lwc_eq, temp_eq, time_eq = variables
    result = z1 * (lwc_eq / 1000) ** z_lwc * (273.15 - temp_eq) ** z_temp * time_eq**z_time + z2
    return result


initial_guess = [0.02, 0.5, -0.5, 0.1, 0.006]
rmse_best = np.inf
mask = t_a == 273.15
independent_vars_filtered = [x[~mask] for x in [lwc, t_a, time]]
delta_cd_filtered = delta_cd[~mask]
csv_path = os.path.join(path, "one_part_limited.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["rmse_total"] + ["rmse_other_half"] + ["constant"] + ["lwc"] + ["temp"] + ["time"] + ["offset"])
for i in range(25):
    mask2 = np.array([True] * (len(delta_cd_filtered) // 2) + [False] * (len(delta_cd_filtered) - len(delta_cd_filtered) // 2))
    np.random.shuffle(mask2)
    independent_vars_filtered_2 = [x[mask2] for x in independent_vars_filtered]
    filtered_delta_cd_2 = delta_cd_filtered[mask2]
    [params, _] = curve_fit(f=one_part_limited, xdata=independent_vars_filtered_2, ydata=filtered_delta_cd_2, p0=initial_guess, maxfev=5000)
    independent_vars_filtered_other_half = [x[~mask2] for x in independent_vars_filtered]
    filtered_delta_cd_other_half = delta_cd_filtered[~mask2]
    prediction_other_half = one_part_limited(independent_vars_filtered_other_half, *params)
    rmse_other_half = np.sqrt(np.mean((filtered_delta_cd_other_half - prediction_other_half) ** 2))
    prediction = one_part_limited(independent_vars_filtered, *params)
    rmse = np.sqrt(np.mean((delta_cd_filtered - prediction) ** 2))
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([rmse] + [rmse_other_half] + params.tolist())
    if rmse_other_half < rmse_best:
        rmse_best = rmse_other_half
        prediction_best = prediction
        params_best = params
print(
    f"One-part limited: {params_best[0]} * (lwc/1000)^{params_best[1]} * (temp-273.15)^{params_best[2]} * (time)^{params_best[3]} + {params_best[4]}"
)
print(f"RMSE other half: {rmse_best}")
analysis(simulation=delta_cd_filtered, predicted=prediction_best)
plotting(prediction_best, delta_cd_filtered, n0[~mask], os.path.join(path, "one_part_limited_new.png"))

def one_part_lwc_existing(variables):
    lwc_eq, temp_eq, time_eq = variables
    result = 0.028873881152887713 * (lwc_eq / 1000) ** 0.7297609710044436 * (273.15 - temp_eq) ** -0.44905204351196704 * time_eq ** 0.6031221076418914 + 0.006313113507623483
    return result

independent_vars_filtered = [x[~mask] for x in [lwc, t_a, time]]
prediction = one_part_lwc_existing(independent_vars_filtered)
delta_cd_filtered = delta_cd[~mask]
analysis(delta_cd_filtered, prediction)
plotting(prediction, delta_cd_filtered, n0[~mask], os.path.join(path, "one_part_lwc_existing.png"))
