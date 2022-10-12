# %% [markdown]
# CA: The code for CA simulations is based on the algorithm in "A cellular automata model for forest fire spread prediction: The case of the wildfire that swept through Spetses Island in 1990" (Author: A. Alexandridis a, D. Vakalis b, C.I. Siettos c, G.V. Bafas a) and the work of https://github.com/XC-Li/Parallel_CellularAutomaton_Wildfire (Author: Xiaochi (George) Li)

import argparse
import copy
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
# %%
import sys
from PIL import Image
from PIL import ImageOps
from matplotlib import animation as animation
from matplotlib import cm

# %%
parser = argparse.ArgumentParser(description='Perform CA Fire Prediction')
parser.add_argument(
    '-i', '--index', help='fire index in testset', type=int, default=0)
args = parser.parse_args()

fire_index = args.index
# from google.colab import drive
# drive.mount('/content/drive')

# %%
fire_list = ['Perry_2018_NAT', 'Ranch_2018_NAT', 'Rim_2018_NAT',
             'River_2018_NAT', 'Roosevelt_2018_NAT',
             'Ryan_2018_NAT', 'Sharps_2018_NAT', 'Stone_2018_NAT',
             'Whaleback_2018_NAT', 'Woolsey_2018_NAT',
             'Briceburg_2019_NAT', 'Camp_2019_NAT', 'Caples_2019_NAT',
             'Easy_2019_NAT', 'Fishhawk_2019_NAT',
             'Jordan_2019_NAT', 'Kincade_2019_NAT', 'Lime_2019_NAT',
             'Lincoln_2019_NAT', 'Middle_2019_NAT',
             'Neck_2019_NAT', 'Ranch_2019_NAT', 'South_2019_NAT',
             'Springs_2019_NAT', 'Taboose_2019_NAT',
             'Thomas_2019_NAT', 'Tick_2019_NAT', 'Valley_2019_NAT',
             'Walker_2019_NAT', 'Woodbury_2019_NAT',
             'Apple_2020_NAT']

# %%
# fire_index = 0
fire_name = fire_list[fire_index]
with open(f"./data/{fire_index}_info_{fire_name}.json", 'r') as load_f:
    fire_info = json.load(load_f)
print(fire_info)

# Reading operating parameters
total_day = fire_info['total_day']
wind_v = fire_info['wind_v']
wind_u = fire_info['wind_u']
start_from = 2

def read_ignition(path, shape):
    # Change to gray scale and resize to shape
    ignition = Image.open(path).convert('L')
    ignition = ignition.resize(shape)
    ignition = ImageOps.invert(ignition)
    # Convert to range 0 ~ 255
    ignition = np.array(ignition)
    ignition[ignition > 0.] = 3.
    ignition[ignition <= 0] = 2.
    return ignition


forest = Image.open(f'./data/{fire_index}_canopy_{fire_name}.tif')
altitude = Image.open(f'./data/{fire_index}_slope_{fire_name}.tif')
density = Image.open(f'./data/{fire_index}_density_{fire_name}.tif')
ignition = read_ignition(
    f'./data/{fire_index}_ignition_{fire_name}_{start_from}.png', forest.size)

# %%
forest = np.array(forest)

altitude = np.array(altitude) / np.max(altitude)

density = np.array(density)

density = np.round(density / np.max(density))

density[density < 0.] = 0.

forest[forest < -999.] = 0.

forest = forest / np.max(forest)

n_row = forest.shape[0]
n_col = forest.shape[1]

number_MC = 20
#################################################################
generation = 40 * (total_day - start_from) + 1
n_row = forest.shape[0]
n_col = forest.shape[1]


# %%


def colormap(i, array):
    np_array = np.array(array)
    plt.imshow(np_array, interpolation="none", cmap=cm.plasma)
    plt.title(i)
    plt.show()


# %%


def init_vegetation():
    veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            veg_matrix[i][j] = 1
    return veg_matrix


# %%


def init_density():
    den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            den_matrix[i][j] = 1.0
    return den_matrix


# %%


def init_altitude():
    alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            alt_matrix[i][j] = 1
    return alt_matrix


# %%


def init_forest():
    forest = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            forest[i][j] = 2
    # ignite_col = int(n_col//2)
    # ignite_row = int(n_row//2)
    ignite_col = int(n_col // 2)
    ignite_row = int(100)
    for row in range(ignite_row - 1, ignite_row + 1):
        for col in range(ignite_col - 1, ignite_col + 1):
            forest[row][col] = 3
    # forest[ignite_row-2:ignite_row+2][ignite_col-2:ignite_col+2] = 3
    return forest


# %%


def print_forest(forest):
    for i in range(n_row):
        for j in range(n_col):
            sys.stdout.write(str(forest[i][j]))
        sys.stdout.write("\n")


# %%


def tg(x):
    return math.degrees(math.atan(x))


# %%


def get_slope(altitude_matrix):
    slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for row in range(n_row):
        for col in range(n_col):
            sub_slope_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            if row == 0 or row == n_row - 1 or col == 0 or col == n_col - 1:  # margin is flat
                slope_matrix[row][col] = sub_slope_matrix
                continue
            current_altitude = altitude_matrix[row][col]
            sub_slope_matrix[0][0] = tg(
                (current_altitude - altitude_matrix[row - 1][col - 1]) / 1.414)
            sub_slope_matrix[0][1] = tg(
                current_altitude - altitude_matrix[row - 1][col])
            sub_slope_matrix[0][2] = tg(
                (current_altitude - altitude_matrix[row - 1][col + 1]) / 1.414)
            sub_slope_matrix[1][0] = tg(
                current_altitude - altitude_matrix[row][col - 1])
            sub_slope_matrix[1][1] = 0
            sub_slope_matrix[1][2] = tg(
                current_altitude - altitude_matrix[row][col + 1])
            sub_slope_matrix[2][0] = tg(
                (current_altitude - altitude_matrix[row + 1][col - 1]) / 1.414)
            sub_slope_matrix[2][1] = tg(
                current_altitude - altitude_matrix[row + 1][col])
            sub_slope_matrix[2][2] = tg(
                (current_altitude - altitude_matrix[row + 1][col + 1]) / 1.414)
            slope_matrix[row][col] = sub_slope_matrix
    return slope_matrix


# %%
V = np.linalg.norm([wind_u, wind_v])  # need to find the true wind data
p_h = 0.58
a = 0.078
c_1 = 0.045
c_2 = 0.131


##############################################################################
def calc_pw(theta, c_1, c_2, V):
    t = math.radians(theta)
    ft = math.exp(V * c_2 * (math.cos(t) - 1))
    return math.exp(c_1 * V) * ft


def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = np.sqrt(np.power(
        v1[0], 2) + np.power(v1[1], 2)) * np.sqrt(
        np.power(v2[0], 2) + np.power(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (np.arccos(cos) / pi) * 180


def get_thetas(u, v):
    thetas_coordinate = []
    for i in range(0, 3):
        for j in range(0, 3):
            thetas_coordinate.append([j - 1, 1 - i])
    # print(thetas_coordinate)
    thetas = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            thetas[i, j] = np.round(angle_of_vector(
                thetas_coordinate[i * 3 + j], [u, v]), decimals=4)
    return thetas


def get_wind():
    wind_matrix = [[0 for col in [0, 1, 2]] for row in [0, 1, 2]]

    thetas = get_thetas(wind_u, wind_v)
    print('wind_angles')
    print(thetas)
    # thetas = [[0, 180, 180],  #need to define the exact angle
    #           [180, 0, 180],
    #           [180, 180, 0]]

    for row in [0, 1, 2]:
        for col in [0, 1, 2]:
            wind_matrix[row][col] = calc_pw(thetas[row][col], c_1, c_2, V)
    wind_matrix[1][1] = 0
    return wind_matrix


def burn_or_not_burn(abs_row, abs_col, neighbour_matrix, p_h, a):
    p_veg = vegetation_matrix[abs_row][abs_col]
    p_den = {0: -0.4, 1: 0, 2: 0.3}[density_matrix[abs_row][abs_col]]
    for row in [0, 1, 2]:
        for col in [0, 1, 2]:
            # we only care there is a neighbour that is burning
            if neighbour_matrix[row][col] == 3:
                # print(row,col)
                slope = slope_matrix[abs_row][abs_col][row][col]
                p_slope = math.exp(a * slope)
                p_wind = wind_matrix[row][col]
                p_burn = p_h * (0.5 + p_veg * 10.) * \
                         (1 + p_den) * p_wind * p_slope
                if p_burn > random.random():
                    return 3  # start burning
    return 2  # not burning


def update_forest(old_forest):
    result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
    for row in range(1, n_row - 1):
        for col in range(1, n_col - 1):

            if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                # no fuel or burnt down
                result_forest[row][col] = old_forest[row][col]
            if old_forest[row][col] == 3:
                if random.random() < 0.4:
                    # TODO need to change back here
                    result_forest[row][col] = 3
                else:
                    result_forest[row][col] = 4
            if old_forest[row][col] == 2:
                neighbours = [
                    [row_vec[col_vec] for col_vec in range(col - 1, col + 2)]
                    for row_vec in old_forest[row - 1:row + 2]]
                # print(neighbours)
                result_forest[row][col] = burn_or_not_burn(
                    row, col, neighbours, p_h, a)
    return result_forest


#############################################################################

fields_1_sim = np.zeros((1, 100))

vegetation_matrix = forest

density_matrix = density.tolist()

altitude_matrix = altitude.tolist()

wind_matrix = get_wind()

new_forest = ignition.tolist()

slope_matrix = get_slope(altitude_matrix)

ims = []

###########################################################
# custormize colorbar

cmap = mpl.colors.ListedColormap(['orange', 'yellow', 'green', 'red'])
cmap.set_over('0.25')
cmap.set_under('0.75')
bounds = [1.0, 2.03, 2.35, 3.5, 5.1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

############################################################
import datetime

start = datetime.datetime.now()

for i in range(generation):
    sys.stdout.write(
        f'\rGeneration: {i + 1}/{generation} Day: {i // 40 + start_from} / {generation // 40 + start_from}')
    new_forest = copy.deepcopy(update_forest(new_forest))
    forest_array = np.array(new_forest)
    if i > 0 and i % 40 == 0:
        # plt.imshow(forest + forest_array, cmap=cmap, norm=norm, interpolation="none")
        np.save(
            f'./result/{fire_index}_fire_range_{fire_name}_{i // 40 + start_from}.npy',
            forest_array)
        plt.imshow(forest + forest_array, cmap=cmap, norm=norm,
                   interpolation="none", vmin=0., vmax=4.)
        plt.axis('off')
        plt.savefig(
            f'./result/{fire_index}_total_{fire_name}_{i // 40 + start_from}.png',
            format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
    print('\nburning', np.sum(forest_array == 4))
    # plt.show()
    # plt.close()

end = datetime.datetime.now()

result = {'runtime': (end - start).total_seconds()}
print(result)
with open(f"./result/time/{fire_index}_time_{fire_name}.json", 'w') as write_f:
    json.dump(result, write_f, indent=4, ensure_ascii=False)
# %%

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.ListedColormap(['orange', 'yellow', 'green'])
cmap.set_over('0.25')
cmap.set_under('0.75')
bounds = [1.0, 2.03, 2.35, 3.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cmap = mpl.cm.cool
# norm = mpl.colors.Normalize(vmin=0, vmax=10)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal')  # label='Vegetation density')
plt.xticks(ticks=[1.0, 2.03, 2.35, 3.5], labels=[
    'low', '', '', 'high'], fontsize=25)
plt.title("vegetation density", fontsize=30)
plt.savefig(f'./result/{fire_index}_colorbar_{fire_name}.png',
            format='png', bbox_inches='tight')
plt.legend()
