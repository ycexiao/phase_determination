import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import phase_component_determination
import json


def read_data(file_path):
    data = pd.read_csv(
        file_path, sep=r"\s+", comment="#", header=None, names=["r", "G"]
    )
    x_array = data["r"].values
    y_array = data["G"].values
    return [x_array, y_array]


ref_region_1_file = "data/RefPosition1-Gr-Frame7555-map58-51-ROI58-51.csv"
ref_region_2_file = "data/RefPosition2-Gr-Frame12406-map58-84-ROI58-84.csv"
target_region_file = "data/Gr-Frame3145-map58-21-ROI58-21.csv"
target_phase = read_data(target_region_file)
ref_phase1 = read_data(ref_region_1_file)
ref_phase2 = read_data(ref_region_2_file)
# Plotting the phases
# Found that peak is between 1 and 2
# plt.plot(target_phase[0], target_phase[1], label="Target Phase")
# plt.plot(ref_phase1[0], ref_phase1[1], label="Reference Phase 1")
# plt.plot(ref_phase2[0], ref_phase2[1], label="Reference Phase 2")
# plt.legend()
# plt.show()
region1 = 1 - np.random.rand(4, 10) * 0.1
region2 = 0.5 + np.random.rand(2, 10) * 0.1
region3 = 0 + np.random.rand(4, 10) * 0.1
region = np.vstack([region1, region2, region3])
# plt.imshow(region, cmap="viridis")
# plt.colorbar(label="Region Intensity")
# plt.show()
target_region = [
    [
        [ref_phase1[0], a_1 * ref_phase1[1] + (1 - a_1) * ref_phase2[1]]
        for a_1 in slice
    ]
    for slice in region
]
ref_region1 = [
    [[ref_phase1[0], ref_phase1[1]] for a_1 in slice] for slice in region
]
ref_region2 = [
    [[ref_phase2[0], ref_phase2[1]] for a_1 in slice] for slice in region
]
# main.main(
#     target_region,
#     ref_region1,
#     ref_region2,
#     mode="relaxed",
#     save_name="two_phase_results.json",
# )
# main.main(
#     target_region,
#     ref_region1,
#     mode="relaxed",
#     save_name="phase1_results.json",
# )
# main.main(
#     target_region,
#     ref_region2,
#     mode="relaxed",
#     save_name="phase2_results.json",
# )

with open("two_phase_results.json", "r") as f:
    data0 = json.load(f)
with open("phase1_results.json", "r") as f:
    data1 = json.load(f)
with open("phase2_results.json", "r") as f:
    data2 = json.load(f)

phase_component_determination.error_visualization(
    data1, title="Phase 1 Results"
)
phase_component_determination.error_visualization(
    data2, title="Phase 2 Results"
)
phase_component_determination.error_visualization(
    data0, title="Two Phase Results (Original)"
)
