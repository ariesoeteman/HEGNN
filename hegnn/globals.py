import os
import sys

package_dir = os.path.dirname(os.path.abspath(__file__))
all_data_dir = os.path.join(package_dir, "all_data")
data_dir = os.path.join(package_dir, "all_data", "data")
results_dir = os.path.join(package_dir, "results")
custom_data_dir = os.path.join(package_dir, "all_data", "custom_data")
csl_data_dir = os.path.join(package_dir, "all_data", "CSL")
srg_data_dir = os.path.join(package_dir, "all_data", "SR_graphs")

model_dir = os.path.join(custom_data_dir, "saved_models")

config_dir = os.path.join(package_dir, "config")
config_path = os.path.join(config_dir, "config.yaml")
plot_dir = os.path.join(package_dir, "plots")
final_results_dir = os.path.join(plot_dir, "final_results")
