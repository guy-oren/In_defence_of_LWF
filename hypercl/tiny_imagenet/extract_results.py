import argparse
import os
import re
from collections import defaultdict
import json

import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--n_try", type=int, default=10)

    args = parser.parse_args()

    res_dict = defaultdict(list)

    for d in os.listdir(args.res_dir):
        d_full_path = os.path.join(args.res_dir, d)

        with open(os.path.join(d_full_path, "config.json")) as ifp:
            config = json.load(ifp)

        experiment = "tiny_imagnet"

        approach = "hypercl_" + config["hnet_arch"] + "_" + config["net_type"]
        seed = config["random_seed"]

        if config["disable_data_augmentation"]:
            augment = "no_augment"
        else:
            augment = config["aug_ver"]

        with open(os.path.join(d_full_path, "performance_summary.txt")) as ifp:
            lines = ifp.readlines()

            # check if finished
            if lines[-1].split()[-1] != "1":
                print("{} did not finished".format(d))
                continue

            num_tasks = config["num_tasks"]

            last_acc = [float(s) for s in lines[num_tasks - 1].split()]

            avg_all_acc = np.mean(last_acc)

            current_acc = [float(lines[i].split()[i]) for i in range(num_tasks)]

            avg_all_bwt = np.mean(np.array(last_acc[:-1]) - np.array(current_acc[:-1]))

            task_size = 200 // config["num_tasks"]

        res_dict["{}_{}_{}_{}".format(approach, experiment, task_size, augment)].append({"avg_all_acc": avg_all_acc,
                                                               "avg_all_bwt": avg_all_bwt,
                                                               "seed": seed})

    for key, values in res_dict.items():
        if len(values) < args.n_try:
            print("\n\nNot all runs finished for {}".format(key))
            continue
        elif len(values) > args.n_try:
            print("\n\nThere are redundent runs for {}".format(key))
            continue

        print("\n\nResults for {}:".format(key))

        all_runs_avg_all_acc = [v["avg_all_acc"] for v in values]
        all_runs_avg_all_bwt = [v["avg_all_bwt"] for v in values]

        all_runs_avg_all_acc_mu = np.mean(all_runs_avg_all_acc)
        all_runs_avg_all_acc_std = np.std(all_runs_avg_all_acc)
        print("avg_all_acc: {:.1f} \pm {:.1f}".format(all_runs_avg_all_acc_mu, all_runs_avg_all_acc_std))

        all_runs_avg_all_bwt_mu = np.mean(all_runs_avg_all_bwt)
        all_runs_avg_all_bwt_std = np.std(all_runs_avg_all_bwt)
        print("avg_all_bwt: {:.1f} \pm {:.1f}".format(all_runs_avg_all_bwt_mu, all_runs_avg_all_bwt_std))




