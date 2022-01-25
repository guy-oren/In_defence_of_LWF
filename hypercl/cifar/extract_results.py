import argparse
import os
import re
from collections import defaultdict
import json

import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--n_try", type=int, default=5)

    args = parser.parse_args()

    res_dict = defaultdict(list)

    for d in os.listdir(args.res_dir):
        d_full_path = os.path.join(args.res_dir, d)

        with open(os.path.join(d_full_path, "config.json")) as ifp:
            config = json.load(ifp)

        if config["use_cifar10"]:
            experiment = "cifar"
        else:
            experiment = "only_cifar100"

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

            if experiment == "only_cifar100":
                num_tasks = config["num_tasks"]

                last_acc = [float(s) for s in lines[num_tasks - 1].split()]
                avg_cifar100_acc = np.mean(last_acc)
                avg_all_acc = avg_cifar100_acc

                current_acc = [float(lines[i].split()[i]) for i in range(num_tasks)]

                avg_cifar100_bwt = np.mean(np.array(last_acc[:-1]) - np.array(current_acc[:-1]))

                avg_all_bwt = avg_cifar100_bwt

                task_size = 100 // num_tasks

            else:
                raise NotImplementedError


                last_acc = [float(s) for s in lines[0].split()[1:]]
                avg_cifar100_acc = np.mean(last_acc[1:])
                avg_all_acc = np.mean(last_acc)

                current_acc = [float(s) for s in lines[1].split()[1:]]

                avg_cifar100_bwt = np.mean(np.array(last_acc[1:-1]) - np.array(current_acc[1:-1]))

                avg_all_bwt = np.mean(np.array(last_acc[:-1]) - np.array(current_acc[:-1]))

                task_size = 100 // (config["num_tasks"] - 1)

        res_dict["{}_{}_{}_{}".format(approach, experiment, task_size, augment)].append({"avg_all_acc": avg_all_acc,
                                                               "avg_all_bwt": avg_all_bwt,
                                                               "avg_cifar100_acc": avg_cifar100_acc,
                                                               "avg_cifar100_bwt": avg_cifar100_bwt,
                                                               "seed": seed})

    for key, values in res_dict.items():
        if len(values) < args.n_try:
            print("\n\nNot all runs finished for {}".format(key))
            continue
        elif len(values) > args.n_try:
            print("\n\nThere are redundent runs for {}".format(key))
            continue

        print("\n\nResults for {}:".format(key))

        all_runs_avg_cifar100_acc = [v["avg_cifar100_acc"] for v in values]
        all_runs_avg_cifar100_bwt = [v["avg_cifar100_bwt"] for v in values]

        all_runs_avg_cifar100_acc_mu = np.mean(all_runs_avg_cifar100_acc)
        all_runs_avg_cifar100_acc_std = np.std(all_runs_avg_cifar100_acc)
        print("avg_cifar100_acc: {:.1f} \pm {:.1f}".format(all_runs_avg_cifar100_acc_mu,
                                                         all_runs_avg_cifar100_acc_std))

        all_runs_avg_cifar100_bwt_mu = np.mean(all_runs_avg_cifar100_bwt)
        all_runs_avg_cifar100_bwt_std = np.std(all_runs_avg_cifar100_bwt)
        print("avg_cifar100_bwt: {:.1f} \pm {:.1f}".format(all_runs_avg_cifar100_bwt_mu,
                                                       all_runs_avg_cifar100_bwt_std))
