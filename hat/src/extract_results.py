import argparse
import os
import re
from collections import defaultdict

import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n_try", type=int, default=5)

    args = parser.parse_args()

    res_dict = defaultdict(list)

    for f in os.listdir(args.res_dir):
        if f.startswith("."):
            continue

        f_split = f.split("_")
        if args.data == "cifar":
            if f_split[0] == "only":
                experiment = "_".join(f_split[:2])
                approach_id = 2
            else:
                experiment = f_split[0]
                approach_id = 1
            total_classes = 100
        elif args.data == "tiny_imagenet":
            experiment = "_".join(f_split[:2])
            approach_id = 2
            total_classes = 200
        else:
            raise NotImplementedError

        approach = f_split[approach_id]
        seed = f_split[approach_id + 1]

        if f_split[approach_id + 2] == "no":
            augment = "no_augment"
        else:
            augment = f_split[approach_id + 2]

        f_full_path = os.path.join(args.res_dir, f)

        with open(f_full_path) as ifp:
            lines = ifp.readlines()

            if experiment == "only_cifar100" or experiment == "tiny_imagenet":
                last_acc = [float(s) for s in lines[-1].split()]

                task_size = total_classes // len(lines)

                if np.sum(last_acc) > 0:
                    # finished
                    avg_all_acc = np.mean(last_acc)

                    current_acc = []
                    for t, l in enumerate(lines):
                        current_acc.append([float(s) for s in lines[t].split()][t])

                    avg_all_bwt = np.mean(np.array(last_acc[:-1]) - np.array(current_acc[:-1]))

                else:
                    print("{}_{}_{}_{}_{} not finished".format(approach, experiment, task_size, augment, seed))
                    continue

            else:
                raise NotImplementedError

        res_dict["{}_{}_{}_{}".format(approach, experiment, task_size, augment)].append({"avg_all_acc": avg_all_acc,
                                                               "avg_all_bwt": avg_all_bwt})

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
        print("avg_all_acc: {:.1f} \\pm {:.1f}".format(all_runs_avg_all_acc_mu * 100, all_runs_avg_all_acc_std * 100))

        all_runs_avg_all_bwt_mu = np.mean(all_runs_avg_all_bwt)
        all_runs_avg_all_bwt_std = np.std(all_runs_avg_all_bwt)
        print("avg_all_bwt: {:.1f} \\pm {:.1f}".format(all_runs_avg_all_bwt_mu * 100, all_runs_avg_all_bwt_std * 100))
