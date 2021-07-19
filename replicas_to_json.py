# -*- coding: utf-8 -*-
import os

import json
import argparse

from progress.bar import Bar

def convert_to_json(k):
    coeffs = []
    vals = {}

    # VBS paper replicas were with a different convention,
    # need to add a multiplicative factor
    rescale_fact = 1.0
    if "VBS" in k:
        rescale_fact = 1/0.246 ** 2
        print("Rescale factor: ", rescale_fact)

    counterbar = Bar(r"Reading replica", max=len(os.listdir(k)) - 1)
    for i, filename in enumerate( os.listdir(k)):
        if filename.startswith("SMEFT_coeffs_") is False or ( filename.endswith(
            "_0.txt"
        ) and "VBS" not in k):
            continue

        file = open(f"{k}/{filename}", "r")
        data = file.readlines()
        if i == 0:
            coeffs = [str(i) for i in data[0].split()]
            for c in coeffs:
                if c not in vals:
                    vals[c] = []
        # sanity check
        for op in [str(i) for i in data[0].split()]:
            if op not in coeffs:
                raise UserWarning(f"Fonund a new op in replica, {filename}: {op}")

        temp = dict(zip(coeffs, [rescale_fact * float(i) for i in data[1].split()]))
        for c in coeffs:
            vals[c].append(temp[c])
        file.close()

        # cleaning
        os.remove(f"{k}/{filename}")
        counterbar.next()
    counterbar.finish()
    return vals

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fitcards",
        nargs="+",
        required=True,
        default=None,
        type=str,
        help="fit names",
    )

    fits = list(parser.parse_args().fitcards)

    for fit in fits:
        if "SNS" in fit:
            vals = {}
            # read replicas in each folder
            for op_dir in os.listdir(fit):
                if op_dir.startswith('O') is False:
                    continue
                temp = convert_to_json(f"{fit}/{op_dir}")
                # clean
                for op in os.listdir(fit):
                    if op != op_dir and op.startswith('O'):
                        temp.pop(op)
                vals.update(temp)
        else:
            vals = convert_to_json(fit)

        with open(f"{fit}/posterior.json", "w") as f:
            json.dump(vals, f)
