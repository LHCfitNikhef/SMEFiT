from os import listdir
from os import remove

import json
import argparse

from progress.bar import Bar

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

    for k in fits:
        coeffs = []
        vals = {}

        bar = Bar(r"Reading replica", max=len(listdir(k)) - 1)
        for filename in listdir(k):
            if filename.startswith("SMEFT_coeffs_") is False or filename.endswith(
                "_0.txt"
            ):
                continue

            file = open(f"{k}/{filename}", "r")
            data = file.readlines()
            if filename.endswith("_1.txt"):
                coeffs = [str(i) for i in data[0].split()]
                for c in coeffs:
                    if c not in vals.keys():
                        vals[c] = []
            # else:
            # for op in list(data[0]):
            #    if op not in coeffs:
            #        raise UserWarning(f"Fonund a new op in replica, {filename}: {op}")
            temp = dict(zip(coeffs, [float(i) for i in data[1].split()]))
            for c in coeffs:
                vals[c].append(temp[c])
            file.close()

            # cleaning
            #remove(f"{k}/{filename}")
            bar.next()
        bar.finish()
        with open(f"{k}/postetior.json", "w") as f:
            json.dump(vals, f)
