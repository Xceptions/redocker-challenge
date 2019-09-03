import click
import numpy as np
import pandas as pd
import os


def _save_datasets(train, test, outdir):
    """Save data sets into nice directory structure"""
    train_file = "{}/train.csv".format(outdir)
    test_file = "{}/test.csv".format(outdir)
    with open(train_file, 'w') as train_path:
        print(train.to_csv(index=False), file=train_path)
    with open(test_file, 'w') as test_path:
        print(test.to_csv(index=False), file=test_path)



@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data = pd.read_csv(in_csv)
    split_1, split_2 = np.split(data, [int(.8*len(data))])
    train = pd.DataFrame(split_1)
    test = pd.DataFrame(split_2)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
