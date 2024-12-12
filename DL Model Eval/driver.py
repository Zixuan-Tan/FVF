import os
import sys
import tempfile
import numpy as np
import pandas as pd
from itertools import zip_longest
import multiprocessing as mp
from tqdm import tqdm

NO_PADDING = object()


def chunked(iterable, n=1000, padding=NO_PADDING, progress=False):
    """return iterable in chunks of at most n items"""
    args = [iter(iterable)] * n

    if padding is NO_PADDING:
        ret = (filter(None, chunk) for chunk in zip_longest(*args))
    else:
        ret = zip_longest(*args, fillvalue=padding)

    if progress:
        from tqdm import tqdm

        return tqdm(ret)

    return ret


df = pd.read_parquet(os.path.dirname(__file__) + "/data-input.parquet")

model = sys.argv[1]

if model == "linevul":
    from linevul.main import test as linevul_test

    all_labels = []
    for chunk in chunked(df.iterrows(), n=200000, progress=True):
        codes = [row["code"] for idx, row in chunk]
        labels = linevul_test(codes)
        all_labels.extend(labels)
    np.save("linevul.npy", np.array(all_labels))

if model == "devign" or model == "ivdetect":
    sys.path.append(os.path.join(os.path.dirname(__file__), "devign"))
    from devign.main import run_test_inner as devign_test
    from devign.util import run_joern

    sys.path.append(os.path.join(os.path.dirname(__file__), "ivdetect"))
    from ivdetect.ivmain import run_test_inner as ivdetect_test

    devign_all_labels = []
    ivdetect_all_labels = []
    skip = 0
    if os.path.exists("devign-checkpoint.npy"):
        devign_all_labels = list(np.load("devign-checkpoint.npy"))
        ivdetect_all_labels = list(np.load("ivdetect-checkpoint.npy"))
        skip = len(devign_all_labels)

    for chunk in chunked(df.iterrows(), n=200, progress=True):
        codes = [row["code"] for idx, row in chunk]

        if skip > 0:
            print(f"skipping {len(codes)} samples")
            skip -= len(codes)
            continue

        with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
            print(d)
            dataset = []
            for i, src in enumerate(codes):
                path = os.path.join(d, f"{i}.c")
                with open(path, "w") as f:
                    f.write(src)
                dataset.append({"id": i, "path": path})

            with mp.Pool() as pool:
                pool.map(
                    run_joern,
                    tqdm([d["path"] for d in dataset], desc="run joern"),
                    chunksize=1,
                )

            devign_labels = devign_test(dataset)
            ivdetect_labels = ivdetect_test(dataset)

        devign_all_labels.extend(devign_labels)
        ivdetect_all_labels.extend(ivdetect_labels)

        np.save("devign-checkpoint.npy", np.array(devign_all_labels))
        np.save("ivdetect-checkpoint.npy", np.array(ivdetect_all_labels))

    np.save("devign.npy", np.array(devign_all_labels))
    np.save("ivdetect.npy", np.array(ivdetect_all_labels))

if model == "vulberta":
    from vulberta.driver import test as vulberta_test

    all_cnn_labels = []
    all_mlp_labels = []
    for chunk in chunked(df.iterrows(), n=200000, progress=True):
        codes = [row["code"] for idx, row in chunk]
        cnn_labels, mlp_labels = vulberta_test(codes)
        all_cnn_labels.extend(cnn_labels)
        all_mlp_labels.extend(mlp_labels)
    np.save("vulberta_cnn.npy", np.array(all_cnn_labels))
    np.save("vulberta_mlp.npy", np.array(all_mlp_labels))
