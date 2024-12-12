import numpy as np
import os
import json
import tempfile
import subprocess

DIR = os.path.dirname(__file__)


def test(func_codes):
    with tempfile.NamedTemporaryFile(dir="/dev/shm/", suffix=".jsonl") as f:
        for func_code in func_codes:
            obj = {"project": "xx", "commit_id": "xx", "func": func_code, "target": 0}
            f.write(json.dumps(obj).encode())
            f.write(b"\n")
        f.flush()

        subprocess.check_call(
            args=f"./.venv/bin/python vulberta_test.py {f.name}",
            shell=True,
            cwd=DIR,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
        )

    cnn_y_preds = np.load(DIR + "/cnn_test_y_preds.npy")
    mlp_y_preds = np.load(DIR + "/mlp_test_y_preds.npy")
    os.unlink(DIR + "/cnn_test_y_preds.npy")
    os.unlink(DIR + "/mlp_test_y_preds.npy")

    return cnn_y_preds.tolist(), mlp_y_preds.tolist()
