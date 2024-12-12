import numpy as np
import os
import json
import tempfile
import subprocess

DIR = os.path.dirname(__file__)
TEST_SCRIPT = os.path.join(DIR, "run_test.sh")


def test(func_codes: list[str]) -> list[bool]:
    with tempfile.NamedTemporaryFile(dir="/dev/shm/", suffix=".jsonl") as f:
        for func_code in func_codes:
            lines = func_code.split("\n")
            lines = [line.lstrip() for line in lines]
            func_code = "\n".join(lines)
            f.write(json.dumps({"func": func_code, "target": 0}).encode())
            f.write(b"\n")
        f.flush()

        subprocess.check_call(
            f"{TEST_SCRIPT} {f.name}",
            shell=True,
            cwd=DIR,
        )

    y_preds = np.load(DIR + "/test_y_preds.npy")
    os.unlink(DIR + "/test_y_preds.npy")
    os.unlink(DIR + "/test_y_trues.npy")

    return y_preds.tolist()
