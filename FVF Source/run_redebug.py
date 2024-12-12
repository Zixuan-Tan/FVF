import os
import tempfile
import subprocess as sp

import fire
from util import escape
from loguru import logger
from joblib import Parallel, delayed
from utils.gitw import TempRepo
from project_list import projects


PYTHON2 = "python2"
REDBEUGPY = "redebug/redebug.py"


def run_redebug(src_path, patch_path, result_path):
    out = tempfile.NamedTemporaryFile(delete=False)
    err = tempfile.NamedTemporaryFile(delete=False)
    ret = sp.call(
        f"{PYTHON2} {REDBEUGPY} {patch_path} {src_path} {result_path}",
        shell=True,
        stdout=out,
        stderr=err,
    )
    if ret == 0:
        os.remove(out.name)
        os.remove(err.name)

    return ret, (out.name, err.name)


def clean(target, version):
    result = f"./result/redebug_{target}_{escape(version)}.jsonl"
    if os.path.exists(result):
        os.unlink(result)


def process_row(target, version, src_path, patch_dir):
    """Checkout and run redebug"""
    result = f"./result/redebug_{target}_{escape(version)}.jsonl"
    if os.path.exists(result):
        return

    logger.info(f"processing {target} {version}")

    with TempRepo(src_path, version) as src_clone:
        ret, (out, err) = run_redebug(src_clone, patch_dir, result)

    if ret != 0:
        logger.info(f"error in {target} {version}: {ret} {out} {err}")


def run_all():
    tasks = []
    for row in projects:
        target = row["target"]
        version = row["version"]
        src_path = row["project_dir"]
        patch_dir = row["patch_dir"]
        tasks.append(delayed(process_row)(target, version, src_path, patch_dir))

    Parallel(n_jobs=3)(tasks)


if __name__ == "__main__":
    fire.Fire()
