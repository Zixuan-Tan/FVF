import json
import os
import shutil
import fire
import run_vuddy_util
from run_vuddy_util import escape
from loguru import logger
from utils.gitw import TempRepo
from joblib import Parallel, delayed


def clean(target, version):
    output = f"./result/vuddy_{target}_{escape(version)}.jsonl"
    if os.path.exists(output):
        os.unlink(output)

    exploded = f"./vuddy_tmp/{target}_{escape(version)}/"
    if os.path.exists(exploded):
        shutil.rmtree(exploded)


def run_vuddy(target, src_path, version):
    output = f"./result/vuddy_{target}_{escape(version)}.jsonl"
    if os.path.exists(output):
        return

    exploded = f"./vuddy_tmp/{target}_{escape(version)}/"
    if not os.path.exists(exploded):
        with TempRepo(src_path, version) as src_clone:
            run_vuddy_util.explode(src_clone, exploded)

    basename = os.path.basename(exploded.rstrip("/"))
    hidx = os.path.join(exploded, "hidx", f"hashmark_4_{basename}.hidx")
    if not os.path.exists(hidx):
        ret, param = run_vuddy_util.run_hmark(exploded)
        if ret != 0:
            logger.info(f"error in {src_path}: {ret} {param}")
            return -1

        run_vuddy_util.patch_hidx(hidx)

    res, params = run_vuddy_util.upload_hidx(hidx)
    if res != 0:
        resp = params
        logger.info(f"error in {target}: {res} {resp.request.url} {resp.status_code}")  # type: ignore
        return -2
    tree_result = params["result"]  # type: ignore
    with open(output, "w") as f:
        for entry in tree_result:
            json.dump(entry, f)
            f.write("\n")


def run_all(projects):
    Parallel()(
        delayed(run_vuddy)(row["target"], row["project_dir"], row["version"])
        for row in projects
    )


if __name__ == "__main__":
    fire.Fire()
