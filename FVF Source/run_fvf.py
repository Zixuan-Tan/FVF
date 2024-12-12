import os
import fire
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from datetime import datetime, timedelta
from utils.diff import SimpleDiff
from fvf import (
    get_file_diff_seq,
    get_func_diff_seq,
    vdf_judge,
)

# cleanup previous log
os.system("rm -f ./run_fvf.*.log")

logger.remove()
logger.add("run_fvf.log", rotation="10MB")


def get_fvf_result(
    trepo,
    file_path: str,
    func_name: str,
    version: str,
    vul_change_log: list[SimpleDiff],
    patch_time: datetime,
):
    hist_commit_ids, target_change_log = get_func_diff_seq(
        trepo,
        file_path,
        func_name,
        version,
        patch_time - timedelta(days=1),
    )
    judge_result, idx, ridx, midx = vdf_judge(
        vul_change_log,
        target_change_log,
    )
    if "fixed" in judge_result:
        return judge_result

    hist_commit_ids, target_change_log = get_file_diff_seq(
        trepo,
        file_path,
        func_name,
        version,
        patch_time - timedelta(days=1),
    )
    judge_result, idx, ridx, midx = vdf_judge(
        vul_change_log,
        target_change_log,
    )
    return judge_result


def run_fvf(
    _id,
    cve_id,
    commit_id,
    vul_file_path,
    vul_func_name,
    target,
    version,
    file_path,
    func_name,
):
    try:
        vul_change_log, patch_time = query_cve_patch(
            cve_id, commit_id, vul_file_path, vul_func_name
        )
        assert (
            vul_change_log
        ), f"No patch log: {cve_id} {commit_id} {vul_file_path}:{vul_func_name}"

        trepo = get_project_dir(target, version)
        assert trepo, f"Cannot find {target} {version}"
        trepo = Repo(trepo)

        judge_result = get_fvf_result(
            trepo,
            file_path,
            func_name,
            version,
            vul_change_log,
            patch_time,
        )
        return _id, judge_result
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        logger.error("{} {}", type(e), e)
        return _id, "error"


def test(
    cve_id,
    commit_id,
    vul_file_path,
    vul_func_name,
    target,
    version,
    file_path,
    func_name,
):
    _, result = run_fvf(
        "",
        cve_id,
        commit_id,
        vul_file_path,
        vul_func_name,
        target,
        version,
        file_path,
        func_name,
    )
    return result


def get_vuddy_fvf_result():
    for d in vuddy.find():
        _id = (d["_id"],)
        cve_id = d["cve_id"]
        commit_id = d["commit_id"]
        project = d["project"]
        version = d["version"]
        vul_file_path = d["vul_fname"]  # only basename
        file_path = d["file_path"].replace("#", ".").rsplit("/", 1)[0].split("/", 1)[1]
        func_code = d["func_code"]

        try:
            func_name = tst.extract_functions(func_code, "C")[0].name
            assert func_name, f"Cannot find function name: {cve_id} {commit_id}"

            _id, judge_result = run_fvf(
                _id,
                cve_id,
                commit_id,
                vul_file_path,
                func_name,
                project,
                version,
                file_path,
                func_name,
            )
            vuddy.update_one(
                {"_id": d["_id"]},
                {"$set": {"fvf_result": judge_result}},
            )
        except Exception as e:
            logger.error("{} {}", type(e), e)


def get_mvp_fvf_result():
    mvp_results = []

    logger.info("Processing mvp, total {}", mvp.count_documents({}))

    for d in mvp.find():
        _id = d["_id"]
        cve_id = d["train_vul_id"]
        commit_id = d["train_commit_id"]
        project = d["project"]
        version = d["version"]
        vul_file_path = d["vul_file_path"]
        vul_func_name = d["vul_func_name"]
        file_path = d["file_path"]
        func_name = d["func_name"]

        result = delayed(run_fvf)(
            _id,
            cve_id,
            commit_id,
            vul_file_path,
            vul_func_name,
            project,
            version,
            file_path,
            func_name,
        )
        mvp_results.append(result)

    mvp_results = Parallel(n_jobs=16, backend="multiprocessing")(mvp_results)

    for _id, result in tqdm(mvp_results):
        mvp.update_one(
            {"_id": _id},
            {"$set": {"fvf_result": result}},
        )


def get_redebug_fvf_result():
    redebug_results = []

    logger.info("Processing redebug, total {}", redebug.count_documents({}))

    for d in redebug.find():
        _id = d["_id"]
        cve_id = d["cve_id"]
        commit_id = d["commit_id"]
        project = d["project"]
        version = d["version"]
        vul_file_path = d["vul_file_path"]  # only basename
        vul_func_name = d["vul_func_name"]
        file_path = d["file_path"]
        func_name = vul_func_name

        result = delayed(run_fvf)(
            _id,
            cve_id,
            commit_id,
            vul_file_path,
            vul_func_name,
            project,
            version,
            file_path,
            func_name,
        )
        redebug_results.append(result)

    redebug_results = Parallel(n_jobs=16, backend="multiprocessing")(redebug_results)

    for _id, result in tqdm(redebug_results):
        redebug.update_one(
            {"_id": _id},
            {"$set": {"fvf_result": result}},
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "vuddy": get_vuddy_fvf_result,
            "mvp": get_mvp_fvf_result,
            "redebug": get_redebug_fvf_result,
            "test": test,
        }
    )
