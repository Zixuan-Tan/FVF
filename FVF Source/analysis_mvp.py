# %%
from datetime import timedelta
import difflib
from math import ceil
from joblib import Parallel, delayed
from loguru import logger
import orjson
import pandas as pd
from scipy.stats import norm
from utils import codeparser, linguist, diff, repo_manager
from utils.gitw import Repo, GitError
import test_repo
from test_repo import vul_sig_repos, target2repo
from fvf import get_file_diff_seq, get_func_diff_seq, vdf_judge


def calculate_sample_size(
    population_size, confidence_level=0.95, confidence_interval=0.05
):
    Z = norm.ppf(1 - (1 - confidence_level) / 2)
    p = 0.5
    E = confidence_interval
    n = (Z**2 * p * (1 - p)) / (E**2)
    if population_size is not None:
        n = (n * population_size) / (n + population_size - 1)
    return ceil(n)


def read_jsonl(file_path):
    with open(file_path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def find_commit(commit_id):
    for name, repo in vul_sig_repos.items():
        try:
            return name, repo, repo.commit(commit_id)
        except GitError as e:
            pass
    return None, None, None


def get_function(file_path, func_name):
    with open(file_path, "rb") as f:
        src = f.read()
    language = linguist.detect_code_language(src, file_path)
    funcs = codeparser.extract_functions(src, language)
    for func in funcs:
        if func.name == func_name:
            return func
    return None


def export_mvp_result():
    projects = pd.read_csv("mvp_fast_matcher_input.csv")

    dataset = []

    for idx, row in projects.iterrows():
        project = row["target"]
        version = row["version"]

        data = read_jsonl(f"data/mvp_result/mvp_{project}_{version}.jsonl")
        df = pd.DataFrame(data)

        def filter_bad(row):
            return not any(
                bad_vul in row["train_func_sig"]["src_func_path"]
                for bad_vul in [
                    "nfsd_break_deleg_cb",
                    "tctx_inflight",
                    "check_leaf_item",
                ]
            )

        df = df[df.apply(filter_bad, axis=1)]
        print(idx, row)
        print("Result size:", len(df))

        sample = df

        sample.to_csv(
            f"mvp_result/mvp_{project}_{version}_for_label_v2.csv",
            index=False,
        )

        for idx, row in sample.iterrows():
            cve_id = row["train_vul_id"]
            vulpatch_commit_id = row["train_commit_id"]
            vul_sig = row["train_func_sig"]
            vul_func_path, vul_func_name = vul_sig["src_func_path"].split(":")

            n, r, c = find_commit(vulpatch_commit_id)
            if not c:
                logger.error(f"Commit {vulpatch_commit_id} not found")
                continue

            bf = c.before_file(vul_func_path)
            f = c.file(vul_func_path)
            try:
                bm = bf.method(vul_func_name)
            except GitError as e:
                logger.error(e)
                continue
            # m = f.method(vul_func_name)

            project_sig = row["test_func_sig"]
            project_func_path, project_func_name = project_sig["path"].split(":")
            project_func_path = project_func_path.split("/", 3)[-1]
            project_repo = target2repo[project + "-" + version]

            pc = project_repo.commit(version)
            pf = pc.file(project_func_path)
            try:
                pm = pf.method(project_func_name)
            except GitError as e:
                logger.error(e)
                continue

            item = {
                "vul": {
                    "vulnId": cve_id,
                    "project": n,
                    "vulpatch_commit_id": vulpatch_commit_id,
                    "vul_func_path": vul_func_path,
                    "name": vul_func_name,
                    "funcBody": bm.code,
                    "startLine": bm.start_line,
                    "endLine": bm.end_line,
                },
                "src": {
                    "project": project,
                    "version": version,
                    "project_func_path": project_func_path,
                    "name": project_func_name,
                    "funcBody": pm.code,
                    "startLine": pm.start_line,
                    "endLine": pm.end_line,
                },
            }
            dataset.append(item)

    open("mvp_result_v2.json", "wb").write(orjson.dumps(dataset))

    csvdataset = []

    for item in dataset:
        csvdataset.append(
            {
                "vulnId": item["vul"]["vulnId"],
                "project": item["vul"]["project"],
                "src_project": item["src"]["project"],
                "vulpatch_commit_id": item["vul"]["vulpatch_commit_id"],
                "src_version": item["src"]["version"],
                "vul_func_path": item["vul"]["vul_func_path"],
                "src_func_path": item["src"]["project_func_path"],
                "vul_func_name": item["vul"]["name"],
                "src_func_name": item["src"]["name"],
                "vul_func": item["vul"]["funcBody"],
                "src_func": item["src"]["funcBody"],
            }
        )

    csvdataset = pd.DataFrame(csvdataset)

    csvdataset.to_csv("mvp_result_v2.csv", index=False)


# export_mvp_result()

# %%
csvdataset = pd.read_csv("mvp_result_v2.csv")


def judge(
    trepo,
    tfile_path,
    tfunc_name,
    tcommit_id,
    vrepo,
    vcommit_id,
    vfile_path,
    vfunc_name,
):
    vcommit = vrepo.commit(vcommit_id)
    vfix_date = vcommit.author_date - timedelta(days=1)
    vfunc_code = vcommit.before_file(vfile_path).method(
        vfunc_name, _remove_comments=True
    )
    ffunc_code = vcommit.file(vfile_path).method(vfunc_name, _remove_comments=True)

    vul_seq = []
    unidiff = list(
        difflib.unified_diff(
            vfunc_code.code_lines,
            ffunc_code.code_lines,
            lineterm="",
        )
    )
    vul_sdiff = diff.extract_add_del_lines(unidiff)
    if vul_sdiff:
        vul_seq.append(vul_sdiff)

    if vul_seq == []:
        # what? no fix?
        return "err", -1, -1, -1

    tcommits, tfunc_seq = get_func_diff_seq(
        trepo, tfile_path, tfunc_name, tcommit_id, vfix_date
    )
    judge = vdf_judge(vul_seq, tfunc_seq, False, True, False)
    if "fixed" in judge:
        return judge

    # double check
    tcommits, tfile_seq = get_file_diff_seq(
        trepo, tfile_path, tfunc_name, tcommit_id, vfix_date
    )
    return vdf_judge(vul_seq, tfile_seq, False, True, False)


def process(idx, row):
    try:
        result = judge(
            trepo=target2repo[row["src_project"] + "-" + row["src_version"]],
            tfile_path=row["src_func_path"],
            tfunc_name=row["src_func_name"],
            tcommit_id=row["src_version"],
            vrepo=vul_sig_repos[row["project"]],
            vcommit_id=row["vulpatch_commit_id"],
            vfile_path=row["vul_func_path"],
            vfunc_name=row["vul_func_name"],
        )
        # logger.info(
        #     "Src: {} {} {}\nVuln: {} {} {} {} {}\nResult: {}",
        #     row["src_project"] + "-" + row["src_version"],
        #     row["src_func_path"],
        #     row["src_func_name"],
        #     row["vulnId"],
        #     row["project"],
        #     row["vulpatch_commit_id"],
        #     row["vul_func_path"],
        #     row["vul_func_name"],
        #     result,
        # )

        return idx, result[0]
    except Exception as e:
        # logger.error(e)
        return idx, "err"


def run_fvf():
    if "result" not in csvdataset.columns:
        csvdataset["result"] = None

    tasks = []

    for idx, row in csvdataset.iterrows():
        if row["result"]:
            continue

        # bad matches
        if row["vul_func_name"] in {
            "nfsd_break_deleg_cb",
            "tctx_inflight",
            "check_leaf_item",
        }:
            continue

        task = delayed(process)(idx, row.to_dict())
        tasks.append(task)

    for idx, res in Parallel(n_jobs=32, verbose=10, return_as="generator")(tasks):
        print(idx, res)
        csvdataset.loc[idx, "result"] = res
        csvdataset.to_csv("mvp_result_v2.csv", index=False)


# run_fvf()

# %%
judge(
    target2repo["linux-v5.15.105"],
    "net/9p/trans_xen.c",
    "xen_9pfs_front_free",
    "v5.15.105",
    vul_sig_repos["linux"],
    "ea4f1009408efb4989a0f139b70fb338e7f687d0",
    "net/9p/trans_xen.c",
    "xen_9pfs_front_free",
)

# %%
judge(
    target2repo["linux-oh-2f4ecbf3"],
    "io_uring/io_uring.c",
    "io_prep_async_work",
    "2f4ecbf3",
    vul_sig_repos["linux"],
    "4379bf8bd70b5de6bba7d53015b0c36c57a634ee",
    "fs/io_uring.c",
    "io_prep_async_work",
)

# %%
"v5.15.105" in target2repo["linux-v5.15.105"].commit(
    "ea4f1009408efb4989a0f139b70fb338e7f687d0"
).tags_contain

# %%
test_repo.linuxstable.commit("4379bf8bd70b5de6bba7d53015b0c36c57a634ee").tags_contain

# %%


def load_label():
    labeled = pd.read_excel("mvp_result_labeled.xlsx")

    for idx, row in labeled.iterrows():
        find = labeled[
            (labeled["vulnId"] == row["vulnId"])
            & (labeled["project"] == row["project"])
            & (labeled["src_project"] == row["src_project"])
            & (labeled["vulpatch_commit_id"] == row["vulpatch_commit_id"])
            & (labeled["src_version"] == row["src_version"])
            & (labeled["vul_func_path"] == row["vul_func_path"])
            & (labeled["src_func_path"] == row["src_func_path"])
            & (labeled["vul_func_name"] == row["vul_func_name"])
            & (labeled["src_func_name"] == row["src_func_name"])
            & (labeled["vul_func"] == row["vul_func"])
            & (labeled["src_func"] == row["src_func"])
        ]
        find = find.iloc[0]
        csvdataset.loc[idx, "label"] = find["label"]

    csvdataset.to_excel("mvp_result_final.xlsx", index=False)


# %%
if __name__ == "__main__":
    import fire

    fire.Fire()
