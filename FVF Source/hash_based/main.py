"""
Given a project and revision, clone the project and checkout the revision, then
return all functions in all source files in the revision.
"""

import difflib
import hashlib
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict
from datetime import timedelta
from functools import lru_cache
import orjson
from loguru import logger
from simhash import Simhash, SimhashIndex
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstractions import abst_method1, abst_method2, abst_method3
from fvf import get_file_diff_seq, get_func_diff_seq, vdf_judge
from utils.gitw import Repo
from utils import diff, linguist, codeparser
import vul_db


@lru_cache
def get_lsh_index(abst_method):
    col_name = f"abst{abst_method[-1]}_lsh"
    all_lsh = vul_db.execute(
        f"""SELECT func_id, {col_name} FROM vuln_func WHERE data_type=0"""
    )
    return SimhashIndex([(fnid, Simhash(int(lsh))) for fnid, lsh in all_lsh], k=3)


def do_scan(project_id, branch, abst_method, comp_method):
    logger.info("Processing {} {} {} {}", project_id, branch, abst_method, comp_method)

    repo = Repo.open_project(project_id)
    repo.checkout(branch)

    tcommit = repo.commit(branch)
    tcommit_id = tcommit.hexsha
    target_date = tcommit.author_date

    n_files = 0
    n_funcs = 0
    n_matches = 0

    results = []

    for file_path, lang in linguist.traverse_src_files(
        repo.working_tree_dir, ["C", "C++", "Java"], relative=True
    ):
        # print(file_path, lang)
        n_files += 1
        ext = os.path.splitext(file_path)[1]
        with open(repo.working_tree_dir + "/" + file_path, "rb") as f:
            src_code = f.read()

        for func in codeparser.extract_functions(src_code, lang):
            n_funcs += 1
            # export to file
            # export_name = f"{project_id}_{revision[:8]}_{file_path.replace('/','+')}_{func.start_line}{ext}"func
            # with open(outputdir + "/" + export_name, "wb") as f:
            #     f.write(func.code_bytes)

            # abstract
            abst_code = eval(abst_method)(func.code_bytes, lang)

            # filter short functions
            abst_code_join = b"".join(abst_code)
            if b"){}" in abst_code_join:
                continue
            if re.search(rb"\)\{return\w+;\}", abst_code_join):
                # publicVTYPEGSYM0(){returnGSYM1;}
                # VTYPEFNAME(VTYPE*FPARAM0){returnNUM;}
                continue

            # query
            if comp_method == "sha1":
                abst_sig = hashlib.sha1(abst_code_join).hexdigest()
                matches = vul_db.execute(
                    f"""
                    SELECT project_id, func_id, func FROM vuln_func WHERE abst{abst_method[-1]}_sha1 = %s
                """,
                    (abst_sig,),
                )
            else:
                abst_ft = [
                    (abst_code[i] + abst_code[i + 1] + abst_code[i + 2]).decode(
                        errors="ignore"
                    )
                    for i in range(0, len(abst_code) - 2)
                ]
                abst_lsh = Simhash(abst_ft, f=64)
                lsh_index = get_lsh_index(abst_method)
                near = lsh_index.get_near_dups(abst_lsh)
                matches = [
                    vul_db.execute(
                        """SELECT project_id, func_id, func FROM vuln_func WHERE func_id = %s""",
                        (fnid,),
                    )[0]
                    for fnid in near
                ]

            # fvf filter
            res_filtered = []
            for match in matches:
                vrepo = Repo.open_project(match[0])
                vfunc_id = match[1]
                vfunc_code = match[2]
                vcommit_id, _, vfile_path, vstart_line = vfunc_id.split(":")
                vfix_date = vrepo.commit(vcommit_id).author_date - timedelta(days=1)

                # the target function is elder than the vulnerable function
                # it's a true positive, really historical vulnerability.
                if target_date < vfix_date:
                    continue

                # # the vulnerable function is in the same file
                # # it's high likely a false positive
                # if file_path == vfile_path:
                #     continue

                # query fixed func
                ffunc_code = vul_db.execute(
                    """
                    SELECT vf1.func
                    FROM vuln_func vf1
                    WHERE (vf1.project_id, vf1.commit_id, vf1.vuln_ids, vf1.func_fullname) IN (
                        SELECT vf.project_id, vf.commit_id, vf.vuln_ids, vf.func_fullname
                        FROM vuln_func vf
                        WHERE vf.func_id = %s
                    ) AND vf1.data_type = 1
                """,
                    (vfunc_id,),
                )
                if not ffunc_code:
                    # the vulnerable function is deleted in the fix
                    continue
                ffunc_code = ffunc_code[0][0]

                vfunc_code = codeparser.remove_comments(vfunc_code, lang)
                ffunc_code = codeparser.remove_comments(ffunc_code, lang)

                vul_seq = []
                unidiff = list(
                    difflib.unified_diff(
                        codeparser.splitlines(vfunc_code),
                        codeparser.splitlines(ffunc_code),
                        lineterm="",
                    )
                )
                vul_sdiff = diff.extract_add_del_lines(unidiff)
                if vul_sdiff:
                    vul_seq.append(vul_sdiff)

                if vul_seq == []:
                    # what? no fix?
                    break

                tcommits, tfunc_seq = get_func_diff_seq(
                    repo.repo, file_path, func.name, tcommit_id, vfix_date
                )
                judge = vdf_judge(vul_seq, tfunc_seq, False, False, False)
                if "fixed" in judge:
                    # it's false alarm, fixed, not vulnerable
                    res_filtered.append(match)
                    continue

                tcommits, tfile_seq = get_file_diff_seq(
                    repo.repo, file_path, func.name, tcommit_id, vfix_date
                )
                judge = vdf_judge(vul_seq, tfile_seq, False, False, False)
                if "fixed" in judge:
                    # double check: it's false alarm, fixed, not vulnerable
                    res_filtered.append(match)
                    continue

            if not res_filtered:
                continue

            n_matches += len(res_filtered)

            # print
            # if len(res_filtered) > 0:
            #     logger.info("=" * 80)
            #     logger.info(
            #         f"Target benign function: {project_id} {revision}:{file_path}:{func.start_line}"
            #     )
            #     for ln in func.code_lines[:10]:
            #         logger.info(f"\t{ln}")
            #     # logger.info(abst_code_join[:100])
            #     logger.info("-" * 80)
            #     for res in res_filtered:
            #         logger.info(f"Similar vuln function: {res[0]} {res[1]}")
            #         for ln in res[2].splitlines()[:10]:
            #             logger.info(f"\t{ln}")

            results.append(
                {
                    "target": {
                        "func_id": f"{tcommit_id}:1:{file_path}:{func.start_line}",
                        "branch": branch,
                        "project_id": project_id,
                        "code": func.code,
                    },
                    "matches": [
                        {
                            "project_id": res[0],
                            "func_id": res[1],
                            "code": res[2],
                        }
                        for res in res_filtered
                    ],
                }
            )

    logger.success(
        "Done {} {} {} {}, {} files, {} functions, {} matches",
        project_id,
        branch,
        abst_method,
        comp_method,
        n_files,
        n_funcs,
        n_matches,
    )
    return results


def check_scan_save(project_id, branch, abst_method, comp_method):
    try:
        branch_commit_id = Repo.open_project(project_id).commit(branch).hexsha
    except (errors.NoRepoError, errors.NoCommitError) as e:
        logger.error(e)
        return

    res = vul_db.execute(
        f"""
        SELECT target_func_id
        FROM pseudo_similar 
        WHERE target_func_id LIKE '{branch_commit_id}:%'
        LIMIT 1
    """,
        conn=conn,
    )
    if len(res) > 0:
        return

    try:
        results = do_scan(project_id, branch, abst_method, comp_method)
    except (errors.NoRepoError, errors.NoCommitError) as e:
        logger.error(f"> {e}")
        return
    except Exception as e:
        logger.error(
            "Failed {} {} {} {} {}", project_id, branch, abst_method, comp_method, e
        )
        return

    for chunk in util.chunked(results, 100):
        vul_db.executemany(
            """
            INSERT INTO pseudo_similar (target_func_id, target_project_id, target_branch, target_code, vuln_func_id, abst_method, comp_method)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
            [
                (
                    r["target"]["func_id"],
                    project_id,
                    r["target"]["branch"],
                    r["target"]["code"],
                    m["func_id"],
                    abst_method,
                    comp_method,
                )
                for r in chunk
                for m in r["matches"]
            ],
            conn=conn,
        )


def process_project(project_id, branches):
    for branch, abst_method, comp_method in branches:
        check_scan_save(project_id, branch, abst_method, comp_method)


def collect_item(wd):
    res = []
    for p, d, files in os.walk(wd):
        for file in files:
            if file.endswith(".jsonl"):
                with open(os.path.join(p, file), "r") as fin:
                    for line in fin:
                        item = orjson.loads(line)
                        if item["fork"]:
                            continue
                        res.append(item)
    return res


task_list = defaultdict(list)

scaning_once_vuln = False
scaning_wild = False

if scaning_once_vuln:
    for (p,) in vul_db.execute("SELECT DISTINCT project_id FROM vuln_func"):
        try:
            repo = Repo.open_project(p)
            for branch in repo.remote_branches:
                if branch.name == "origin/HEAD":
                    continue
                branch_name = branch.name.split("/", 1)[1]
                # project_list.append((p, branch.name, "abst_method1", "lsh"))
                # project_list.append((p, branch.name, "abst_method2", "lsh"))
                task_list[p].append((branch_name, "abst_method3", "lsh"))
        except Exception as e:
            logger.error(f"> collecting {p} error: {e}")
            continue

if scaning_wild:
    for p, g in vul_db.execute("SELECT project_name, group_name FROM popular_projects"):
        project_id = gitparse.make_project_id(g, p)
        try:
            repo = Repo.open_project(project_id)
            for branch in repo.remote_branches:
                if branch.name == "origin/HEAD":
                    continue

                task_list[project_id].append((branch.name, "abst_method3", "lsh"))
        except Exception as e:
            logger.error(f"> collecting {p} error: {e}")
            continue

if not scaning_once_vuln and not scaning_wild:
    # test
    g = "xelerance"
    p = "openswan"
    project_id = gitparse.make_project_id(g, p)
    try:
        repo = Repo.open_project(project_id)
        for branch in repo.remote_branches:
            if branch.name == "origin/HEAD":
                continue

            task_list[project_id].append((branch.name, "abst_method3", "lsh"))
    except Exception as e:
        logger.error(f"> collecting {p} error: {e}")

conn = None


def init_db():
    global conn
    conn = vul_db.get_conn()


for project in task_list:
    workdir = Repo.get_project_path(project[0])
    if os.path.exists(f"{workdir}/.git/index.lock"):
        os.unlink(f"{workdir}/.git/index.lock")

with mp.Pool(32, initializer=init_db, maxtasksperchild=10) as pool:
    task_list_flat = [
        (project_id, versions) for project_id, versions in task_list.items()
    ]
    pool.starmap(process_project, tqdm(task_list_flat), chunksize=1)

# for project_id, revision, abst_method, comp_method in project_list:
