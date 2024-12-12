from datetime import timedelta
import difflib
from utils import diff
from fvf import (
    get_file_diff_seq,
    get_func_diff_seq,
    vdf_judge,
)
from test_repo import target2repo, vul_sig_repos


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


judge_result = judge(
    target2repo["linux-v5.15.105"],
    "mm/mincore.c",
    "mincore_pte_range",
    "v5.15.105",
    vul_sig_repos["linux"],
    "574823bfab82d9d8fa47f422778043fbb4b4f50e",
    "mm/mincore.c",
    "mincore_pte_range",
)

print(judge_result)
