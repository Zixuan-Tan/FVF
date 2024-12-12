import difflib
from functools import lru_cache
import git
from loguru import logger
from typing import List, Optional, Tuple
from utils import diff, codeparser
from utils.gitw import Repo
from utils.diff import SimpleDiff
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def fvf_diff(old: List[str], new: List[str]) -> Optional[SimpleDiff]:
    udiff = list(difflib.unified_diff(old, new, lineterm="", n=0))
    del_lines, add_lines = [], []
    for line in udiff[3:]:
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip():
            add_lines.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---") and line[1:].strip():
            del_lines.append(line[1:].strip())

    return SimpleDiff(del_lines, add_lines)


sm = SmoothingFunction()


@lru_cache(maxsize=1024)
def is_similar_line(line1: str, line2: str):
    if line1 == line2:
        return True
    tokens1 = line1.split()
    tokens2 = line2.split()
    return (
        sentence_bleu(
            [tokens1],
            tokens2,
            smoothing_function=sm.method1,
            weights=(0, 1),
        )
        >= 0.95
    )  # type: ignore


def line_in_patch(line: str, patch: List[str]):
    return any(is_similar_line(line, p) for p in patch)


def is_subdiff(diff1: SimpleDiff, diff2: SimpleDiff, loose=False):
    if not diff1:
        logger.warning("empty diff")

    if loose:
        if all(line_in_patch(x, diff2.del_lines) for x in diff1.del_lines) or all(
            line_in_patch(x, diff2.add_lines) for x in diff1.add_lines
        ):
            return True
    else:
        if all(line_in_patch(x, diff2.del_lines) for x in diff1.del_lines) and all(
            line_in_patch(x, diff2.add_lines) for x in diff1.add_lines
        ):
            return True

    return False


def is_subsequence(subseq: List[SimpleDiff], seq: List[SimpleDiff], loose=False):
    if len(subseq) == 0:
        raise ValueError("subseq is empty")

    sub_idx = 0
    for idx, di in enumerate(seq):
        while is_subdiff(subseq[sub_idx], di, loose):
            sub_idx += 1
            if sub_idx == len(subseq):
                return idx

    return -1


def test_is_subsequence():
    sub = [
        SimpleDiff(["a"], ["b"]),
        SimpleDiff(["e"], ["f"]),
    ]
    seq = [
        SimpleDiff(["a", "a1"], ["b", "b1"]),
        SimpleDiff(["c"], ["d"]),
        SimpleDiff(["e", "c"], ["f"]),
    ]
    seq2 = [
        SimpleDiff(["a", "e", "a1"], ["b", "f", "b1"]),
    ]
    assert is_subsequence(sub, seq) >= 0
    assert is_subsequence(sub, seq2) >= 0


def is_partial_subsequence(sub, seq):
    for subdiff in sub:
        for idx, seqdiff in enumerate(seq):
            if is_subdiff(subdiff, seqdiff):
                return idx

    return -1


def reverse_diff_sequence(diff_sequence: List[SimpleDiff]):
    return list(reversed([sdiff.reverse() for sdiff in diff_sequence]))


def vdf_judge(
    vul_change_log: List[SimpleDiff],
    target_change_log: List[SimpleDiff],
    check_partial_fix=True,
    check_rollback=True,
    check_partial_rollback=True,
):
    fixed = 0  # not fixed
    if (idx := is_subsequence(vul_change_log, target_change_log)) >= 0:
        fixed = 2  # full fixed
    elif (
        check_partial_fix
        and (idx := is_partial_subsequence(vul_change_log, target_change_log)) >= 0
    ):
        fixed = 1  # partial fixed

    rollback = 0
    ridx = -1
    midx = -1
    if check_rollback:
        rev_vul_change_log = reverse_diff_sequence(vul_change_log)
        if (ridx := is_subsequence(rev_vul_change_log, target_change_log)) >= 0:
            rollback = 2  # rollback detected

        if check_partial_rollback:
            if (
                midx := is_subsequence(
                    rev_vul_change_log, target_change_log, loose=True
                )
            ) >= 0:
                rollback = 1  # partial_rollback detected

    if fixed == 2:
        judge_result = "fixed"
    elif fixed == 1:
        judge_result = "partial_fixed"
    else:
        judge_result = "is_vul"

    if rollback == 2:
        judge_result += "+rollback"
    elif rollback == 1:
        judge_result += "+partial_rollback"

    return judge_result, idx, ridx, midx


def parse_diff(output: str):
    output_lines = output.strip().splitlines()
    commit_id = output_lines[0]

    a_path = ""
    for line in output_lines:
        if line.startswith("---"):
            a_path = line[4:]
            if a_path.startswith("a"):
                a_path = a_path[2:]
            break

    b_path = ""
    for line in output_lines:
        if line.startswith("+++"):
            b_path = line[4:]
            if b_path.startswith("b"):
                b_path = b_path[2:]
            break

    for idx, line in enumerate(output_lines):
        if line.startswith("@@"):
            break

    diff_hunks = diff.parse_hunks(output_lines[idx:], a_path, b_path)

    return commit_id, a_path, b_path, diff_hunks


def get_func_diff_seq(
    repo: Repo,
    file_path,
    func_name,
    from_rev,
    after_date,
    backtrack_times=9999999,
) -> Tuple[List[str], List[SimpleDiff]]:
    func_name_last = func_name.rsplit(":", 1)[-1]
    try:
        gitlog_output = repo.git.log(
            # f"--pretty=format:%H", "-s",
            "--pretty=format:__mark_cmt__%H",
            f"-L:{func_name_last}:{file_path}",
            f"--after={after_date.strftime('%Y-%m-%d')}",
            f"-{10*backtrack_times}",
            from_rev,
        ).strip()
    except git.GitCommandError as ex:
        logger.warning(f"git log failed: {ex}")
        return [], []

    gitlog_output = gitlog_output.split("__mark_cmt__")
    if gitlog_output:
        gitlog_output = gitlog_output[1:]

    func_histo_commit_ids = []
    func_histo_file_paths = []
    func_histo_func_names = []
    for output in gitlog_output:
        cur_func_name = (
            func_histo_func_names[-1][0] if func_histo_func_names else func_name
        )

        if "\n" not in output.strip():
            commit_id = output.strip()
            func_histo_commit_ids.append(commit_id)
            func_histo_file_paths.append((file_path, file_path))
            func_histo_func_names.append((cur_func_name, cur_func_name))
            continue

        commit_id, a_path, b_path, hunks = parse_diff(output)
        func_histo_commit_ids.append(commit_id)
        func_histo_file_paths.append((a_path, b_path))
        pattern = f" {cur_func_name}("
        hunk = hunks[0]
        if (
            hunk.lines[0].is_del
            and pattern not in hunk.lines[0].content
            and hunk.lines[1].is_add
            and pattern in hunk.lines[1].content
        ):
            # func name changed
            before_func_name = codeparser.extract_function_name_regex(
                hunk.lines[0].content
            )
            func_histo_func_names.append((before_func_name, cur_func_name))
        else:
            func_histo_func_names.append((cur_func_name, cur_func_name))

    func_histo_commit_ids = func_histo_commit_ids[::-1]  # 从旧到新遍历
    func_histo_file_paths = func_histo_file_paths[::-1]
    func_histo_func_names = func_histo_func_names[::-1]

    final_func_histo_commit_ids = []
    func_diff_seq = []
    for commit_id, this_file_path, this_func_name in zip(
        func_histo_commit_ids, func_histo_file_paths, func_histo_func_names
    ):
        try:
            func1 = (
                repo.commit(commit_id + "~1")
                .file(this_file_path[0])
                .remove_comments()
                .method(this_func_name[0])
            ).code_lines
        except Exception as e:
            logger.error(
                "Func not found: {} {} {} {}",
                commit_id,
                this_file_path[0],
                this_func_name[0],
                e,
            )
            func1 = []
        try:
            func2 = (
                repo.commit(commit_id)
                .file(this_file_path[1])
                .remove_comments()
                .method(this_func_name[1])
            ).code_lines
        except Exception as e:
            logger.error(
                "Func not found: {} {} {} {}",
                commit_id,
                this_file_path[1],
                this_func_name[1],
                e,
            )
            func2 = []

        sdiff = fvf_diff(func1, func2)
        if sdiff:
            func_diff_seq.append(sdiff)
            final_func_histo_commit_ids.append(commit_id)

            if len(final_func_histo_commit_ids) >= backtrack_times:
                break

    return final_func_histo_commit_ids, func_diff_seq


def get_file_diff_seq(
    repo: Repo,
    file_path,
    func_name,
    from_rev,
    after_date,
    backtrack_times=9999999,
) -> Tuple[List[str], List[SimpleDiff]]:
    try:
        gitlog_output = repo.git.log(
            # f"-s",
            "--pretty=format:__mark_cmt__%H",
            "--patch",
            f"--after={after_date.strftime('%Y-%m-%d')}",
            f"-{10*backtrack_times}",
            from_rev,
            "--",
            file_path,
        )
    except git.GitCommandError as ex:
        logger.warning(f"git log failed: {ex}")
        return [], []

    gitlog_output = gitlog_output.split("__mark_cmt__")
    if gitlog_output:
        gitlog_output = gitlog_output[1:]

    file_histo_commit_ids = []
    file_histo_file_paths = []
    file_histo_func_names = []
    for output in gitlog_output:
        cur_func_name = (
            file_histo_func_names[-1][0] if file_histo_func_names else func_name
        )

        if "\n" not in output.strip():
            commit_id = output.strip()
            file_histo_commit_ids.append(commit_id)
            file_histo_file_paths.append((file_path, file_path))
            file_histo_func_names.append((cur_func_name, cur_func_name))
            continue

        commit_id, a_path, b_path, hunks = parse_diff(output)
        file_histo_commit_ids.append(commit_id)
        file_histo_file_paths.append((a_path, b_path))
        pattern = f" {cur_func_name}("
        for hunk in hunks:
            if (
                hunk.lines[0].is_del
                and pattern not in hunk.lines[0].content
                and hunk.lines[1].is_add
                and pattern in hunk.lines[1].content
            ):
                # func name changed
                a_func_name = codeparser.extract_function_name_regex(
                    hunk.lines[0].content
                )
                file_histo_func_names.append((a_func_name, cur_func_name))
                break
        else:
            file_histo_func_names.append((cur_func_name, cur_func_name))

    file_histo_commit_ids = file_histo_commit_ids[::-1]  # 从旧到新遍历
    file_histo_file_paths = file_histo_file_paths[::-1]
    file_histo_func_names = file_histo_func_names[::-1]

    final_func_histo_commit_ids = []
    func_diff_seq = []
    for commit_id, this_file_path, this_func_name in zip(
        file_histo_commit_ids, file_histo_file_paths, file_histo_func_names
    ):
        try:
            func1 = (
                repo.commit(commit_id + "~1")
                .file(this_file_path[0])
                .remove_comments()
                .method(this_func_name[0])
            ).code_lines
        except Exception as e:
            logger.error(
                "Func not found: {} {} {} {}",
                commit_id,
                this_file_path[0],
                this_func_name[0],
                e,
            )
            func1 = []
        try:
            func2 = (
                repo.commit(commit_id)
                .file(this_file_path[1])
                .remove_comments()
                .method(this_func_name[1])
            ).code_lines
        except Exception as e:
            logger.error(
                "Func not found: {} {} {} {}",
                commit_id,
                this_file_path[1],
                this_func_name[1],
                e,
            )
            func2 = []

        sdiff = fvf_diff(func1, func2)
        if sdiff:
            func_diff_seq.append(sdiff)
            final_func_histo_commit_ids.append(commit_id)

            if len(final_func_histo_commit_ids) >= backtrack_times:
                break

    return final_func_histo_commit_ids, func_diff_seq
