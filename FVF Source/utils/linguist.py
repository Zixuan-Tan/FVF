import os
import subprocess
from typing import List, Union
import orjson
import tempfile


def detect_code_language(src: str | bytes, hint_file_path: str):
    basename = os.path.basename(hint_file_path)
    if isinstance(src, str):
        src = src.encode("utf-8")

    with tempfile.NamedTemporaryFile(suffix=basename) as tmp_f:
        tmp_f.write(src)
        tmp_f.flush()

        output = subprocess.check_output(
            ["github-linguist", "-j", tmp_f.name],
            stderr=subprocess.STDOUT,
        )
        res = orjson.loads(output)
        return res[tmp_f.name]["language"]


def detect_file_language(file_path: str):
    with open(file_path, "rb") as f:
        src = f.read()
    return detect_code_language(src, file_path)


def guess_language_by_filename(path):
    if path.endswith((".cc", ".cpp", ".hh", ".hpp")):
        return "C++"
    elif path.endswith((".c", ".h")):
        return "C"
    elif path.endswith((".java", ".jar")):
        return "Java"
    return None


def traverse_files(dir_path=".", relative=False):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if relative:
                yield os.path.relpath(os.path.join(root, file), dir_path)
            else:
                yield os.path.join(root, file)


def traverse_src_files(
    dir_path: str, langs: Union[str, List[str]], relative: bool = False
):
    if isinstance(langs, str):
        langs = [langs]
    for file in traverse_files(dir_path, relative):
        lang = detect_file_language(file)
        if lang in langs:
            yield file, lang


if __name__ == "__main__":
    import fire

    fire.Fire()
