from collections import Counter
from functools import lru_cache
import subprocess
from pathlib import Path
import json
import numpy as np
import pandas as pd
import re
import networkx as nx
import os
import tempfile
from loguru import logger
import time

DIR = os.path.dirname(os.path.abspath(__file__))
JOERN_DIR = DIR + "/joern-cli-1.1.919"


type_map = {
    "UNKNOWN": 0,
    "METHOD": 1,
    "METHOD_PARAMETER_IN": 2,
    "BLOCK": 3,
    "External Function Call": 4,
    "Comparison Operator": 5,
    "IDENTIFIER": 6,
    "Assignment Operator": 7,
    "RETURN": 8,
    "LITERAL": 9,
    "METHOD_RETURN": 10,
    "METHOD_PARAMETER_OUT": 11,
    "IF": 12,
    "Arithmetic Operator": 13,
    "Builtin Function Call": 14,
    "Access Operator": 15,
    "FIELD_IDENTIFIER": 16,
    "Other Operator": 17,
    "LOCAL": 18,
    "Logical Operator": 19,
    "Cast Operator": 20,
    "WHILE": 21,
    "ELSE": 22,
    "FOR": 23,
    "GOTO": 24,
    "JUMP_TARGET": 25,
    "SWITCH": 26,
    "BREAK": 27,
    "DO": 28,
    "CONTINUE": 29,
    "TYPE_DECL": 30,
    "MEMBER": 31,
}

type_one_hot = np.eye(len(type_map))
etype_map = {"AST": 0, "CDG": 1, "REACHING_DEF": 2, "CFG": 3, "EVAL_TYPE": 4, "REF": 5}


def run_joern(filepath: str):
    """Extract graph using Joern."""
    with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
        trial = 0
        while trial < 2:
            res = subprocess.run(
                f"{JOERN_DIR}/joern --script {DIR}/get_func_graph.sc --params='filename={filepath}'",
                shell=True,
                cwd=d,
                capture_output=True,
            )
            if res.returncode != 0:
                outfile = f"joern-{time.time()}.out"
                logger.error(f"Joern failed on {filepath}, saving to {outfile}")
                with open(outfile, "wb") as f:
                    f.write(res.stdout)
                    f.write(res.stderr)
                trial += 1
            else:
                break


def get_node_edges(filepath: str):
    """Get node and edges given filepath (must run after run_joern).

    nodes:
        id, _label, name,       code,                                   lineNumber, controlStructureType,       node_label
        8   METHOD  DetachBlob  void* DetachBlob (BlobInfo *blob_info)  1.0                                     METHOD_1.0: void* DetachBlob (BlobInfo *blob_i...
    edges:
        innode, outnode, etype, dataflow, id_x, line_out, id_y, line_in
        120     119        AST            119     25.0   120    25.0
    """
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name

    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)

    edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
    edges = edges.fillna("")

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)

    nodes = pd.DataFrame.from_records(nodes)
    if "controlStructureType" not in nodes.columns:
        nodes["controlStructureType"] = ""
    nodes = nodes.fillna("")
    nodes = nodes[
        ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
    ]
    nodes = nodes[(nodes["name"] != "<global>")]
    nodes = nodes[~nodes["_label"].apply(lambda x: "META" in x)]

    # Assign line number to local variables
    # with open(filepath, "r") as f:
    #     code = f.readlines()
    # lmap = assign_line_num_to_local(nodes, edges, code)
    # nodes.lineNumber = nodes.apply(
    #     lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    # )
    # nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    # typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]
    #
    # linemap = nodes.set_index("id").to_dict()["lineNumber"]
    # for e in edges.itertuples():
    #     if type(e.outnode) == str:
    #         lineNum = linemap[e.innode]
    #         node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
    #         nodes = nodes.append(
    #             {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
    #             ignore_index=True,
    #         )
    return nodes, edges


def rdg(edges, gtype):
    """Reduce graph given type."""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            # | (edges.etype == "EVAL_TYPE")
            # | (edges.etype == "REF")
        ]
    raise Exception("Incorrect gtype.")


def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def tokenize(s):
    """Tokenise according to IVDetect.

    Tests:
    s = "FooBar fooBar foo bar_blub23/x~y'z"
    """
    spec_char = re.compile(r"[^a-zA-Z0-9\s]")
    camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    spec_split = re.split(spec_char, s)
    space_split = " ".join(spec_split).split()

    def camel_case_split(identifier):
        return [i.group(0) for i in re.finditer(camelcase, identifier)]

    camel_split = [i for j in [camel_case_split(i) for i in space_split] for i in j]
    remove_single = [i for i in camel_split if len(i) > 1]
    return " ".join(remove_single)


def type_2_type(info):
    for _ in range(1):
        if info["_label"] == "CALL":
            if "<operator>" in info["name"]:
                if "assignment" in info["name"]:
                    new_type = "Assignment Operator"
                    break
                if (
                    "addition" in info["name"]
                    or "subtraction" in info["name"]
                    or "division" in info["name"]
                    or "Plus" in info["name"]
                    or "Minus" in info["name"]
                    or "minus" in info["name"]
                    or "plus" in info["name"]
                    or "modulo" in info["name"]
                    or "multiplication" in info["name"]
                ):
                    new_type = "Arithmetic Operator"
                    break
                if (
                    "lessThan" in info["name"]
                    or "greaterThan" in info["name"]
                    or "EqualsThan" in info["name"]
                    or "equals" in info["name"]
                ):
                    new_type = "Comparison Operator"
                    break
                if (
                    "FieldAccess" in info["name"]
                    or "IndexAccess" in info["name"]
                    or "fieldAccess" in info["name"]
                    or "indexAccess" in info["name"]
                ):
                    new_type = "Access Operator"
                    break
                if (
                    "logical" in info["name"]
                    or "<operator>.not" in info["name"]
                    or "<operator>.or" in info["name"]
                    or "<operator>.and" in info["name"]
                    or "conditional" in info["name"]
                ):
                    new_type = "Logical Operator"
                    break
                if "<operator>.cast" in info["name"]:
                    new_type = "Cast Operator"
                    break
                if "<operator>" in info["name"]:
                    new_type = "Other Operator"
                    break
            elif info["name"] in l_funcs:
                new_type = "Builtin Function Call"
                break
            else:
                new_type = "External Function Call"
                break
        if info["_label"] == "CONTROL_STRUCTURE":
            new_type = info["controlStructureType"]
            break
        new_type = info["_label"]
    return new_type


def etype_2_id(etype):
    return etype_map[etype]


def count_labels(nodes):
    """Get info about nodes."""
    label_info = []
    for _, info in nodes.iterrows():
        new_type = type_2_type(info)
        label_info.append(new_type)

    counter = Counter(label_info)
    return counter, label_info


@lru_cache(maxsize=128)
def feature_extraction(filepath, graph_type="all", return_nodes=False):
    """Extract graph feature (basic).

    _id = svddc.BigVulDataset.itempath(177775)
    _id = svddc.BigVulDataset.itempath(180189)
    _id = svddc.BigVulDataset.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get CPG
    n, e = get_node_edges(filepath)
    # n, e = ne_groupnodes(n, e)

    # Filter nodes
    e = rdg(e, graph_type)
    n = drop_lone_nodes(n, e)
    counter, label_info = count_labels(n)
    # Return node metadata
    if return_nodes:
        return n

    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()

    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.id_y.map(iddict)
    e.outnode = e.id_x.map(iddict)

    e = e[e.innode.notnull() & e.outnode.notnull()]

    src_edges = e.innode.tolist()
    dst_edges = e.outnode.tolist()
    edges_type = [etype_2_id(t) for t in e.etype.tolist()]

    # # Map edge types
    # etypes = e.etype.tolist()
    # d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    # debug('etypes')
    # print(etypes)
    # print(d)
    # etypes = [d[i] for i in etypes]
    # print(etypes)
    # assert False
    # # Return plain-text code, line number list, innodes, outnodes
    return (
        n.code.tolist(),
        n.lineNumber.tolist(),
        label_info,
        src_edges,
        dst_edges,
        edges_type,
        counter,
    )


l_funcs = set(
    [
        "StrNCat",
        "getaddrinfo",
        "_ui64toa",
        "fclose",
        "pthread_mutex_lock",
        "gets_s",
        "sleep",
        "_ui64tot",
        "freopen_s",
        "_ui64tow",
        "send",
        "lstrcat",
        "HMAC_Update",
        "__fxstat",
        "StrCatBuff",
        "_mbscat",
        "_mbstok_s",
        "_cprintf_s",
        "ldap_search_init_page",
        "memmove_s",
        "ctime_s",
        "vswprintf",
        "vswprintf_s",
        "_snwprintf",
        "_gmtime_s",
        "_tccpy",
        "*RC6*",
        "_mbslwr_s",
        "random",
        "__wcstof_internal",
        "_wcslwr_s",
        "_ctime32_s",
        "wcsncat*",
        "MD5_Init",
        "_ultoa",
        "snprintf",
        "memset",
        "syslog",
        "_vsnprintf_s",
        "HeapAlloc",
        "pthread_mutex_destroy",
        "ChangeWindowMessageFilter",
        "_ultot",
        "crypt_r",
        "_strupr_s_l",
        "LoadLibraryExA",
        "_strerror_s",
        "LoadLibraryExW",
        "wvsprintf",
        "MoveFileEx",
        "_strdate_s",
        "SHA1",
        "sprintfW",
        "StrCatNW",
        "_scanf_s_l",
        "pthread_attr_init",
        "_wtmpnam_s",
        "snscanf",
        "_sprintf_s_l",
        "dlopen",
        "sprintfA",
        "timed_mutex",
        "OemToCharA",
        "ldap_delete_ext",
        "sethostid",
        "popen",
        "OemToCharW",
        "_gettws",
        "vfork",
        "_wcsnset_s_l",
        "sendmsg",
        "_mbsncat",
        "wvnsprintfA",
        "HeapFree",
        "_wcserror_s",
        "realloc",
        "_snprintf*",
        "wcstok",
        "_strncat*",
        "StrNCpy",
        "_wasctime_s",
        "push*",
        "_lfind_s",
        "CC_SHA512",
        "ldap_compare_ext_s",
        "wcscat_s",
        "strdup",
        "_chsize_s",
        "sprintf_s",
        "CC_MD4_Init",
        "wcsncpy",
        "_wfreopen_s",
        "_wcsupr_s",
        "_searchenv_s",
        "ldap_modify_ext_s",
        "_wsplitpath",
        "CC_SHA384_Final",
        "MD2",
        "RtlCopyMemory",
        "lstrcatW",
        "MD4",
        "MD5",
        "_wcstok_s_l",
        "_vsnwprintf_s",
        "ldap_modify_s",
        "strerror",
        "_lsearch_s",
        "_mbsnbcat_s",
        "_wsplitpath_s",
        "MD4_Update",
        "_mbccpy_s",
        "_strncpy_s_l",
        "_snprintf_s",
        "CC_SHA512_Init",
        "fwscanf_s",
        "_snwprintf_s",
        "CC_SHA1",
        "swprintf",
        "fprintf",
        "EVP_DigestInit_ex",
        "strlen",
        "SHA1_Init",
        "strncat",
        "_getws_s",
        "CC_MD4_Final",
        "wnsprintfW",
        "lcong48",
        "lrand48",
        "write",
        "HMAC_Init",
        "_wfopen_s",
        "wmemchr",
        "_tmakepath",
        "wnsprintfA",
        "lstrcpynW",
        "scanf_s",
        "_mbsncpy_s_l",
        "_localtime64_s",
        "fstream.open",
        "_wmakepath",
        "Connection.open",
        "_tccat",
        "valloc",
        "setgroups",
        "unlink",
        "fstream.put",
        "wsprintfA",
        "*SHA1*",
        "_wsearchenv_s",
        "ualstrcpyA",
        "CC_MD5_Update",
        "strerror_s",
        "HeapCreate",
        "ualstrcpyW",
        "__xstat",
        "_wmktemp_s",
        "StrCatChainW",
        "ldap_search_st",
        "_mbstowcs_s_l",
        "ldap_modify_ext",
        "_mbsset_s",
        "strncpy_s",
        "move",
        "execle",
        "StrCat",
        "xrealloc",
        "wcsncpy_s",
        "_tcsncpy*",
        "execlp",
        "RIPEMD160_Final",
        "ldap_search_s",
        "EnterCriticalSection",
        "_wctomb_s_l",
        "fwrite",
        "_gmtime64_s",
        "sscanf_s",
        "wcscat",
        "_strupr_s",
        "wcrtomb_s",
        "VirtualLock",
        "ldap_add_ext_s",
        "_mbscpy",
        "_localtime32_s",
        "lstrcpy",
        "_wcsncpy*",
        "CC_SHA1_Init",
        "_getts",
        "_wfopen",
        "__xstat64",
        "strcoll",
        "_fwscanf_s_l",
        "_mbslwr_s_l",
        "RegOpenKey",
        "makepath",
        "seed48",
        "CC_SHA256",
        "sendto",
        "execv",
        "CalculateDigest",
        "memchr",
        "_mbscpy_s",
        "_strtime_s",
        "ldap_search_ext_s",
        "_chmod",
        "flock",
        "__fxstat64",
        "_vsntprintf",
        "CC_SHA256_Init",
        "_itoa_s",
        "__wcserror_s",
        "_gcvt_s",
        "fstream.write",
        "sprintf",
        "recursive_mutex",
        "strrchr",
        "gethostbyaddr",
        "_wcsupr_s_l",
        "strcspn",
        "MD5_Final",
        "asprintf",
        "_wcstombs_s_l",
        "_tcstok",
        "free",
        "MD2_Final",
        "asctime_s",
        "_alloca",
        "_wputenv_s",
        "_wcsset_s",
        "_wcslwr_s_l",
        "SHA1_Update",
        "filebuf.sputc",
        "filebuf.sputn",
        "SQLConnect",
        "ldap_compare",
        "mbstowcs_s",
        "HMAC_Final",
        "pthread_condattr_init",
        "_ultow_s",
        "rand",
        "ofstream.put",
        "CC_SHA224_Final",
        "lstrcpynA",
        "bcopy",
        "system",
        "CreateFile*",
        "wcscpy_s",
        "_mbsnbcpy*",
        "open",
        "_vsnwprintf",
        "strncpy",
        "getopt_long",
        "CC_SHA512_Final",
        "_vsprintf_s_l",
        "scanf",
        "mkdir",
        "_localtime_s",
        "_snprintf",
        "_mbccpy_s_l",
        "memcmp",
        "final",
        "_ultoa_s",
        "lstrcpyW",
        "LoadModule",
        "_swprintf_s_l",
        "MD5_Update",
        "_mbsnset_s_l",
        "_wstrtime_s",
        "_strnset_s",
        "lstrcpyA",
        "_mbsnbcpy_s",
        "mlock",
        "IsBadHugeWritePtr",
        "copy",
        "_mbsnbcpy_s_l",
        "wnsprintf",
        "wcscpy",
        "ShellExecute",
        "CC_MD4",
        "_ultow",
        "_vsnwprintf_s_l",
        "lstrcpyn",
        "CC_SHA1_Final",
        "vsnprintf",
        "_mbsnbset_s",
        "_i64tow",
        "SHA256_Init",
        "wvnsprintf",
        "RegCreateKey",
        "strtok_s",
        "_wctime32_s",
        "_i64toa",
        "CC_MD5_Final",
        "wmemcpy",
        "WinExec",
        "CreateDirectory*",
        "CC_SHA256_Update",
        "_vsnprintf_s_l",
        "jrand48",
        "wsprintf",
        "ldap_rename_ext_s",
        "filebuf.open",
        "_wsystem",
        "SHA256_Update",
        "_cwscanf_s",
        "wsprintfW",
        "_sntscanf",
        "_splitpath",
        "fscanf_s",
        "strpbrk",
        "wcstombs_s",
        "wscanf",
        "_mbsnbcat_s_l",
        "strcpynA",
        "pthread_cond_init",
        "wcsrtombs_s",
        "_wsopen_s",
        "CharToOemBuffA",
        "RIPEMD160_Update",
        "_tscanf",
        "HMAC",
        "StrCCpy",
        "Connection.connect",
        "lstrcatn",
        "_mbstok",
        "_mbsncpy",
        "CC_SHA384_Update",
        "create_directories",
        "pthread_mutex_unlock",
        "CFile.Open",
        "connect",
        "_vswprintf_s_l",
        "_snscanf_s_l",
        "fputc",
        "_wscanf_s",
        "_snprintf_s_l",
        "strtok",
        "_strtok_s_l",
        "lstrcatA",
        "snwscanf",
        "pthread_mutex_init",
        "fputs",
        "CC_SHA384_Init",
        "_putenv_s",
        "CharToOemBuffW",
        "pthread_mutex_trylock",
        "__wcstoul_internal",
        "_memccpy",
        "_snwprintf_s_l",
        "_strncpy*",
        "wmemset",
        "MD4_Init",
        "*RC4*",
        "strcpyW",
        "_ecvt_s",
        "memcpy_s",
        "erand48",
        "IsBadHugeReadPtr",
        "strcpyA",
        "HeapReAlloc",
        "memcpy",
        "ldap_rename_ext",
        "fopen_s",
        "srandom",
        "_cgetws_s",
        "_makepath",
        "SHA256_Final",
        "remove",
        "_mbsupr_s",
        "pthread_mutexattr_init",
        "__wcstold_internal",
        "StrCpy",
        "ldap_delete",
        "wmemmove_s",
        "_mkdir",
        "strcat",
        "_cscanf_s_l",
        "StrCAdd",
        "swprintf_s",
        "_strnset_s_l",
        "close",
        "ldap_delete_ext_s",
        "ldap_modrdn",
        "strchr",
        "_gmtime32_s",
        "_ftcscat",
        "lstrcatnA",
        "_tcsncat",
        "OemToChar",
        "mutex",
        "CharToOem",
        "strcpy_s",
        "lstrcatnW",
        "_wscanf_s_l",
        "__lxstat64",
        "memalign",
        "MD2_Init",
        "StrCatBuffW",
        "StrCpyN",
        "CC_MD5",
        "StrCpyA",
        "StrCatBuffA",
        "StrCpyW",
        "tmpnam_r",
        "_vsnprintf",
        "strcatA",
        "StrCpyNW",
        "_mbsnbset_s_l",
        "EVP_DigestInit",
        "_stscanf",
        "CC_MD2",
        "_tcscat",
        "StrCpyNA",
        "xmalloc",
        "_tcslen",
        "*MD4*",
        "vasprintf",
        "strxfrm",
        "chmod",
        "ldap_add_ext",
        "alloca",
        "_snscanf_s",
        "IsBadWritePtr",
        "swscanf_s",
        "wmemcpy_s",
        "_itoa",
        "_ui64toa_s",
        "EVP_DigestUpdate",
        "__wcstol_internal",
        "_itow",
        "StrNCatW",
        "strncat_s",
        "ualstrcpy",
        "execvp",
        "_mbccat",
        "EVP_MD_CTX_init",
        "assert",
        "ofstream.write",
        "ldap_add",
        "_sscanf_s_l",
        "drand48",
        "CharToOemW",
        "swscanf",
        "_itow_s",
        "RIPEMD160_Init",
        "CopyMemory",
        "initstate",
        "getpwuid",
        "vsprintf",
        "_fcvt_s",
        "CharToOemA",
        "setuid",
        "malloc",
        "StrCatNA",
        "strcat_s",
        "srand",
        "getwd",
        "_controlfp_s",
        "olestrcpy",
        "__wcstod_internal",
        "_mbsnbcat",
        "lstrncat",
        "des_*",
        "CC_SHA224_Init",
        "set*",
        "vsprintf_s",
        "SHA1_Final",
        "_umask_s",
        "gets",
        "setstate",
        "wvsprintfW",
        "LoadLibraryEx",
        "ofstream.open",
        "calloc",
        "_mbstrlen",
        "_cgets_s",
        "_sopen_s",
        "IsBadStringPtr",
        "wcsncat_s",
        "add*",
        "nrand48",
        "create_directory",
        "ldap_search_ext",
        "_i64toa_s",
        "_ltoa_s",
        "_cwscanf_s_l",
        "wmemcmp",
        "__lxstat",
        "lstrlen",
        "pthread_condattr_destroy",
        "_ftcscpy",
        "wcstok_s",
        "__xmknod",
        "pthread_attr_destroy",
        "sethostname",
        "_fscanf_s_l",
        "StrCatN",
        "RegEnumKey",
        "_tcsncpy",
        "strcatW",
        "AfxLoadLibrary",
        "setenv",
        "tmpnam",
        "_mbsncat_s_l",
        "_wstrdate_s",
        "_wctime64_s",
        "_i64tow_s",
        "CC_MD4_Update",
        "ldap_add_s",
        "_umask",
        "CC_SHA1_Update",
        "_wcsset_s_l",
        "_mbsupr_s_l",
        "strstr",
        "_tsplitpath",
        "memmove",
        "_tcscpy",
        "vsnprintf_s",
        "strcmp",
        "wvnsprintfW",
        "tmpfile",
        "ldap_modify",
        "_mbsncat*",
        "mrand48",
        "sizeof",
        "StrCatA",
        "_ltow_s",
        "*desencrypt*",
        "StrCatW",
        "_mbccpy",
        "CC_MD2_Init",
        "RIPEMD160",
        "ldap_search",
        "CC_SHA224",
        "mbsrtowcs_s",
        "update",
        "ldap_delete_s",
        "getnameinfo",
        "*RC5*",
        "_wcsncat_s_l",
        "DriverManager.getConnection",
        "socket",
        "_cscanf_s",
        "ldap_modrdn_s",
        "_wopen",
        "CC_SHA256_Final",
        "_snwprintf*",
        "MD2_Update",
        "strcpy",
        "_strncat_s_l",
        "CC_MD5_Init",
        "mbscpy",
        "wmemmove",
        "LoadLibraryW",
        "_mbslen",
        "*alloc",
        "_mbsncat_s",
        "LoadLibraryA",
        "fopen",
        "StrLen",
        "delete",
        "_splitpath_s",
        "CreateFileTransacted*",
        "MD4_Final",
        "_open",
        "CC_SHA384",
        "wcslen",
        "wcsncat",
        "_mktemp_s",
        "pthread_mutexattr_destroy",
        "_snwscanf_s",
        "_strset_s",
        "_wcsncpy_s_l",
        "CC_MD2_Final",
        "_mbstok_s_l",
        "wctomb_s",
        "MySQL_Driver.connect",
        "_snwscanf_s_l",
        "*_des_*",
        "LoadLibrary",
        "_swscanf_s_l",
        "ldap_compare_s",
        "ldap_compare_ext",
        "_strlwr_s",
        "GetEnvironmentVariable",
        "cuserid",
        "_mbscat_s",
        "strspn",
        "_mbsncpy_s",
        "ldap_modrdn2",
        "LeaveCriticalSection",
        "CopyFile",
        "getpwd",
        "sscanf",
        "creat",
        "RegSetValue",
        "ldap_modrdn2_s",
        "CFile.Close",
        "*SHA_1*",
        "pthread_cond_destroy",
        "CC_SHA512_Update",
        "*RC2*",
        "StrNCatA",
        "_mbsnbcpy",
        "_mbsnset_s",
        "crypt",
        "excel",
        "_vstprintf",
        "xstrdup",
        "wvsprintfA",
        "getopt",
        "mkstemp",
        "_wcsnset_s",
        "_stprintf",
        "_sntprintf",
        "tmpfile_s",
        "OpenDocumentFile",
        "_mbsset_s_l",
        "_strset_s_l",
        "_strlwr_s_l",
        "ifstream.open",
        "xcalloc",
        "StrNCpyA",
        "_wctime_s",
        "CC_SHA224_Update",
        "_ctime64_s",
        "MoveFile",
        "chown",
        "StrNCpyW",
        "IsBadReadPtr",
        "_ui64tow_s",
        "IsBadCodePtr",
        "getc",
        "OracleCommand.ExecuteOracleScalar",
        "AccessDataSource.Insert",
        "IDbDataAdapter.FillSchema",
        "IDbDataAdapter.Update",
        "GetWindowText*",
        "SendMessage",
        "SqlCommand.ExecuteNonQuery",
        "streambuf.sgetc",
        "streambuf.sgetn",
        "OracleCommand.ExecuteScalar",
        "SqlDataSource.Update",
        "_Read_s",
        "IDataAdapter.Fill",
        "_wgetenv",
        "_RecordsetPtr.Open*",
        "AccessDataSource.Delete",
        "Recordset.Open*",
        "filebuf.sbumpc",
        "DDX_*",
        "RegGetValue",
        "fstream.read*",
        "SqlCeCommand.ExecuteResultSet",
        "SqlCommand.ExecuteXmlReader",
        "main",
        "streambuf.sputbackc",
        "read",
        "m_lpCmdLine",
        "CRichEditCtrl.Get*",
        "istream.putback",
        "SqlCeCommand.ExecuteXmlReader",
        "SqlCeCommand.BeginExecuteXmlReader",
        "filebuf.sgetn",
        "OdbcDataAdapter.Update",
        "filebuf.sgetc",
        "SQLPutData",
        "recvfrom",
        "OleDbDataAdapter.FillSchema",
        "IDataAdapter.FillSchema",
        "CRichEditCtrl.GetLine",
        "DbDataAdapter.Update",
        "SqlCommand.ExecuteReader",
        "istream.get",
        "ReceiveFrom",
        "_main",
        "fgetc",
        "DbDataAdapter.FillSchema",
        "kbhit",
        "UpdateCommand.Execute*",
        "Statement.execute",
        "fgets",
        "SelectCommand.Execute*",
        "getch",
        "OdbcCommand.ExecuteNonQuery",
        "CDaoQueryDef.Execute",
        "fstream.getline",
        "ifstream.getline",
        "SqlDataAdapter.FillSchema",
        "OleDbCommand.ExecuteReader",
        "Statement.execute*",
        "SqlCeCommand.BeginExecuteNonQuery",
        "OdbcCommand.ExecuteScalar",
        "SqlCeDataAdapter.Update",
        "sendmessage",
        "mysqlpp.DBDriver",
        "fstream.peek",
        "Receive",
        "CDaoRecordset.Open",
        "OdbcDataAdapter.FillSchema",
        "_wgetenv_s",
        "OleDbDataAdapter.Update",
        "readsome",
        "SqlCommand.BeginExecuteXmlReader",
        "recv",
        "ifstream.peek",
        "_Main",
        "_tmain",
        "_Readsome_s",
        "SqlCeCommand.ExecuteReader",
        "OleDbCommand.ExecuteNonQuery",
        "fstream.get",
        "IDbCommand.ExecuteScalar",
        "filebuf.sputbackc",
        "IDataAdapter.Update",
        "streambuf.sbumpc",
        "InsertCommand.Execute*",
        "RegQueryValue",
        "IDbCommand.ExecuteReader",
        "SqlPipe.ExecuteAndSend",
        "Connection.Execute*",
        "getdlgtext",
        "ReceiveFromEx",
        "SqlDataAdapter.Update",
        "RegQueryValueEx",
        "SQLExecute",
        "pread",
        "SqlCommand.BeginExecuteReader",
        "AfxWinMain",
        "getchar",
        "istream.getline",
        "SqlCeDataAdapter.Fill",
        "OleDbDataReader.ExecuteReader",
        "SqlDataSource.Insert",
        "istream.peek",
        "SendMessageCallback",
        "ifstream.read*",
        "SqlDataSource.Select",
        "SqlCommand.ExecuteScalar",
        "SqlDataAdapter.Fill",
        "SqlCommand.BeginExecuteNonQuery",
        "getche",
        "SqlCeCommand.BeginExecuteReader",
        "getenv",
        "streambuf.snextc",
        "Command.Execute*",
        "_CommandPtr.Execute*",
        "SendNotifyMessage",
        "OdbcDataAdapter.Fill",
        "AccessDataSource.Update",
        "fscanf",
        "QSqlQuery.execBatch",
        "DbDataAdapter.Fill",
        "cin",
        "DeleteCommand.Execute*",
        "QSqlQuery.exec",
        "PostMessage",
        "ifstream.get",
        "filebuf.snextc",
        "IDbCommand.ExecuteNonQuery",
        "Winmain",
        "fread",
        "getpass",
        "GetDlgItemTextCCheckListBox.GetCheck",
        "DISP_PROPERTY_EX",
        "pread64",
        "Socket.Receive*",
        "SACommand.Execute*",
        "SQLExecDirect",
        "SqlCeDataAdapter.FillSchema",
        "DISP_FUNCTION",
        "OracleCommand.ExecuteNonQuery",
        "CEdit.GetLine",
        "OdbcCommand.ExecuteReader",
        "CEdit.Get*",
        "AccessDataSource.Select",
        "OracleCommand.ExecuteReader",
        "OCIStmtExecute",
        "getenv_s",
        "DB2Command.Execute*",
        "OracleDataAdapter.FillSchema",
        "OracleDataAdapter.Fill",
        "CComboBox.Get*",
        "SqlCeCommand.ExecuteNonQuery",
        "OracleCommand.ExecuteOracleNonQuery",
        "mysqlpp.Query",
        "istream.read*",
        "CListBox.GetText",
        "SqlCeCommand.ExecuteScalar",
        "ifstream.putback",
        "readlink",
        "CHtmlEditCtrl.GetDHtmlDocument",
        "PostThreadMessage",
        "CListCtrl.GetItemText",
        "OracleDataAdapter.Update",
        "OleDbCommand.ExecuteScalar",
        "stdin",
        "SqlDataSource.Delete",
        "OleDbDataAdapter.Fill",
        "fstream.putback",
        "IDbDataAdapter.Fill",
        "_wspawnl",
        "fwprintf",
        "sem_wait",
        "_unlink",
        "ldap_search_ext_sW",
        "signal",
        "PQclear",
        "PQfinish",
        "PQexec",
        "PQresultStatus",
    ]
)
