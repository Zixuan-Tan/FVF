from functools import lru_cache
import subprocess
from pathlib import Path
import json
import pandas as pd
import re
import networkx as nx
import os
import tempfile

DIR = os.path.dirname(os.path.abspath(__file__))
JOERN_DIR = DIR + "/joern-cli-1.1.919"


def run_joern(filepath: str):
    """Extract graph using Joern."""
    with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
        subprocess.check_call(
            f"{JOERN_DIR}/joern --script {DIR}/get_func_graph.sc --params='filename={filepath}'",
            shell=True,
            cwd=d,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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


@lru_cache(maxsize=128)
def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation."""
    nodes, edges = get_node_edges(filepath)

    # F1. Generate tokenized subtoken sequences
    # get node with longest code per line
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "code"]].copy()
    # subseq.code = subseq.local_type + " " + subseq.code
    # subseq = subseq.drop(columns="local_type")
    subseq = subseq[~subseq.eq("").any(axis=1)]
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    subseq = subseq.sort_values("lineNumber")
    subseq.code = subseq.code.apply(tokenize)
    # { 1: "code1", 2: "code2", ... }
    subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. Line to AST
    ast_edges = rdg(edges, "ast")
    ast_nodes = drop_lone_nodes(nodes, ast_edges)
    ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)

    # 转换为行内边：行号，行内节点index，行内节点index
    # 行内index: assign number to each node in each line
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    # 行内edges
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]
    # node id (from joern) -> 行内index
    nodeid2inlineno = pd.Series(
        ast_nodes["lineidx"].values, index=ast_nodes.id
    ).to_dict()
    ast_edges.innode = ast_edges.innode.map(nodeid2inlineno)
    ast_edges.outnode = ast_edges.outnode.map(nodeid2inlineno)
    # line_no: [from1, from2, ...]  [to1, to2, ...]
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})

    ast_nodes.code = ast_nodes.code.fillna("").apply(tokenize)
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )
    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})

    ast = ast_edges.join(ast_nodes, how="inner")
    if ast.empty:
        return [], []

    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    ast = ast.to_dict()["ast"]

    # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # incoming connections (parent node), then add an edge from that node to the node
    # with id = 0 (unless it is zero itself).
    # DEBUG:
    # import sastvd.helpers.graphs as svdgr
    # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    for k, v in ast.items():
        allnodes = nodes_per_line[k]
        outnodes = v[0]
        innodes = v[1]
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        parentnodes = [i for i in outnodes if i not in innodes]
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    reftype_edges = rdg(edges, "reftype")
    reftype_nodes = drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))
    varnametypes = list()
    # for cc in reftype_cc:
    #     cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
    #     if sum(cc_nodes["_label"] == "IDENTIFIER") == 0:
    #         continue
    #     if sum(cc_nodes["_label"] == "TYPE") == 0:
    #         continue
    #     var_type = cc_nodes[cc_nodes["_label"] == "TYPE"]
    #     print('varTYPE',var_type)
    for cc in reftype_cc:
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        if sum(cc_nodes["_label"] == "IDENTIFIER") == 0:
            continue
        if sum(cc_nodes["_label"] == "TYPE") == 0:
            continue
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].head(1).name.item()
        # code_ = cc_nodes[cc_nodes["_label"] != "IDENTIFIER"].code.item()
        # name_ = cc_nodes[cc_nodes["_label"] != "IDENTIFIER"].name.item()
        # var_type = code_[0:code_.find(name_)].strip()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    nametypes.type = nametypes.type.apply(tokenize)
    nametypes.name = nametypes.name.apply(tokenize)
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    nametypes = nametypes.to_dict()["nametype"]

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = rdg(edgesline, "pdg")
    nodesline = drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])

    if len(edgesline) > 0:
        edgesline["etype"] = edgesline.apply(
            lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
        )
        edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
        edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot(index="innode", columns="etype", values="outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    # print(filepath)
    # print('subseq:', subseq)
    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    # 过滤掉不在pdg中的节点
    pdg_nodes = pdg_nodes[pdg_nodes.id.isin(list(data.keys()) + list(control.keys()))]

    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_nodes.data = pdg_nodes.data.map(
        lambda x: " ".join([subseq[i] for i in x if i in subseq])
    )
    pdg_nodes.control = pdg_nodes.control.map(
        lambda x: " ".join([subseq[i] for i in x if i in subseq])
    )
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    return pdg_nodes, pdg_edges
