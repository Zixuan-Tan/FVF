import multiprocessing as mp
import os
import tempfile
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import word2vec
from dgl.dataloading import GraphDataLoader
from model import DevignModel
from tqdm import tqdm

DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 12345
hidden_size = 200
input_size = 135  # 132
lr = 0.0001
weight_decay = 0.001
dropout_rate = 0.3
epochs = 500
num_steps = 6
train_batch_size = 128
test_batch_size = 256
max_patience = 10
val_every = 100
log_every = 20

W2VMODEL = DIR + "/w2v.model"


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
    "TRY": 32,
    "METHOD_REF": 33,
    "NAMESPACE_BLOCK": 34,
}

type_one_hot = np.eye(len(type_map))
etype_map = {"AST": 0, "CDG": 1, "REACHING_DEF": 2, "CFG": 3, "EVAL_TYPE": 4, "REF": 5}


failed_cases = mp.Queue()


class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.w2v = word2vec.MyWord2Vec(W2VMODEL)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Override getitem."""
        item = self.df.iloc[idx]
        code, lineno, nt, ei, eo, et, _ = util.feature_extraction(item.path)

        if len(lineno) == 1:
            failed_cases.put(item.id)
            print(f"Empty graph for {idx} {item.path}")
            return dgl.DGLGraph()

        g = dgl.graph((eo, ei))
        try:
            # g.ndata["_LINE"] = torch.Tensor(np.array(lineno).astype(int))
            g.ndata["_LABEL"] = torch.Tensor([False] * len(lineno))
            g.ndata["_SAMPLE"] = torch.Tensor([idx] * len(lineno))
        except:
            failed_cases.put(item.id)
            print(f"Failed graph for {idx} {item.path}")
            return dgl.DGLGraph()

        # node features
        assert g.num_nodes() == len(lineno)
        text_feats = self.w2v.get_embeddings_list(code)
        structure_feats = [type_one_hot[type_map[node_type] - 1] for node_type in nt]
        node_feats = np.concatenate([structure_feats, text_feats], axis=1)

        # debug('node_feats')
        # print('node f', len(node_feats), len(code))
        g.ndata["node_feat"] = torch.Tensor(np.array(node_feats))
        g.ndata["_WORD2VEC"] = torch.Tensor(np.array(node_feats))
        g.ndata["_LABEL"] = torch.Tensor([False] * len(lineno))
        g.edata["_ETYPE"] = torch.Tensor(np.array(et)).long()
        # Add edges between each node and itself to preserve old node representations
        return dgl.add_self_loop(g)


def test(model, test_dl):
    model.eval()
    all_probs = torch.empty((0)).float().to(DEVICE)
    all_ids = torch.empty((0)).float().to(DEVICE)
    with torch.no_grad():
        for test_batch in test_dl:
            test_batch = test_batch.to(DEVICE)
            test_ids = dgl.max_nodes(test_batch, "_SAMPLE")
            test_probs2 = model(test_batch)
            test_probs = test_probs2.argmax(dim=1)
            all_probs = torch.cat([all_probs, test_probs])
            all_ids = torch.cat([all_ids, test_ids])

        # test_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
        probs = all_probs.detach().cpu().numpy()
        preds = [1 if i > 0.5 else 0 for i in probs]
    failed = 0
    while not failed_cases.empty():
        failed += 1
        preds[failed_cases.get()] = -1
    print(f"Failed {failed} cases")
    return preds


def run_test_inner(dataset: list[dict]):
    df = pd.DataFrame(dataset)
    test_ds = Dataset(df)

    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
    test_dl = GraphDataLoader(test_ds, batch_size=test_batch_size, **dl_args)

    model = DevignModel(input_dim=input_size, output_dim=hidden_size)
    model.to(DEVICE)

    model.load_state_dict(torch.load(DIR + "/best_f1.model", map_location=DEVICE))

    return test(model, test_dl=test_dl)


def run_test(data: list[str]):
    with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
        dataset = []
        for i, src in enumerate(data):
            path = os.path.join(d, f"{i}.c")
            print(path)
            with open(path, "w") as f:
                f.write(src)
            dataset.append({"id": i, "path": path})

        with mp.Pool() as pool:
            pool.map(util.run_joern, [d["path"] for d in dataset])

        return run_test_inner(dataset)

