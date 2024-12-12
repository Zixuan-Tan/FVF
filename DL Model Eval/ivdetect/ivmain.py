import multiprocessing as mp
import os
import tempfile
import dgl
import glove
import numpy as np
import pandas as pd
import torch
import ivutil
from dgl.dataloading import GraphDataLoader
from ivmodel import IVDetect
from tqdm import tqdm

DIR = os.path.dirname(__file__)
EMB_DICT, _ = glove.glove_dict(DIR + "/glove/")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

do_train = False
do_test = True
n_gpu = 1
seed = 12345
hidden_size = 128
input_size = 200
lr = 0.0001
dropout_rate = 0.3
epochs = 100
num_conv_layers = 3
eval_batch_size = 32
max_patience = 10
val_every = 100
log_every = 20
train_batch_size = 16
test_batch_size = 32

failed_cases = mp.Queue()


class Dataset:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        """Override getitem."""
        item = self.df.iloc[idx]
        try:
            n, e = ivutil.feature_extraction(item.path)
        except Exception as ex:
            print(ex)
            failed_cases.put(item.id)
            print(f"Failed for data {item.id}")
            return dgl.DGLGraph()
        if not len(n):
            failed_cases.put(item.id)
            print(f"Failed for data {item.id}")
            return dgl.DGLGraph()

        g = dgl.graph(e)
        g.ndata["_SAMPLE"] = torch.Tensor([idx] * len(n))
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())
        g.ndata["_LABEL"] = torch.Tensor([False] * len(n))

        # Add edges between each node and itself to preserve old node representations
        # print(g.number_of_nodes(), g.number_of_edges())
        return dgl.add_self_loop(g)

    def __len__(self):
        """Override len."""
        return len(self.df)

    def item(self, sampleid):
        """Get item data."""
        itempath = self.df.iloc[sampleid].path
        n, _ = ivutil.feature_extraction(itempath)
        n.subseq = n.subseq.apply(lambda x: glove.get_embeddings(x, EMB_DICT, 200))
        n.nametypes = n.nametypes.apply(lambda x: glove.get_embeddings(x, EMB_DICT, 200))
        n.data = n.data.apply(lambda x: glove.get_embeddings(x, EMB_DICT, 200))
        n.control = n.control.apply(lambda x: glove.get_embeddings(x, EMB_DICT, 200))

        def ast_dgl(row_ast, row_id):
            if len(row_ast) == 0:
                return None
            """
            row example
            [[0, 0, 0, 0, 0, 0], 
             [1, 2, 3, 4, 5, 6], 
             ['int alloc addbyter int output FILE data', 'int output', 'FILE data', '', 'int', 'int output', 'FILE data']]

            """
            outnode, innode, ndata = row_ast
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(np.array(glove.get_embeddings_list(ndata, EMB_DICT, 200)))
            g.ndata["_ID"] = torch.Tensor([sampleid] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([row_id] * g.number_of_nodes())
            return g

        asts = []
        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}


def test(model, test_dl, test_ds):
    model.eval()
    all_pred = torch.empty((0, 2)).long().to(DEVICE)
    with torch.no_grad():
        for test_batch in tqdm(test_dl, total=len(test_dl)):
            test_batch = test_batch.to(DEVICE)
            test_logits = model(test_batch, test_ds)
            all_pred = torch.cat([all_pred, test_logits])

        # test_mets = get_metrics_logits(all_true, all_pred)
        sm_logits = torch.nn.functional.softmax(all_pred, dim=1)
        pos_logits = sm_logits[:, 1].detach().cpu().numpy()
        preds = [1 if i > 0.5 else 0 for i in pos_logits]
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

    model = IVDetect(input_size=input_size, hidden_size=hidden_size)
    model.to(DEVICE)

    model.load_state_dict(torch.load(DIR + "/best_f1.model", map_location=DEVICE))

    return test(model, test_dl=test_dl, test_ds=test_ds)


def run_test(data):
    with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
        dataset = []
        for i, src in enumerate(data):
            path = os.path.join(d, f"{i}.c")
            print(path)
            with open(path, "w") as f:
                f.write(src)
            dataset.append({"id": i, "path": path})

        with mp.Pool(4) as pool:
            pool.map(ivutil.run_joern, [d["path"] for d in dataset])

        return run_test_inner(dataset)
