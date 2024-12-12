import os
import random
import re
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchtext
import torchtext.vocab as vocab

# from clang import *
from clang import cindex
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer, normalizers, processors
from tokenizers.models import BPE
from tokenizers.normalizers import Replace, StripAccents
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import TemplateProcessing
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaModel


# Tokenizer
class MyTokenizer:
    cidx = cindex.Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        # Tokkenize using clang
        tok = []
        tu = self.cidx.parse("tmp.c", args=[""], unsaved_files=[("tmp.c", str(normalized_string.original))], options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()

            if spelling == "":
                continue

            # Keyword no need

            # Punctuations no need

            # Literal all to BPE

            # spelling = spelling.replace(' ', '')
            tok.append(NormalizedString(spelling))

        return tok

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)


class myCNN(nn.Module):
    def __init__(self, EMBED_SIZE, EMBED_DIM):
        super(myCNN, self).__init__()

        pretrained_weights = RobertaModel.from_pretrained("VulBERTa").embeddings.word_embeddings.weight

        self.embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=True, padding_idx=1)

        self.conv1 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200 * 3, 256)  # 500
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = F.max_pool1d(x1, x1.shape[2])
        x2 = F.max_pool1d(x2, x2.shape[2])
        x3 = F.max_pool1d(x3, x3.shape[2])

        x = torch.cat([x1, x2, x3], dim=1)

        # flatten the tensor
        x = x.flatten(1)

        # apply mean over the last dimension
        # x = torch.mean(x, -1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cleaner(code):
    # Remove code comments
    pat = re.compile(r"(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)")
    code = re.sub(pat, "", code)
    code = re.sub("\n", "", code)
    code = re.sub("\t", "", code)
    return code


class TabularDataset_From_List(torchtext.data.Dataset):
    def __init__(self, input_list, format, fields, skip_header=False, **kwargs):
        make_example = {"json": torchtext.data.Example.fromJSON, "dict": torchtext.data.Example.fromdict}[
            format.lower()
        ]

        examples = [make_example(item, fields) for item in input_list]

        if make_example in (torchtext.data.Example.fromdict, torchtext.data.Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset_From_List, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root=".data", train=None, validation=None, test=None, **kwargs):
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(train, **kwargs)
        val_data = None if validation is None else cls(validation, **kwargs)
        test_data = None if test is None else cls(test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def evaluate_testing(all_pred, all_labels):
    def getClass(x):
        return x.index(max(x))

    probs = pd.Series(all_pred)
    all_predicted = probs.apply(getClass)
    all_predicted.reset_index(drop=True, inplace=True)
    vc = pd.value_counts(all_predicted == all_labels)

    probs2 = []
    for x in probs:
        probs2.append(x[1])

    confusion = sklearn.metrics.confusion_matrix(y_true=all_labels, y_pred=all_predicted)
    print("Confusion matrix: \n", confusion)

    try:
        tn, fp, fn, tp = confusion.ravel()
        print("\nTP:", tp)
        print("FP:", fp)
        print("TN:", tn)
        print("FN:", fn)

        # Performance measure
        print("\nAccuracy: " + str(sklearn.metrics.accuracy_score(y_true=all_labels, y_pred=all_predicted)))
        print("Precision: " + str(sklearn.metrics.precision_score(y_true=all_labels, y_pred=all_predicted)))
        print("F-measure: " + str(sklearn.metrics.f1_score(y_true=all_labels, y_pred=all_predicted)))
        print("Recall: " + str(sklearn.metrics.recall_score(y_true=all_labels, y_pred=all_predicted)))
        print(
            "Precision-Recall AUC: " + str(sklearn.metrics.average_precision_score(y_true=all_labels, y_score=probs2))
        )
        print("AUC: " + str(sklearn.metrics.roc_auc_score(y_true=all_labels, y_score=probs2)))
        print("MCC: " + str(sklearn.metrics.matthews_corrcoef(y_true=all_labels, y_pred=all_predicted)))
    except:
        print("This is multiclass prediction")
    return all_predicted


def softmax_accuracy(probs, all_labels):
    def getClass(x):
        return x.index(max(x))

    all_labels = all_labels.tolist()
    probs = pd.Series(probs.tolist())
    all_predicted = probs.apply(getClass)
    all_predicted.reset_index(drop=True, inplace=True)
    vc = pd.value_counts(all_predicted == all_labels)
    try:
        acc = vc[1] / len(all_labels)
    except:
        if vc.index[0] == False:
            acc = 0
        else:
            acc = 1
    return (acc, all_predicted)


class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings["input_ids"]) == len(self.encodings["attention_mask"]) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


seed = 1234

os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multigpu = False
if device == torch.device("cuda"):
    multigpu = torch.cuda.device_count() > 1
print("Device: ", device)
print("MultiGPU: ", multigpu)

# Training & vocab parameters
DATA_PATH = "data"
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = VOCAB_SIZE + 2
EMBED_DIM = 768  # 768


# Load pre-trained tokenizers
vocab, merges = BPE.read_file(vocab="tokenizer/drapgh-vocab.json", merges="tokenizer/drapgh-merges.txt")
my_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
my_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>", special_tokens=[("<s>", 0), ("<pad>", 1), ("</s>", 2), ("<unk>", 3), ("<mask>", 4)]
)
my_tokenizer.enable_truncation(max_length=1024)
my_tokenizer.enable_padding(
    direction="right", pad_id=1, pad_type_id=0, pad_token="<pad>", length=1024, pad_to_multiple_of=None
)

import sys

m3 = pd.read_json(sys.argv[1], orient="records", lines=True)

m3.func = m3.func.apply(cleaner)

test_encodings = my_tokenizer.encode_batch(m3.func)
test_encodings = [{"func": enc.ids, "target": lab} for enc, lab in zip(test_encodings, m3.target.tolist())]

test_data = TabularDataset_From_List(
    test_encodings,
    "dict",
    fields={
        "func": ("codes", torchtext.data.Field(batch_first=True, fix_length=1024, use_vocab=False)),
        "target": ("label", torchtext.data.LabelField(dtype=torch.long, is_target=True, use_vocab=False)),
    },
)

test_iterator = torchtext.data.BucketIterator(test_data, batch_size=1, sort=False, shuffle=False)

#
# Vulberta-CNN
#
model = myCNN(EMBED_SIZE, EMBED_DIM)
checkpoint = torch.load("VB-CNN_devign/model_ep_17.tar", map_location=device)

new_state_dict = OrderedDict()
for k, v in checkpoint["model_state_dict"].items():
    new_state_dict[k[7:]] = v

model.load_state_dict(new_state_dict)

model.eval()
model.to(device)
with torch.no_grad():
    all_pred = []
    for batch in test_iterator:
        batch.codes, batch.label = batch.codes.to(device), batch.label.to(device)
        output_test = model(batch.codes).squeeze(1)
        all_pred += output_test.tolist()

y_preds = [0 if p0 > p1 else 1 for p0, p1 in all_pred]
np.save("cnn_test_y_preds.npy", np.array(y_preds))

#
# Vulberta-MLP
#
my_tokenizer.enable_truncation(max_length=1024)
my_tokenizer.enable_padding(
    direction="right", pad_id=1, pad_type_id=0, pad_token="<pad>", length=None, pad_to_multiple_of=None
)

test_encodings = my_tokenizer.encode_batch(m3.func)
test_encodings = {
    "input_ids": [enc.ids for enc in test_encodings],
    "attention_mask": [enc.attention_mask for enc in test_encodings],
}
test_dataset = MyCustomDataset(test_encodings, m3.target.tolist())
test_loader = DataLoader(test_dataset, batch_size=2)


model = RobertaForSequenceClassification.from_pretrained("./VB-MLP_devign")
print(model.num_parameters())

all_pred = []
model.eval()
model.to(device)
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        acc_val, pred = softmax_accuracy(torch.nn.functional.softmax(outputs[1], dim=1), labels)
        all_pred += pred.tolist()


np.save("mlp_test_y_preds.npy", np.array(all_pred))
