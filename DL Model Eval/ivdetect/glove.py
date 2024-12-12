"""Python wrapper to call StanfordNLP Glove."""

from util import tokenize
import numpy as np


def glove_dict(dict_path, cache=True):
    """Load glove embeddings and vocab.

    Example:
    vectors_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    """
    # Read into dict
    embeddings_dict = {}
    with open(dict_path + "/vectors.txt", "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # Read vocab
    with open(dict_path + "/vocab.txt", "r") as f:
        vocab = [i.split()[0] for i in f.readlines()]
        vocab = dict([(j, i) for i, j in enumerate(vocab)])

    return embeddings_dict, vocab


def get_embeddings(text: str, emb_dict: dict, emb_size: int = 100):
    """Get embeddings from text, zero vectors for OoV.

    Args:
        text (str): Text as string. Should be preprocessed, tokenised by space.
        emb_dict (dict): Dict with key = token and val = embedding.
        emb_size (int, optional): Size of embedding. Defaults to 100.

    Returns:
        np.array: Array of embeddings, shape (seq_length, emb_size)
    """
    return [
        emb_dict[i] if i in emb_dict else np.full(emb_size, 0.001) for i in text.split()
    ]


def get_embeddings_list(li: list, emb_dict: dict, emb_size: int = 100) -> list:
    """Get embeddings from a list of sentences, then average.

    Args:
        li (list): List of sentences.
        emb_dict (dict): Dict with key = token and val = embedding.
        emb_size (int, optional): Size of embedding. Defaults to 100.

    Example:
    li = ['static long ec device ioctl xcmd struct cros ec dev ec void user arg',
        'struct cros ec dev ec',
        'void user arg',
        'static long',
        'struct cros ec dev ec',
        'void user arg',
        '']

    glove_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    emb_dict, _ = glove_dict(glove_path)
    emb_size = 200
    """
    li = [tokenize(i) for i in li]
    li = [i if len(i) > 0 else "<EMPTY>" for i in li]
    return [np.mean(get_embeddings(i, emb_dict, emb_size), axis=0) for i in li]
