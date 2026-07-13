# embedding.py - step 2 of the pipeline: abstracts go in, vectors come out
#
# computers cant compare texts, but they can compare numbers. MiniLM turns
# each abstract into 384 numbers that capture what its about. similar papers
# get similar numbers. thats what lets us do "how alike are these two papers"
# as plain math later.
#
# strict rule: this file knows nothing about csvs, users, rooms or paths.
# main.py reads the users file, hands us texts, we hand back vectors. thats it.

# numpy = the standard library for big tables of numbers. clustering expects
# the vectors in exactly this form
import numpy as np


def embed(abstracts, model_name="sentence-transformers/all-MiniLM-L6-v2",
          show_progress=True):
    """list of abstract texts in -> one length-1 vector per text, same order."""

    # abstracts     - a plain list of texts
    # model_name    - all-MiniLM-L6-v2 by default, the winner of my model
    #                 choice experiment. its an argument so it can be swapped
    # show_progress - loading bar on or off

    # torch is heavy and slow to load, so we import it only when this function
    # is actually called. never called = never loaded. and python only pays
    # this price once - the second call sees its already loaded and skips it
    import torch
    from sentence_transformers import SentenceTransformer

    # pick the chip that does the math. nvidia and amd (rocm) cards say yes
    # here and get the fast path. apple, intel, no card = cpu. the cpu gives
    # the exact same numbers, just slower
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # say where its running so a slow run isnt a mystery
    print(f"embedding {len(abstracts)} abstracts on {device}...")

    # load the model. the very first run downloads it (~90 MB, one time),
    # after that it loads from disk instantly
    model = SentenceTransformer(model_name, device=device)

    # the actual work. normalize_embeddings=True scales every vector to
    # length 1 so similarity becomes a simple dot product - clustering and
    # grasp both assume this, do not remove it
    vectors = model.encode(list(abstracts),
                           normalize_embeddings=True,
                           show_progress_bar=show_progress)

    # return as float32 (half the memory, plenty precise). one row per paper,
    # 384 columns, same order as the input - row 7 belongs to abstract 7
    return np.asarray(vectors, dtype=np.float32)