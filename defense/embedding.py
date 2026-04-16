import torch

def vec(model):
    """
    Flatten the parameters of the model that require gradients into a 1D vector.
    """
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])

def load_vec(model, v):
    """
    Reload the 1D vector back into the model parameters.
    """
    p = 0
    for t in model.parameters():
        if not t.requires_grad:
            continue
        n = t.numel()
        t.data.copy_(v[p:p+n].view_as(t))
        p += n

def embed_watermark(v, idxs, bits, alpha):
    """
    Embed a bit sequence into the vector 'v' at specified indices.
    Logic: Flip and replace the parameter only if the polarities are inconsistent.
    """
    for j, i in enumerate(idxs):
        tgt = 1.0 if bits[j] else -1.0
        if (v[i] > 0) != (tgt > 0):
            v[i] = alpha * tgt
    return v