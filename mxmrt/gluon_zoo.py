import os

from mxnet.gluon.model_zoo import vision
import mxnet as mx

import gluoncv as cv

from . import conf, utils

def load_inception_v3(ctx):
    return vision.inception_v3(pretrained=True, ctx=ctx, prefix="")
def save_inception_v3():
    graph = load_inception_v3(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/inception_v3.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/inception_v3.params')

def load_mobilenet1_0(ctx):
    return vision.mobilenet1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet1_0():
    graph = load_mobilenet1_0(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/mobilenet1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/mobilenet1_0.params')

def load_mobilenet_v2_1_0(ctx):
    return vision.mobilenet_v2_1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet_v2_1_0():
    graph = load_mobilenet_v2_1_0(mx.cpu())
    sym = graph(mx.sym.var('data'))
    with open('./data/mobilenet_v2_1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_parameters('./data/mobilenet_v2_1_0.params')

def load_resnet18_v1_yolo():
    return cv.model_zoo.get_model('yolo3_resnet18_v1_voc',
            pretrained=False, pretrained_base=True,
            ctx=mx.gpu())

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    classes : int
        Number of classes for the output layer.

    Returns
    -------
    HybridBlock
        The model.
    """
    return cv.model_zoo.get_model(name, pretrained=True,
            ctx=mx.gpu(), **kwargs)

def save_model(name, data_dir=None, **kwargs):
    net = get_model(name, **kwargs)
    sym = net(mx.sym.var('data'))
    if isinstance(sym, tuple):
        sym = mx.sym.Group([*sym])

    data_dir = conf.MRT_MODEL_ROOT if data_dir is None else data_dir
    prefix = os.path.join(data_dir, name)
    sym_path, prm_path = utils.extend_fname(prefix)

    # os.path.join(conf.MRT_MODEL_ROOT, name)
    # sym_path = sym_path if sym_path else "./data/%s.json"%name
    # prm_path = prm_path if prm_path else "./data/%s.params"%name
    with open(sym_path, "w") as fout:
        fout.write(sym.tojson())
    net.collect_params().save(prm_path)
    return sym_path, prm_path
