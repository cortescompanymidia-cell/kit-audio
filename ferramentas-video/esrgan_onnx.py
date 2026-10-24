#!/usr/bin/env python3
"""Converte o Real-ESRGAN "realesr-general-x4v3" (.pth) para ONNX sem precisar do PyTorch.

O .pth é um zip com os tensores e um pickle que os descreve. Este script lê os pesos direto com numpy
e monta a rede SRVGGNetCompact (conv 3→64, 32 blocos conv 64→64 + PReLU, conv 64→48, pixel shuffle ×4,
somada à imagem ampliada por vizinho mais próximo).

Uso: python3 esrgan_onnx.py realesr-general-x4v3.pth realesr-general-x4v3.onnx
"""
import pickle, sys, zipfile, numpy as np


def carregar_pth(caminho):
    z = zipfile.ZipFile(caminho)
    raiz = z.namelist()[0].split('/')[0]
    tipos = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'DoubleStorage': np.float64, 'LongStorage': np.int64}

    class Storage:
        def __init__(self, tipo, chave): self.dados = np.frombuffer(z.read(f'{raiz}/data/{chave}'), tipos[tipo])

    def rebuild(storage, offset, shape, stride, *_):
        if not shape: return storage.dados[offset].copy()
        st = tuple(s * storage.dados.itemsize for s in stride)
        return np.lib.stride_tricks.as_strided(storage.dados[offset:], shape, st).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, nome):
            if nome == '_rebuild_tensor_v2': return rebuild
            if nome in tipos: return nome
            if mod == 'collections' and nome == 'OrderedDict':
                import collections; return collections.OrderedDict
            return super().find_class(mod, nome)

        def persistent_load(self, pid):
            _, tipo, chave, *_ = pid
            return Storage(tipo if isinstance(tipo, str) else tipo.__name__, chave)

    d = U(z.open(f'{raiz}/data.pkl')).load()
    return d.get('params', d.get('params_ema', d))


def montar_onnx(w, saida):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nos, inits = [], []
    conv_idx = sorted({int(k.split('.')[1]) for k in w if k.endswith('.weight') and w[k].ndim == 4})
    prelu_idx = sorted({int(k.split('.')[1]) for k in w if k.endswith('.weight') and w[k].ndim == 1})
    x = 'entrada'
    for i in sorted(conv_idx + prelu_idx):
        if i in conv_idx:
            inits += [numpy_helper.from_array(w[f'body.{i}.weight'].astype(np.float32), f'w{i}'),
                      numpy_helper.from_array(w[f'body.{i}.bias'].astype(np.float32), f'b{i}')]
            nos.append(helper.make_node('Conv', [x, f'w{i}', f'b{i}'], [f'c{i}'], pads=[1, 1, 1, 1]))
            x = f'c{i}'
        else:
            inits.append(numpy_helper.from_array(w[f'body.{i}.weight'].astype(np.float32).reshape(-1, 1, 1), f'a{i}'))
            nos.append(helper.make_node('PRelu', [x, f'a{i}'], [f'p{i}']))
            x = f'p{i}'
    nos.append(helper.make_node('DepthToSpace', [x], ['ps'], blocksize=4, mode='CRD'))
    inits.append(numpy_helper.from_array(np.array([1, 1, 4, 4], np.float32), 'esc'))
    nos.append(helper.make_node('Resize', ['entrada', '', 'esc'], ['base'], mode='nearest'))
    nos.append(helper.make_node('Add', ['ps', 'base'], ['saida']))
    g = helper.make_graph(nos, 'srvgg', [helper.make_tensor_value_info('entrada', TensorProto.FLOAT, [1, 3, None, None])],
                          [helper.make_tensor_value_info('saida', TensorProto.FLOAT, [1, 3, None, None])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(m)
    onnx.save(m, saida)


if __name__ == '__main__':
    pesos = carregar_pth(sys.argv[1])
    montar_onnx(pesos, sys.argv[2])
    print('ok', sys.argv[2])
