from pathlib import Path
from functools import lru_cache
import os

@lru_cache(None)
def extension(gain=0.25, alpha=0.98, lamb=2.0):
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    from torch.utils.cpp_extension import load
    p = Path(__file__).resolve().parent / 'kernels'
    return load(name='sutro_persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3_warmstart_featureaug_grokfast__bn_stagepack' + str(int(gain * 100)) + '_' + str(int(alpha * 10000)) + '_' + str(int(lamb * 100)), sources=[str(p / 'persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3_warmstart_featureaug_grokfast_bindings.cpp'), str(p / 'persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3_warmstart_featureaug_grokfast.cu')], extra_cflags=['-O0', '-g0'], extra_cuda_cflags=['-DGROK_ALPHA=' + str(float(alpha)) + 'f', '-DGROK_LAMB=' + str(float(lamb)) + 'f', '-DSKIP_GAIN=' + str(float(gain)) + 'f', '-O3', '-lineinfo', '--ptxas-options=-v'], extra_ldflags=['-lcublas', '-lcublasLt'], verbose=False)

def classify(x, y, q, width=256, members=4, batch=512, views=1, steps=400, lr1=0.3, lr2=0.3, head_lr=0.3, momentum=0.9, seed=42, use_graph=True, schedule='none', weight_decay=0.0001, dropout=0.0, input_noise=0.0, input_scale=-1.0, optimizer='nesterov', mixup=0.0, init_gain=1.0, head_gain=0.0, initialization='gaussian', swa_fraction=0.0, dictionary_scale=1.0, data_noise=0.5, skip_gain=0.25, initial_weights=None, projection=None, scale_jitter=0.0, raw_noise=0.0, axis_jitter=0.0, mask_prob=0.0, rotation=0.0, grok_alpha=0.98, grok_lamb=2.0):
    if initial_weights is None or projection is None:
        raise ValueError('initial weights and projection required')
    schedule_id = {'none': 0, 'cosine': 1, 'step': 2}[schedule]
    if optimizer == 'momentum':
        schedule_id += 3
    if optimizer == 'sgd':
        momentum = 0.0
    init_kind = {'gaussian': 0, 'paired': 1, 'hadamard': 2, 'xavier': 3, 'kaiming': 4, 'xavier_relu': 5, 'data_centroid': 6, 'data_sample': 7}[initialization]
    return extension(skip_gain, grok_alpha, grok_lamb).classify(x, y, q, width, members, batch, views, steps, lr1, lr2, head_lr, momentum, seed, use_graph, schedule_id, weight_decay, dropout, input_noise, input_scale, mixup, init_gain, head_gain, init_kind, swa_fraction, dictionary_scale, data_noise, initial_weights, projection, scale_jitter, raw_noise, axis_jitter, mask_prob, rotation)
