import torch
import centered_halfnorm_small_members as centered_halfnorm
import embedding_graph, learned_ridge_head
import persistent_bf16_deep3_batchnorm16_staged_back128 as teacher_model
import persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3_warmstart_featureaug_grokfast as student_model
CONFIG = dict(lr1=0.4, lr2=0.4, head_lr=0.4, momentum=0.95, dropout=0.1, weight_decay=0.001, mixup=0.2, input_scale=2.0, initialization='data_sample', dictionary_scale=0.0, data_noise=0.25, swa_fraction=0.25, schedule='cosine')

def classify(x, y, q):
    student_steps = 3000
    student_lr = 0.4 / 1.5
    grok_alpha = 0.98
    grok_lamb = 0.5
    means = torch.stack([x[y == c].mean(0) if bool((y == c).any()) else torch.zeros_like(x[0]) for c in range(10)])
    projection = torch.nn.functional.normalize(means, dim=1).T
    scale = x.square().sum(1).mean().sqrt().clamp_min(1e-08)

    def augment(t):
        return torch.cat((t, t @ projection / scale), dim=1).contiguous()
    initial = []
    queries = torch.cat((x, q))
    N = len(queries)

    def fit(xx, yy, steps, is_student=False, seed=42):
        backend = student_model if is_student else teacher_model
        E, width, batch = (4, 512, 2048) if is_student else (8, 256, 1024)
        extra = {'projection': (projection / scale).contiguous(), 'raw_noise': 0.04, 'initial_weights': initial, 'grok_alpha': grok_alpha, 'grok_lamb': grok_lamb} if is_student else {}
        backend.classify(augment(xx) if is_student else xx, yy, augment(queries) if is_student else queries, **extra, **dict(CONFIG, lr1=student_lr if is_student else CONFIG['lr1'], lr2=student_lr if is_student else CONFIG['lr2'], head_lr=student_lr if is_student else CONFIG['head_lr'], steps=steps, members=E, width=width, batch=batch, views=2, seed=seed, input_noise=0.0 if is_student else 0.15))
        state = (backend.extension(0.25, grok_alpha, grok_lamb) if is_student else backend.extension()).debug_last_state()
        if not is_student and seed == 42:
            w1 = state[5].reshape(8, 256, 128).reshape(4, 512, 128).clone()
            w1[:, :, 60:] = 0
            w2 = torch.zeros((4, 512, 512), device=x.device, dtype=torch.bfloat16)
            w3 = torch.zeros_like(w2)
            for dest, src in ((w2, state[6]), (w3, state[28])):
                src = src.reshape(8, 256, 256)
                dest[:, :256, :256] = src[0::2]
                dest[:, 256:, 256:] = src[1::2]
            head = torch.zeros((4, 512, 16), device=x.device, dtype=torch.bfloat16)
            head[:, :, :10] = state[7].reshape(4, 512, 10) * 0.5
            initial[:] = [w1.reshape(2048, 128), w2.reshape(2048, 512), head.reshape(2048, 16), w3.reshape(2048, 512)]
        z = state[30][:E * N].contiguous()
        temp = 100.0
        if is_student:
            z = centered_halfnorm.transform(z, state[11][:E * N].contiguous(), E, len(x))
            E = 1
            temp = 10.0
        p = learned_ridge_head.extension().knn(z, y, E, 5, temp)
        p = (p / p.sum(1, keepdim=True).clamp_min(1e-20)).contiguous()
        return embedding_graph.extension().run(z, y, p, E, 5, temp, 0.9, 3)
    teacher = (fit(x, y, 1000).clone() + fit(x, y, 250, seed=1042)) / 2
    confidence, pseudo = teacher.max(1)
    ids = torch.nonzero(confidence >= 0.999, as_tuple=False).flatten()
    order = torch.arange(len(q), device=q.device)
    if ids.numel():
        ids = ids[order % ids.numel()]
        px, py = (q[ids], pseudo[ids])
    else:
        px, py = (x[order % len(x)], y[order % len(x)])
    return fit(torch.cat((x, px)), torch.cat((y, py)), student_steps, True).argmax(1)
