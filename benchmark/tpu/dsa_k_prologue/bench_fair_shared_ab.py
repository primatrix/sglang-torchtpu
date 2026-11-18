"""Controlled, test-only shared/private attention comparison.

Frozen kernel: 48709f1 (identical to the f9b2ed3 Q-layout experiment).
Every timed variant writes identical FP8 KV, receives explicit donated caches
and explicit physical indices, and returns (BF16 output[..., :512], caches).
The primary comparison pins Q row-major on both sides. Intersections and slot
mapping are offline; this measures KV write + attention, NOT a full dedup chain.
Real files contain only real Top-K: Q/KV are deterministic synthetic inputs.
"""

import argparse
import gc
import glob
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import numpy as np

import bench_fp8_attention_ab as att
from bench_shared_private import atoms_for
from check_physical_topk import to_physical

att.FP = PAGE = 64
CHUNK, BATCH, TOPK = 128, 16, 2048
# (DMA rows, atoms per owner batch). Shared (32, 16) anchors the two sweeps.
COMBINED_CONFIGS = ((16, 16), (32, 16), (64, 16), (32, 8), (32, 32))


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def partition(raw, g, quantum=256):
    """Prove a disjoint, lossless partition before sending any rows to TPU."""
    t, k = raw.shape
    if t % g or CHUNK % g or BATCH % g:
        raise ValueError("g must divide tokens, chunk and batch")
    ordered = np.sort(raw, axis=1)
    if np.any(ordered < 0) or np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError("This experiment requires full, unique nonnegative Top-K")
    common = []
    for start in range(0, t, g):
        rows = ordered[start]
        for other in ordered[start + 1:start + g]:
            rows = np.intersect1d(rows, other, assume_unique=True)
        common.append(rows)
    counts = np.array([len(x) for x in common])
    ks = int(counts.min()) // quantum * quantum
    stats = dict(g=g, ks=ks, quantum=quantum, inter_min=int(counts.min()),
                 inter_mean=float(counts.mean()), inter_max=int(counts.max()),
                 rows_fraction=1 - (g - 1) / g * ks / k,
                 rows_fraction_if_mean=1 - (g - 1) / g * float(counts.mean()) / k)
    if ks == 0:
        return stats, None
    shared = np.stack([x[:ks] for x in common])
    private = np.stack([np.setdiff1d(ordered[i], shared[i // g], assume_unique=True)
                        for i in range(t)])
    reordered = np.concatenate([np.repeat(shared, g, axis=0), private], axis=1)
    if not np.array_equal(np.sort(reordered, axis=1), ordered):
        raise AssertionError("Partition changed the selected KV multiset")
    return stats, (shared, private, reordered)


def differences(out, ref):
    a, b = np.asarray(out, np.float64), np.asarray(ref, np.float64)
    delta = a - b
    return dict(finite=bool(np.isfinite(a).all()), max_abs=float(np.abs(delta).max()),
                nrmse=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
                max_normalized=float(np.abs(delta).max() / max(np.abs(b).max(), 1e-30)),
                p99_rel=float(np.percentile(np.abs(delta) / np.maximum(np.abs(b), 1e-3), 99)))


def reference(c, raw):
    """FP64 attention on the actual FP8 cache, plus unquantized BF16 reference."""
    import ml_dtypes

    sample = np.unique(np.r_[np.linspace(0, c['T'] - 1, 24).astype(int),
                             [0, 1, 2, 3, 7, 8, 15, 16, 127, 128]])
    quantized, unquantized = [], []
    prefix = int(c['prefixes'][0])
    for i in sample:
        pos = raw[i][raw[i] >= 0]
        slots = c['pi'][pos // PAGE] * PAGE + pos % PAGE
        lat = c['prefix_kv'][slots].astype(np.float64)
        rope = c['prefix_pe'][slots].astype(np.float64)
        fresh = pos >= prefix
        lat[fresh] = c['kv_c'][pos[fresh] - prefix].astype(np.float64)
        rope[fresh] = c['k_pe'][pos[fresh] - prefix].astype(np.float64)
        ql = c['q_latent'][i].astype(np.float64)
        qr = c['q_rope'][i].astype(np.float64)
        for quantize, dest in ((True, quantized), (False, unquantized)):
            kl, kr = lat, rope
            if quantize:
                kl = np.clip(lat, -448, 448).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
                kr = np.clip(rope, -448, 448).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
            scores = (ql @ kl.T + qr @ kr.T) * float(c['sm_scale'])
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            dest.append(weights @ kl)
    return sample, np.stack(quantized), np.stack(unquantized)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--gather-phase', action='store_true', help='Compare SC atom schedules with unchanged output order')
    p.add_argument('--rope-half', action='store_true', help='Compare baseline and half-size intermediate RoPE buffer at phase 5')
    p.add_argument('--combined', action='store_true', help='Measure compact phase-5 gather window and owner batch size')
    p.add_argument('--longpipe', action='store_true', help='Fold compact32 owner rounds into one longer pipeline')
    p.add_argument('--files', default='synthetic,raw-0096-layer0,raw-0097-layer10,raw-0099-layer40')
    p.add_argument('--groups', default='2,4,8,16')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--profile-iters', type=int, default=10)
    a = p.parse_args()
    if sum((a.gather_phase, a.rope_half, a.combined, a.longpipe)) > 1:
        p.error('choose at most one gather experiment')
    gather_test = a.gather_phase or a.rope_half or a.combined or a.longpipe
    os.environ.setdefault('JAX_PLATFORMS', 'tpu')
    import jax
    import jax.numpy as jnp
    from jax.experimental.layout import Format, Layout
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import prepare_gather_indices, sparse_ragged_paged_attention
    from sglang_tpu.attention.pallas.mla_sparse.native_sc_gather import dsa_gather_native_sc
    from xplane_summary import _planes, _union_ps

    root = Path(a.out_dir)
    for sub in ('benchmark', 'compiler/hlo', 'profiling/xprof'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    metrics = root / 'benchmark/metrics.jsonl'

    def emit(row):
        line = json.dumps(row, allow_nan=False)
        with metrics.open('a') as f:
            f.write(line + '\n')
        print('FAIR ' + line, flush=True)

    c = att.case('pr2198_extend_c1', seed=0)
    t, h = c['T'], att.H
    sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
    put = lambda x: jax.device_put(x, sh)
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, kcu.KVCacheLayout.SPARSECORE, c['P'], PAGE, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, kcu.KVCacheLayout.SPARSECORE, c['P'], PAGE, 64)
    q8 = lambda x: jnp.clip(jnp.asarray(x).astype(jnp.float32), -448, 448).astype(jnp.float8_e4m3fn)
    n = c['P'] * PAGE
    fill = jax.jit(lambda no, ro, kv, pe: kcu.update_sparse_mla_kv_cache_jax(
        no, ro, kv, pe, jnp.array([n], jnp.int32), jnp.arange(c['P'], dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    cache = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                 q8(c['prefix_kv']), q8(c['prefix_pe']))
    initial = tuple(np.asarray(x) for x in cache)
    del cache
    kvc, kpe = q8(c['kv_c']), q8(c['k_pe'])
    kvl, pi, cuq = (put(c[k]) for k in ('kv', 'pi', 'cu_q'))
    dist = put(np.array([0, 0, 1], np.int32))
    q_host = np.concatenate([c['q_latent'], c['q_rope']], axis=-1)
    q_inputs = dict(default=put(q_host), rowmajor=jax.device_put(q_host, fmt),
                    split=(jax.device_put(c['q_latent'], fmt), jax.device_put(c['q_rope'], fmt)))
    manifest = dict(schema_version=1, workflow='operator-optimization',
                    operator_family='sparse_attention', operator_name='dsa_shared_private',
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    kernel_base='48709f10366ae638fba548fd8ecadb30c35f9895',
                    dimensions=dict(tokens=t, heads=h, top_k=TOPK, page_size=PAGE,
                                    query_chunk=CHUNK, batch_size=BATCH, k_scale=1.0),
                    timing_scope='FP8 KV write + attention, final output512 and caches; full tuple sync',
                    exclusions=['QKV projections', 'Top-K/indexer', 'host intersection', 'slot mapping', 'model E2E'],
                    q_sha256=digest(q_host), cache_sha256=[digest(x) for x in initial],
                    current_kv_sha256=[digest(c['kv_c']), digest(c['k_pe'])],
                    devices=[str(x) for x in jax.devices()], active_device=str(jax.devices()[0]),
                    args=vars(a), acceptance={'max_sampled_error_above_base': 0.01,
                                            'nrmse_sampled_above_base': 0.005})
    if a.gather_phase:
        manifest.update(operator_name='dsa_gather_atom_phase', phases=[0, 1, 5], q_layout='split',
                        acceptance={'full_output_and_cache_bit_exact': True, 'nope_and_rope_gather_bit_exact': True},
                        gather_scope='repeated isolated 128-query chunk, first/middle/last of 16; not summed')
    if a.rope_half:
        manifest.update(operator_name='dsa_half_rope_output', phase=5, q_layout='split',
                        acceptance={'full_output_and_cache_bit_exact': True, 'gather_bytes_exact': True})
    if a.combined:
        manifest.update(operator_name='dsa_compact_window_atoms', phase=5, q_layout='split', compact_rope=True,
                        reference_commit='c07522f2fc3b2c4907578ed762d36ef84823a849',
                        rows_per_dma_and_atoms=COMBINED_CONFIGS,
                        acceptance={'full_output_and_cache_bit_exact': True, 'both_gather_outputs_bit_exact': True})
    if a.longpipe:
        manifest.update(operator_name='dsa_original_vs_final', phase=5, q_layout='split',
                        variants=['original_phase5','compact32_a32','compact32_fold4'], device_index_repack_included=True,
                        acceptance={'full_output_and_cache_bit_exact': True})
    (root.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    emit(dict(kind='manifest', **manifest))
    paths = {Path(f).stem: f for f in glob.glob(a.raw_dir + '/**/raw-*.npy', recursive=True)}
    failures = []

    def physical(rows):
        return to_physical(rows, c['pi'], c['ppr'], c['cu_q'], page=PAGE)

    def attend(q, no, ro, rows, group=1, lse=False, phase=0, rope_half=False, window=16, atoms=None, fold=1):
        nt = rows.shape[0]
        return sparse_ragged_paged_attention(
            q, no, ro, rows, None, jnp.array([0, nt], jnp.int32), dist,
            sm_scale=float(c['sm_scale']), k_scale=1.0, cache_layout='sparsecore',
            gather_and_attention_chunk_size=CHUNK // group,
            attention_kernel_batch_size=BATCH // group,
            sparsecore_atoms_per_batch=atoms_for(CHUNK // group, rows.shape[1]) if atoms is None else atoms,
            sparsecore_gather_wait_mode='overlap_rope', return_lse=lse,
            sparsecore_phase_stride=phase, compact_rope_output=rope_half, sparsecore_rows_per_dma=window,
            sparsecore_batch_fold=fold)

    def make_call(layout, rows, tag, group=1, phase=0, rope_half=False, window=16, atoms=None, fold=1):
        def step(no, ro, qin, kv, pe, indices):
            no, ro = kcu.update_sparse_mla_kv_cache(
                no, ro, kv, pe, kvl, pi, cuq, nope_spec=ns, rope_spec=rs)
            q = jnp.concatenate(qin, axis=-1) if layout == 'split' else qin
            if group == 1:
                out = attend(q, no, ro, indices, phase=phase, rope_half=rope_half, window=window, atoms=atoms, fold=fold)[..., :512]
            else:
                shared, private = indices
                if private.shape[1] == 0:
                    out = attend(q.reshape(t // group, group * h, -1), no, ro,
                                 shared, group).reshape(t, h, -1)[..., :512]
                else:
                    o1, l1 = attend(q.reshape(t // group, group * h, -1), no, ro,
                                    shared, group, True)
                    o2, l2 = attend(q, no, ro, private, 1, True)
                    o1 = o1.reshape(t, h, -1)[..., :512].astype(jnp.float32)
                    o2 = o2[..., :512].astype(jnp.float32)
                    l1, l2 = l1.reshape(t, h, 1), l2[..., None]
                    m = jnp.maximum(l1, l2)
                    w1, w2 = jnp.exp(l1 - m), jnp.exp(l2 - m)
                    out = ((o1 * w1 + o2 * w2) / (w1 + w2)).astype(q.dtype)
            return out, no, ro

        qformat = (fmt, fmt) if layout == 'split' else (fmt if layout == 'rowmajor' else sh)
        rformat = (sh, sh) if group > 1 else sh
        jf = jax.jit(step, donate_argnums=(0, 1),
                     in_shardings=(sh, sh, qformat, sh, sh, rformat))
        state = {'cache': tuple(put(x) for x in initial), 'last': None}
        rows = jax.tree.map(put, rows)
        start = time.perf_counter()
        executable = jf.lower(*state['cache'], q_inputs[layout], kvc, kpe, rows).compile()
        compile_s = time.perf_counter() - start
        hlo = executable.as_text()
        (root / f'compiler/hlo/{tag}.txt').write_text(hlo)

        def call():
            # Match the historical jitted dispatch; the compiled object is only
            # used for inspection. AOT calls require exact physical input layouts.
            state['last'] = jf(*state['cache'], q_inputs[layout], kvc, kpe, rows)
            state['cache'] = state['last'][1:]
            return state['last']

        def sync():
            jax.block_until_ready(state['last'])

        call()
        sync()
        out = np.asarray(state['last'][0])
        cache_hash = [digest(np.asarray(x)) for x in state['cache']]
        info = dict(compile_time_s=compile_s, output_shape=list(out.shape),
                    input_formats=str(executable.input_formats), cache_sha256=cache_hash,
                    q_copy_lines=[line.strip() for line in hlo.splitlines()
                                  if '= bf16[2048,64,576]' in line and ' copy(' in line],
                    output_sha256=digest(out))
        return dict(call=call, sync=sync, out=out, info=info, state=state)

    def measure(calls):
        # Alternate order across rounds; every branch is precompiled and warm.
        for f in calls.values():
            for _ in range(3):
                f['call']()
            f['sync']()
        keys = list(calls)
        times = {key: [] for key in keys}
        for rep in range(a.repeats):
            order = keys[rep % len(keys):] + keys[:rep % len(keys)]
            for key in order:
                f = calls[key]
                f['sync']()
                start = time.perf_counter()
                for _ in range(a.iters):
                    f['call']()
                f['sync']()
                times[key].append((time.perf_counter() - start) * 1e6 / a.iters)
        return {key: dict(us=statistics.median(values), us_runs=values,
                          us_range=[min(values), max(values)]) for key, values in times.items()}

    def profile(f, tag):
        if not a.profile_iters:
            return {}
        dest = root / 'profiling/xprof' / tag
        f['sync']()
        jax.profiler.start_trace(str(dest))
        for _ in range(a.profile_iters):
            f['call']()
        f['sync']()
        jax.profiler.stop_trace()
        coverage = []
        for path in dest.rglob('*.xplane.pb'):
            for plane, names, lines in _planes(memoryview(path.read_bytes())):
                if not plane.startswith('/device:TPU'):
                    continue
                for line, events in lines:
                    if line == 'XLA Modules' and events:
                        coverage.append(dict(file=path.name, plane=plane, events=len(events),
                                             module_names=sorted({names.get(e[0], '?') for e in events}),
                                             busy_us_per_call=_union_ps(events) / 1e6 / a.profile_iters))
        # Never sum top-N modules across devices or present a truncated count as total.
        return dict(profile_coverage=coverage, expected_module_calls=a.profile_iters,
                    profile_complete=bool(coverage) and all(x['events'] == a.profile_iters for x in coverage))

    def phase_experiment(name, full, base, base_error):
        phases = (0, 1, 3, 5, 7)
        calls = {0: base}
        for phase in phases[1:]:
            cand = make_call('split', full, f'{name}_phase{phase}', phase=phase)
            if (not np.array_equal(cand['out'].view(np.uint16), base['out'].view(np.uint16))
                    or cand['info']['cache_sha256'] != base['info']['cache_sha256']):
                raise AssertionError(f'phase={phase}: full attention output or written cache mismatch')
            calls[phase] = cand
        times = measure(calls)
        emit(dict(kind='phase_chain', sample=name, phases=phases, paired_timings=times,
                  output_and_cache_exact=True, fp8_reference=base_error,
                  output_sha256=base['info']['output_sha256'], cache_sha256=base['info']['cache_sha256'],
                  physical_topk_sha256=digest(full)))

        # Use exactly the bytes and sanitized physical slots the full step reads.
        # Take slices after sanitization: the scatter-padding hash uses global row numbers.
        cache_args = base['state']['cache']
        _, slots = prepare_gather_indices(put(full), None, cuq, page_size=PAGE, total_num_pages=c['P'])
        live = put(np.int32(CHUNK * TOPK))
        def micro(rows, phase):
            last = [None]
            def run():
                last[0] = dsa_gather_native_sc(*cache_args, rows, atoms_per_batch=16,
                                               out_size=CHUNK * TOPK, wait_mode='overlap_rope',
                                               num_live_indices=live, live_multiple=BATCH * TOPK,
                                               phase_stride=phase)
                return last[0]
            return {'call': run, 'sync': lambda: jax.block_until_ready(last[0])}

        for chunk in (0, t // CHUNK // 2, t // CHUNK - 1):
            chunk_rows = slots[chunk * CHUNK:(chunk + 1) * CHUNK]
            calls = {phase: micro(chunk_rows, phase) for phase in phases}
            expected = tuple(np.asarray(x) for x in calls[0]['call']())
            for phase in phases[1:]:
                actual = calls[phase]['call']()
                if not all(np.array_equal(np.asarray(x), y) for x, y in zip(actual, expected, strict=True)):
                    raise AssertionError(f'chunk={chunk}, phase={phase}: gathered bytes mismatch')
            hashes = [digest(x) for x in expected]
            del expected, actual
            times = measure(calls)
            emit(dict(kind='phase_gather', sample=name, chunk=chunk, phases=phases, paired_timings=times,
                      nope_and_rope_exact=True, output_sha256=hashes,
                      physical_slots_sha256=digest(np.asarray(chunk_rows)), scope='one repeated chunk'))

    def rope_half_experiment(name, full, base, base_error):
        cand = make_call('split', full, name + '_ropehalf', phase=5, rope_half=True)
        if (not np.array_equal(cand['out'].view(np.uint16), base['out'].view(np.uint16))
                or cand['info']['cache_sha256'] != base['info']['cache_sha256']):
            raise AssertionError('compact RoPE: attention output or written cache mismatch')
        times = measure({'baseline': base, 'compact_rope': cand})
        emit(dict(kind='rope_half_chain', sample=name, paired_timings=times,
                  output_and_cache_exact=True, fp8_reference=base_error, phase=5))

        # Compare one complete gather of 128 queries before the consumer reads it.
        _, slots = prepare_gather_indices(put(full), None, cuq, page_size=PAGE, total_num_pages=c['P'])
        slots = slots[8 * CHUNK:9 * CHUNK]
        live = put(np.int32(CHUNK * TOPK))
        def gather_half(half):
            last = [None]
            def run():
                last[0] = dsa_gather_native_sc(*base['state']['cache'], slots, out_size=CHUNK * TOPK,
                                               atoms_per_batch=16, wait_mode='overlap_rope',
                                               num_live_indices=live, live_multiple=BATCH * TOPK,
                                               phase_stride=5, compact_rope_output=half)
                return last[0]
            return {'call': run, 'sync': lambda: jax.block_until_ready(last[0])}

        calls = {name_: gather_half(half) for name_, half in [('baseline', False), ('compact_rope', True)]}
        n0, r0 = [np.asarray(x) for x in calls['baseline']['call']()]
        n1, r1 = [np.asarray(x) for x in calls['compact_rope']['call']()]
        orig = r0.reshape(-1, 16, 128)
        half = r1.reshape(-1, 8, 128)
        if (not np.array_equal(n0, n1) or np.any(orig[:, :, 64:])
                or not np.array_equal(half[:, :, :64], orig[:, :8, :64])
                or not np.array_equal(half[:, :, 64:], orig[:, 8:, :64])):
            raise AssertionError('compact RoPE: gather buffer differs after canonicalization')
        emit(dict(kind='rope_half_gather', sample=name, chunk=8, phase=5,
                  nope_and_canonical_rope_exact=True, rope_bytes=[r0.nbytes, r1.nbytes],
                  paired_timings=measure(calls)))

    def combined_experiment(name, full, base, base_error):
        # All variants share the same compact physical output and the same input slots.
        # Check the SC buffer separately so a consistently permuted K/V list cannot pass unnoticed.
        _, slots = prepare_gather_indices(put(full), None, cuq, page_size=PAGE, total_num_pages=c['P'])
        slots = slots[8 * CHUNK:9 * CHUNK]
        live = put(np.int32(CHUNK * TOPK))
        chains, micros = {}, {}
        def gather(rows, atoms, fold):
            last = [None]
            def call():
                last[0] = dsa_gather_native_sc(*base['state']['cache'], slots, out_size=CHUNK * TOPK,
                                               atoms_per_batch=atoms, rows_per_dma=rows, phase_stride=5,
                                               compact_rope_output=True, num_live_indices=live,
                                               live_multiple=BATCH * TOPK, batch_fold=fold)
                return last[0]
            return {'call': call, 'sync': lambda: jax.block_until_ready(last[0])}

        if a.longpipe:
            explicit = {'phase5': base}
            for tag, atoms, nfold in [('compact32_a32',32,1), ('compact32_fold4',16,4)]:
                trial = make_call('split', full, f'{name}_{tag}', phase=5, rope_half=True,
                                  window=32, atoms=atoms, fold=nfold)
                if (not np.array_equal(base['out'].view(np.uint16), trial['out'].view(np.uint16))
                        or base['info']['cache_sha256'] != trial['info']['cache_sha256']):
                    raise AssertionError(f'{tag}: difference from original native phase5')
                explicit[tag] = trial
            emit(dict(kind='final_chain', sample=name, paired_timings=measure(explicit),
                      canonical_input_sha256=digest(full), full_output_and_cache_exact=True,
                      native_output_sha256=base['info']['output_sha256'], native_cache_sha256=base['info']['cache_sha256']))
            return
        configs = ([(32, 16, f) for f in (1, 2, 4, 8)] if a.longpipe else
                   [(w, n, 1) for w, n in COMBINED_CONFIGS])
        for window, atoms, fold in configs:
            control = (window, atoms, fold) == configs[0]
            tag = f'fold{fold}' if a.longpipe else f'w{window}_a{atoms}'
            try:
                if control:
                    variant = base
                else:
                    variant = make_call('split', full, f'{name}_{tag}', phase=5, rope_half=True,
                                        window=window, atoms=atoms, fold=fold)
                if (not np.array_equal(base['out'].view(np.uint16), variant['out'].view(np.uint16))
                        or base['info']['cache_sha256'] != variant['info']['cache_sha256']):
                    raise AssertionError('attention output or persistent cache mismatch')
                micro = gather(window, atoms, fold)
                hashes = [digest(np.asarray(x)) for x in micro['call']()]
                if control:
                    oracle = hashes
                elif hashes != oracle:
                    raise AssertionError('NoPE or compact RoPE gather bytes mismatch')
                chains[tag], micros[tag] = variant, micro
            except Exception as err:
                if control:
                    raise
                failures.append(f'{name}/{tag}: {err!r}')
                emit(dict(kind='failure', sample=name, variant=tag, error=repr(err)))
                traceback.print_exc()

        # micro() obtains a fresh cache handle after donated chain calls update state.
        exp_name = 'longpipe' if a.longpipe else 'combined'
        for kind, calls in [(exp_name + '_chain', chains), (exp_name + '_gather', micros)]:
            timings = measure(calls)
            emit(dict(kind=kind, sample=name, measured=list(calls), compact=True, phase=5,
                      fp8_reference=base_error, full_output_and_cache_exact=True, both_gather_buffers_exact=True,
                      gather_chunk=8, physical_topk_sha256=digest(full), output_sha256=base['info']['output_sha256'],
                      gather_sha256=oracle, paired_timings=timings))

    for name in a.files.split(','):
        print('FAIR_PROGRESS ' + name, flush=True)
        try:
            raw = c['topk'].copy() if name == 'synthetic' or (gather_test and name == 'masked') else np.load(paths[name]).astype(np.int32)
            assert raw.shape == (t, TOPK)
            assert (raw >= 0).all() and (raw <= c['prefixes'][0] + np.arange(t)[:, None]).all()
            assert np.all(np.diff(np.sort(raw, axis=1), axis=1) > 0)
            if name == 'masked' and gather_test:
                lengths = np.resize([1, 63, 64, 127, 128, 1025, 2047, 2048], t)
                raw[np.arange(TOPK)[None, :] >= lengths[:, None]] = -1
            full = physical(raw)
            sample, ref_quant, ref_bf = reference(c, raw)
            layout = 'split' if gather_test else 'rowmajor'
            compact = a.combined
            base = make_call(layout, full, name + '_base', phase=5 if a.rope_half or a.longpipe or compact else 0,
                             rope_half=compact, window=16)
            base_error = differences(base['out'][sample], ref_quant)
            if (not np.isfinite(base['out']).all() or not base_error['finite']
                    or base_error['max_normalized'] > 0.03):
                raise AssertionError(f'Baseline FP8-cache reference failed: {base_error}')
            if compact or a.longpipe:
                combined_experiment(name, full, base, base_error)
                del base
                gc.collect()
                continue
            if a.rope_half:
                rope_half_experiment(name, full, base, base_error)
                del base
                gc.collect()
                continue
            if a.gather_phase:
                phase_experiment(name, full, base, base_error)
                del base
                gc.collect()
                continue
            base_row = dict(kind='baseline', sample=name, variant='q_rowmajor',
                            topk_sha256=digest(raw), reference_queries=sample.tolist(),
                            error_vs_quantized=base_error,
                            error_vs_bf16=differences(base['out'][sample], ref_bf), **base['info'])
            base_row.update(measure({'baseline': base})['baseline'])
            base_row.update(profile(base, name + '_base'))
            base_row['latency_ms'] = base_row['us'] / 1000
            emit(base_row)

            # Equal output/cache and timing boundaries, different Q input layouts only.
            for layout in ('default', 'split'):
                other = make_call(layout, full, name + '_q_' + layout)
                exact = bool(np.array_equal(other['out'].view(np.uint16), base['out'].view(np.uint16)))
                if not exact or other['info']['cache_sha256'] != base['info']['cache_sha256']:
                    raise AssertionError('Layout-only control changed output or cache')
                times = measure({'baseline': base, layout: other})
                emit(dict(kind='layout_control', sample=name, variant='q_' + layout,
                          exact_vs_rowmajor=exact, paired_timings=times, **other['info'],
                          latency_ms=times[layout]['us'] / 1000))
                del other
                gc.collect()

            for g in map(int, a.groups.split(',')):
                tag = name + '_g' + str(g)
                try:
                    start = time.perf_counter()
                    stats, parts = partition(raw, g)
                    stats['host_partition_s_excluded'] = time.perf_counter() - start
                    row = dict(kind='shared_private', sample=name, variant='g' + str(g), **stats)
                    if parts is None:
                        emit(dict(row, status='skipped_zero_static_ks'))
                        continue
                    shared, private, reordered = parts
                    row['atoms'] = [atoms_for(CHUNK // g, shared.shape[1]),
                                    atoms_for(CHUNK, private.shape[1]) if private.shape[1] else None]
                    row['partition_exact'] = True
                    candidate = make_call('rowmajor', (physical(shared), physical(private)), tag, group=g)
                    ordered_base = make_call('rowmajor', physical(reordered), tag + '_ordered_base')
                    row['error_vs_base'] = differences(candidate['out'], base['out'])
                    row['error_vs_reordered_base'] = differences(candidate['out'], ordered_base['out'])
                    row['error_vs_quantized'] = err = differences(candidate['out'][sample], ref_quant)
                    row['error_vs_bf16'] = differences(candidate['out'][sample], ref_bf)
                    row['cache_equal'] = (candidate['info']['cache_sha256'] == base['info']['cache_sha256']
                                          == ordered_base['info']['cache_sha256'])
                    row['numerics_pass'] = (err['finite'] and row['error_vs_base']['finite']
                                            and np.isfinite(ordered_base['out']).all().item()
                                            and row['cache_equal']
                                            and err['max_normalized'] <= base_error['max_normalized'] + 0.01
                                            and err['nrmse'] <= base_error['nrmse'] + 0.005)
                    if not row['numerics_pass']:
                        emit(dict(row, status='numerical_failure'))
                        failures.append(tag + ':numerics')
                        continue
                    times = measure({'baseline': base, 'reordered_baseline': ordered_base, 'split': candidate})
                    row.update(candidate['info'])
                    row.update(paired_timings=times, latency_ms=times['split']['us'] / 1000,
                               ratio_vs_base=statistics.median([x / y for x, y in zip(times['split']['us_runs'], times['baseline']['us_runs'])]),
                               ratio_vs_reordered=statistics.median([x / y for x, y in zip(times['split']['us_runs'], times['reordered_baseline']['us_runs'])]),
                               status='measured')
                    row.update(profile(candidate, tag))
                    emit(row)
                    del candidate, ordered_base
                except Exception as e:
                    failures.append(tag + ':' + repr(e))
                    emit(dict(kind='failure', sample=name, variant='g' + str(g), error=repr(e)))
                    traceback.print_exc()
                gc.collect()
            del base
        except Exception as e:
            failures.append(name + ':' + repr(e))
            emit(dict(kind='failure', sample=name, error=repr(e)))
            traceback.print_exc()
        jax.clear_caches()
        gc.collect()
    emit(dict(kind='completion', status='failed' if failures else 'completed', failures=failures))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
