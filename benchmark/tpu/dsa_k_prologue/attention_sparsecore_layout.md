# TorchTPU DSA attention 优化：SparseCore KV 布局

日期：2026-11-17
PR：sgl-project/sglang-torchtpu#2246（Draft，头提交 e8be99e，分支 `feat/dsa-sparsecore-kv-layout`）
依赖：sgl-project/sglang-torchtpu#2070（Tarun-Arora，OPEN）
脚本：`benchmark/tpu/dsa_k_prologue/bench_target_baseline.py`、`bench_target_gather_sweep.py`、`bench_sc_diag.py`、`hlo_copy_report.py`

## 结论

- 打开 `SGLANG_TPU_DSA_SPARSECORE_KV_CACHE=1` 后，KV cache 改用 SparseCore 布局（uint32），attention 走 native SparseCore gather。
- PR 2198 的 extend 形状上，attention（KV 写入 + sparse attention）从 4.40 ms 降到 3.70 ms（逻辑 top-k，−16%）；配合物理 slot 降到 3.49～3.55 ms（−19%～−21%）。
- verify / decode 持平，在波动范围内。
- 四个形状、所有变体的输出都和原版 Target 逐字节一致。
- 每 token 字节数不变，KV 容量不受影响。
- 后续（2026-11-18，见文末“top-k 去重”一节）：extend 剩下的约 2.9 ms 是 SC gather 的每行开销。真实 GLM-5.2 top-k 在 128 query 内去重后只剩约 4%，SC 标记表去重 16 个 chunk 只要约 270 us，但 dense-over-distinct 的 attention（不分组和分组都试过）比现在每 chunk 约 220 us 的 gather + attention 更慢。VMEM 内按行 sparse 读取也测了（exp-85fudprrt4）：TensorCore 最快 1.46 ns/行，是 SC 从 HBM gather（0.547 ns/行）的 2.7 倍；LLO bundle（exp-7uv7xzb1cp）显示每行至少一条标量 `sld` 取行号，完美流水下限约 0.57-0.63 ns/行，仍高于 SC gather，KV 加载的去重路线暂停。在现有 Pallas 能力下，extend gather 这条线的收益就是上面的 −16%～−21%。

## 背景

上一篇 profile 拆分（exp-0ptaq1t3hy）的结论：extend 下 4.3 ms 里 TensorCore 只忙 0.6～1.0 ms，32 个 SparseCore TEC 各忙 2.4～3.0 ms，瓶颈是 SparseCore gather。所以先扫 gather 参数。

## 第一步：gather 参数扫描（exp-cdkg2zm04b，提交 d9dd16a）

直接用 JAX 调用 kernel，wall time 中位数，µs：

| 用例 | 原版 Target | TensorCore 布局最优 | SparseCore 布局最优 |
|---|---|---|---|
| extend_c1 | 4329 | 4181（物理 slot，chunk 128 / batch 16，−3.4%） | **3577**（chunk 128 / atoms 16 / overlap_rope，−17.4%） |
| verify_b4 | 141.1 | **129.1**（物理 slot，batch 4，−8.5%） | 131.2（atoms 4 / overlap_rope） |
| decode_b4 | 82.2 | 80～85，在波动内 | 84～93 |

- TensorCore 布局能调的只有 chunk 和 batch。batch 32 编译失败（VMEM 93.6 MB > 64 MB），chunk 512 变慢。
- SparseCore 布局：`atoms_per_batch` 16 最好；`overlap_rope` 比 `before_rope` 快 2%～4%；chunk 64 和 128 接近，256 变慢。verify 下 atoms 16 不能整除（24 × 2048 行），代码里自动降到 8。
- 只有 SparseCore 布局能让 extend 有明显收益，所以把它接进 KV pool。

## 第二步：两种布局

| | TensorCore 布局（原有） | SparseCore 布局（新增） |
|---|---|---|
| NoPE | `u8[P, page, 4, 128]` | `u32[P, page, 128]` |
| RoPE | `u8[P, page/4, 4, 128]` | `u32[P, page/4, 128]` |
| gather | `dsa_gather`（TensorCore 格式） | `native_sc_gather.dsa_gather_native_sc` |
| 每 token 字节 | 512 + 128 | 512 + 128（相同） |

代码改动（全部在 `sglang_tpu/` 下，core 没有新增 `is_tpu()` 分支）：

- `envs.py`：新增 `SGLANG_TPU_DSA_SPARSECORE_KV_CACHE`，默认关闭。
- `kv_pool.py`：`TpuSplitMLATokenToKVPool` 在 flag 打开时按 u32 形状分配。`bytes_per_token` 不变。
- `mla_sparse_fused.py`：
  - `cache_layout_of(cache_nope)`：按 dtype 判断布局，u32 就是 SparseCore。kernel 不读 env，布局由 cache 自己说明。
  - SparseCore 布局用扫参得到的设置：chunk 128、atoms 16；`_sparsecore_atoms` 在 chunk 不能整除时把 atoms 减半（extend 16、verify 8、decode 4）。
  - SparseCore 布局下不走 q-block（q-block kernel 按 uint8 TensorCore 页做 DMA）。
- `topk_is_physical`（同一个 PR）：top-k 直接给物理 slot（`page * page_size + offset`），跳过 `gather_page_ids`，省约 150～210 µs。

## 第三步：经过 torch op 后变慢的问题

### 现象

同样的 kernel 设置，直接调 JAX 时 extend 3.6 ms，经过 torch op（`torch.compile(backend="tpu")`）却要 **23.2 ms**（exp-42iw4cidov）。

### 定位（HLO dump，exp-wr4cl7mbh6）

- torch op 的编译结果里多出 16 个 `copy`，每个约 1046 µs，复制的是 `u32[262144,128]`，源头是同一个 `empty` buffer。直接调 JAX 时没有这些 copy。
- 原因：`core_map_helper.kernel` 用 `lax.empty` + `new_ref` + `core_map` 预先分配 gather 的输出，降成无输入、无副作用的 `AllocateBuffer`。torch_tpu 把 16 个 chunk 内联进一张图后跑 CSE，16 个相同的 `AllocateBuffer` 被合并成一个。每个 chunk 的 gather 又要原地写这个 buffer，XLA 的 CopyInsertion 只能在每次 gather 前复制一份。

### 两种修法的对比

| 修法 | 做法 | extend_c1 µs |
|---|---|---|
| 未修 | | 23200 |
| `optimization_barrier` | 隔开各 chunk 的 buffer | 无效 |
| 我们的修法 1 | chunk i 写进 chunk i−2 已读完的 gather 输出（`out_init`） | 4834 |
| 我们的修法 2（exp-cn1qjruygs / exp-vx111hhabc，提交 16dd48b） | 再让前两个 chunk 各用独立 buffer（第 1 块零填充） | 3650.8 |
| **#2070（exp-tl8dtiawq8，提交 fcba0af）** | helper 改用 `api.jit(pl.kernel(...))`，输出直接是 `tpu_custom_call` 的结果，不再生成 `AllocateBuffer` | **3505.5** |

- 我们的修法只在 attention 里绕开问题，还剩一个 `q_rope` copy 和第 1 块的零填充。
- #2070 从根上去掉了 `AllocateBuffer`，比我们的修法再快约 140 µs，改动更小，也覆盖 MoE 里用同一个 helper 的 kernel（`ragged_gather_v2`、`ragged_gather_reduce_v2`）。
- 所以采用 #2070，删掉我们的 `out_init` / chunk i−2 复用逻辑（提交 18d2f01）。
- #2070 还没合并。它合并之前，SparseCore 布局经 torch op 仍是约 23 ms，所以 PR 2246 依赖它。

## 结果

PR 分支（基于上游 main e1c3e43）加上 #2070 的 helper，exp-a6oeqeo568，测试分支提交 0299c60。四形状，wall time 中位数，µs；所有列 `exact_vs_target` 都是 true：

| 用例 | 原版 Target | 物理 slot | SparseCore + 逻辑 top-k | SparseCore + 物理 slot |
|---|---|---|---|---|
| extend_c1 | 4417.4 | 4222.7 | 3701.8（−16.2%） | **3488.3（−21.0%）** |
| extend_c2 | 4403.7 | 4228.1 | 3699.4（−16.0%） | **3547.7（−19.4%）** |
| verify_b4 | 179.5 | 177.2 | 166.0 | 172.2 |
| decode_b4 | 115.4 | 126.0 | 130.3 | 120.4 |

- extend：逻辑 top-k 比物理 slot 慢 150～210 µs，就是 `gather_page_ids` 的开销。
- verify / decode：各列差距在 15 µs 以内，每列自身的波动范围有 15～20 µs，算持平。这两个形状的瓶颈仍是 kernel 外的小 op，换布局帮不上。
- 精度：`max_rel` / `median_rel`（对 f64 参考）和原版完全相同，因为输出逐字节一致。
- 提交 fcba0af 上抓到的 SparseCore + 物理 slot 的 extend_c1 profile：device module 3412.8 µs，TensorCore 线上 attention 669.4 µs。

## 测试

在 TPU 上通过（exp-8fm0fiemrs，提交 18d2f01，164 个用例；exp-a6oeqeo568，PR 分支，44 个用例）：

| 文件 | 新增用例 | 防的是什么 |
|---|---|---|
| `test_mla_fp8_kv.py`（CPU） | `test_split_pool_sparsecore_layout_is_uint32_with_the_same_bytes` | flag 打开后 nope / rope 是 u32 独立 tensor，每层和总字节数与 u8 布局相同，否则 KV 预算算错 |
| `test_mla_sparse_native_sc_attention.py` | `test_sparsecore_layout_spans_chunks_of_two_sizes` | 448 token 分 4 个 chunk（128/128/128/64），最后一块较短，结果和 TensorCore 布局一致 |
| `test_sc_ragged_gather_v2.py`、`test_sc_ragged_gather_reduce_v2.py` | 无（已有，72 + 64 个） | 换上 #2070 的 helper 后 MoE 的 SparseCore kernel 仍然正确 |

## 局限和待办

- 依赖 #2070 合并。
- `topk_is_physical` 目前在 main 上没有调用方，等能输出物理 slot 的 indexer 进来后才用到。
- 不能和 `SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT` 一起用：`sparse_mla_prefill_window` 写死 TensorCore 布局。
- sgl-project/sglang-torchtpu#2212（DRZDONG，OPEN）会把 SparseCore 布局的 RoPE 压到 64 B/token，合并后 u32 rope 的形状和 `kv_pool.py` 的分配要跟着改。
- 没有做 GLM-5.2 双机端到端验证。
- PR 2198 已被作者关闭，改走加载时动态转 channelwise FP8。这改的是权重，不影响 attention 读的 KV cache；新的 serving 配置出来后要核对 page size、每个 DP 的 token 数和 verify token 数。

## 下一步

- 给 SparseCore 布局做 profile 拆分：extend 下 gather 和 attention 的时间与重叠情况，判断能否让下一块的 gather 和当前块的 attention 重叠。
- verify / decode：torch op 比直接调 JAX 慢约 35 µs，查清是哪些小 op（FP8 量化、KV 写入、slice），再考虑合并。

## SparseCore 布局的 profile 拆分（exp-w37z40wjy3，提交 b88c969）

`bench_sc_diag.py`，每个变体一个进程，物理 slot，10 次调用的 xprof 平均；三列输出 sha 一致。单位 us。

| 用例 | 变体 | wall | device module | TC op 忙 | tc_attention | sc_gather（TC 线） | kv_store | other |
|---|---|---|---|---|---|---|---|---|
| extend_c1 | jax_sc | 3479 | 3404 | 246 | 116 | 2 | 7 | 19 |
| extend_c1 | torch_sc | 3513 | 3413 | 302 | 93 | 9 | 0 | 42 |
| extend_c1 | torch_tc | 4242 | 4128 | 264 | 42 | 4 | 19 | 59 |
| verify_b4 | jax_sc | 131 | 106 | 80 | 35 | 26 | 7 | 8 |
| verify_b4 | torch_sc | 165 | 127 | 80 | 24 | 20 | 5 | 24 |
| verify_b4 | torch_tc | 165 | 129 | 81 | 24 | 22 | 5 | 23 |
| decode_b4 | jax_sc | 89 | 61 | 45 | 11 | 12 | 8 | 13 |
| decode_b4 | torch_sc | 114 | 65 | 47 | 11 | 12 | 5 | 14 |
| decode_b4 | torch_tc | 123 | 61 | 43 | 11 | 12 | 2 | 12 |

- extend：3.4 ms 里 TensorCore 只忙约 0.25-0.30 ms，约 91% 时间在等 SparseCore gather。torch op 与直接 JAX 的 device 时间相同（3413 对 3404），torch 路径已无额外开销。搬运约 2048 x 2048 x 640 B = 2.7 GB，折合约 0.8 TB/s。
- verify：torch op 比直接 JAX 多 35 us，其中 device 21 us（`other` 多出约 15 us：`fusion.31` 8 us 及若干 1-2 us 的 index fusion，只在 torch 路径出现），host 约 14 us。sweep 里 verify 的 129 us 是直接 JAX 的数，与这里 jax_sc 131 us 一致。
- decode：device 只差 4 us，wall 差 25 us，主要是 host 下发开销。
- 口径：SparseCore 线统计（`sc_lines`）把非 TEC 线也算了进去（max 超过 module 跨度），本次不可用；extend 的 gather 占比以 module 跨度减 TC 忙碌时间为准。

## SparseCore gather 瓶颈拆分（exp-h1pj43wtu2，测试分支 bc77f66，带 #2212）

`bench_sc_gather_micro.py` 单独调 `dsa_gather_native_sc`，随机索引，atoms 16，us：

| 测量 | 结果 |
|---|---|
| 行数 65,536 / 131,072 / 262,144 / 524,288 | 72.2 / 108.1 / 179.4 / 322.8 |
| RoPE 128 B 对比 64 B（262,144 行） | 178.9 对比 176.5（-1.3%） |
| RoPE 128 B 对比 64 B（524,288 行） | 322.7 对比 316.3（-2.0%） |
| atoms 4 / 8 / 16 / 32 / 64 | 198.7 / 188.2 / 180.4 / 186.4 / 176.6 |
| 相邻 query 部分重复 / 全部同一行 | 194.2 / 10,420 |
| dense copy，640 B 对比 576 B 每行 | 116.5 对比 106.3（约 1.43 TB/s） |

拟合：每次调用约 **36 us 固定开销 + 0.547 ns/行**（三段斜率 0.544～0.548，线性很好）。

- 字节数几乎不影响耗时：每行少 64 B，每行只省约 0.012 ns。dense copy 少 9% 字节就快 9%，gather 不是。瓶颈是每行一次的 indirect DMA，不是 HBM 带宽或访问局部性。
- extend 一步（2048 query x 2048 行，16 个 chunk）：16 x 36 = 0.58 ms 固定开销 + 4.19M 行 x 0.547 ns = 2.29 ms，合计 2.87 ms，和 profile 里 TensorCore 空等的约 3.1 ms 对得上。
- atoms 到 16 已饱和，32 / 64 在波动内。
- 部分重复的索引慢 8%；全部同一行慢 57 倍（重复行串行化）。

可做的方向：
1. 增大 chunk，摊薄每次调用的 36 us：256 query 的一次 gather 是 322.8 us，比两次 128 的 358.8 us 少 36 us，extend 估计能省约 0.29 ms（约 8%）。但之前扫参里 chunk 256 端到端变慢（exp-cdkg2zm04b），要先查清慢在 attention 还是 VMEM。
2. 减少行数：2.29 ms 的每行部分只能靠少取行（同一批 query 的 top-k 去重后共享），改动大。

### chunk / attention batch 扫参（exp-cmhqg1nu3s，测试分支 d4672d9，extend_c1）

SparseCore 布局 + 物理 slot，atoms 16，`overlap_rope`。wall time 中位数，us；所有能编译的变体输出都与原版逐字节一致。

| chunk | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| 128 | 3541 | **3487 / 3504**（两次） | VMEM 93.6 MB > 64 MB，编译失败 |
| 256 | 3606 | 3599 | 编译失败 |
| 512 | 3779 | 3766 | 编译失败 |

原版 Target 为 4356 us。

- chunk 越大越慢：SC TEC 忙碌时间从约 2.4-2.6 ms（chunk 128）升到约 2.8-2.9 ms（256 / 512）。微基准拟合的每次调用约 36 us 固定开销在 attention 流程里拿不到，单次 gather 随 chunk 变大超线性变慢。
- batch 8 在每种 chunk 下都比 batch 16 慢；batch 32 都超 VMEM。
- 结论：默认 chunk 128 / batch 16 已是最优，调 chunk 和 batch 这条路结束。

## 按 chunk 去重 top-k 行

gather 的耗时几乎全在每行一次的 indirect DMA 上（见上节），所以剩下的办法是少取行：同一个 chunk 里多个 query 选中的同一行只 gather 一次。本节记录这条路的测量。结论：真实 top-k 的重复率很高，SparseCore 上的去重也足够快；但 attention 改成对去重后的行做 dense 计算（不分组或按 8-32 个 query 分组）都比现在慢，这条路只剩“VMEM 里按行 sparse 读”没有测。

### 真实 top-k 的重复率（GLM-5.2 双机 e2e）

方法：基于 exp-lhz0s984yj 的 runner（310b2ec，block-FP8 + `--tpu-fp8-runtime channelwise`，TP/DP/EP16，page 64），在 `_forward_mla` 里对非投机 extend 的 `topk_indices` 加 host 回调（`sglang_tpu/dsa_topk_stats.py`，`direct_compile.insert_callback`，torch compile 不关），只在 attention-DP rank 0 记录。client 向 DP0 发 3 个 53,248 token 的自然代码 prompt（49,152 + 4,096），输出 1。代码在测试分支 `test/glm52-topk-dump`，不进 PR。

128 query 一个 chunk（exp-fj8ahrq5pz，3c32a50，400 次调用，层 0/10/20/40/60）：

| 层 | 去重后占比（全部 chunk） | 深上下文 chunk | 深上下文 p10-p90 | 相邻 query 重叠 |
|---|---|---|---|---|
| 0 | 3.0% | 4.0% | 2.3-6.0% | 75% |
| 10 | 3.0% | 3.9% | 3.6-4.2% | 80% |
| 20 | 2.3% | 2.8% | 2.5-3.0% | 77% |
| 40 | 3.8% | 4.9% | 4.1-6.0% | 66% |
| 60 | 2.9% | 3.6% | 2.9-4.4% | 74% |
| 全部 | 3.0% | 3.9% | | 75% |

按组大小（exp-s2a0c6e4ss，f47b9df）。`GROUPS` 日志行被截断，平均值只来自层 0/10/20，层 20 缺 g=8 和 g=64：

| 每组 query 数 | 平均去重后行数 | dense 计算量（相对 sparse 2048 行） | 去重后剩下的 gather 行数 |
|---|---|---|---|
| 4 | 3083 | 1.51x | 37.6% |
| 8 | 3871 | 1.89x | 23.6% |
| 16 | 4674 | 2.28x | 14.3% |
| 32 | 5803 | 2.83x | 8.9% |
| 64 | 7916 | 3.87x | 6.0% |
| 128 | 9294 | 4.54x | 3.5% |

局限：只测了约 53K 的上下文和自然代码输入；上下文更长时可选行更多，去重率可能变低。

### XLA 去重原型（exp-d7dy0xiqxz，测试分支 a2c1a50，`bench_sc_dedup_proto.py`）

一个 chunk（128 x 2048 行），us：

| 去重后行数 | 全量 gather | `jnp.unique` | 只 gather 去重后的行 | XLA `take` 展开回每个 query |
|---|---|---|---|---|
| 8192 | 185 | 2303 | 70 | 1015 |
| 12274 | 185 | 2307 | 63 | 1014 |
| 16221 | 186 | 2306 | 69 | 1017 |

- gather 去重后的行省约 2/3，剩下的基本是每次调用的固定开销。
- `jnp.unique`（排序）每个 chunk 2.3 ms，`take` 展开 1.0 ms，都不能用。

### SparseCore 标记表去重（exp-drw56fnr3r，测试分支 62243ca，`bench_sc_dedup_mark.py`）

每个 subcore 负责一个 chunk：top-k 行号在 TileSPMEM 的 bitmap 里标记（`addupdate_scatter`），按字算前缀计数，`store_compressed` 输出有序的去重行，再由前缀计数加字内低位个数算反向索引。6 个用例和 numpy 的结果完全一致。

| chunk 数 | 去重后 8192 行 | 12288 行 | 16384 行 |
|---|---|---|---|
| 1 | 233 us | 235 us | 235 us |
| 16（并行） | 270 us | 269 us | 270 us |

- 一次 extend 的 16 个 chunk 合计约 0.27 ms，比 XLA unique（16 x 2.3 ms）快两个数量级，和去重后行数基本无关。
- 实现时踩到的两个限制：`lax.population_count` 在 SC vector subcore 上没有 lowering（改用移位/与/加的 SWAR 写法）；HBM DMA 的偏移必须按字对齐（`p_pad` 取整到 1024，按整块写回）。
- 只用了 32 个 subcore 里的 16 个；单 chunk 233 us 的构成没有 profile。

### attention 读去重后的行

方案：每个 query 对整组去重后的行做 dense flash attention，用自己的 top-k 做 mask（mask 由 `select_bits` 在 SC 上生成，45-50 us），不展开成 26 万行。误差：bf16 约 0.3-0.4%，fp8 约 4-5%（相对 f32 sparse 参考）。

不分组，128 query 一组（exp-opt2scvpai，f1f2590，`bench_sc_dedup_attn.py`），tk=1024，us：

| 去重后行数 | dense 计算量 | bf16 | fp8 |
|---|---|---|---|
| 8192 | 4x | 357 | 280 |
| 12288 | 6x | 527 | 414 |
| 16384 | 8x | 693 | 532 |

按组（exp-ag2x4yuziu，01c2ff1，`bench_sc_dedup_attn_group.py`），去重后行数取真实统计的均值，tk=512，us：

| 每组 query 数 | 去重后行数 | dense 计算量 | bf16 | fp8 | TFLOPS（fp8） |
|---|---|---|---|---|---|
| 8 | 3871 | 2.0x | 310 | 245 | 297 |
| 16 | 4674 | 2.5x | 308 | 261 | 350 |
| 32 | 5803 | 3.0x | 343 | 311 | 353 |

对照：现在每个 chunk 的 gather 加 attention 约 220 us（3.49 ms / 16）。

- 不分组时 MXU 跑到 410-550 TFLOPS（fp8），但多做 4-8 倍计算。
- 分组把多余计算降到 2-3 倍，但每个 grid step 的计算变少，效率降到 200-350 TFLOPS。最好的 g=8 fp8 光 attention 就要 245 us，加上 SC 去重和 gather 去重后的行，总时间比现在长。
- dense-over-distinct（不分组和按组）都放弃。

### VMEM 内按行 sparse 读（exp-85fudprrt4，测试分支 e8b5848，`bench_tc_vmem_gather.py`）

设想：去重后的行（约 1 万行 x 640 B ≈ 6.4 MB）放进 VMEM，attention 按每个 query 的反向索引只读它的 2048 行，计算量和现在一样，HBM 流量只剩去重后的行。留给这一步的上限约 0.42 ns/行（SC 去重约 17 us/chunk + gather 去重后的行约 65 us + attention 约 23 us 之外，剩约 110 us）。

微基准只测 TensorCore 读 128 x 2048 = 262,144 行（每行 512 B NoPE），不做 attention。去重后行数 4K / 8K / 16K 结果几乎一样，下表取 8K；所有 gather 变体与 numpy 逐值一致。

| 变体 | us/chunk | ns/行 |
|---|---|---|
| `contig`：同样行数的连续切片（不 gather，下限） | 47 | 0.18 |
| `vld`：每行一次动态 sublane load，qb=4，unroll 16 | **384** | **1.46** |
| `vld`：qb=1，unroll 8 | 419 | 1.60 |
| `dma_vmem`：每行一次 VMEM→VMEM DMA | 1932 | 7.37 |
| `dma_hbm`：每行一次 HBM→VMEM DMA | 3036 | 11.6 |
| 对照：SC 从 HBM gather（`dsa_gather_native_sc`） | 约 180 | 0.547 |

- 最快的 `vld` 是 SC gather 的 2.7 倍，离 0.42 ns/行的上限差 3.5 倍；单是读行就超过现在每 chunk 的 gather + attention（约 220 us）。
- 瓶颈是每行一次标量取索引加一次动态 sublane load，不是 VMEM 带宽（连续读同样多数据只要 47 us）。按行 DMA 更慢，单次 DMA 开销在微秒级。
- 第一次提交（exp-4besr7rwle）编译失败：索引按 1 / 4 行分块放 SMEM，不满足最后两维为 8 / 128 倍数的要求；改为一维整块放 SMEM 后通过。

### 微基准与接入现有 attention 的差别

目标是复用现有 attention 的 QK、softmax、PV 和精度，只改 KV 加载。现有 kernel（`mla_sparse/kernel.py`）的输入是每个 query 一块连续的 `[2048, 640]` tile：SC 把每个 query 的 2048 行 gather 到 HBM，attention 每个 batch 发两次整块 DMA（`_fetch_bkv_batch`，双缓冲），`unpack_bkv` 在 VMEM 里 bitcast、拼 RoPE、pad 到 640、mask 尾部后进两个 einsum。所以不对并集做 dense 时，每个 chunk 都要把 262,144 行按 query 摆好；去重只减少从 HBM 读的行，摆放的行数不变。

| `vld` 384 us 里的部分 | 估计 | 接入后 |
|---|---|---|
| 去重后的行 HBM→VMEM（整块 BlockSpec，≤ 8 MB） | 几 us | 仍需要，便宜 |
| 每行：SMEM 取行号、动态 sublane load、store 到 buf | 约 337 us | 仍需要，attention 读的就是这块 buf |
| 对 buf 求和（防止被优化掉） | 在 contig 的 47 us 内 | 可去掉，QK 直接读 buf |
| 输出写回、同步 | 可忽略 | 已有 |
| RoPE 的每行摆放 | 没算 | 要加，每行再一次 load |

- 按块加载：每个 query 的行号分散，相邻 query 重叠约 75% 不会让同一个 query 内的行变连续，一次 load 仍只装一行（一个 vreg 的 1/8）。
- 换 SC 摆放（去重后的行放 SPMEM，再按 query 摆）：SC gather 的成本按行算，与字节数和索引是否连续基本无关（RoPE 128 B→64 B 只快 1.3%），从 SPMEM 读也只省字节。

### bundle 核对（exp-7uv7xzb1cp，测试分支 5a6a6c9，`bench_tc_vmem_gather_llo.py`）

`--xla_jf_dump_llo_text` dump 的 `vld` 变体（qb=1，unroll 8）`final_bundles`，内层循环 `0x46`–`0x57`：8 行 18 个 bundle，每行 2.25 个。每行固定四条：`sld`（SMEM 读行号）→ `scalar_lea`（算 VMEM 地址）→ `vld [..] ss:$0 sm:$0xff`（按标量地址读一行）→ `vst [..] sm:$0x1`（写目标 buf 的 1 个 sublane）。前 7 个 bundle 基本只在串行发 `sld`，迭代之间没有软件流水。

- 实测反推时钟：372 us / 262,144 行 / 2.25 bundle ≈ 1.6 GHz。
- 理想流水下 `sld` / `vld` / `vst` 每 bundle 各一条，下限每行 1 个 bundle，即 0.57-0.63 ns/行（1.6-1.75 GHz），只算 NoPE。unroll 16 从 1.60 降到 1.46 ns/行，在往这个方向走。
- 这个下限去不掉：TensorCore 没有按向量索引的 gather load（`vld.idx` 只在 SparseCore TEC 上），每行地址都要一次标量 `sld`，dump 里每个 bundle 只发一条。接入后还要加 RoPE 的 load 和拼 tile。

| 项目 | ns/行 |
|---|---|
| 现在的 SC 从 HBM gather | 0.547 |
| 去重方案留给 TC 摆放的上限 | 约 0.42 |
| TC 逐行读理论下限（只算 NoPE，完美流水） | 0.57-0.63 |
| TC 实测 | 1.42-1.60 |

TC 逐行读的下限已经超过现在整条 SC gather，更超过去重留出的预算，所以没有写融合原型，也不再做 KV 并集上的 dense attention。

### 结论

extend gather 的路线都测过：减字节（#2212）1-3%；atoms / chunk / batch 默认已最优；去重后 dense attention、分组 dense 都比现在慢；复用现有 attention、只改 KV 加载的去重方案，卡在 TensorCore 逐行摆放每行至少一个 bundle（0.57-0.63 ns/行）。这条路线暂停。在现有 Pallas 能力下，extend 的收益就是 SparseCore 布局的 −16%～−21%。

重新打开的条件（目前都没有 Pallas 接口）：
- TensorCore 上暴露一条指令读多行的向量索引 load；
- SC 能把去重后的行按每个 query 的顺序直接写进 TC VMEM（ISA 里的 `TC VMEM <-> SC SPMEM` 直连 DMA），且每行开销明显低于 0.547 ns。

微基准 manifest 从这次起改为挂载 GCS 上的 torch_tpu wheel（`/wheels/torch-tpu`，与 GLM-5.2 e2e runner 相同，SHA256 `6e12816e…` 校验通过），不再分片传输；pod 启动到出结果不到 1 分钟。

参考：SparseCore ISA 文档（Outline「SparseCore」及子文档，libtpu 0.0.50）。相关通路：`vst.idx.ret.add`（TileSPMEM 上返回旧值的原子加）、`vunique` / `vdupcnt` / `vsort`（单个 16 lane 向量内）、`stream.indirect.gather` + `ssetifvalue`（硬件跳过哨兵索引）、SC 发起的 `TC VMEM <-> SC SPMEM` 直连 DMA（不经 HBM）。Pallas `plsc` 是否暴露后两项没有核对。
