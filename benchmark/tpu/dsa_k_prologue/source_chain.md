# Source DSA 链路（sglang-jax-private c948bdd）与迁移计划

V = 读代码核实；I = 推断。路径相对 `sglang-jax-private/python/sgl_jax/srt/`。

## 部署实际启用的路径
共同：`--attention-backend dsa_sparse`、`--dsa-sparse-impl exact`、`--dsa-topk-impl radix`、
prologue `fused`、`--dsa-hlo-overlap` 开、`--page-size 64`。wk 为 channel-wise W8A16。

两种 KV 配置（V，c948bdd）：
- 默认 serve（`benchmark/glm52/*.yaml`、`*_runner_*.sh`）：KV `bf16`，attention 精度 `bf16`。
- FP8 交付（`benchmark/glm52/delivery/benchmark/run_16chip_decode90k.sh`）：KV `fp8_e4m3`，
  attention 精度 `fp8-qk-pv`，tp/dp/ep 32，NEXTN 5 步。`DsaConfig.resolve` 由此选
  `pipeline="split"`、`cache_read_mode="packed"`：走 `exact_attention.py` + `fp8_attention.py`，
  不走 HLO overlap kernel 和 `expand_cache`。KV layout 脚本未断言（默认 `fused`）；
  `fp8_kv_cache.py` 写明 serving 用 split。

**M4-S 迁移的是 FP8 交付路径，KV layout 选 split**（用户决定）。

## 模块
| # | 模块 | 输入 | 输出 / cache | dtype 约定 | 关键代码 |
|---|---|---|---|---|---|
| S1 | Index-K prologue | hidden bf16 [T,H]，wk f8 [128,K] + f32 scale[128]，LN w/b bf16，cos/sin [T,32]，H128 f32 | index cache bf16 [P,64,128]，存 rope+Hadamard 后的 key | W8A16 dot，LN eps 1e-5，无量化 | `kernels/dsa/fused_fp8_k_projection_scatter.py`，`dsa_indexer_ops.py:354-461` |
| S2 | Index-Q + head weights | qr（q_a_layernorm 后），hidden | q bf16 [T,32,128]（rope+Hadamard），w bf16 [T,32] | — | `kernels/dsa/fused_q_rope_hadamard.py`，`glm5_moe.py:362-411` |
| S3 | Index 打分 | q，index cache bf16，w，page table | score f32 [T,max_kv]，-inf 填充 | bf16 dot f32 累加，ReLU，乘 w 求和 | `kernels/dsa/paged_score.py`，`indexer.py:807` |
| S4 | Top-k | score f32 | int32 [T,2048]，物理 slot（page*64+off），-1 填充 | — | `kernels/dsa/topk.py`（radix v2） |
| S5 | MLA KV 写入（split FP8） | latent [T,512]，k_rope [T,64] | latent u8 [P,64,4,128]，RoPE u8 [P,16,4,128]；共享 f32 行 scale 存在 RoPE 字节 [64:68] | FP8 E4M3 per-token，scale = absmax/448 | `kernels/dsa/fp8_kv_cache.py`（`quantize_split_fp8_kv_cache`），`dsa_cache_ops.scatter_paged_cache`（RoPE `tokens_per_group=4`） |
| S6 | 稀疏 attention（fp8-qk-pv） | q_latent，q_rope，split cache，物理 slot | out | SparseCore gather + TensorCore；Q 内部量化 FP8，QK 与 PV 都在 FP8；pipeline split，packed 直读 | `exact_attention.py`（`split_gather_buffers=True`），`fp8_attention.py`，`gather.py` |

耦合：S1/S3 绑定 bf16 index cache（pool 强制）；S4 物理 slot 绑定 S6 的校验；
S5 行格式绑定 expand/gather/attention；page 64 绑定 S1 fused kernel 与 64 对齐的 prefill 切块。

## 当前 M1（FP8 Ported）与 Source 的差异
| 项 | Source | 当前 M1 | 性质 |
|---|---|---|---|
| index cache | bf16 [P,64,128] | FP8 UE8M0 记录 u8[P,32,4,256] | 改变源方案 |
| Hadamard | 有 | 无 | 改变源方案 |
| wk dot | W8A16 | W8A8（激活量化到 FP8） | 改变源方案 |
| LN eps | 1e-5 | 1e-6 | 需核对（I） |
| LN 参数 | bf16 | f32 | 改变源方案（小） |
| page size | 64 | 128 | 待定：框架适配还是改变方案 |
| 框架 | JAX jit | torch_tpu custom op + lax.cond 回退 | 框架适配 |

## 已知差异（记录，不处理）

### S1–S4 串联：非对齐 extend 下 torch op 与 JAX 调用差约 1 个 bf16 ulp

- 现象：unaligned extend（T=2000）时，key store 和 query tail 的 XLA 回退路径（`project_keys_xla`、`query_tail_xla`），torch op 与 JAX 直接调用的输出差约 1 个 bf16 ulp，top-k 有少量行不同。对齐 extend 和 decode 走 Pallas kernel，逐字节一致。
- 已排除：scorer（用 torch 侧的 cache 和 q 跑 JAX scorer，top-k 完全一致）；matmul 默认精度（`"highest"` 不消除差异）；`jax.export` 降级（导出再调用与直接 jit 逐字节一致）；excess precision（两边 HLO 中 fusion 内运算 dtype 相同）。
- 推测原因（未最终确认）：torch_tpu 编译时 bf16 elementwise 的融合边界不同，fusion 内中间值不舍入。完整 torch op 与逐阶段 torch op 之间也差 42 个元素。
- 影响：两边对 f64 精确分的选择偏差都约 0.0017，超过 2% 的比例为 0，选择质量不受影响。
- 实验：exp-3r882ygc3n、exp-b43xl43c7b、exp-fuhu2bl2uo。

### S1–S4 串联：extend 2048 下 torch op 多约 634 µs `radix_topk_v2` 等待

- torch op 的 profile 多一个 `radix_topk_v2.call-done.2`（约 634 µs），JAX 侧没有这段等待。推测是异步 SparseCore 调用和下一块打分的重叠在 torch op 里丢了。留到性能优化阶段处理。

## M4-S 结果：split FP8 KV 写入 + fp8-qk-pv exact attention（S5–S6）

配置：Source FP8 交付路径（`fp8_e4m3` KV、split 布局、`fp8-qk-pv`，`DsaConfig.resolve` 选 `pipeline="split"`、`cache_read_mode="packed"`）。
Ported-S 在 page 128 上用 2×64 页视图调用迁移的 Source 代码（`dsa_fp8_attention.py`、torch op `sglang_tpu_pallas::dsa_fp8_attention`）。
单芯片 v7x，64 个 head，K=2048，bench `bench_fp8_attention_ab.py`。

正确性（V）：
- Ported-S（JAX）对 Source：三个用例（extend 2048、unaligned extend T=2000、decode b64）的 split KV cache（latent + RoPE）和 attention 输出都逐字节一致。
- torch op 对 Ported-S JAX：三个用例输出和 cache 都逐字节一致。
- dispatch 测试（`test_dsa_fp8_attention_dispatch.py`）：CPU 进程按 TPU7x 导出，prefill 和 decode 计划都含 `dsa_sparse_core_gather` 和 `dsa_tensor_core_attention_fp8`；不打补丁时导出报错，不会悄悄回退。

耗时与精度（module 时间 µs；精度为相对同一 f64 参考的中位 / 最大相对误差，参考用未量化的 bf16 K/V）：

| 用例 | Source | Ported-S JAX | Ported-S torch op | Target |
|---|---|---|---|---|
| extend 2048 | 34497，7.6% / 19.0% | 34510，逐字节一致 | 34497 | per-query 4407，6.1% / 9.7% |
| unaligned extend | 33917，7.6% / 10.4% | 33915，逐字节一致 | 33896 | q-block 4286，6.4% / 11.7% |
| decode b64 | 2091，7.9% / 13.0% | 2074，逐字节一致 | 1520 | per-query 777（取自 exp-rgqv6jaw33），6.5% / 9.3% |

说明：
- Target 是不同格式（FP8 E4M3 per-tensor，`k_scale=1`，TensorCore 布局），不做逐字节对比，只比耗时和对同一参考的误差。
- Target extend 2048 的 q-block prefill kernel（`sparse_mla_prefill_qblock_tokens`）在 64 head 单芯片下 VMEM 不够（需要 93 MB，上限 64 MB），所以表中是它的 per-query kernel，工作量与其他几方相同。按每卡 4 个 head 跑 q-block 能编译，498 µs；每卡 4 head 是按 64 head / tp16 的假设（I），GLM-5.2 的实际 head 数在两边仓库里都没找到，且这个耗时只有 1/16 工作量，不能和 64 head 的行直接比。
- Source 的 prefill 时间里约 25.7 ms 是一个 `gather_fusion`（`s32[4194304]` = T×K），decode 里同一个 op 约 718 µs；三边一样，不是迁移引入的。根因见下一节：它是逻辑 top-k 到物理 slot 的映射，Source 部署不走这一步。
- JAX bench 里 decode 有两个整 cache 的 `copy`（约 540 µs），torch op 里 cache donate + alias 后就没有了；Target 的 decode 行也包含这两个 copy，去掉后约 240 µs（I）。
- 结论（已被下一节修正）：迁移的 Source attention 与 Source 完全一致。表中 Source 列包含部署不付的逻辑映射，不能代表 Source 部署的耗时。
- 实验：exp-7kvikla8m9、exp-45eqx0fnzf、exp-rgqv6jaw33、exp-u4cubhb90c（提交 a7c795f）。

### `gather_fusion` 的来源：逻辑 top-k 映射（Source 部署不走）

- 代码（V）：bench 一直给 attention 喂序列内逻辑 top-k，backend 用 `_logical_topk_to_physical_slots` 映射成物理行，其中 `page_indices[safe_ptr]` 是 [T,K] 的 gather，形状与 `gather_fusion` 一致。Source 部署的 exact 模式里 `topk_indices_are_physical=True`（`model_runner.py:860`），indexer 直接输出物理 slot（`glm5_moe.py:683` `output_physical_slots=True`），backend 只做 `PhysicalTopKValidation`（比较 + `where`，无 gather）。
- 实测（exp-97x4ruzrrh，提交 2e1c339）：按部署路径跑 Source（输入换成物理 slot），输出与原 Source 逐字节一致，`gather_fusion` 消失。

| 用例 | Source（逻辑 top-k，上表） | Source 部署路径（物理 top-k） | 单独跑映射 | Target |
|---|---|---|---|---|
| extend 2048 | 34503 | 8682 | 44076 | per-query 4407 |
| unaligned extend | 33919 | 8759 | 25235 | q-block 4286 |
| decode b64 | 2091 | 1365（含约 560 µs 两个 cache copy） | 732 | 777（同样含 copy） |

- 修正后的结论：Source 部署路径比 Target 的 sparse MLA 慢约 2×（prefill 8682 对 4407 µs；decode 1365 对 777 µs，两边都含 cache copy），不是 8×。
- 对迁移的影响：已按 Source 部署改（提交 dc0b007）。bf16 indexer op 加 `physical`（scorer `output_physical_slots=True`，framework slot p×128+o 与 Source page-64 slot 同值），FP8 attention op 加 `topk_is_physical`（只做 `PhysicalTopKValidation`）。两者默认关，尚未接入 attention backend。
- 验证（exp-lt9gre9714，`check_physical_topk.py`）：
  - indexer `physical=True` 的输出逐行等于 `physical=False` 输出按页表映射的结果，两种模式写出的 cache 逐字节相同（extend 2048、unaligned、decode b64；每行 2048 个有效 slot）。
  - attention op `topk_is_physical=True` 的输出和 cache 与逻辑路径的 Ported-S JAX 逐字节一致。torch op module 时间：extend 2048 34502 → 8669 µs，unaligned 33900 → 8752 µs，decode b64 1519 → 791 µs（torch op 无 cache copy；Target decode 去掉 copy 约 240 µs，I）。
- 单独跑映射比融在 attention 里还慢（44 ms 对 25.7 ms），原因未查；部署不走这一步，所以不再追。

### 接入 sparse MLA backend（`SGLANG_TPU_DSA_SOURCE_FP8_ATTENTION`）

- 开关默认关。打开后 bf16 indexer 输出物理 slot（`physical=True`），`_forward_mla` 走 `sglang_tpu_pallas::dsa_fp8_attention`（`topk_is_physical=True`），只做 `PhysicalTopKValidation`。
- 启动时拒绝未验证的配置：未开 `SGLANG_TPU_DSA_BF16_INDEXER`、kpool≠1、KV 不是 `fp8_e4m3`、投机解码、无 rope、attention TP>1。MIXED batch 在 trace 时报 `NotImplementedError`（Source 的 MIXED 走 bf16 QK + HLO overlap，未迁移）。
- op 在 backend `__init__` 里建好：在 `torch.compile` 图里第一次建 op 会因 `inspect.signature` 的 `id()` 报 Dynamo `Unsupported`（exp-ysc3sht8pz，提交 03a94d2 修复）。
- 验证（V，exp-s8jck2ktyv，提交 03a94d2）：`check_source_attention_backend.py` 经 `_forward_mla`（编译）跑 extend 2048、unaligned extend、decode b64，输出和 split cache 与 Ported-S JAX 逐字节一致；MIXED 报错；`test_dsa_bf16_indexer_wiring.py` 8/8 通过。
- 未做：attention TP>1 的 token split，MIXED / 投机解码路径，真实模型层上的端到端。

## 整层 S1–S6 三方对比

配置：单芯片 v7x，64 head，K=2048，bench `bench_dsa_layer_ab.py`（提交 5dd9d33，exp-rii7xivn5y）。
一层 = index key 写入、打分、top-k，然后 split FP8 KV 写入和 attention。各方按部署方式跑：
Source 用物理 slot（`output_physical_slots=True` + `PhysicalTopKValidation`）；Ported-S 用
`physical=True` / `topk_is_physical=True`（torch 版是两个 op 放在一张编译图里，等于 backend
同时开 `SGLANG_TPU_DSA_BF16_INDEXER=1` 和 `SGLANG_TPU_DSA_SOURCE_FP8_ATTENTION=1`）；Target 是
FP8 UE8M0 indexer + per-query sparse MLA（64 head 下 q-block 的 VMEM 不够）。

正确性（V）：
- Ported-S JAX 对 Source：extend 2048 和 decode b64 的 index cache、slot、split KV、输出都逐字节一致。
  unaligned extend 没有 Source 数（Source 的 key store 在这里走未随包提供的 XLA chain）。
- torch op 对 Ported-S JAX：extend 2048 和 decode b64 逐字节一致；unaligned 的 index cache 和输出不一致，
  是已知的约 1 bf16 ulp key-store 差异（见"已知差异"），split KV 一致。

耗时（module µs）与质量：

| 用例 | Source | Ported-S JAX | Ported-S torch | Target |
|---|---|---|---|---|
| extend 2048 | 10739 | 10749 | 10723 | 5449 |
| unaligned extend | — | 11298 | 11267 | 4905 |
| decode b64 | 1286 | 1233 | 1217 | 446 |

| 用例 | 选择偏差 max / 超 2% 的比例（Ported-S） | 同（Target，对自己的 FP8 key） | attention 中位 / 最大相对误差（Ported-S / Target） | Target 与 Ported-S 选中行重合 |
|---|---|---|---|---|
| extend 2048 | 0.0024 / 0 | 0.024 / 28% | 7.8% / 13.0%，6.6% / 9.3% | 34% |
| unaligned extend | 0.0017 / 0 | 0.027 / 31% | 7.5% / 10.5%，5.9% / 8.8% | 34% |
| decode b64 | 0.0027 / 0 | 0.023 / 16% | 7.7% / 13.9%，7.0% / 12.2% | 25% |

说明：
- 选择偏差 = 1 − 选中的最低精确分 / 第 k 大精确分（f64，各方对自己存的 key 和 query）。attention 误差是对
  未量化 K/V 在各自选中行上的 f64 attention。
- 迁移的整层与 Source 一致，且 torch op 没有额外开销（10723 对 10739 µs）。
- 整层上 Ported-S 比 Target 慢约 2×（prefill）和约 2.7×（decode）。Ported-S 的选择质量明显更好
  （超 2% 偏差的比例为 0，Target 为 16–31%）；attention 精度两边相近，Target 略好。
- 两边选中的行只重合 25–34%：key 链不同（Target W8A8、eps 1e-6、无 Hadamard、FP8 UE8M0 存储），
  所以 Target 的 attention 是在另一组行上算的。随机输入下这个数只说明两边选择差得多，不代表模型质量。
- 第一轮（exp-eugdo5kysz，6825f63）Source 那一方在 extend 2048 编译时 SparseCore lowering 报
  `RET_CHECK ... Could not find buffer`（radix_topk_v2）。把 page table、token slot、`cu_kv_lens`
  从 jit 闭包常量改成 jit 参数后通过。是常量折进 SparseCore 调用导致的推断（I），未确认。

## PR 2198 基线形状下的整层对比（exp-jgbl4d045u，提交 bb6a055）

形状按 PR sgl-project/sglang-torchtpu#2198（exp-6mtkc6t2qi，TP16 / attn-DP16 / EP16）每个 attention-DP rank 推算（I）：H=64，attention TP 1，随机输入，单芯片，不起服务。Ported-S / Target page 128，Source page 64（基线本身是 page 64）。

device module 时间，µs：

| 用例 | 形状 | Source | Ported-S JAX | Ported-S torch op | Target |
|---|---|---|---|---|---|
| extend_c1 | 1 x 2048，prefix 49,152 | 12981 | 13016 | 12865 | 9012 |
| extend_c2 | 1 x 2048，prefix 51,200 | 12970 | 13003 | 12855 | 9085 |
| verify_b4 | 4 x 6，context 约 90K | 不可跑 | 445 | 440 | 1288 |
| decode_b4 | 4 x 1，context 约 90K | 208 | 216 | 228 | 211 |

- 正确性：extend_c1 / c2、decode_b4 上 Ported-S JAX 对 Source 四项（index cache、slot、split KV、输出）逐字节一致；torch op 对 JAX 在 extend 两个用例、decode_b4 上逐字节一致。verify_b4 的 prefix 不是 64 的倍数，key store 走 XLA fallback，torch op 的 index cache 和输出与 JAX 不一致（已知约 1 bf16 ulp，split KV 一致），只报告。verify 下 Source 的 key store 是未随包的 XLA chain，不可跑。
- extend：Ported-S 比 Target 慢约 1.44x（小形状下是约 2x）。Ported-S 的大头是 `dsa_paged_extend_score_block_paged`（每块约 1.7 ms）和 attention；Target 的大头是 indexer：`StreamIdxTC` 2.3-2.4 ms + SparseCore top-k 1.6 ms。
- verify：Ported-S 比 Target 快约 2.9x。Target 的 `StreamIdxTC-m-bq_128` 对 24 行占 1025 us，推测是 128 行 query block 的填充（I，未查）。
- decode：三方持平（208-228 us）。
- 选择偏差（对各自 indexer 的 f64 精确分，最大值）：Ported-S 0.11%-0.24%，超过 2% 的比例 0；Target 1.9%-2.3%，超过 2% 的比例 0-22%。Target 与 Ported-S 选中行的重合度 3%-9%（随机输入，长 context 下分数更平）。
- attention 中位相对误差（对未量化 K/V 的 f64 参考）：Ported-S 6.7%-7.6%，Target 5.2%-6.6%。

## 多机模型端到端（推迟到整个 DSA 迁完并完成整体对比之后）

- 资源：GLM-5.2 FP8（约 762 GB）需要 2 台 tpu7x 主机、tp16（`benchmark/tpu/recipe/configs/glm_5_2/server.py`）。目前所有 Falcon manifest 都是单 pod（`2x2x1`），需要一个 2 主机 job。
- 权重：`gs://inference-model-storage-sgl/model_scope/GLM5.2-fp8-channel-wise`。Falcon pod 读同一 bucket 里的 wheel 时一直回退到 `falcon exp cp`，大概率也读不了权重；`falcon exp cp` 不适合 762 GB。需要给 Falcon job 的身份开该 bucket 的读权限（类似 `iam_grant_request.md`）。
- 对比方式：同一 recipe 分别以 `SGLANG_TPU_DSA_BF16_INDEXER=1` 和关闭运行，比较输出质量和吞吐。
- 在此之前用单层或小模型验证 `indexer_forward` 的真实调用路径。

## 集群状况（记录）

- 2026-11-17 00:25 起，`falcon exp cp` 传 torch_tpu wheel（约 99 MB）在 c1 和 c2 上都失败（`transfer agent disconnected`、`did not reach a terminal state`、`connection reset by peer`），涉及 exp-70ldncj77u、exp-r1ld2gwaet（c1）和 exp-ey8mviy8ww（c2）。不是单个集群的问题。
- 01:00 在 exp-ey8mviy8ww 上测试：小文件（几字节、1 MB、20 MB）都传成功，各 40～76 s；随后重传 wheel，213 s 后 `connection reset by peer`。所以是大文件传输不通，小文件正常。
- 本机 `falcon v0.1.0`（commit 9e69585，2026-06-11 构建）。
- 可选办法：等传输服务恢复；或把 wheel 切成小于 20 MB 的分片分别传，在 pod 里拼接后校验 SHA256；或给 Falcon job 身份开 wheel 所在 GCS bucket 的读权限（manifest 已有 GCS 下载路径，同 `iam_grant_request.md`）。
- 采用分片：wheel 切成 7 片（6×15 MB + 1×5 MB），M4-S manifest 在 pod 里拼接后校验 SHA256（提交 3a9a0a1 起）。之后 exp-7kvikla8m9、exp-45eqx0fnzf、exp-rgqv6jaw33、exp-u4cubhb90c 都成功，个别分片要重试 1～2 次。c1 和 c2 都可用。
- 同一时段本机到 github.com 的连接也不稳定（`Empty reply from server`、443 连接超时），push 需要多次重试。
