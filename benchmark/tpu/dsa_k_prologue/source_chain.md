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
