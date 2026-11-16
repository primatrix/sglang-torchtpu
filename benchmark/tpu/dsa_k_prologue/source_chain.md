# Source DSA 链路（sglang-jax-private c948bdd）与迁移计划

V = 读代码核实；I = 推断。路径相对 `sglang-jax-private/python/sgl_jax/srt/`。

## 部署实际启用的路径
`--attention-backend dsa_sparse`、`--dsa-sparse-impl exact`、`--dsa-topk-impl radix`、
prologue `fused`、`--dsa-hlo-overlap` 开、attention 精度 `bf16`、KV `fp8_e4m3`、
`--page-size 64`、KV layout `fused`（DCP/offload 才用 split）。wk 为 channel-wise W8A16。

## 模块
| # | 模块 | 输入 | 输出 / cache | dtype 约定 | 关键代码 |
|---|---|---|---|---|---|
| S1 | Index-K prologue | hidden bf16 [T,H]，wk f8 [128,K] + f32 scale[128]，LN w/b bf16，cos/sin [T,32]，H128 f32 | index cache bf16 [P,64,128]，存 rope+Hadamard 后的 key | W8A16 dot，LN eps 1e-5，无量化 | `kernels/dsa/fused_fp8_k_projection_scatter.py`，`dsa_indexer_ops.py:354-461` |
| S2 | Index-Q + head weights | qr（q_a_layernorm 后），hidden | q bf16 [T,32,128]（rope+Hadamard），w bf16 [T,32] | — | `kernels/dsa/fused_q_rope_hadamard.py`，`glm5_moe.py:362-411` |
| S3 | Index 打分 | q，index cache bf16，w，page table | score f32 [T,max_kv]，-inf 填充 | bf16 dot f32 累加，ReLU，乘 w 求和 | `kernels/dsa/paged_score.py`，`indexer.py:807` |
| S4 | Top-k | score f32 | int32 [T,2048]，物理 slot（page*64+off），-1 填充 | — | `kernels/dsa/topk.py`（radix v2） |
| S5 | MLA KV 写入 | latent [T,512]，k_rope [T,64] | u8 [P,64,4,pair_dim]，每 token 一个 f32 scale（偏移 576） | FP8 E4M3 per-token | `kernels/dsa/fp8_kv_cache.py` |
| S6 | 稀疏 attention | q_latent，q_rope，cache，物理 slot | out | prefill 先 expand 成 bf16，decode 直读 FP8；HLO overlap | `exact_attention_hlo_overlap.py`，`expand_cache.py` |

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
