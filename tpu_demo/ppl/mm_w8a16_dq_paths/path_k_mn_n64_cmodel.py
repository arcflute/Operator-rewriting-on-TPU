import tilelang
import tilelang.language as T
import torch


def path_k_mn_n64(
    M=32,
    K=256,
    N=128,
    group_size=128,
    block_M=32,
    block_N=64,
    block_K=128,
    dtype_left="float16",
    dtype_right="e4m3_float8",
    dtype_compute="float16",
    dtype_scale="float16",
    dtype_out="float16",
    accum_dtype="float32",
):
    groups_n = (N + group_size - 1) // group_size
    groups_k = (K + group_size - 1) // group_size

    @T.prim_func
    def main_kernel_inner(
        G_left: T.Tensor((M, K), dtype_left),
        G_right_fp8: T.Tensor((N, K), dtype_right),
        G_scale: T.Tensor((groups_n, groups_k), dtype_scale),
        G_out: T.Tensor((M, N), dtype_out),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            left_shared = T.alloc_shared((block_M, block_K), dtype_left)

            right_fp8 = T.alloc_shared((block_N, block_K), dtype_right)
            right_fp16 = T.alloc_shared((block_N, block_K), dtype_compute)

            scale_shared = T.alloc_shared((1, 1), dtype_scale)
            scale_bcast = T.alloc_shared((block_N, 1), dtype_scale)

            right_scaled = T.alloc_shared((block_N, block_K), dtype_compute)

            out_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            out_partial = T.alloc_shared((block_M, block_N), accum_dtype)
            out_cast = T.alloc_shared((block_M, block_N), dtype_out)

            T.ppl_fill(out_shared, T.float32(0))

            for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.ppl_copy(G_left[by * block_M, kk * block_K], left_shared)
                T.ppl_copy(G_right_fp8[bx * block_N, kk * block_K], right_fp8)
                T.ppl_copy(right_fp8, right_fp16)

                T.ppl_copy(G_scale[0, kk], scale_shared)
                T.ppl_npu_bcast(scale_bcast, scale_shared)
                T.ppl_mul(right_scaled, right_fp16, scale_bcast)

                T.ppl_fill(out_partial, T.float32(0))
                T.ppl_gemm(left_shared, right_scaled, out_partial, transpose_B=True)
                T.ppl_add(out_shared, out_shared, out_partial)

            T.ppl_copy(out_shared, out_cast)
            T.ppl_copy(out_cast, G_out[by * block_M, bx * block_N])

    return main_kernel_inner


kernel = tilelang.compile(
    path_k_mn_n64(),
    out_idx=-1,
    target="tpu",
    mode="cmodel",
)

M, K, N = 32, 256, 128
group_size = 128
block_K = 128
groups_n = (N + group_size - 1) // group_size
groups_k = (K + group_size - 1) // group_size

torch.manual_seed(1)

left = torch.randn(M, K, dtype=torch.float16)

right_f32 = torch.randn(N, K, dtype=torch.float32) * 0.5
right_fp8 = right_f32.to(torch.float8_e4m3fn)

scale = torch.randn(groups_n, groups_k, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, dtype=torch.float16)

print("start run path_k_mn_n64 fp8->fp16 K-split + MN-tile matmul cmodel...")
res = kernel(left, right_fp8, scale, out)
print("kernel return:", res)

# PyTorch 参考实现：按 K group 分段乘 scale，再整体 matmul
right_fp16_ref = right_fp8.to(torch.float16)
right_scaled_ref = torch.empty((N, K), dtype=torch.float32)

for kg in range(groups_k):
    k0 = kg * block_K
    k1 = min((kg + 1) * block_K, K)
    right_scaled_ref[:, k0:k1] = right_fp16_ref[:, k0:k1].float() * scale[0, kg].float()

ref = left.float() @ right_scaled_ref.transpose(0, 1)
ref = ref.to(torch.float16)

print("scale:")
print(scale)

print("\nout sample [:4, :8]:")
print(out[:4, :8])

print("\nref sample [:4, :8]:")
print(ref[:4, :8])

print("\nout boundary sample [:4, 60:68]:")
print(out[:4, 60:68])

print("\nref boundary sample [:4, 60:68]:")
print(ref[:4, 60:68])

print("\nout tail sample [:4, 120:128]:")
print(out[:4, 120:128])

print("\nref tail sample [:4, 120:128]:")
print(ref[:4, 120:128])

diff = ref.float() - out.float()
abs_diff = diff.abs()

print("\n=== 差异分析 ===")
print("out 是否全有限:", torch.isfinite(out.float()).all().item())
print("ref 是否全有限:", torch.isfinite(ref.float()).all().item())
print("diff 是否全有限:", torch.isfinite(diff).all().item())

print("out 最大绝对值:", out.float().abs().max().item())
print("ref 最大绝对值:", ref.float().abs().max().item())

print("最大差异:", abs_diff.max().item())
print("平均差异 double:", abs_diff.double().mean().item())
print("check close:", torch.allclose(out.float(), ref.float(), atol=1e-1, rtol=1e-1))

bad_mask = (~torch.isfinite(diff)) | (abs_diff > 0.1)
bad_idx = bad_mask.nonzero()

print("\n异常元素个数:", bad_idx.shape[0])
print("前 20 个异常位置和值:")

for idx in bad_idx[:20]:
    i, j = idx.tolist()
    print(
        f"pos=({i},{j}), "
        f"out={out[i, j].item()}, "
        f"ref={ref[i, j].item()}, "
        f"diff={diff[i, j].item()}"
    )