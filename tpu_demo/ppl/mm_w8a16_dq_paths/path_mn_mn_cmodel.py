import tilelang
import tilelang.language as T
import torch


def path_mn_mn(
    M=32,
    K=128,
    N=128,
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
    @T.prim_func
    def main_kernel_inner(
        G_left: T.Tensor((M, K), dtype_left),
        G_right_fp8: T.Tensor((N, K), dtype_right),
        G_scale: T.Tensor((1, 1), dtype_scale),
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
            out_cast = T.alloc_shared((block_M, block_N), dtype_out)

            T.ppl_fill(out_shared, T.float32(0))

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.ppl_copy(G_left[by * block_M, k * block_K], left_shared)

                # global FP8 -> local FP8
                T.ppl_copy(G_right_fp8[bx * block_N, k * block_K], right_fp8)

                # local FP8 -> local FP16
                T.ppl_copy(right_fp8, right_fp16)

                # scale: global -> local
                T.ppl_copy(G_scale[0, 0], scale_shared)

                # scale: (1,1) -> (64,1)
                T.ppl_npu_bcast(scale_bcast, scale_shared)

                # right_scaled[n,k] = right_fp16[n,k] * scale[0,0]
                T.ppl_mul(right_scaled, right_fp16, scale_bcast)

                # out = left @ right_scaled.T
                T.ppl_gemm(left_shared, right_scaled, out_shared, transpose_B=True)

            T.ppl_copy(out_shared, out_cast)
            T.ppl_copy(out_cast, G_out[by * block_M, bx * block_N])

    return main_kernel_inner


kernel = tilelang.compile(
    path_mn_mn(),
    out_idx=-1,
    target="tpu",
    mode="cmodel",
)

M, K, N = 32, 128, 128

torch.manual_seed(0)

# left 用 FP16
left = torch.randn(M, K, dtype=torch.float16)

# right 先生成 float32，再量化成 FP8，避免极端值影响
right_f32 = torch.randn(N, K, dtype=torch.float32) * 0.5
right_fp8 = right_f32.to(torch.float8_e4m3fn)

scale = torch.randn(1, 1, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, dtype=torch.float16)

print("start run TileLang path_mn_mn fp8->fp16 block-scale matmul cmodel...")
res = kernel(left, right_fp8, scale, out)
print("kernel return:", res)

# 参考实现：FP8 先转 FP16，再乘 scale，再做 matmul
right_fp16_ref = right_fp8.to(torch.float16)
right_scaled_ref = right_fp16_ref.float() * scale[0, 0].float()
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