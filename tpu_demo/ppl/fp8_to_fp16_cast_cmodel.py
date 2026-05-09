import tilelang
import tilelang.language as T
import torch


def fp8_to_fp16_cast(
    M=64,
    K=128,
    dtype_in="e4m3_float8",
    dtype_out="float16",
):
    @T.prim_func
    def main_kernel_inner(
        G_right_fp8: T.Tensor((M, K), dtype_in),
        G_right_fp16: T.Tensor((M, K), dtype_out),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            right_fp8 = T.alloc_shared((M, K), dtype_in)
            right_fp16 = T.alloc_shared((M, K), dtype_out)

            T.ppl_copy(G_right_fp8[0, 0], right_fp8)
            T.ppl_copy(right_fp8, right_fp16)
            T.ppl_copy(right_fp16, G_right_fp16[0, 0])

    return main_kernel_inner


kernel = tilelang.compile(
    fp8_to_fp16_cast(),
    out_idx=-1,
    target="tpu",
    mode="cmodel",
)

M, K = 64, 128
torch.manual_seed(0)

# 先用较小范围，避免 FP8 溢出或特殊值干扰
right_f32 = torch.randn(M, K, dtype=torch.float32) * 0.5
right_fp8 = right_f32.to(torch.float8_e4m3fn)

out = torch.zeros(M, K, dtype=torch.float16)

print("start run FP8 -> FP16 cast cmodel...")
res = kernel(right_fp8, out)
print("kernel return:", res)

ref = right_fp8.to(torch.float16)

print("\nout sample [:4, :8]:")
print(out[:4, :8])

print("\nref sample [:4, :8]:")
print(ref[:4, :8])

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
print("check close:", torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3))

bad_mask = (~torch.isfinite(diff)) | (abs_diff > 1e-3)
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