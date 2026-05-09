import os
import tilelang
import tilelang.language as T


def path_mn_k_lower_only(
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

            # mn_k 的关键：out_shared 先清零，后面每个 K block 都累加进去
            T.ppl_fill(out_shared, T.float32(0))

            for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                # left[:, kk block]
                T.ppl_copy(G_left[by * block_M, kk * block_K], left_shared)

                # right[:, kk block]，global FP8 -> local FP8
                T.ppl_copy(G_right_fp8[bx * block_N, kk * block_K], right_fp8)

                # local FP8 -> local FP16
                T.ppl_copy(right_fp8, right_fp16)

                # 当前固定 N=128, group_size=128，所以 N 方向 scale group 恒为 0
                # K=256, group_size=128，所以 kk=0/1 对应两个 K group
                T.ppl_copy(G_scale[0, kk], scale_shared)

                # scale: (1,1) -> (64,1)
                T.ppl_npu_bcast(scale_bcast, scale_shared)

                # right_scaled[n,k] = right_fp16[n,k] * scale[0, kk]
                T.ppl_mul(right_scaled, right_fp16, scale_bcast)

                # 每个 K block 先算到 out_partial，再显式累加到 out_shared
                T.ppl_fill(out_partial, T.float32(0))
                T.ppl_gemm(left_shared, right_scaled, out_partial, transpose_B=True)
                T.ppl_add(out_shared, out_shared, out_partial)

            T.ppl_copy(out_shared, out_cast)
            T.ppl_copy(out_cast, G_out[by * block_M, bx * block_N])

    return main_kernel_inner


if __name__ == "__main__":
    func = path_mn_k_lower_only()
    artifact = tilelang.lower(func)

    out_dir = "/mnt2/users/tilelanguser8/tilelang-tpu/tpu_demo/ppl/generated"
    os.makedirs(out_dir, exist_ok=True)

    kernel_source = getattr(artifact, "kernel_source", None)
    if kernel_source is None:
        kernel_source = str(artifact)

    out_path = os.path.join(out_dir, "path_mn_k_kernel.c")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kernel_source)

    print("path_mn_k lower only done.")
    print("generated file:", out_path)
    print("\n===== first 1200 chars =====")
    print(kernel_source[:1200])