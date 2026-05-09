import os
import tilelang
import tilelang.language as T


def fp8_to_fp16_cast_lower_only(
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

            # Global FP8 -> Local FP8
            T.ppl_copy(G_right_fp8[0, 0], right_fp8)

            # 关键测试：Local FP8 -> Local FP16
            # 看 codegen 能不能生成 tpu_bdc_cast
            T.ppl_copy(right_fp8, right_fp16)

            # Local FP16 -> Global FP16
            T.ppl_copy(right_fp16, G_right_fp16[0, 0])

    return main_kernel_inner


if __name__ == "__main__":
    func = fp8_to_fp16_cast_lower_only()
    artifact = tilelang.lower(func)

    out_dir = "/mnt2/users/tilelanguser8/tilelang-tpu/tpu_demo/ppl/generated"
    os.makedirs(out_dir, exist_ok=True)

    kernel_source = getattr(artifact, "kernel_source", None)
    if kernel_source is None:
        kernel_source = str(artifact)

    out_path = os.path.join(out_dir, "fp8_to_fp16_cast_kernel.c")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kernel_source)

    print("lower only done.")
    print("generated file:", out_path)
    print("\n===== first 1200 chars =====")
    print(kernel_source[:1200])