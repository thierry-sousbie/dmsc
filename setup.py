import glob
import os
import subprocess
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

cpu_compile_args = ["-O3", "-std=c++17"]
cpu_link_args = []
no_gpu = False
tbb_include = None
tbb_lib = None


def generate_metal_headers():
    """Converts raw .metal files into C++ string headers for compilation."""
    metal_files = glob.glob("dmsc/csrc/gpu/metal/*.metal")
    if not metal_files:
        print("[SETUP] No .metal files found in dmsc/csrc/gpu/metal/")
        return

    for metal_src_path in metal_files:
        metal_hdr_path = metal_src_path.replace(".metal", "_metal.h")

        with open(metal_src_path) as f:
            shader_code = f.read()

        base_name = os.path.basename(metal_src_path).replace(".metal", "").upper()
        macro_name = f"{base_name}_METAL_SRC"
        header = (
            "// AUTO-GENERATED DURING BUILD. DO NOT EDIT.\n"
            "#pragma once\n\n"
            f'const char* {macro_name} = R"METAL_SHADER(\n'
            f"{shader_code}"
            '\n)METAL_SHADER";\n'
        )

        if os.path.exists(metal_hdr_path):
            with open(metal_hdr_path) as f:
                if f.read() == header:
                    continue

        print(f"[SETUP] Generating C++ header for {metal_src_path}...")
        with open(metal_hdr_path, "w") as f:
            f.write(header)


if tbb_lib is None or tbb_include is None:
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix", "tbb"]).decode().strip()
        tbb_include = os.path.join(brew_prefix, "include")
        tbb_lib = os.path.join(brew_prefix, "lib")
    except Exception:
        tbb_include = "/usr/local/include"
        tbb_lib = "/usr/local/lib"

gpu_ext = None
if sys.platform == "darwin":
    os.environ["CC"] = "clang++"
    os.environ["CXX"] = "clang++"

    try:
        sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode("utf-8").strip()
    except Exception:
        sdk_path = None

    if sdk_path:
        cpu_compile_args.extend(["-isysroot", sdk_path])
        cpp_header_path = os.path.join(sdk_path, "usr", "include", "c++", "v1")
        cpu_compile_args.extend([f"-I{cpp_header_path}"])

    mac_target = "-mmacosx-version-min=12.0"
    cpu_compile_args.extend([mac_target, "-Xpreprocessor", "-fopenmp"])
    cpu_link_args.extend([mac_target])

    gpu_compile_args = cpu_compile_args.copy()
    gpu_link_args = cpu_link_args.copy() + ["-framework", "Metal", "-framework", "Foundation"]

    generate_metal_headers()

    gpu_ext = CppExtension(
        name="dmsc.csrc.dmsc_gpu",
        sources=[
            "dmsc/csrc/dmsc_gpu.cpp",
            "dmsc/csrc/gpu/metal/metal_backend.mm",  # Objective-C++ bridge
        ],
        extra_compile_args=gpu_compile_args,
        extra_link_args=gpu_link_args,  # Uses the new GPU link args
        include_dirs=[tbb_include],
        library_dirs=[tbb_lib],
        libraries=["tbb"],
    )
else:
    cpu_compile_args.extend(["-fopenmp"])

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        print("\n" + "=" * 60)
        print("[WARNING] TORCH_CUDA_ARCH_LIST is not set.")
        print("PyTorch may crash if it cannot auto-detect a GPU.")
        print('export TORCH_CUDA_ARCH_LIST="$VER"')
        print("Injecting default versions: 7.5;8.0;8.6;8.9;9.0")
        print("=" * 60 + "\n")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0"

    gpu_ext = CUDAExtension(
        name="dmsc.csrc.dmsc_gpu",
        sources=[
            "dmsc/csrc/dmsc_gpu.cpp",
            "dmsc/csrc/gpu/cuda/cell_groups.cu",
            "dmsc/csrc/gpu/cuda/critical_points.cu",
            "dmsc/csrc/gpu/cuda/gradient.cu",
            "dmsc/csrc/gpu/cuda/trace_from_saddles.cu",
            "dmsc/csrc/gpu/cuda/arcs_simplification.cu",
            "dmsc/csrc/gpu/cuda/trace_raw_arcs_geometry.cu",
        ],
        extra_compile_args={"cxx": cpu_compile_args, "nvcc": ["-O3", "-lineinfo"]},
        # extra_compile_args={"cxx": cpu_compile_args, "nvcc": ["-G", "-lineinfo"]},
        include_dirs=[tbb_include],
        library_dirs=[tbb_lib],
        libraries=["tbb"],
    )

ext_modules = [
    CppExtension(
        name="dmsc.csrc.dmsc_cpu",
        sources=["dmsc/csrc/dmsc_cpu.cpp"],
        extra_compile_args=cpu_compile_args,
        extra_link_args=cpu_link_args,  # <--- Added here
        include_dirs=[tbb_include],
        library_dirs=[tbb_lib],
        libraries=["tbb"],
    ),
]

if gpu_ext is not None and not no_gpu:
    ext_modules.append(gpu_ext)

setup(
    name="dmsc",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
