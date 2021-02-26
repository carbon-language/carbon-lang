# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark rules to bootstrap Clang (and LLVM).

These rules are loaded as part of the `WORKSPACE`, and used by
`clang_configuration.bzl`. The llvm-project submodule is used for the build.
"""

FORCE_LOCAL_BOOTSTRAP_ENV = "CARBON_FORCE_LOCAL_BOOTSTRAP_BUILD"

def _run(
        repository_ctx,
        cmd,
        timeout = 10,
        environment = {},
        quiet = True):
    """Runs the provided `cmd`, checks for failure, and returns the result."""
    exec_result = repository_ctx.execute(
        cmd,
        timeout = timeout,
        environment = environment,
        quiet = quiet,
    )
    if exec_result.return_code != 0:
        fail("Unable to run command successfully: %s" % str(cmd))

    return exec_result

def _detect_system_clang(repository_ctx):
    """Detects whether a system-provided clang can be used.

    Returns a tuple of (is_clang, environment).
    """

    # If the user provides an explicit `CC` environment variable, use that as
    # the compiler.
    cc = repository_ctx.os.environ.get("CC")
    cxx = repository_ctx.os.environ.get("CXX")
    if cc or cxx:
        version_output = _run(repository_ctx, [cc, "--version"]).stdout
        return "clang" in version_output, {}

    # If we can build our Clang toolchain using a system-installed Clang, try
    # to do so.
    system_clang = repository_ctx.which("clang")
    if system_clang:
        return True, {
            "CC": str(system_clang),
            "CXX": str(system_clang) + "++",
        }
    return False, {}

def _get_cmake_defines(repository_ctx, is_clang):
    """Returns a long list of cmake defines for the bootstrap."""
    modules_setting = "OFF"
    if is_clang:
        modules_setting = "ON"

    static_link_cxx = "ON"
    unstable_libcxx_abi = "ON"
    if repository_ctx.os.name.lower().startswith("mac os"):
        # macOS doesn't support the static C++ standard library linking. Turn
        # it off here, and disable the unstable libc++ ABI as we will also be
        # unable to use it later on.
        static_link_cxx = "OFF"
        unstable_libcxx_abi = "OFF"

    return [
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;libcxx;libcxxabi;compiler-rt;libunwind",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_ENABLE_MODULES=" + modules_setting,
        "-DLLVM_STATIC_LINK_CXX_STDLIB=" + static_link_cxx,
        "-DLLVM_TARGETS_TO_BUILD=AArch64;X86",
        "-DLIBCXX_ABI_UNSTABLE=" + unstable_libcxx_abi,
        "-DLIBCXX_ENABLE_ASSERTIONS=OFF",
        "-DLIBCXXABI_ENABLE_ASSERTIONS=OFF",

        # Disable components of the build that we don't use while building Carbon.
        "-DCLANG_ENABLE_ARCMT=OFF",
        "-DCLANG_INCLUDE_TESTS=OFF",
        "-DCLANG_TOOL_APINOTES_TEST_BUILD=OFF",
        "-DCLANG_TOOL_ARCMT_TEST_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_CHECK_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_DIFF_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_EXTDEF_MAPPING_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_FUZZER_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_IMPORT_TEST_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_OFFLOAD_WRAPPER_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_SCAN_DEPS_BUILD=OFF",
        "-DCLANG_TOOL_CLANG_SHLIB_BUILD=OFF",
        "-DCLANG_TOOL_C_ARCMT_TEST_BUILD=OFF",
        "-DCLANG_TOOL_C_INDEX_TEST_BUILD=OFF",
        "-DCLANG_TOOL_DIAGTOOL_BUILD=OFF",
        "-DCLANG_TOOL_LIBCLANG_BUILD=OFF",
        "-DCLANG_TOOL_SCAN_BUILD_BUILD=OFF",
        "-DCLANG_TOOL_SCAN_VIEW_BUILD=OFF",
        "-DLLVM_BUILD_UTILS=OFF",
        "-DLLVM_ENABLE_BINDINGS=OFF",
        "-DLLVM_ENABLE_LIBXML2=OFF",
        "-DLLVM_ENABLE_OCAMLDOC=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_DOCS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_GO_TESTS=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_UTILS=OFF",
        "-DLLVM_TOOL_BUGPOINT_BUILD=OFF",
        "-DLLVM_TOOL_BUGPOINT_PASSES_BUILD=OFF",
        "-DLLVM_TOOL_DSYMUTIL_BUILD=OFF",
        "-DLLVM_TOOL_GOLD_BUILD=OFF",
        "-DLLVM_TOOL_LLC_BUILD=OFF",
        "-DLLVM_TOOL_LLI_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_AS_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_BCANALYZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CAT_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CFI_VERIFY_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CONFIG_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CVTRES_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CXXDUMP_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CXXFILT_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_CXXMAP_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_C_TEST_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_DIFF_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_DWARFDUMP_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_ELFABI_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_EXEGESIS_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_EXTRACT_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_GO_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_GSYMUTIL_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_IFS_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_ISEL_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_ITANIUM_DEMANGLE_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_JITLINK_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_JITLISTENER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LIBTOOL_DARWIN_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LINK_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LIPO_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LTO2_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LTO_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_MCA_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_MC_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_ML_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_MICROSOFT_DEMANGLE_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_MT_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_OPT_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_PDBUTIL_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_PROFDATA_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_PROFGEN_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_RC_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_READOBJ_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_REDUCE_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_RTDYLD_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_SHLIB_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_SIZE_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_SPECIAL_CASE_LIST_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_SPLIT_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_STRESS_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_STRINGS_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_XRAY_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_YAML_NUMERIC_PARSER_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_YAML_PARSER_FUZZER_BUILD=OFF",
        "-DLLVM_TOOL_LTO_BUILD=OFF",
        "-DLLVM_TOOL_OBJ2YAML_BUILD=OFF",
        "-DLLVM_TOOL_OPT_BUILD=OFF",
        "-DLLVM_TOOL_OPT_VIEWER_BUILD=OFF",
        "-DLLVM_TOOL_REMARKS_SHLIB_BUILD=OFF",
        "-DLLVM_TOOL_SPLIT_FILE_BUILD=OFF",
        "-DLLVM_TOOL_VERIFY_USELISTORDER_BUILD=OFF",
        "-DLLVM_TOOL_YAML2OBJ_BUILD=OFF",
    ]

def _local_cmake_build_clang_toolchain(repository_ctx):
    """Locally build the LLVM toolchain with CMake and Ninja.

    This is used as a fallback for when we can't download a prebuilt set of
    binaries and libraries for a particular platform.
    """
    repository_ctx.report_progress("Configuring Clang toolchain bootstrap...")
    is_clang, environment = _detect_system_clang(repository_ctx)

    cmake = repository_ctx.which("cmake")
    if not cmake:
        fail("`cmake` not found: is it installed?")
    ninja = repository_ctx.which("ninja")
    if not ninja:
        fail("`ninja` not found: is it installed?")

    workspace_dir = repository_ctx.path(repository_ctx.attr._workspace).dirname
    llvm_dir = repository_ctx.path("%s/third_party/llvm-project/llvm" %
                                   workspace_dir)

    repository_ctx.report_progress(
        "Running CMake for the Clang toolchain build...",
    )
    cmake_args = [cmake, "-G", "Ninja", str(llvm_dir)]
    cmake_args += _get_cmake_defines(repository_ctx, is_clang)
    _run(
        repository_ctx,
        cmake_args,
        timeout = 600,
        environment = environment,
        # This is very slow, so print output as a form of progress.
        quiet = False,
    )

    # Run ninja for the final build.
    repository_ctx.report_progress("Building the Clang toolchain...")
    _run(
        repository_ctx,
        [ninja],
        timeout = 10800,
        # This is very slow, so print output as a form of progress.
        quiet = False,
    )

def _download_prebuilt_toolchain(repository_ctx):
    """Downloads and extracts an LLVM build for the current platform.

    Returns `True` when a toolchain can be successfully downloaded.
    """
    repository_ctx.report_progress("Checking for a downloadable toolchain...")
    os = repository_ctx.os.name
    if os == "linux":
        url = "https://github.com/mmdriley/llvm-builds/releases/download/r32/llvm-linux.tar.xz"
        sha256 = "db9f2698aa84935efca3402bdebada127de16f6746adbe54d4cdb7e3b8fec5f3"
    elif os == "mac os x":
        url = "https://github.com/mmdriley/llvm-builds/releases/download/r32/llvm-macos.tar.xz"
        sha256 = "937b81c235977ed2b265baf656f30b7a03c33b6299090d91beb72c2b41846673"
    elif os.startswith("windows"):
        url = "https://github.com/mmdriley/llvm-builds/releases/download/r32/llvm-windows.tar.xz"
        sha256 = "b6b015f9f2fcfb79381004e6a3ae925df4fb827cf7e07f3d5b0b66210fddd172"
    else:
        print(("No prebuilt LLVM toolcahin to download for {}, falling back " +
               "to a local build. This may be very slow!").format(os))
        return False

    repository_ctx.report_progress("Downloading and extracting a toolchain...")
    repository_ctx.download_and_extract(url, sha256 = sha256)
    return True

def _bootstrap_clang_toolchain_impl(repository_ctx):
    """Bootstrap a fresh Clang and LLVM toolchain for use.

    This will first try to download a pre-built archive of the LLVM toolchain
    if one is available.

    Otherwise will locally build the toolchain using CMake out of the LLVM
    submodule.
    """
    force_local_build = False
    if FORCE_LOCAL_BOOTSTRAP_ENV in repository_ctx.os.environ:
        print("Forcing a local bootstrap build. This may be very slow!")
        force_local_build = True

    if force_local_build or not _download_prebuilt_toolchain(repository_ctx):
        # Fallback to a local build.
        _local_cmake_build_clang_toolchain(repository_ctx)

    # Create an empty BUILD file to mark the package. The files are used without
    # Bazel labels directly pointing at them.
    repository_ctx.file("BUILD")

bootstrap_clang_toolchain = repository_rule(
    implementation = _bootstrap_clang_toolchain_impl,
    configure = True,
    attrs = {
        # We use a label pointing at the workspace file to compute the
        # workspace directory.
        "_workspace": attr.label(
            default = Label("//:WORKSPACE"),
            allow_single_file = True,
        ),
    },
    environ = ["CC", "CXX", FORCE_LOCAL_BOOTSTRAP_ENV],
)
