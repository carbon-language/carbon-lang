# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark repository rules to detect and configure Clang (and LLVM) toolchain.

These rules should be run from the `WORKSPACE` file to substitute appropriate
configured values into a `clang_detected_variables.bzl` file that can be used
by the actual toolchain configuration.
"""

# Tools that we verify are present as part of the detected Clang & LLVM toolchain.
_CLANG_LLVM_TOOLS = [
    "llvm-ar",
    "ld.lld",
    "clang-cpp",
    "clang",
    "clang++",
    "llvm-dwp",
    "llvm-cov",
    "llvm-nm",
    "llvm-objcopy",
    "llvm-strip",
]

def _run(
        repository_ctx,
        cmd,
        timeout = 10,
        environment = {},
        quiet = True,
        working_directory = ""):
    """Runs the provided `cmd`, checks for failure, and returns the result."""
    exec_result = repository_ctx.execute(
        cmd,
        timeout = timeout,
        environment = environment,
        quiet = quiet,
        # Need to convert path objects to a string.
        working_directory = str(working_directory),
    )
    if exec_result.return_code != 0:
        fail("Unable to run command successfully: %s" % str(cmd))

    return exec_result

def _bootstrap_clang_toolchain_impl(repository_ctx):
    """Returns the path a bootstrapped Clang executable.

    This bootstraps Clang and the rest of the LLVM toolchain from the LLVM
    submodule.
    """
    # If we can build our Clang toolchain using a system-installed Clang, try
    # to do so. However, if the user provides an explicit `CC` environment
    # variable, just use that as the system C++ compiler.
    is_clang = False
    environment = {}
    cc = repository_ctx.os.environ.get("CC")
    if not cc:
        system_clang = repository_ctx.which("clang")
        if system_clang:
            is_clang = True
            environment.update(CC = str(system_clang))
    else:
        version_output = _run(repository_ctx, [cc, "--version"]).stdout
        if "clang" not in version_output:
            is_clang = True

    cmake = repository_ctx.which("cmake")
    if not cmake:
        fail("`cmake` not found: is it installed?")
    ninja = repository_ctx.which("ninja")
    if not ninja:
        fail("`ninja` not found: is it installed?")

    workspace_dir = repository_ctx.path(repository_ctx.attr._workspace).dirname
    llvm_dir = repository_ctx.path("%s/third_party/llvm-project/llvm" %
                                   workspace_dir)
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
    cmake_args = [
        cmake,
        "-G",
        "Ninja",
        str(llvm_dir),
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;libcxx;libcxxabi;compiler-rt;libunwind",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_ENABLE_MODULES=" + modules_setting,
        "-DLLVM_STATIC_LINK_CXX_STDLIB=" + static_link_cxx,
        "-DLLVM_TARGETS_TO_BUILD=AArch64;X86",
        "-DLIBCXX_ABI_UNSTABLE=" + unstable_libcxx_abi,
        "-DLIBCXX_ENABLE_ASSERTIONS=OFF",
        "-DLIBCXXABI_ENABLE_ASSERTIONS=OFF",

        # Disable components of the build that we'll never need.
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
    repository_ctx.report_progress("Running CMake for the LLVM toolchain build...")
    _run(
        repository_ctx,
        cmake_args,
        timeout = 600,
        environment = environment,
        # This is very slow, so print output as a form of progress.
        quiet = False,
    )

    repository_ctx.report_progress("Building the LLVM toolchain...")

    # Run ninja for the final build.
    _run(
        repository_ctx,
        [ninja],
        timeout = 3600,
        # This is very slow, so print output as a form of progress.
        quiet = False,
    )

    clang = repository_ctx.path("bin/clang")
    if not clang.exists:
        fail("`%s` still not found after building the LLVM toolchain" % clang)

    # Create an empty BUILD file to mark the package, the files are used without
    # Bazel labels directly pointing at them.
    repository_ctx.file("BUILD", content="")

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
    environ = ["CC"],
)
