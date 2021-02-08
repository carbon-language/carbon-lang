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
        quiet = True,
        working_directory = ""):
    """Runs the provided `cmd`, checks for failure, and returns the result."""
    exec_result = repository_ctx.execute(
        cmd,
        timeout = timeout,
        quiet = quiet,
        # Need to convert path objects to a string.
        working_directory = str(working_directory),
    )
    if exec_result.return_code != 0:
        fail("Unable to run command successfully: %s" % str(cmd))

    return exec_result

<<<<<<< HEAD
def _detect_or_build_clang(repository_ctx):
    """Returns the path to a Clang executable if it can find one.

    This looks for third_party/llvm-project/build/bin/clang. If that doesn't
    exist, it will be build it.
    """

    cmake = repository_ctx.which("cmake")
    if not cmake:
        fail("`cmake` not found: is it installed?")
    ninja = repository_ctx.which("ninja")
    if not ninja:
        fail("`ninja` not found: is it installed?")

    workspace_dir = repository_ctx.path(repository_ctx.attr._workspace).dirname
    llvm_dir = repository_ctx.path("%s/third_party/llvm-project/llvm" %
                                   workspace_dir)
    repository_ctx.report_progress("Running CMake for the LLVM toolchain build...")
    cmake_args = [
        cmake,
        "-G",
        "Ninja",
        str(llvm_dir),
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;libcxx;libcxxabi;compiler-rt;libunwind",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLIBCXX_ABI_UNSTABLE=ON",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_STATIC_LINK_CXX_STDLIB=ON",
        "-DLLVM_TARGETS_TO_BUILD=AArch64;X86",
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
    _run(
        repository_ctx,
        cmake_args,
        timeout = 600,
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

    return clang

||||||| parent of c4e8c4c (Split bootstrap from bazel toolchain configuration.)
def _detect_or_build_clang(repository_ctx):
    """Returns the path to a Clang executable if it can find one.

    This looks for third_party/llvm-project/build/bin/clang. If that doesn't
    exist, it will be build it.
    """

    # If we can build our Clang toolchain using a system-installed Clang, try
    # to do so. However, if the user provides an explicit `CC` environment
    # variable, just use that as the system C++ compiler.
    environment = {}
    if not repository_ctx.os.environ.get("CC"):
        system_clang = repository_ctx.which("clang")
        if system_clang:
            environment.update(CC = str(system_clang))

    cmake = repository_ctx.which("cmake")
    if not cmake:
        fail("`cmake` not found: is it installed?")
    ninja = repository_ctx.which("ninja")
    if not ninja:
        fail("`ninja` not found: is it installed?")

    workspace_dir = repository_ctx.path(repository_ctx.attr._workspace).dirname
    llvm_dir = repository_ctx.path("%s/third_party/llvm-project/llvm" %
                                   workspace_dir)
    repository_ctx.report_progress("Running CMake for the LLVM toolchain build...")
    cmake_args = [
        cmake,
        "-G",
        "Ninja",
        str(llvm_dir),
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;libcxx;libcxxabi;compiler-rt;libunwind",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLIBCXX_ABI_UNSTABLE=ON",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_STATIC_LINK_CXX_STDLIB=ON",
        "-DLLVM_TARGETS_TO_BUILD=AArch64;X86",
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

    return clang

=======
>>>>>>> c4e8c4c (Split bootstrap from bazel toolchain configuration.)
def _validate_clang(repository_ctx, clang):
    """Validates that the discovered clang is correctly set up."""
    version_output = _run(repository_ctx, [clang, "--version"]).stdout
    if "clang" not in version_output:
        fail(("Selected Clang executable (`%s`) does not appear to actually " +
              "be Clang.") % clang)

    # Make sure this is part of a complete Clang and LLVM toolchain.
    for tool in _CLANG_LLVM_TOOLS:
        if not clang.dirname.get_child(tool).exists:
            fail(("Couldn't find executable `%s` that is expected to be part " +
                  "of the Clang and LLVM toolchain detected with `%s`.") %
                 (tool, clang))

def _compute_clang_resource_dir(repository_ctx, clang):
    """Runs the `clang` binary to get its resource dir."""
    output = _run(
        repository_ctx,
        [clang, "-no-canonical-prefixes", "--print-resource-dir"],
    ).stdout

    # The only line printed is this path.
    return output.splitlines()[0]

def _compute_mac_os_sysroot(repository_ctx):
    """Runs `xcrun` to extract the correct sysroot."""
    xcrun = repository_ctx.which("xcrun")
    if not xcrun:
        fail("`xcrun` not found: is Xcode installed?")
    output = _run(repository_ctx, [xcrun, "--show-sdk-path"]).stdout
    return output.splitlines()[0]

def _compute_clang_cpp_include_search_paths(repository_ctx, clang, sysroot):
    """Runs the `clang` binary and extracts the include search paths.

    Returns the resulting paths as a list of strings.
    """

    # The only way to get this out of Clang currently is to parse the verbose
    # output of the compiler when it is compiling C++ code.
    cmd = [
        clang,
        # Avoid canonicalizing away symlinks.
        "-no-canonical-prefixes",
        # Extract verbose output.
        "-v",
        # Just parse the input, don't generate outputs.
        "-fsyntax-only",
        # Force the language to be C++.
        "-x",
        "c++",
        # Read in an empty input file.
        "/dev/null",
    ]
    if repository_ctx.os.name.lower().startswith("mac os"):
        if not sysroot:
            fail("Must provide a sysroot on macOS!")
        cmd.append("--sysroot=" + sysroot)
    else:
        cmd.append("-stdlib=libc++")

    # Note that verbose output is on stderr, not stdout!
    output = _run(repository_ctx, cmd).stderr.splitlines()

    # Return the list of directories printed for system headers. These are the
    # only ones that Bazel needs us to manually provide. We find these by
    # searching for a begin and end marker. We also have to strip off a leading
    # space from each path.
    include_begin = output.index("#include <...> search starts here:") + 1
    include_end = output.index("End of search list.", include_begin)
    return [
        repository_ctx.path(s.lstrip(" "))
        for s in output[include_begin:include_end]
    ]

def _configure_clang_toolchain_impl(repository_ctx):
    # First just symlink in the untemplated parts of the toolchain repo.
    repository_ctx.symlink(repository_ctx.attr._clang_toolchain_build, "BUILD")
    repository_ctx.symlink(
        repository_ctx.attr._clang_cc_toolchain_config,
        "cc_toolchain_config.bzl",
    )

    # Run the bootstrapped clang to detect relevant features for the toolchain.
    clang = repository_ctx.path(repository_ctx.attr.clang)
    if clang.basename != "clang":
        fail("The provided Clang binary is not spelled `clang`, but: %s" % clang.basename)
    # Adjust this to the "clang++" binary to ensure we get the correct behavior
    # when configuring it.
    clang = repository_ctx.path(str(clang) + "++")
    _validate_clang(repository_ctx, clang)
    resource_dir = _compute_clang_resource_dir(repository_ctx, clang)
    sysroot_dir = None
    if repository_ctx.os.name.lower().startswith("mac os"):
        sysroot_dir = _compute_mac_os_sysroot(repository_ctx)
    include_dirs = _compute_clang_cpp_include_search_paths(
        repository_ctx,
        clang,
        sysroot_dir,
    )

    repository_ctx.template(
        "clang_detected_variables.bzl",
        repository_ctx.attr._clang_detected_variables_template,
        substitutions = {
            "{LLVM_BINDIR}": str(clang.dirname),
            "{CLANG_RESOURCE_DIR}": resource_dir,
            "{CLANG_INCLUDE_DIRS_LIST}": str([str(path) for path in include_dirs]),
            "{SYSROOT}": str(sysroot_dir),
        },
        executable = False,
    )

configure_clang_toolchain = repository_rule(
    implementation = _configure_clang_toolchain_impl,
    configure = True,
    attrs = {
        "_clang_toolchain_build": attr.label(
            default = Label("//bazel/cc_toolchains:clang_toolchain.BUILD"),
            allow_single_file = True,
        ),
        "_clang_cc_toolchain_config": attr.label(
            default = Label("//bazel/cc_toolchains:clang_cc_toolchain_config.bzl"),
            allow_single_file = True,
        ),
        "_clang_detected_variables_template": attr.label(
            default = Label("//bazel/cc_toolchains:clang_detected_variables.tpl.bzl"),
            allow_single_file = True,
        ),
        # This must point at the `clang` binary inside a full LLVM toolchain
        # installation.
        "clang": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
    environ = ["CC"],
)
