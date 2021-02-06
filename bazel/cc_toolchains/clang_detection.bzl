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

def _detect_or_build_clang(repository_ctx):
    """Returns the path to a Clang executable if it can find one.

    This looks for third_party/llvm-project/build/bin/clang. If that doesn't
    exist, it will be build it.
    """
    llvm_root = repository_ctx.path("%s/third_party/llvm-project" %
                                    repository_ctx.attr.workspace_dir)
    clang = repository_ctx.path("%s/build/bin/clang" % llvm_root)
    if clang.exists:
        return clang

    cmake = repository_ctx.which("cmake")
    if not cmake:
        fail("`cmake` not found: is it installed?")
    mkdir = repository_ctx.which("mkdir")
    if not cmake:
        fail("`mkdir` not found: unsupported OS?")
    ninja = repository_ctx.which("ninja")
    if not ninja:
        fail("`ninja` not found: is it installed?")

    llvm_dir = repository_ctx.path("%s/llvm" % llvm_root)
    if not llvm_dir.exists:
        fail(
            ("`%s` not found: are submodules initialized? " +
             "(git submodule update --init)") %
            llvm_dir,
        )

    repository_ctx.report_progress("Clang/LLVM not found, starting build.")
    build_dir = repository_ctx.path("%s/build" % llvm_root)
    if not build_dir.exists:
        _run(repository_ctx, [mkdir, build_dir])

    cmake_args = [
        cmake,
        "-G",
        "Ninja",
        "../llvm",
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;libcxx;libcxxabi;compiler-rt;libunwind",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLIBCXX_ABI_UNSTABLE=ON",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLIBCXX_ENABLE_ASSERTIONS=OFF",
        "-DLIBCXXABI_ENABLE_ASSERTIONS=OFF",
    ]
    _run(
        repository_ctx,
        cmake_args,
        timeout = 600,
        # This is very slow, so print output as a form of progress.
        quiet = False,
        working_directory = build_dir,
    )

    # Run ninja for the final build.
    _run(
        repository_ctx,
        [ninja],
        timeout = 3600,
        # This is very slow, so print output as a form of progress.
        quiet = False,
        working_directory = build_dir,
    )

    if not clang.exists:
        fail("`%s` still not found after building LLVM" % clang)

    return clang

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

def _compute_clang_cpp_include_search_paths(repository_ctx, clang):
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
        # Use libc++ rather than any other standard library.
        "-stdlib=libc++",
        # Force the language to be C++.
        "-x",
        "c++",
        # Read in an empty input file.
        "/dev/null",
    ]

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

def _detect_clang_toolchain_impl(repository_ctx):
    # First just symlink in the untemplated parts of the toolchain repo.
    repository_ctx.symlink(repository_ctx.attr._clang_toolchain_build, "BUILD")
    repository_ctx.symlink(
        repository_ctx.attr._clang_cc_toolchain_config,
        "cc_toolchain_config.bzl",
    )

    clang = _detect_or_build_clang(repository_ctx)
    _validate_clang(repository_ctx, clang)
    resource_dir = _compute_clang_resource_dir(repository_ctx, clang)
    include_dirs = _compute_clang_cpp_include_search_paths(
        repository_ctx,
        clang,
    )

    repository_ctx.template(
        "clang_detected_variables.bzl",
        repository_ctx.attr._clang_detected_variables_template,
        substitutions = {
            "{LLVM_BINDIR}": str(clang.dirname),
            "{CLANG_RESOURCE_DIR}": resource_dir,
            "{CLANG_INCLUDE_DIRS_LIST}": str([str(path) for path in include_dirs]),
        },
        executable = False,
    )

detect_clang_toolchain = repository_rule(
    implementation = _detect_clang_toolchain_impl,
    configure = True,
    local = True,
    attrs = {
        "workspace_dir": attr.string(mandatory = True),
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
    },
    environ = ["CC"],
)
