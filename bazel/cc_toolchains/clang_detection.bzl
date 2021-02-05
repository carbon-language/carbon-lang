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

def _run(repository_ctx, cmd):
    """Runs the provided `cmd`, checks for failure, and returns the result."""
    exec_result = repository_ctx.execute(cmd, timeout = 10)
    if exec_result.return_code != 0:
        fail("Unable to run command successfully: %s" % str(cmd))

    return exec_result

def _find_clang(repository_ctx):
    """Returns the path to a Clang executable if it can find one.

    This assumes the `CC` environment variable points to a Clang binary or
    looks for one on the path.
    """
    clang = repository_ctx.path("%s/third_party/llvm-project/build/bin/clang" %
                                repository_ctx.attr.workspace_dir)
    if not clang.exists:
        cc_env = repository_ctx.os.environ.get("CC")
        if not cc_env:
            # Without a specified `CC` name, simply look for `clang`.
            clang = repository_ctx.which("clang")
        elif "/" not in cc_env:
            # Lookup relative `CC` names according to the system `PATH`.
            clang = repository_ctx.which(cc_env)
        else:
            # An absolute `CC` path is simply be used directly.
            clang = repository_ctx.path(cc_env)
            if not clang.exists:
                fail(("The `CC` environment variable is set to a path (`%s`) " +
                      "that doesn't exist.") % cc_env)

    # Check if either of the `which` invocations fail.
    if not clang:
        missing = "`clang`"
        if cc_env:
            missing = "`%s` (from the `CC` environment variable)" % cc_env
        fail("Unable to find the %s executable on the PATH." % missing)

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

    return clang

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

    clang = _find_clang(repository_ctx)
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
