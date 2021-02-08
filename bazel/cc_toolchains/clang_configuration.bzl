# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark repository rules to configure Clang (and LLVM) toolchain for Bazel.

These rules should be run from the `WORKSPACE` file to substitute appropriate
configured values into a `clang_detected_variables.bzl` file that can be used
by the actual toolchain configuration.
"""

def _run(repository_ctx, cmd):
    """Runs the provided `cmd`, checks for failure, and returns the result."""
    exec_result = repository_ctx.execute(cmd)
    if exec_result.return_code != 0:
        fail("Unable to run command successfully: %s" % str(cmd))

    return exec_result

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
        # Always use libc++.
        "-stdlib=libc++",
    ]

    # We need to use a sysroot to correctly represent a run on macOS.
    if repository_ctx.os.name.lower().startswith("mac os"):
        if not sysroot:
            fail("Must provide a sysroot on macOS!")
        cmd.append("--sysroot=" + sysroot)

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
)
