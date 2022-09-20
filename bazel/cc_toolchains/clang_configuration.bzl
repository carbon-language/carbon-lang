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

def _clang_version(version_output):
    """Returns clang's major version number, or None if not found."""
    version_prefix = "clang version "
    version_start = version_output.find(version_prefix)
    if version_start == -1:
        # No version
        return None
    version_start += len(version_prefix)

    # Find a dot to indicate something like 'clang version 14.0.6'.
    version_dot = version_output.find(".", version_start)
    if version_dot == -1:
        return None

    # Make sure the dot was on the same line as the version.
    if version_output.find("\n", version_start) < version_dot:
        return None

    # Return the version as int.
    return int(version_output[version_start:version_dot])

def _detect_system_clang(repository_ctx):
    """Detects whether the system-provided clang can be used.

    Returns a tuple of (is_clang, environment).
    """

    # If the user provides an explicit `CC` environment variable, use that as
    # the compiler. This should point at the `clang` executable to use.
    cc = repository_ctx.os.environ.get("CC")
    cc_path = None
    if cc:
        cc_path = repository_ctx.path(cc)
        if not cc_path.exists:
            cc_path = repository_ctx.which(cc)
    if not cc_path:
        cc_path = repository_ctx.which("clang")
    if not cc_path:
        fail("Cannot find clang or CC (%s); either correct your path or set the CC environment variable" % cc)
    version_output = _run(repository_ctx, [cc_path, "--version"]).stdout
    if "clang" not in version_output:
        fail("Searching for clang or CC (%s), and found (%s), which is not a Clang compiler" % (cc, cc_path))
    return (cc_path, _clang_version(version_output))

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

    # Create an empty temp file for Clang to use
    if repository_ctx.os.name.lower().startswith("windows"):
        repository_ctx.file("_temp", "")

    # Read in an empty input file. If we are building from
    # Windows, then we create an empty temp file. Clang
    # on Windows does not like it when you pass a non-existent file.
    if repository_ctx.os.name.lower().startswith("windows"):
        repository_ctx.file("_temp", "")
        input_file = repository_ctx.path("_temp")
    else:
        input_file = "/dev/null"

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
        input_file,
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

    # Suffix present on framework paths.
    framework_suffix = " (framework directory)"
    return [
        repository_ctx.path(s.lstrip(" ").removesuffix(framework_suffix))
        for s in output[include_begin:include_end][
    ] + ["/usr/lib/llvm-15/lib/clang/"]

def _configure_clang_toolchain_impl(repository_ctx):
    # First just symlink in the untemplated parts of the toolchain repo.
    repository_ctx.symlink(repository_ctx.attr._clang_toolchain_build, "BUILD")
    repository_ctx.symlink(
        repository_ctx.attr._clang_cc_toolchain_config,
        "cc_toolchain_config.bzl",
    )

    # Find a Clang C++ compiler, and where it lives. We need to walk symlinks
    # here as the other LLVM tools may not be symlinked into the PATH even if
    # `clang` is. We also insist on finding the basename of `clang++` as that is
    # important for C vs. C++ compiles.
    (clang, clang_version) = _detect_system_clang(repository_ctx)
    clang = clang.realpath.dirname.get_child("clang++")

    # Compute the various directories used by Clang.
    resource_dir = _compute_clang_resource_dir(repository_ctx, clang)
    sysroot_dir = None
    if repository_ctx.os.name.lower().startswith("mac os"):
        sysroot_dir = _compute_mac_os_sysroot(repository_ctx)
    include_dirs = _compute_clang_cpp_include_search_paths(
        repository_ctx,
        clang,
        sysroot_dir,
    )

    # We expect that the LLVM binutils live adjacent to llvm-ar.
    # First look for llvm-ar adjacent to clang, so that if found,
    # it is most likely to match the same version as clang.
    # Otherwise, try PATH.
    arpath = clang.dirname.get_child("llvm-ar")
    if not arpath.exists:
        arpath = repository_ctx.which("llvm-ar")
        if not arpath:
            fail("`llvm-ar` not found in PATH or adjacent to clang")

    # By default Windows uses '\' in its paths. These will be
    # interpreted as escape characters and fail the build, thus
    # we must manually replace the backslashes with '/'
    if repository_ctx.os.name.lower().startswith("windows"):
        resource_dir = resource_dir.replace("\\", "/")
        include_dirs = [str(s).replace("\\", "/") for s in include_dirs]

    repository_ctx.template(
        "clang_detected_variables.bzl",
        repository_ctx.attr._clang_detected_variables_template,
        substitutions = {
            "{LLVM_BINDIR}": str(arpath.dirname),
            "{CLANG_BINDIR}": str(clang.dirname),
            "{CLANG_VERSION}": str(clang_version),
            "{CLANG_RESOURCE_DIR}": resource_dir,
            "{CLANG_INCLUDE_DIRS_LIST}": str(
                [str(path) for path in include_dirs],
            ),
            "{SYSROOT}": str(sysroot_dir),
        },
        executable = False,
    )

configure_clang_toolchain = repository_rule(
    implementation = _configure_clang_toolchain_impl,
    configure = True,
    local = True,
    attrs = {
        "_clang_toolchain_build": attr.label(
            default = Label("//bazel/cc_toolchains:clang_toolchain.BUILD"),
            allow_single_file = True,
        ),
        "_clang_cc_toolchain_config": attr.label(
            default = Label(
                "//bazel/cc_toolchains:clang_cc_toolchain_config.bzl",
            ),
            allow_single_file = True,
        ),
        "_clang_detected_variables_template": attr.label(
            default = Label(
                "//bazel/cc_toolchains:clang_detected_variables.tpl.bzl",
            ),
            allow_single_file = True,
        ),
    },
    environ = ["CC"],
)
