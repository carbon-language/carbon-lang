# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for constructing install information."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mklink", "strip_prefix")
load("//toolchain/base:llvm_tools.bzl", "LLVM_MAIN_TOOLS", "LLVM_TOOL_ALIASES")

_clang_aliases = [
    "clang",
    "clang++",
    "clang-cl",
    "clang-cpp",
]

# TODO: Add remaining aliases of LLD for Windows and WASM when we have support
# for them wired up through the busybox.
_lld_aliases = [
    "ld.lld",
    "ld64.lld",
]

_llvm_binaries = _clang_aliases + _lld_aliases + [
    tool.bin_name
    for tool in LLVM_MAIN_TOOLS.values()
] + [
    "llvm-" + alias
    for (_, aliases) in LLVM_TOOL_ALIASES.items()
    for alias in aliases
]

def _toolchain_llvm_binaries_impl(ctx):
    outputs = []
    for bin in _llvm_binaries:
        out = ctx.actions.declare_file(ctx.attr.prefix + "llvm/bin/" + bin)
        ctx.actions.symlink(
            output = out,
            target_file = ctx.files.carbon_binary[0],
        )
        outputs.append(out)

    return [DefaultInfo(files = depset(direct = outputs))]

toolchain_llvm_binaries = rule(
    doc = "Creates symlinks for LLVM binaries pointing to a single Carbon binary.",
    implementation = _toolchain_llvm_binaries_impl,
    attrs = {
        "carbon_binary": attr.label(
            allow_single_file = True,
            #executable = True,
            mandatory = True,
            #cfg = None,
        ),
        "prefix": attr.string(default = ""),
    },
)

def _removeprefix_or_fail(s, prefix):
    if prefix == "":
        return s
    new_s = s.removeprefix(prefix)
    if new_s == s:
        fail("Unable to remove prefix '{0}' from '{1}'".format(prefix, s))
    return new_s

def _get_pkg_relative_path(file):
    path = file.path
    if file.root.path != "":
        path = _removeprefix_or_fail(path, file.root.path + "/")
    if file.owner.workspace_root != "":
        path = _removeprefix_or_fail(path, file.owner.workspace_root + "/")
    return _removeprefix_or_fail(path, file.owner.package + "/")

def _toolchain_files_impl(ctx):
    prefix = ctx.attr.prefix
    outputs = []
    for src in ctx.files.srcs:
        rel_path = _get_pkg_relative_path(src)
        rel_path = _removeprefix_or_fail(rel_path, ctx.attr.remove_prefix)
        if rel_path in ctx.attr.renames:
            rel_path = ctx.attr.renames[rel_path]
        out = ctx.actions.declare_file("{0}{1}".format(prefix, rel_path))
        ctx.actions.symlink(output = out, target_file = src)
        outputs.append(out)

    return [DefaultInfo(files = depset(outputs))]

toolchain_files = rule(
    doc = "Arranges files into a directory structure by symlinking them with prefix removal and renames.",
    implementation = _toolchain_files_impl,
    attrs = {
        "prefix": attr.string(default = ""),
        "remove_prefix": attr.string(default = ""),
        "renames": attr.string_dict(default = {}),
        "srcs": attr.label_list(allow_files = True),
    },
)

def _filtered_files_impl(ctx):
    include_set = set()
    for filter_src in ctx.files.filter_to_srcs:
        rel_path = _get_pkg_relative_path(filter_src)
        include_set.add(_removeprefix_or_fail(rel_path, ctx.attr.filter_to_srcs_prefix))

    outputs = []
    for src in ctx.files.srcs:
        rel_path = _get_pkg_relative_path(src)
        if _removeprefix_or_fail(rel_path, ctx.attr.srcs_prefix) in include_set:
            outputs.append(src)

    return [DefaultInfo(files = depset(outputs))]

filtered_files = rule(
    doc = "Filters a set of files based on their relative paths matching another set of files.",
    implementation = _filtered_files_impl,
    attrs = {
        "filter_to_srcs": attr.label_list(),
        "filter_to_srcs_prefix": attr.string(default = ""),
        "srcs": attr.label_list(),
        "srcs_prefix": attr.string(default = ""),
    },
)

def filtered_toolchain_files(name, prefix, remove_prefix, srcs_groups):
    """A collection of filtered toolchain files from overlapping `srcs`."""
    name_base = name.removesuffix("_srcs")
    if name_base == "" or name_base == name:
        fail("Invalid name for building a set of filtered toolchain files.")
    toolchain_files(
        name = name_base + "_all_srcs",
        srcs = srcs_groups,
        prefix = prefix,
        remove_prefix = remove_prefix,
    )

    for srcs in srcs_groups:
        suffix = srcs.partition(":")[2].removeprefix(name_base)
        filtered_files(
            name = name_base + suffix,
            srcs = [":" + name_base + "_all_srcs"],
            srcs_prefix = prefix,
            filter_to_srcs = [srcs],
            filter_to_srcs_prefix = remove_prefix,
        )

def toolchain_pkg_filegroup(name, carbon_busybox, srcs, tags = []):
    """Given a CMake-style install prefix[1], the hierarchy looks like:

    - prefix/bin: Binaries intended for direct use.
    - prefix/lib/carbon: Private data and files.
    - prefix/lib/carbon/core: The `Core` package files.
    - prefix/lib/carbon/llvm/bin: LLVM binaries.

    This will be how installs are provided on Unix-y platforms, and is loosely
    based on the FHS (Filesystem Hierarchy Standard). See the CMake install prefix
    documentation[1] for more details.

    [1]: https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html
    """
    pkg_srcs = []

    # Separately handle the busybox so that we can set its attributes.
    pkg_files(
        name = name + "_busybox",
        srcs = [carbon_busybox],
        prefix = "lib/carbon",
        attributes = pkg_attributes(mode = "0755"),
        tags = tags,
    )
    pkg_srcs.append(":" + name + "_busybox")

    pkg_files(
        name = name + "_srcs",
        srcs = srcs,
        prefix = "lib/carbon",
        strip_prefix = strip_prefix.from_pkg(),
        tags = tags,
    )
    pkg_srcs.append(":" + name + "_srcs")

    for bin in _llvm_binaries:
        pkg_mklink(
            name = name + "_llvm_symlink_" + bin,
            link_name = "lib/carbon/llvm/bin/" + bin,
            target = "../../carbon-busybox",
            tags = tags,
        )
        pkg_srcs.append(":" + name + "_llvm_symlink_" + bin)

    pkg_mklink(
        name = name + "_bin_symlink",
        link_name = "bin/carbon",
        target = "../lib/carbon/carbon-busybox",
        tags = tags,
    )
    pkg_srcs.append(":" + name + "_bin_symlink")

    pkg_filegroup(name = name, srcs = pkg_srcs, tags = tags)
