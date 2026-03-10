# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides variables and rules to work with Clang's runtime library sources.

These are organized into groups based on the Clang runtimes providing them and
how they are built:
- CRT: The C language runtimes not provided by the C standard library, currently
  just infrastructure for global initialization and teardown.
- Builtins: The compiler builtins library mirroring `libgcc` that provides
  function definitions for operations not reliably available in hardware but
  needed by Clang.
- Libc++ and libc++abi: The C++ standard library and its ABI components.
- Libunwind: The unwinding library.

Future runtimes we plan to add support for but not yet included:
- Sanitizers
- Profiling runtimes
"""

load("@llvm-project//compiler-rt:compiler-rt.bzl", "builtins_copts", "crt_copts")
load("@llvm-project//libcxx:libcxx_library.bzl", "libcxx_and_abi_copts")
load("@llvm-project//libunwind:libunwind_library.bzl", "libunwind_copts")
load("//bazel/cc_rules:defs.bzl", "cc_library")

CRT_FILES = {
    "crtbegin_src": "@llvm-project//compiler-rt:builtins_crtbegin_src",
    "crtend_src": "@llvm-project//compiler-rt:builtins_crtend_src",
}

BUILTINS_SRCS_FILEGROUPS = [
    "@llvm-project//compiler-rt:builtins_aarch64_srcs",
    "@llvm-project//compiler-rt:builtins_i386_srcs",
    "@llvm-project//compiler-rt:builtins_x86_64_srcs",
]

BUILTINS_TEXTUAL_SRCS_FILEGROUPS = [
    "@llvm-project//compiler-rt:builtins_aarch64_textual_srcs",
    "@llvm-project//compiler-rt:builtins_i386_textual_srcs",
    "@llvm-project//compiler-rt:builtins_x86_64_textual_srcs",
]

RUNTIMES_HDRS_FILEGROUPS = [
    "@llvm-project//libc:libcxx_shared_headers_hdrs",
    "@llvm-project//libcxx:libcxx_hdrs",
    "@llvm-project//libcxxabi:libcxxabi_hdrs",
    "@llvm-project//libunwind:libunwind_hdrs",
]

RUNTIMES_SRCS_FILEGROUPS = [
    "@llvm-project//libcxx:libcxx_linux_srcs",
    "@llvm-project//libcxx:libcxx_macos_srcs",
    "@llvm-project//libcxx:libcxx_win32_srcs",
    "@llvm-project//libcxxabi:libcxxabi_srcs",
    "@llvm-project//libunwind:libunwind_srcs",
]

RUNTIMES_TEXTUAL_SRCS_FILEGROUPS = [
    "@llvm-project//libcxxabi:libcxxabi_textual_srcs",
]

RUNTIMES_PREFIXES = {
    "libcxx_hdrs": "libcxx/",
    "libcxx_linux_srcs": "libcxx/",
    "libcxx_macos_srcs": "libcxx/",
    "libcxx_shared_headers_hdrs": "libc/internal/",
    "libcxx_win32_srcs": "libcxx/",
    "libcxxabi_hdrs": "libcxxabi/",
    "libcxxabi_srcs": "libcxxabi/",
    "libcxxabi_textual_srcs": "libcxxabi/",
    "libunwind_hdrs": "libunwind/",
    "libunwind_srcs": "libunwind/",
}

def _get_name(target):
    return target.split(":")[1]

def _format_one_per_line(list):
    return "\n" + "\n".join([
        '    "{0}",'.format(item)
        for item in list
    ]) + "\n"

def _builtins_path(file):
    """Returns the install path for a file in CompilerRT's builtins library."""
    path = file.path

    # Skip to the relative path below the workspace root.
    path = path.rpartition(file.owner.workspace_root + "/")[2]

    # And now we can predictably remove the `compiler-rt/lib` prefix.
    path = path.removeprefix("compiler-rt/lib/")
    if not path.startswith("builtins/"):
        fail("Not a builtins-relative path for: {0}".format(file.path))
    return path

def _runtimes_path(file):
    """Returns the install path for a file in a normal runtimes library."""
    return file.owner.name

def _get_path(file_attr, to_path_fn):
    files = file_attr[DefaultInfo].files.to_list()
    if len(files) > 1:
        fail("Expected a single file and got {0} files.".format(len(files)))

    return '"{0}"'.format(to_path_fn(files[0]))

def _get_paths(files_attr, to_path_fn, prefix = ""):
    files = []
    for src in files_attr:
        files.extend(src[DefaultInfo].files.to_list())
        files.extend(src[DefaultInfo].default_runfiles.files.to_list())

    return _format_one_per_line([
        "{0}{1}".format(prefix, to_path_fn(f))
        for f in files
    ])

def _get_substitutions(ctx):
    key_attr = lambda k: getattr(ctx.attr, "_" + k)
    return {
        "BUILTINS_COPTS": _format_one_per_line(builtins_copts),
        "CRT_COPTS": _format_one_per_line(crt_copts),
        "LIBCXX_AND_ABI_COPTS": _format_one_per_line(libcxx_and_abi_copts),
        "LIBUNWIND_COPTS": _format_one_per_line(libunwind_copts),
    } | {
        k.upper(): _get_path(key_attr(k), _builtins_path)
        for k in CRT_FILES.keys()
    } | {
        name.upper(): _get_paths(key_attr(name), _builtins_path)
        for name in [_get_name(g) for g in (
            BUILTINS_SRCS_FILEGROUPS + BUILTINS_TEXTUAL_SRCS_FILEGROUPS
        )]
    } | {
        # Other runtimes are installed under separate directories named the
        # same as their key.
        name.upper(): _get_paths(
            key_attr(name),
            _runtimes_path,
            RUNTIMES_PREFIXES[name],
        )
        for name in [_get_name(g) for g in (
            RUNTIMES_HDRS_FILEGROUPS + RUNTIMES_SRCS_FILEGROUPS +
            RUNTIMES_TEXTUAL_SRCS_FILEGROUPS
        )]
    }

def _generate_runtimes_build_info_h_rule(ctx):
    h_file = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.expand_template(
        template = ctx.file._template_file,
        output = h_file,
        substitutions = _get_substitutions(ctx),
    )
    return [DefaultInfo(files = depset([h_file]))]

generate_runtimes_build_info_h = rule(
    implementation = _generate_runtimes_build_info_h_rule,
    attrs = {
        "_" + k: attr.label(default = v, allow_single_file = True)
        for k, v in CRT_FILES.items()
    } | {
        "_" + _get_name(g): attr.label_list(default = [g], allow_files = True)
        for g in (
            BUILTINS_SRCS_FILEGROUPS +
            BUILTINS_TEXTUAL_SRCS_FILEGROUPS +
            RUNTIMES_HDRS_FILEGROUPS +
            RUNTIMES_SRCS_FILEGROUPS +
            RUNTIMES_TEXTUAL_SRCS_FILEGROUPS
        )
    } | {
        "_template_file": attr.label(
            default = "runtimes_build_info.tpl.h",
            allow_single_file = True,
        ),
    },
)

def generate_runtimes_build_info_cc_library(name, deps = [], **kwargs):
    """Generates a `runtimes_build_info.h` header and a `cc_library` rule.

    This first generates the header file with variables describing the runtimes
    build info from Clang, and then a `cc_library` that exports that header.

    The `cc_library` rule name is the provided `name` and should be depended on
    by code that includes the generated header. The `kwargs` are expanded into
    the `cc_library` in case other attributes need to be configured there.
    """
    generate_runtimes_build_info_h(name = "runtimes_build_info.h")
    cc_library(
        name = name,
        hdrs = ["runtimes_build_info.h"],
        deps = [
            # For StringRef.h
            "@llvm-project//llvm:Support",
        ] + deps,
        **kwargs
    )
