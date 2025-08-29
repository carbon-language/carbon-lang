# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides variables and rules to work with Clang's runtime library sources.

These are organized into groups based on the runtime functionality:
- CRT: The C language runtimes not provided by the C standard library, currently
  just infrastructure for global initialization and teardown.
- Builtins: The compiler builtins library mirroring `libgcc` that provides
  function definitions for operations not reliably available in hardware but
  needed by Clang.

Future runtimes we plan to add support for but not yet included:
- Libunwind
- Libc++ and libc++abi
- Sanitizers
- Profiling runtimes
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

CRT_FILES = {
    "crtbegin_src": "@llvm-project//compiler-rt:builtins_crtbegin_src",
    "crtend_src": "@llvm-project//compiler-rt:builtins_crtend_src",
}

BUILTINS_FILEGROUPS = {
    "aarch64_srcs": "@llvm-project//compiler-rt:builtins_aarch64_srcs",
    "bf16_srcs": "@llvm-project//compiler-rt:builtins_bf16_srcs",
    "generic_srcs": "@llvm-project//compiler-rt:builtins_generic_srcs",
    "i386_srcs": "@llvm-project//compiler-rt:builtins_i386_srcs",
    "macos_srcs": "@llvm-project//compiler-rt:builtins_macos_atomic_srcs",
    "tf_srcs": "@llvm-project//compiler-rt:builtins_tf_srcs",
    "x86_64_srcs": "@llvm-project//compiler-rt:builtins_x86_64_srcs",
    "x86_arch_srcs": "@llvm-project//compiler-rt:builtins_x86_arch_srcs",
    "x86_fp80_srcs": "@llvm-project//compiler-rt:builtins_x86_fp80_srcs",
}

_TEMPLATE = """
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generated header file of strings describing the Clang runtime library source
// files.
//
// See toolchain/driver/runtime_sources.bzl for more details.

#ifndef CARBON_TOOLCHAIN_BASE_RUNTIME_SOURCES_H_
#define CARBON_TOOLCHAIN_BASE_RUNTIME_SOURCES_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon::RuntimeSources {{

constexpr inline llvm::StringLiteral CrtBegin = {crtbegin_src};
constexpr inline llvm::StringLiteral CrtEnd = {crtend_src};

constexpr inline llvm::StringLiteral BuiltinsGenericSrcs[] = {{
{generic_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsMacosSrcs[] = {{
{macos_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsBf16Srcs[] = {{
{bf16_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsTfSrcs[] = {{
{tf_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsX86ArchSrcs[] = {{
{x86_arch_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsX86Fp80Srcs[] = {{
{x86_fp80_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsAarch64Srcs[] = {{
{aarch64_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsX86_64Srcs[] = {{
{x86_64_srcs}
}};
constexpr inline llvm::StringLiteral BuiltinsI386Srcs[] = {{
{i386_srcs}
}};

}}  // namespace Carbon::RuntimeSources

#endif  // CARBON_TOOLCHAIN_BASE_RUNTIME_SOURCES_H_
"""

def _builtins_path(file):
    """Returns the runtime install path for a file in CompilerRT's builtins library."""

    # The CompilerRT package has the builtins runtime sources in the
    # "lib/builtins/" subdirectory, and we install into a "builtins/"
    # subdirectory, so just remove the "lib/" prefix from the package-relative
    # label name.
    return file.owner.name.removeprefix("lib/")

def _get_path(file_attr, to_path_fn):
    files = file_attr[DefaultInfo].files.to_list()
    if len(files) > 1:
        fail(msg = "Expected a single file and got {0} files.".format(len(files)))

    return '"{0}"'.format(to_path_fn(files[0]))

def _get_paths(files_attr, to_path_fn):
    files = []
    for src in files_attr:
        files.extend(src[DefaultInfo].files.to_list())
        files.extend(src[DefaultInfo].default_runfiles.files.to_list())

    return "\n".join([
        '    "{0}",'.format(to_path_fn(f))
        for f in files
    ])

def _generate_runtime_sources_h_rule(ctx):
    h_file = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(h_file, _TEMPLATE.format(**({
        k: _get_path(getattr(ctx.attr, "_" + k), _builtins_path)
        for k in CRT_FILES.keys()
    } | {
        k: _get_paths(getattr(ctx.attr, "_" + k), _builtins_path)
        for k in BUILTINS_FILEGROUPS.keys()
    })))
    return [DefaultInfo(files = depset([h_file]))]

generate_runtime_sources_h = rule(
    implementation = _generate_runtime_sources_h_rule,
    attrs = {
        "_" + k: attr.label(default = v, allow_single_file = True)
        for k, v in CRT_FILES.items()
    } | {
        "_" + k: attr.label_list(default = [v], allow_files = True)
        for k, v in BUILTINS_FILEGROUPS.items()
    },
)

def generate_runtime_sources_cc_library(name, **kwargs):
    """Generates a `runtime_sources.h` header and a `cc_library` rule for it.

    This first generates the header file with variables describing the runtime
    sources from Clang, and then a `cc_library` that exports that header.

    The `cc_library` rule name is the provided `name` and should be depended on
    by code that includes the generated header. The `kwargs` are expanded into
    the `cc_library` in case other attributes need to be configured there.
    """
    generate_runtime_sources_h(name = "runtime_sources.h")
    cc_library(
        name = name,
        hdrs = ["runtime_sources.h"],
        **kwargs
    )
