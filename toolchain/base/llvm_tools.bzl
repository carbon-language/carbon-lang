# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides variables and rules to automate working with LLVM's CLI tools."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

# The main LLVM command line tools, including their "primary" name, binary name,
# and the library dependency required to use them.
LLVM_MAIN_TOOLS = {
    "ar": struct(bin_name = "llvm-ar", lib = "@llvm-project//llvm:llvm-ar-lib"),
    "cgdata": struct(bin_name = "llvm-cgdata", lib = "@llvm-project//llvm:llvm-cgdata-lib"),
    "cxxfilt": struct(bin_name = "llvm-cxxfilt", lib = "@llvm-project//llvm:llvm-cxxfilt-lib"),
    "debuginfod-find": struct(bin_name = "llvm-debuginfod-find", lib = "@llvm-project//llvm:llvm-debuginfod-find-lib"),
    "dsymutil": struct(bin_name = "dsymutil", lib = "@llvm-project//llvm:dsymutil-lib"),
    "dwp": struct(bin_name = "llvm-dwp", lib = "@llvm-project//llvm:llvm-dwp-lib"),
    "gsymutil": struct(bin_name = "llvm-gsymutil", lib = "@llvm-project//llvm:llvm-gsymutil-lib"),
    "ifs": struct(bin_name = "llvm-ifs", lib = "@llvm-project//llvm:llvm-ifs-lib"),
    "libtool-darwin": struct(bin_name = "llvm-libtool-darwin", lib = "@llvm-project//llvm:llvm-libtool-darwin-lib"),
    "lipo": struct(bin_name = "llvm-lipo", lib = "@llvm-project//llvm:llvm-lipo-lib"),
    "ml": struct(bin_name = "llvm-ml", lib = "@llvm-project//llvm:llvm-ml-lib"),
    "mt": struct(bin_name = "llvm-mt", lib = "@llvm-project//llvm:llvm-mt-lib"),
    "nm": struct(bin_name = "llvm-nm", lib = "@llvm-project//llvm:llvm-nm-lib"),
    "objcopy": struct(bin_name = "llvm-objcopy", lib = "@llvm-project//llvm:llvm-objcopy-lib"),
    "objdump": struct(bin_name = "llvm-objdump", lib = "@llvm-project//llvm:llvm-objdump-lib"),
    "profdata": struct(bin_name = "llvm-profdata", lib = "@llvm-project//llvm:llvm-profdata-lib"),
    "rc": struct(bin_name = "llvm-rc", lib = "@llvm-project//llvm:llvm-rc-lib"),
    "readobj": struct(bin_name = "llvm-readobj", lib = "@llvm-project//llvm:llvm-readobj-lib"),
    "sancov": struct(bin_name = "sancov", lib = "@llvm-project//llvm:sancov-lib"),
    "size": struct(bin_name = "llvm-size", lib = "@llvm-project//llvm:llvm-size-lib"),
    "symbolizer": struct(bin_name = "llvm-symbolizer", lib = "@llvm-project//llvm:llvm-symbolizer-lib"),
}

# A collection of additional alias names that should be available for the main
# tools. The key is the main tool with the support for these names, followed by
# a list of the aliased names.
#
# Note that we don't track separate binary names for the alias names as those
# are always formed by prepending `llvm-` for the aliases.
LLVM_TOOL_ALIASES = {
    "ar": ["ranlib", "lib", "dlltool"],
    "objcopy": ["bitcode-strip", "install-name-tool", "strip"],
    "objdump": ["otool"],
    "rc": ["windres"],
    "readobj": ["readelf"],
    "symbolizer": ["addr2line"],
}

_DEF_FILE_TEMPLATE = """
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is a generated X-macro header for defining LLVM tools. It does not use
// `#include` guards, and instead is designed to be `#include`ed after the
// x-macro is defined in order for its inclusion to expand to the desired
// output. Macro definitions are cleaned up at the end of this file.
//
// Each X-macro takes four arguments:
// - `Id` is the identifier-shaped PascalCased tool name.
// - `Name` is a string literal of the tool name.
// - `BinName` is a string literal of the binary name of the tool when installed
//    as a stand-alone command line tool.
// - `MainFn` is the function symbol name used to run the tool as-if its `main`.
//
// There are three X-macros available:
// - `CARBON_LLVM_TOOL` is available for every tool.
//   - `CARBON_LLVM_MAIN_TOOL` is available for each tool with a distinct
//     `MainFn` symbol name.
//   - `CARBON_LLVM_ALIAS_TOOL` is available for each tool that is an alias of
//     some other tool. It's `MainFn` will be the alias-target symbol name.
//
// See toolchain/driver/llvm_tools.bzl for more details.

#ifndef CARBON_LLVM_TOOL
#define CARBON_LLVM_TOOL(Id, Name, BinName, MainFn)
#endif

#ifndef CARBON_LLVM_MAIN_TOOL
#define CARBON_LLVM_MAIN_TOOL(Id, Name, BinName, MainFn) \\
  CARBON_LLVM_TOOL(Id, Name, BinName, MainFn)
#endif

#ifndef CARBON_LLVM_ALIAS_TOOL
#define CARBON_LLVM_ALIAS_TOOL(Id, Name, BinName, MainFn) \\
  CARBON_LLVM_TOOL(Id, Name, BinName, MainFn)
#endif

{}

#undef CARBON_LLVM_TOOL
#undef CARBON_LLVM_MAIN_TOOL
#undef CARBON_LLVM_ALIAS_TOOL
"""

_DEF_MACRO_TEMPLATE = """
CARBON_LLVM_{kind}TOOL({id}, "{name}", "{bin_name}", {main_fn})
""".strip()

def _build_def_macro(kind, name, bin_name, main_info):
    id = "".join([w.title() for w in name.split("-")])
    main_fn = main_info.bin_name.replace("-", "_") + "_main"
    return _DEF_MACRO_TEMPLATE.format(
        kind = kind,
        id = id,
        name = name,
        bin_name = bin_name,
        main_fn = main_fn,
    )

def _generate_llvm_tools_def_rule(ctx):
    def_lines = []

    for name, tool_info in LLVM_MAIN_TOOLS.items():
        def_lines.append(_build_def_macro("MAIN_", name, tool_info.bin_name, tool_info))

    for target, aliases in LLVM_TOOL_ALIASES.items():
        tool_info = LLVM_MAIN_TOOLS[target]
        for alias in aliases:
            bin_name = "llvm-" + alias
            def_lines.append(_build_def_macro("ALIAS_", alias, bin_name, tool_info))

    def_file = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(def_file, _DEF_FILE_TEMPLATE.format("\n".join(def_lines)))
    return [DefaultInfo(files = depset([def_file]))]

generate_llvm_tools_def_rule = rule(
    implementation = _generate_llvm_tools_def_rule,
    attrs = {},
)

def generate_llvm_tools_def(name, out, **kwargs):
    """Generates the LLVM tools `.def` file.

    This first generates the `.def` file into the `out` filename, and then
    synthesizes a `cc_library` rule exporting that file in its `textual_hdrs`.

    The `cc_library` rule name is the provided `name` and should be depended on
    by code that includes the generated file. The `kwargs` are expanded into the
    `cc_library` in case other attributes need to be configured there.

    The two-step process is necessary to avoid trying to compile or otherwise
    process the generated file as something other than a textual header.
    """
    generate_llvm_tools_def_rule(name = out)
    cc_library(
        name = name,
        textual_hdrs = [out],
        **kwargs
    )
