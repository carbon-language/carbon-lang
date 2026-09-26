# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_library", "py_test")

# The TextMate tokenizer. See tmlanguage.py for why `re` can replace Oniguruma
# here.
py_library(
    name = "tmlanguage",
    testonly = True,
    srcs = ["tmlanguage.py"],
    imports = ["."],
)

py_library(
    name = "render_sample",
    testonly = True,
    srcs = ["render_sample.py"],
    imports = ["."],
    deps = [":tmlanguage"],
)

# Structural checks with no goldens to regenerate. They fail when the grammar
# stops covering a `token_kind.def` spelling, or leaves a `begin`/`end` region
# unclosed at end of file.
py_test(
    name = "grammar_test",
    size = "small",
    srcs = ["grammar_test.py"],
    data = glob(["Samples/*.carbon"]) + [
        "//examples:carbon_files",
        "//toolchain/lex:token_kind.def",
        "//utils/vscode:carbon.tmLanguage.json",
    ],
    deps = [":tmlanguage"],
)

# conformance_test.py has no target here: it needs node. See README.md.

# The scope of every token in the samples, so a diff shows which construct
# changed. Regenerate with:
#   ./utils/textmate/golden_test.py --update
py_test(
    name = "golden_test",
    size = "small",
    srcs = ["golden_test.py"],
    data = glob([
        "Samples/*.carbon",
        "Samples/*.svg",
        "testdata/highlight/*.scopes",
    ]) + ["//utils/vscode:carbon.tmLanguage.json"],
    deps = [
        ":render_sample",
        ":tmlanguage",
    ],
)
