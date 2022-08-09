# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def generate_md_tests(srcs, tool):
    """Macro to generate `lit` tests from Markdown fenced code blocks.

    Args:
      srcs: List of Markdown files to analyze.
      tool: Executable target that generates Carbon testcases.
    """
    tests = []
    testcfg = "//explorer/testdata:lit.cfg.py"
    for src in srcs:
        basepath = src.split(".")[0]
        tarpath = basepath + ".tar"
        genname = "generate_md_testdata_" + basepath.replace("/", "_")
        native.genrule(
            name = genname,
            srcs = [src, testcfg],
            outs = [tarpath],
            tools = [tool],
            cmd = "tarpath=$(location " + tarpath + ") \
                && $(location " + tool + ") --input=$(location " + src + ") --output=$${tarpath%.*} \
                && cp $(location " + testcfg + ") . \
                && tar cf \"$@\" $${tarpath%.*} lit.cfg.py --remove-files",
        )
        test_data = [
            "//explorer",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:not",
            "@llvm-project//llvm:lit",
            testcfg,
            tarpath,
        ]
        test_name = basepath + ".test"
        native.py_test(
            name = test_name,
            srcs = ["//bazel/testing:lit_test.py"],
            main = "//bazel/testing:lit_test.py",
            data = test_data,
            args = ["--tarball_path=$(location " + tarpath + ")", "--"],
        )
        tests.append(test_name)
    native.test_suite(
        name = "doctests",
        tests = tests,
    )
