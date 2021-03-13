# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

filegroup(
    name = "website_content",
    srcs = [
        "licenses.md",
        "//:website_content",
        "//docs:website_content",
        "//proposals:website_content",
    ] + glob(["theme/**"]),
)

filegroup(
    name = "configs",
    srcs = [
        "Gemfile",
        "Gemfile.lock",
        "_config.yml",
    ],
)

py_binary(
    name = "gen_sidebar",
    srcs = ["gen_sidebar.py"],
    python_version = "PY3",
    deps = ["//proposals/scripts:proposals"],
)

py_test(
    name = "gen_sidebar_test",
    srcs = ["gen_sidebar_test.py"],
    data = [":website_content"],
    deps = [":gen_sidebar"],
)

genrule(
    name = "gen_sidebar_html",
    srcs = [":website_content"],
    outs = ["gen_sidebar.html"],
    cmd = "$(location :gen_sidebar) > $@",
    tools = [":gen_sidebar"],
)

genrule(
    name = "jekyll_site_tgz",
    srcs = [
        ":configs",
        ":website_content",
        ":gen_sidebar.html",
    ],
    outs = ["jekyll_site.tgz"],
    cmd = "$(location jekyll_site.sh) $(location :gen_sidebar.html) $@",
    tags = ["manual"],
    tools = ["jekyll_site.sh"],
)

sh_binary(
    name = "serve",
    srcs = ["serve.sh"],
    data = [
        ":configs",
        ":jekyll_site.tgz",
    ],
    tags = ["manual"],
)

sh_binary(
    name = "publish",
    srcs = ["publish.sh"],
    data = [":jekyll_site.tgz"],
    tags = ["manual"],
)
