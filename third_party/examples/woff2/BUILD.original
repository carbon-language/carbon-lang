# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Rules are adapted from CMakeLists.txt.

package(
    default_visibility = ["//visibility:public"],
)

# Common part used by decoder and encoder
cc_library(
    name = "woff2common",
    srcs = [
        "src/table_tags.cc",
        "src/variable_length.cc",
        "src/woff2_common.cc",
    ],
    hdrs = [
        "src/buffer.h",
        "src/file.h",
        "src/port.h",
        "src/round.h",
        "src/store_bytes.h",
        "src/table_tags.h",
        "src/variable_length.h",
        "src/woff2_common.h",
    ],
    strip_include_prefix = "src/",
)

# WOFF2 Decoder
cc_library(
    name = "woff2dec",
    srcs = [
        "src/woff2_dec.cc",
        "src/woff2_out.cc",
    ],
    hdrs = [
        "include/woff2/decode.h",
        "include/woff2/output.h",
    ],
    copts = [
        "-Wno-unused-variable",
        "-Wno-unused-const-variable",
    ],
    strip_include_prefix = "include/",
    deps = [
        ":woff2common",
        "@brotli//:brotlidec",
    ],
)

cc_binary(
    name = "woff2_decompress",
    srcs = ["src/woff2_decompress.cc"],
    deps = [":woff2dec"],
)

# WOFF2 Encoder
cc_library(
    name = "woff2enc",
    srcs = [
        "src/font.cc",
        "src/font.h",
        "src/glyph.cc",
        "src/glyph.h",
        "src/normalize.cc",
        "src/normalize.h",
        "src/transform.cc",
        "src/transform.h",
        "src/woff2_enc.cc",
    ],
    hdrs = [
        "include/woff2/encode.h",
    ],
    copts = [
        "-Wno-unused-variable",
        "-Wno-unused-const-variable",
        "-Wno-sign-compare",
    ],
    strip_include_prefix = "include/",
    deps = [
        ":woff2common",
        "@brotli//:brotlienc",
    ],
)

cc_binary(
    name = "woff2_compress",
    srcs = ["src/woff2_compress.cc"],
    deps = [":woff2enc"],
)

# WOFF2 info
cc_binary(
    name = "woff2_info",
    srcs = [
        "src/font.h",
        "src/woff2_info.cc",
    ],
    copts = ["-Wno-sign-compare"],
    deps = [":woff2common"],
)
