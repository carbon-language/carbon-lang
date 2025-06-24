# BUILD file for Carbon C++ Reference and Pointer Interop (Issue #5426)
#
# This demonstrates how C++ reference/pointer support would be built
# and integrated with Carbon's build system.

load("//bazel/cc_rules:defs.bzl", "cc_library")

# C++ implementation library for reference/pointer support
cc_library(
    name = "cpp_reference_support",
    srcs = ["cpp_reference_support.cpp"],
    hdrs = ["cpp_reference_support.h"],
    # This library provides the C++ side of the interop bridge
    visibility = ["//toolchain/check:__subpackages__"],
)

# Future: Carbon test that imports the C++ library
# This would be enabled once issue #5426 is fully implemented
# carbon_test(
#     name = "reference_pointer_basic_test", 
#     srcs = ["reference_pointer_basic.carbon"],
#     deps = [":cpp_reference_support"],
# ) 