// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/include_paths.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/include_paths.carbon

// --- include_stddef.h

// Check that include paths are properly set up when running Clang.

// Clang provides <stddef.h> as a builtin header.
#include <stddef.h>

ptrdiff_t GetSize();

// --- import_stddef_indirectly.carbon

library "[[@TEST_NAME]]";

import Cpp library "include_stddef.h";

fn CallGetSize() -> Cpp.ptrdiff_t {
  return Cpp.GetSize();
}

// --- import_stddef_directly.carbon

library "[[@TEST_NAME]]";

import Cpp library "stddef.h";

var n: Cpp.size_t = 42;
