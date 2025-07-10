// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
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
inline int GetSizeAsInt() { return GetSize(); }

// --- import_stddef_indirectly.carbon

library "[[@TEST_NAME]]";

import Cpp library "include_stddef.h";

// TODO: `fn CallGetSize() -> Cpp.ptrdiff_t {`
fn CallGetSize() -> i32 {
  // TODO: Call `Cpp.GetSize` directly once we can import the types `long` and `long long`.
  return Cpp.GetSizeAsInt() as i32;
}

// --- fail_todo_import_stddef_directly.carbon

library "[[@TEST_NAME]]";

import Cpp library "stddef.h";

// TODO: Once we can import `unsigned long` / `unsigned long long`, this should work.
// CHECK:STDERR: fail_todo_import_stddef_directly.carbon:[[@LINE+7]]:8: error: semantics TODO: `Unsupported: Type declaration: size_t` [SemanticsTodo]
// CHECK:STDERR: var n: Cpp.size_t = 42;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR: fail_todo_import_stddef_directly.carbon:[[@LINE+4]]:8: note: in `Cpp` name lookup for `size_t` [InCppNameLookup]
// CHECK:STDERR: var n: Cpp.size_t = 42;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR:
var n: Cpp.size_t = 42;
