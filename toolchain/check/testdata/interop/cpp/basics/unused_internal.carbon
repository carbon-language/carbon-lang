// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-Wunused
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/unused_internal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/unused_internal.carbon

// --- unused.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
// CHECK:STDERR: unused.carbon:[[@LINE+4]]:12: warning: unused variable 'n' [CppInteropParseWarning]
// CHECK:STDERR:     9 | static int n = 0;
// CHECK:STDERR:       |            ^
// CHECK:STDERR:
static int n = 0;
// CHECK:STDERR: unused.carbon:[[@LINE+4]]:13: warning: unused function 'f' [CppInteropParseWarning]
// CHECK:STDERR:    14 | static void f() {}
// CHECK:STDERR:       |             ^
// CHECK:STDERR:
static void f() {}
''';

// --- not_unused.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
static int n = 0;
static void f() {}
''';

fn F() -> i32 {
  Cpp.f();
  return Cpp.n;
}
