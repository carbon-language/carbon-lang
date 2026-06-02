// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/var/export/var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/var/export/var.carbon

// --- fail_todo_var.carbon
library "[[@TEST_NAME]]";

import Cpp;

var x: i32 = 123;

inline Cpp '''
int Get() {
  // CHECK:STDERR: fail_todo_var.carbon:[[@LINE+4]]:18: error: no member named 'x' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    13 |   return Carbon::x;
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  return Carbon::x;
}
''';

// --- fail_var_unsupported_type.carbon
library "[[@TEST_NAME]]";

import Cpp;

var x: () = ();

inline Cpp '''
void F() {
  // CHECK:STDERR: fail_var_unsupported_type.carbon:[[@LINE+4]]:11: error: no member named 'x' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    13 |   Carbon::x;
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  Carbon::x;
}
''';
