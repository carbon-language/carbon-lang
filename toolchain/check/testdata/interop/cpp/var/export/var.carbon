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

// --- var.carbon
library "[[@TEST_NAME]]";

import Cpp;

var x: i32 = 123;

inline Cpp '''
int Get() {
  return Carbon::x;
}
''';

// --- fail_var_unsupported_type.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_var_unsupported_type.carbon:[[@LINE+2]]:5-21: error: semantics TODO: `failed to map Carbon type to C++` [SemanticsTodo]
// CHECK:STDERR:
var unsupported_x: () = ();

inline Cpp '''
void F() {
  // CHECK:STDERR: fail_var_unsupported_type.carbon:[[@LINE+2]]:11-23: error: no member named 'unsupported_x' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::unsupported_x;
}
''';
