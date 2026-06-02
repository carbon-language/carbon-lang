// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/export.carbon

// --- var_export.carbon
library "[[@TEST_NAME]]";

import Cpp;

var x: i32;

inline Cpp '''
int Get() {
  // CHECK:STDERR: var_export.carbon:[[@LINE+4]]:18: error: no member named 'x' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    13 |   return Carbon::x;
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  return Carbon::x;
}
''';

fn Run() -> i32 {
  return Cpp.Get();
}

