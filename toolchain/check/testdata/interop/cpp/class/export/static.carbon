// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/static.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/static.carbon

// --- static.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  static var x: i32 = 123;
}

inline Cpp '''
int F() {
  return Carbon::C::x;
}
''';

// --- fail_static_private.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  private static var x: i32 = 123;
}

inline Cpp '''
int F() {
  // CHECK:STDERR: fail_static_private.carbon:[[@LINE+7]]:21: error: 'x' is a private member of 'Carbon::C' [CppInteropParseError]
  // CHECK:STDERR:    17 |   return Carbon::C::x;
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR: fail_static_private.carbon:[[@LINE-8]]:23: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:     5 |   private static var x: i32 = 123;
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR:
  return Carbon::C::x;
}
''';
