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

class PublicClass {
  static var x: i32 = 123;
}

inline Cpp '''
int PublicStatic() {
  return Carbon::PublicClass::x;
}
''';

// --- fail_static_private.carbon
library "[[@TEST_NAME]]";
import Cpp;

class PrivateClass {
  private static var x: i32 = 123;
}

inline Cpp '''
int PrivateStatic() {
  // CHECK:STDERR: fail_static_private.carbon:[[@LINE+3]]:32: error: 'x' is a private member of 'Carbon::PrivateClass' [CppInteropParseError]
  // CHECK:STDERR: fail_static_private.carbon:[[@LINE-6]]:23: note: implicitly declared private here [CppInteropParseNote]
  // CHECK:STDERR:
  return Carbon::PrivateClass::x;
}
''';
