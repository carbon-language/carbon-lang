// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_method_define.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_method_define.carbon

base class B {
  fn F();

  class C {
    fn F();
  }
}

class D {
  extend base: B;
}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+4]]:6: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn D.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn D.F() {}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+4]]:6: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn D.C.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn D.C.F() {}
