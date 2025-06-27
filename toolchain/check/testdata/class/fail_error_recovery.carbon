// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_error_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_error_recovery.carbon

// --- fail_virtual_fn_in_invalid_context.carbon

// CHECK:STDERR: fail_virtual_fn_in_invalid_context.carbon:[[@LINE+4]]:10: error: name `error_not_found` not found [NameNotFound]
// CHECK:STDERR: fn F(N:! error_not_found) {
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(N:! error_not_found) {
  base class C {
    virtual fn Foo[self: Self]() {}
  }

  base class D {
    extend base: C;
  }
}
