// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_scope.carbon

class Class {
  fn F() -> () {
    return ();
  }
}

fn G() -> () {
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+4]]:10: error: name `F` not found [NameNotFound]
  // CHECK:STDERR:   return F();
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return F();
}
