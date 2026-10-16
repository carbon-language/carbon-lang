// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_base_misplaced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_base_misplaced.carbon

base class B {}

// CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+2]]:1-15: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
// CHECK:STDERR:
extend base: B;

fn F() {
  // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+3]]:3-17: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE-2]]:1-8: note: appears in this declaration [ClassSpecificDeclEnclosingScope]
  // CHECK:STDERR:
  extend base: B;
}

class C {
  fn F() {
    // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+3]]:5-19: error: `base` declaration outside class [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE-2]]:3-10: note: appears in this declaration [ClassSpecificDeclEnclosingScope]
    // CHECK:STDERR:
    extend base: B;
  }
}
