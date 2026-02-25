// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_method_redefinition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_method_redefinition.carbon

class Class {
  fn F() {}
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE+7]]:3: error: redefinition of `fn F` [RedeclRedef]
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE-4]]:3: note: previously defined here [RedeclPrevDef]
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  fn F() {}
}
