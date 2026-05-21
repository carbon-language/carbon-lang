// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_keyword_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_keyword_name.carbon

// CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+4]]:13: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: fn F() -> {.class: i32};
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
fn F() -> {.class: i32};

// CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+4]]:13: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: fn G() -> {.return: i32} {
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR:
fn G() -> {.return: i32} {
  // CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+4]]:12: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   return {.return = 5};
  // CHECK:STDERR:            ^~~~~~
  // CHECK:STDERR:
  return {.return = 5};
};
