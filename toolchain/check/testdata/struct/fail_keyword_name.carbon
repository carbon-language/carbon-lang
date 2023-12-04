// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+6]]:13: ERROR: Expected identifier after `.`.
// CHECK:STDERR: fn F() -> {.class: i32};
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+3]]:13: ERROR: Semantics TODO: `Error recovery from keyword name.`.
// CHECK:STDERR: fn F() -> {.class: i32};
// CHECK:STDERR:             ^~~~~
fn F() -> {.class: i32};

// CHECK:STDERR: fail_keyword_name.carbon:[[@LINE+3]]:19: ERROR: Expected identifier after `.`.
// CHECK:STDERR: fn G() { return {.return = 5}; };
// CHECK:STDERR:                   ^~~~~~
fn G() { return {.return = 5}; };

// CHECK:STDOUT: --- fail_keyword_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
