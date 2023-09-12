// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_too_few_values.carbon" {

// CHECK:STDERR: fail_too_few_values.carbon:[[@LINE+3]]:37: Cannot implicitly convert from `{.a: i32}` to `{.a: i32, .b: i32}`.
// CHECK:STDERR: var x: {.a: i32, .b: i32} = {.a = 1};
// CHECK:STDERR:                                     ^
var x: {.a: i32, .b: i32} = {.a = 1};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_25: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: i32, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-4]]_33: i32 = stub_reference %.loc[[@LINE-4]]_35
// CHECK:STDOUT:   %.loc[[@LINE-5]]_36.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc[[@LINE-6]]_36.2: {.a: i32} = struct_literal (%.loc[[@LINE-6]]_33)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
