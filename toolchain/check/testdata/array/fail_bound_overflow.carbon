// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_bound_overflow.carbon" {

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+3]]:32: Invalid array expression.
// CHECK:STDERR: var a: [1; 39999999999999999993];
// CHECK:STDERR:                                ^
var a: [1; 39999999999999999993];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_12: i32 = int_literal 39999999999999999993
// CHECK:STDOUT:   %a: ref <error> = var "a"
// CHECK:STDOUT: }
