// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+3]]:32: ERROR: Invalid array expression.
// CHECK:STDERR: var a: [1; 39999999999999999993];
// CHECK:STDERR:                                ^
var a: [1; 39999999999999999993];

// CHECK:STDOUT: file "fail_bound_overflow.carbon" {
// CHECK:STDOUT:   %.loc10_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 39999999999999999993
// CHECK:STDOUT:   %a: ref <error> = var "a"
// CHECK:STDOUT: }
