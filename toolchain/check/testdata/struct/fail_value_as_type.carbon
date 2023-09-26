// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+3]]:15: ERROR: Cannot implicitly convert from `{.a: i32}` to `type`.
// CHECK:STDERR: var x: {.a = 1};
// CHECK:STDERR:               ^
var x: {.a = 1};

// CHECK:STDOUT: file "fail_value_as_type.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_15.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc10_15.2: {.a: i32} = struct_literal (%.loc10_14)
// CHECK:STDOUT:   %x: ref <error> = var "x"
// CHECK:STDOUT: }
