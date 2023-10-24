// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_too_few_values.carbon:[[@LINE+3]]:36: ERROR: Cannot initialize struct of 2 element(s) from struct with 1 element(s).
// CHECK:STDERR: var x: {.a: i32, .b: i32} = {.a = 1};
// CHECK:STDERR:                                    ^
var x: {.a: i32, .b: i32} = {.a = 1};

// CHECK:STDOUT: file "fail_too_few_values.carbon" {
// CHECK:STDOUT:   %.loc10_25.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_25.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %x.loc10_5.1: ref {.a: i32, .b: i32} = var "x"
// CHECK:STDOUT:   %x.loc10_5.2: ref {.a: i32, .b: i32} = bind_name "x", %x.loc10_5.1
// CHECK:STDOUT:   %.loc10_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_36.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc10_36.2: {.a: i32} = struct_literal (%.loc10_35)
// CHECK:STDOUT:   assign %x.loc10_5.1, <error>
// CHECK:STDOUT: }
