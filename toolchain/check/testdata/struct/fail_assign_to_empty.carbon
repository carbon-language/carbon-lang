// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:20: ERROR: Cannot initialize struct of 0 element(s) from struct with 1 element(s).
// CHECK:STDERR: var x: {} = {.a = 1};
// CHECK:STDERR:                    ^
var x: {} = {.a = 1};

// CHECK:STDOUT: file "fail_assign_to_empty.carbon" {
// CHECK:STDOUT:   %.loc10_9.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_9.2: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_9.3: type = tuple_type ()
// CHECK:STDOUT:   %x: ref {} = var "x"
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_20.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc10_20.2: {.a: i32} = struct_literal (%.loc10_19)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
