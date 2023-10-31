// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+3]]:19: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: let n: i32 = 1 as 2;
// CHECK:STDERR:                   ^
let n: i32 = 1 as 2;

// CHECK:STDOUT: file "fail_not_type.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 2
// CHECK:STDOUT:   %n: i32 = bind_name "n", <error>
// CHECK:STDOUT: }
