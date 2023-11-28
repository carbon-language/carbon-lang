// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:21: ERROR: Cannot convert from `i32` to `(i32, i32) as type` with `as`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_17.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc10_17.3: type = ptr_type (i32, i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_no_conversion.carbon" {
// CHECK:STDOUT:   %.loc10_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_17.2: type = converted %.loc10_17.1, constants.%.loc10_17.2
// CHECK:STDOUT:   %.loc10_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_35: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_17.3: type = converted %.loc10_35, constants.%.loc10_17.2
// CHECK:STDOUT:   %n: (i32, i32) = bind_name n, <error>
// CHECK:STDOUT: }
