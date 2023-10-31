// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:23: ERROR: Cannot convert from `i32` to `(i32, i32) as type` with `as`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                       ^
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: file "fail_no_conversion.carbon" {
// CHECK:STDOUT:   %.loc10_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_17.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc10_17.4: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc10_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_35: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %n: (i32, i32) = bind_name "n", <error>
// CHECK:STDOUT: }
