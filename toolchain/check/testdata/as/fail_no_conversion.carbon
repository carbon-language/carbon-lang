// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:21: ERROR: Cannot convert from `i32` to `(i32, i32)` with `as`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {}
// CHECK:STDOUT:   %.loc10_17: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.2: type = converted %.loc10_17, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_35: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.3: type = converted %.loc10_35, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %n: (i32, i32) = bind_name n, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
