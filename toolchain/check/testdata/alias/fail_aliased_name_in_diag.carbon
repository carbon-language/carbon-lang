// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let a: i32 = 1;
alias b = a;
// CHECK:STDERR: fail_aliased_name_in_diag.carbon:[[@LINE+3]]:8: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: let c: b = 2;
// CHECK:STDERR:        ^
let c: b = 2;

// CHECK:STDOUT: --- fail_aliased_name_in_diag.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.b = %b} [template]
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc7
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b: i32 = bind_alias b, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc12: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
