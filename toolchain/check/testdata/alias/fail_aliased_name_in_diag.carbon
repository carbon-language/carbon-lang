// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias b = bool;
// CHECK:STDERR: fail_aliased_name_in_diag.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `i32` to `bool`.
// CHECK:STDERR: let c: b = 2;
// CHECK:STDERR: ^~~~~~~~~~~~~
let c: b = 2;

// CHECK:STDOUT: --- fail_aliased_name_in_diag.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: type = bind_alias b, bool [template = bool]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %.loc11: i32 = int_literal 2 [template = constants.%.1]
// CHECK:STDOUT:   %c: bool = bind_name c, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
