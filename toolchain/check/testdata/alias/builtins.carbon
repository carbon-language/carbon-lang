// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = i32;
let a_test: a = 0;

alias b = bool;
let b_test: b = false;

// CHECK:STDOUT: --- builtins.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: type = bind_alias a, i32 [template = i32]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = i32]
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   %a_test: i32 = bind_name a_test, %.loc8
// CHECK:STDOUT:   %b: type = bind_alias b, bool [template = bool]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %.loc11: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   %b_test: bool = bind_name b_test, %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
