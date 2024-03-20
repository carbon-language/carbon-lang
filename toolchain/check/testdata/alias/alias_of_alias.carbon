// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = bool;
alias b = a;
alias c = b;
let d: c = false;

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: type = bind_alias a, bool [template = bool]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = bool]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [template = bool]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [template = bool]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = bool]
// CHECK:STDOUT:   %.loc10: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %d: bool = bind_name d, %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
