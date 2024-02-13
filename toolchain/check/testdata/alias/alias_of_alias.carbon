// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = false;
alias b = not a;
alias c = not b;
let d: bool = c;

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b, .c = %c} [template]
// CHECK:STDOUT:   %.loc7: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %a: bool = bind_template_name a, %.loc7 [template = constants.%.1]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %.loc8: bool = not %a.ref [template = constants.%.2]
// CHECK:STDOUT:   %b: bool = bind_template_name b, %.loc8 [template = constants.%.2]
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b [template = constants.%.2]
// CHECK:STDOUT:   %.loc9: bool = not %b.ref [template = constants.%.1]
// CHECK:STDOUT:   %c: bool = bind_template_name c, %.loc9 [template = constants.%.1]
// CHECK:STDOUT:   %c.ref: bool = name_ref c, %c [template = constants.%.1]
// CHECK:STDOUT:   %d: bool = bind_name d, %c.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
