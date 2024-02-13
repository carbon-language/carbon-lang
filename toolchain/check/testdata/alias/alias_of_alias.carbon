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
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a.ref.loc7, .b = %b.ref.loc8, .c = %c.ref.loc9} [template]
// CHECK:STDOUT:   %a.ref.loc7: type = name_ref a, bool [template = bool]
// CHECK:STDOUT:   %a.ref.loc8: type = name_ref a, %a.ref.loc7 [template = bool]
// CHECK:STDOUT:   %b.ref.loc8: type = name_ref b, %a.ref.loc7 [template = bool]
// CHECK:STDOUT:   %b.ref.loc9: type = name_ref b, %b.ref.loc8 [template = bool]
// CHECK:STDOUT:   %c.ref.loc9: type = name_ref c, %b.ref.loc8 [template = bool]
// CHECK:STDOUT:   %c.ref.loc10: type = name_ref c, %c.ref.loc9 [template = bool]
// CHECK:STDOUT:   %.loc10: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %d: bool = bind_name d, %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
