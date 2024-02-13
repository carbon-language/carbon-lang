// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = false;
let a_test: bool = a;
let not_a_test: bool = not a;

alias b = bool;
let b_test: b = false;

let both_test: b = a;
let not_both_test: b = not a;

// CHECK:STDOUT: --- builtins.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %a: bool = bind_template_name a, %.loc7 [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc8: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %a_test: bool = bind_name a_test, %a.ref.loc8
// CHECK:STDOUT:   %a.ref.loc9: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %.loc9: bool = not %a.ref.loc9 [template = constants.%.2]
// CHECK:STDOUT:   %not_a_test: bool = bind_name not_a_test, %.loc9
// CHECK:STDOUT:   %b: type = bind_template_name b, bool [template = bool]
// CHECK:STDOUT:   %b.ref.loc12: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %.loc12: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %b_test: bool = bind_name b_test, %.loc12
// CHECK:STDOUT:   %b.ref.loc14: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %a.ref.loc14: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %both_test: bool = bind_name both_test, %a.ref.loc14
// CHECK:STDOUT:   %b.ref.loc15: type = name_ref b, %b [template = bool]
// CHECK:STDOUT:   %a.ref.loc15: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %.loc15: bool = not %a.ref.loc15 [template = constants.%.2]
// CHECK:STDOUT:   %not_both_test: bool = bind_name not_both_test, %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
