// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: builtins.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias a = false;
// CHECK:STDERR:           ^~~~~
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a.ref.loc10, .b = %b.ref.loc14} [template]
// CHECK:STDOUT:   %.loc10: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc10: <error> = name_ref a, %.loc10 [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc11: <error> = name_ref a, %a.ref.loc10 [template = constants.%.1]
// CHECK:STDOUT:   %a_test: bool = bind_name a_test, <error>
// CHECK:STDOUT:   %a.ref.loc12: <error> = name_ref a, %a.ref.loc10 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   %not_a_test: bool = bind_name not_a_test, <error>
// CHECK:STDOUT:   %b.ref.loc14: type = name_ref b, bool [template = bool]
// CHECK:STDOUT:   %b.ref.loc15: type = name_ref b, %b.ref.loc14 [template = bool]
// CHECK:STDOUT:   %.loc15: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %b_test: bool = bind_name b_test, %.loc15
// CHECK:STDOUT:   %b.ref.loc17: type = name_ref b, %b.ref.loc14 [template = bool]
// CHECK:STDOUT:   %a.ref.loc17: <error> = name_ref a, %a.ref.loc10 [template = constants.%.1]
// CHECK:STDOUT:   %both_test: bool = bind_name both_test, <error>
// CHECK:STDOUT:   %b.ref.loc18: type = name_ref b, %b.ref.loc14 [template = bool]
// CHECK:STDOUT:   %a.ref.loc18: <error> = name_ref a, %a.ref.loc10 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   %not_both_test: bool = bind_name not_both_test, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
