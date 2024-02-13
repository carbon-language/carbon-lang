// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;
// CHECK:STDERR: in_namespace.carbon:[[@LINE+3]]:14: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias NS.a = false;
// CHECK:STDERR:              ^~~~~
alias NS.a = false;

let b: bool = NS.a;

fn F() -> bool {
  return NS.a;
}

// CHECK:STDOUT: --- in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc7, .F = %F} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.a = %a.ref.loc11} [template]
// CHECK:STDOUT:   %.loc11: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc11: <error> = name_ref a, %.loc11 [template = constants.%.1]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %a.ref.loc13: <error> = name_ref a, %a.ref.loc11 [template = constants.%.1]
// CHECK:STDOUT:   %b: bool = bind_name b, <error>
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%.loc7 [template = file.%.loc7]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, file.%a.ref.loc11 [template = constants.%.1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
