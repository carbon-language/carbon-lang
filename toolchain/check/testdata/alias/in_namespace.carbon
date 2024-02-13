// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;
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
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.a = %a} [template]
// CHECK:STDOUT:   %.loc8: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %a: bool = bind_template_name a, %.loc8 [template = constants.%.1]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   %b: bool = bind_name b, %a.ref
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%.loc7 [template = file.%.loc7]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, file.%a [template = constants.%.1]
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
