// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;
alias NS.a = bool;

let b: NS.a = false;

fn F() -> NS.a {
  return false;
}

// CHECK:STDOUT: --- in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: type = bind_alias a, bool [template = bool]
// CHECK:STDOUT:   %NS.ref.loc10: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %a.ref.loc10: type = name_ref a, %a [template = bool]
// CHECK:STDOUT:   %.loc10: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %b: bool = bind_name b, %.loc10
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %NS.ref.loc12: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:     %a.ref.loc12: type = name_ref a, %a [template = bool]
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   return %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
