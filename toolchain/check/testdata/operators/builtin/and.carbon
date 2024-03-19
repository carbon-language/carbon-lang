// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

// CHECK:STDOUT: --- and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:     .And = %And
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %return.var.loc7: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And: <function> = fn_decl @And [template] {
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc11_11.1: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.2: bool = converted %.loc11_11.1, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.2: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   if %.loc11_11.2 br !and.rhs else br !and.result(%.loc11_14.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G [template = file.%G]
// CHECK:STDOUT:   %.loc11_19.1: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc11_14.3: bool = value_of_initializer %.loc11_19.1
// CHECK:STDOUT:   %.loc11_19.2: bool = converted %.loc11_19.1, %.loc11_14.3
// CHECK:STDOUT:   br !and.result(%.loc11_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_14.4: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc11_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
