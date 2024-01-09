// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

// CHECK:STDOUT: --- or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .G = %G, .Or = %Or}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %Or: <function> = fn_decl @Or
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = bool_literal true
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc11_10: <function> = reify_constant %F.ref, file.%F
// CHECK:STDOUT:   %.loc11_11: init bool = call file.%F()
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %.loc11_11
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %.loc11_11, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.3: bool = not %.loc11_14.2
// CHECK:STDOUT:   %.loc11_14.4: bool = bool_literal true
// CHECK:STDOUT:   %.loc11_14.5: bool = reify_constant %.loc11_14.4, %.loc11_14.4
// CHECK:STDOUT:   if %.loc11_14.3 br !or.rhs else br !or.result(%.loc11_14.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G
// CHECK:STDOUT:   %.loc11_17: <function> = reify_constant %G.ref, file.%G
// CHECK:STDOUT:   %.loc11_18: init bool = call file.%G()
// CHECK:STDOUT:   %.loc11_14.6: bool = value_of_initializer %.loc11_18
// CHECK:STDOUT:   %.loc11_14.7: bool = converted %.loc11_18, %.loc11_14.6
// CHECK:STDOUT:   br !or.result(%.loc11_14.7)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc11_14.8: bool = block_arg !or.result
// CHECK:STDOUT:   return %.loc11_14.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
