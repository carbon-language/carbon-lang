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

// CHECK:STDOUT: file "or.carbon" {
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
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", file.%F
// CHECK:STDOUT:   %.loc11_11: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %.loc11_11
// CHECK:STDOUT:   %.loc11_14.2: bool = not %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.3: bool = bool_literal true
// CHECK:STDOUT:   if %.loc11_14.2 br !or.rhs else br !or.result(%.loc11_14.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", file.%G
// CHECK:STDOUT:   %.loc11_18: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc11_14.4: bool = value_of_initializer %.loc11_18
// CHECK:STDOUT:   br !or.result(%.loc11_14.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc11_14.5: bool = block_arg !or.result
// CHECK:STDOUT:   return %.loc11_14.5
// CHECK:STDOUT: }
