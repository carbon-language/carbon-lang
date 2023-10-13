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

// CHECK:STDOUT: file "and.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %And: <function> = fn_decl @And
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
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc11_11.1: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc11_11.2: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc11_11.3: ref bool = temporary %.loc11_11.2, %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.4: bool = bind_value %.loc11_11.3
// CHECK:STDOUT:   %.loc11_14.1: bool = bool_literal false
// CHECK:STDOUT:   if %.loc11_11.4 br !and.rhs else br !and.result(%.loc11_14.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", package.%G
// CHECK:STDOUT:   %.loc11_19.1: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc11_19.2: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc11_19.3: ref bool = temporary %.loc11_19.2, %.loc11_19.1
// CHECK:STDOUT:   %.loc11_19.4: bool = bind_value %.loc11_19.3
// CHECK:STDOUT:   br !and.result(%.loc11_19.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_14.2: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc11_14.2
// CHECK:STDOUT: }
