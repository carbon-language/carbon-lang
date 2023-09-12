// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Not(b: bool) -> bool {
  return not b;
}

// CHECK:STDOUT: file "unary_op.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Not
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = not %b
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
