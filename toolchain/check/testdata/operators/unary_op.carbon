// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "unary_op.carbon" {

fn Not(b: bool) -> bool {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Not
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b: bool) -> bool {
// CHECK:STDOUT: !entry:
  return not b;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: bool = not %b
  // CHECK:STDOUT:   return %.loc[[@LINE-2]]
  // CHECK:STDOUT: }
}
