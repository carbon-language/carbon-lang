// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "code_after_return_value.carbon" {

fn F(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
  return 0;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 0
  // CHECK:STDOUT:   return %.loc[[@LINE-2]]
  // CHECK:STDOUT: }

  // Exercise various forms of control flow to check that we still consider the
  // end of the function unreachable after them.
  var n: i32 = if b then 1 else 2;
  if (b and (b or b)) { }
  if (b) { } else { }

  // Unreachable, no error on missing return.
}
