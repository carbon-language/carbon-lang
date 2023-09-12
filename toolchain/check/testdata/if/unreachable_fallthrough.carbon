// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "unreachable_fallthrough.carbon" {

fn If(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @If
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
  if (b) {
    return 1;
    // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 1
    // CHECK:STDOUT:   return %.loc[[@LINE-2]]
    // CHECK:STDOUT:
    // CHECK:STDOUT: !if.else:
  } else {
    return 2;
    // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 2
    // CHECK:STDOUT:   return %.loc[[@LINE-2]]
    // CHECK:STDOUT: }
  }
  // Missing return here is OK.
}
