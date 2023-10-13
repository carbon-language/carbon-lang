// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn If(b: bool) -> i32 {
  if (b) {
    return 1;
  } else {
    return 2;
  }
  // Missing return here is OK.
}

// CHECK:STDOUT: file "unreachable_fallthrough.carbon" {
// CHECK:STDOUT:   %If: <function> = fn_decl @If
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_reference "b", %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc11
// CHECK:STDOUT: }
