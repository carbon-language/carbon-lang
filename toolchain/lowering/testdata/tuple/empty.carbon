// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty.carbon'
// CHECK:STDOUT: source_filename = "empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %TupleLiteralValue1 = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %1 = load %TupleLiteralType, ptr %TupleLiteralValue1, align 1
// CHECK:STDOUT:   store %TupleLiteralType %1, ptr %var, align 1
// CHECK:STDOUT:   %TupleLiteralValue2 = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %var3 = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %2 = load %TupleLiteralType, ptr %var, align 1
// CHECK:STDOUT:   store %TupleLiteralType %2, ptr %var3, align 1
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: () = ();
  var y: () = x;
  return 0;
}
