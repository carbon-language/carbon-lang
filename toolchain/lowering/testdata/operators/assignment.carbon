// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var a: i32 = 12;
  a = 9;
  var b: (i32, i32);
  b = (1, 2);
}

// CHECK:STDOUT: ; ModuleID = 'assignment.carbon'
// CHECK:STDOUT: source_filename = "assignment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type { {}, {} }
// CHECK:STDOUT: %TupleLiteralType.0 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   store i32 12, ptr %var, align 4
// CHECK:STDOUT:   store i32 -7, ptr %var, align 4
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store {} zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %2 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue, i32 0, i32 1
// CHECK:STDOUT:   store {} zeroinitializer, ptr %2, align 1
// CHECK:STDOUT:   %var1 = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %TupleLiteralValue2 = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %3 = getelementptr inbounds %TupleLiteralType.0, ptr %TupleLiteralValue2, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds %TupleLiteralType.0, ptr %TupleLiteralValue2, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %4, align 4
// CHECK:STDOUT:   %5 = load %TupleLiteralType.0, ptr %TupleLiteralValue2, align 4
// CHECK:STDOUT:   store %TupleLiteralType.0 %5, ptr %var1, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
