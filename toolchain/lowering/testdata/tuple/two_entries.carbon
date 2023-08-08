// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: (i32, i32) = (12, 7);
  var y: (i32, i32) = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type { {}, {} }
// CHECK:STDOUT: %TupleLiteralType.0 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store {} zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %2 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue, i32 0, i32 1
// CHECK:STDOUT:   store {} zeroinitializer, ptr %2, align 1
// CHECK:STDOUT:   %var = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %TupleLiteralValue1 = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %3 = getelementptr inbounds %TupleLiteralType.0, ptr %TupleLiteralValue1, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds %TupleLiteralType.0, ptr %TupleLiteralValue1, i32 0, i32 1
// CHECK:STDOUT:   store i32 7, ptr %4, align 4
// CHECK:STDOUT:   %5 = load %TupleLiteralType.0, ptr %TupleLiteralValue1, align 4
// CHECK:STDOUT:   store %TupleLiteralType.0 %5, ptr %var, align 4
// CHECK:STDOUT:   %TupleLiteralValue2 = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %6 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue2, i32 0, i32 0
// CHECK:STDOUT:   store {} zeroinitializer, ptr %6, align 1
// CHECK:STDOUT:   %7 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue2, i32 0, i32 1
// CHECK:STDOUT:   store {} zeroinitializer, ptr %7, align 1
// CHECK:STDOUT:   %var3 = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %8 = load %TupleLiteralType.0, ptr %var, align 4
// CHECK:STDOUT:   store %TupleLiteralType.0 %8, ptr %var3, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
