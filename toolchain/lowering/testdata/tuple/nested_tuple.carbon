// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  ((i32,), );
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'nested_tuple.carbon'
// CHECK:STDOUT: source_filename = "nested_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type { {} }
// CHECK:STDOUT: %TupleLiteralType.0 = type { %TupleLiteralType }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds %TupleLiteralType, ptr %TupleLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store {} zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %TupleLiteralValue1 = alloca %TupleLiteralType.0, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds %TupleLiteralType.0, ptr %TupleLiteralValue1, i32 0, i32 0
// CHECK:STDOUT:   store ptr %TupleLiteralValue, ptr %2, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
