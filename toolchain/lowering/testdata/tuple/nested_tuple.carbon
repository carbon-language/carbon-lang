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
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %tuple = alloca { %type }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { %type }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %tuple1 = alloca { { %type } }, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { { %type } }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple, ptr %2, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
