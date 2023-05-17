// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/struct/empty.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/struct/empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyStructType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var = alloca %EmptyStructType, align 8
// CHECK:STDOUT:   %var1 = alloca %EmptyStructType, align 8
// CHECK:STDOUT:   store ptr %var, ptr %var1, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {} = {};
  var y: {} = x;
  return 0;
}
