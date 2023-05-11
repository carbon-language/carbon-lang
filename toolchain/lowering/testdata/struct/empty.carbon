// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/struct/empty.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/struct/empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca [0 x i8], align 1
// CHECK:STDOUT:   %1 = alloca [0 x i8], align 1
// CHECK:STDOUT:   store ptr %0, ptr %1, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {} = {};
  var y: {} = x;
  return 0;
}
