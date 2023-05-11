// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/struct/member_access.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/struct/member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %0 = type { double, i32 }
// CHECK:STDOUT: %1 = type { double, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %0, align 8
// CHECK:STDOUT:   %1 = alloca %1, align 8
// CHECK:STDOUT:   store ptr %1, ptr %0, align 8
// CHECK:STDOUT:   %2 = alloca i32, align 4
// CHECK:STDOUT:   %3 = getelementptr inbounds %0, ptr %0, i32 0, i32 1
// CHECK:STDOUT:   store ptr %3, ptr %2, align 8
// CHECK:STDOUT:   %4 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %2, ptr %4, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}
