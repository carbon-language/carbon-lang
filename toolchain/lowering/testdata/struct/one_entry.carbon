// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/struct/one_entry.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/struct/one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %0 = type { i32 }
// CHECK:STDOUT: %1 = type { i32 }
// CHECK:STDOUT: %2 = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %0, align 8
// CHECK:STDOUT:   %1 = alloca %1, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds %1, ptr %1, i32 0, i32 0
// CHECK:STDOUT:   store i32 4, ptr %2, align 4
// CHECK:STDOUT:   store ptr %1, ptr %0, align 8
// CHECK:STDOUT:   %3 = alloca %2, align 8
// CHECK:STDOUT:   store ptr %0, ptr %3, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {.a: i32} = {.a = 4};
  var y: {.a: i32} = x;
  return 0;
}
