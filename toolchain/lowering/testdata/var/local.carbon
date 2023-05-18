// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'local.carbon'
// CHECK:STDOUT: source_filename = "local.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca i32, align 4
// CHECK:STDOUT:   store i32 1, ptr %0, align 4
// CHECK:STDOUT:   ret ptr %0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: i32 = 1;
  return x;
}
