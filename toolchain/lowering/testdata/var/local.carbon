// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/var/local.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/var/local.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   store i32 1, ptr %var, align 4
// CHECK:STDOUT:   ret ptr %var
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: i32 = 1;
  return x;
}
