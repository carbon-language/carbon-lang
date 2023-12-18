// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var n: i32 = 0;
  &n;
}

// CHECK:STDOUT: ; ModuleID = 'address_of_unused.carbon'
// CHECK:STDOUT: source_filename = "address_of_unused.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %n = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %n, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
