// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}

fn Main() {
  return;
  F();
}

// CHECK:STDOUT: ; ModuleID = 'code_after_return.carbon'
// CHECK:STDOUT: source_filename = "code_after_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
