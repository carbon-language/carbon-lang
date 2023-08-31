// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool {
  return false;
}

fn T() -> bool {
  return true;
}

// CHECK:STDOUT: ; ModuleID = 'false_true.carbon'
// CHECK:STDOUT: source_filename = "false_true.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() {
// CHECK:STDOUT:   ret i1 false
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @T() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
