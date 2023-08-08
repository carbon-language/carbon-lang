// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  return;
}

// CHECK:STDOUT: ; ModuleID = 'no_value.carbon'
// CHECK:STDOUT: source_filename = "no_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
