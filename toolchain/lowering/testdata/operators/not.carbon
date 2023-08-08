// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Not(b: bool) -> bool {
  return not b;
}

// CHECK:STDOUT: ; ModuleID = 'not.carbon'
// CHECK:STDOUT: source_filename = "not.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Not(i1 %b) {
// CHECK:STDOUT:   %1 = xor i1 %b, true
// CHECK:STDOUT:   ret i1 %1
// CHECK:STDOUT: }
