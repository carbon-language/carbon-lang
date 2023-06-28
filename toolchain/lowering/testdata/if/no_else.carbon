// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'no_else.carbon'
// CHECK:STDOUT: source_filename = "no_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @F() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @G() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @If(i1 %b) {
// CHECK:STDOUT:   br i1 %b, label %1, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   %F = call %EmptyTupleType @F()
// CHECK:STDOUT:   br label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1, %0
// CHECK:STDOUT:   %G = call %EmptyTupleType @G()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn F() {}
fn G() {}

fn If(b: bool) {
  if (b) {
    F();
  }
  G();
}
