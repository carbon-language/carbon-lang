// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}
fn G() {}

fn If(b: bool) {
  if (b) {
    F();
  }
  G();
}

// CHECK:STDOUT: ; ModuleID = 'no_else.carbon'
// CHECK:STDOUT: source_filename = "no_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @If(i1 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.then, label %if.else
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %if.else
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
