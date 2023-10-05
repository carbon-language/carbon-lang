// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  var b: () = Echo(());
}

// CHECK:STDOUT: ; ModuleID = 'empty_tuple.carbon'
// CHECK:STDOUT: source_filename = "empty_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Echo() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %b = alloca {}, align 8
// CHECK:STDOUT:   call void @Echo()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
