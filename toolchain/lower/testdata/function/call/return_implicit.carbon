// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn MakeImplicitEmptyTuple() {
}

fn Main() {
  var b: () = MakeImplicitEmptyTuple();
}

// CHECK:STDOUT: ; ModuleID = 'return_implicit.carbon'
// CHECK:STDOUT: source_filename = "return_implicit.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @MakeImplicitEmptyTuple() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %b = alloca {}, align 8
// CHECK:STDOUT:   call void @MakeImplicitEmptyTuple()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
