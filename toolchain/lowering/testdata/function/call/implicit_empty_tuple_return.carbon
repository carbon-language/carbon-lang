// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_return.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type {}
// CHECK:STDOUT:

// CHECK:STDOUT: define void @MakeImplicitEmptyTuple() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   call void @MakeImplicitEmptyTuple()
// CHECK:STDOUT:   store i1 false, ptr %var, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn MakeImplicitEmptyTuple() {
}

fn Main() {
  var b: () = MakeImplicitEmptyTuple();
}
