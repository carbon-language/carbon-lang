// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty_tuple.carbon'
// CHECK:STDOUT: source_filename = "empty_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type {}
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %TupleLiteralType @Echo(%TupleLiteralType %a) {
// CHECK:STDOUT:   ret %TupleLiteralType %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %TupleLiteralValue1 = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %1 = load %TupleLiteralType, ptr %TupleLiteralValue1, align 1
// CHECK:STDOUT:   %Echo = call %TupleLiteralType @Echo(%TupleLiteralType %1)
// CHECK:STDOUT:   store %TupleLiteralType %Echo, ptr %var, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  var b: () = Echo(());
}
