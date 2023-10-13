// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;

  return *p;
}

// CHECK:STDOUT: file "basic.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n: ref i32 = var "n"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %n, %.loc8
// CHECK:STDOUT:   %.loc9_13: type = ptr_type i32
// CHECK:STDOUT:   %p: ref i32* = var "p"
// CHECK:STDOUT:   %n.ref: ref i32 = name_reference "n", %n
// CHECK:STDOUT:   %.loc9_17: i32* = address_of %n.ref
// CHECK:STDOUT:   assign %p, %.loc9_17
// CHECK:STDOUT:   %p.ref: ref i32* = name_reference "p", %p
// CHECK:STDOUT:   %.loc11_11: i32* = bind_value %p.ref
// CHECK:STDOUT:   %.loc11_10.1: ref i32 = dereference %.loc11_11
// CHECK:STDOUT:   %.loc11_10.2: i32 = bind_value %.loc11_10.1
// CHECK:STDOUT:   return %.loc11_10.2
// CHECK:STDOUT: }
