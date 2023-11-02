// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Outer {
  class Inner {
    var n: i32;
  }
}

fn F(oi: Outer.Inner) -> i32 {
  return oi.n;
}

fn G(o: Outer) {
  var i: o.Inner;
}

// CHECK:STDOUT: file "nested_name.carbon" {
// CHECK:STDOUT:   class_declaration @Outer, ()
// CHECK:STDOUT:   %Outer: type = class_type @Outer
// CHECK:STDOUT:   %.loc11: type = struct_type {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   class_declaration @Inner, ()
// CHECK:STDOUT:   %Inner: type = class_type @Inner
// CHECK:STDOUT:   %.loc10: type = struct_type {.n: i32}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Inner = <unexpected instref 11>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc9_10.1: type = unbound_field_type Inner, i32
// CHECK:STDOUT:   %.loc9_10.2: <unbound field of class Inner> = field "n", member0
// CHECK:STDOUT:   %n: <unbound field of class Inner> = bind_name "n", %.loc9_10.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .n = %n
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%oi: Inner) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: type = ptr_type {.n: i32}
// CHECK:STDOUT:   %oi.ref: Inner = name_reference "oi", %oi
// CHECK:STDOUT:   %.loc14_12.1: ref i32 = class_field_access %oi.ref, member0
// CHECK:STDOUT:   %.loc14_12.2: i32 = bind_value %.loc14_12.1
// CHECK:STDOUT:   return %.loc14_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%o: Outer) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %o.ref: Outer = name_reference "o", %o
// CHECK:STDOUT:   %Inner.ref: type = name_reference "Inner", @Outer.%Inner
// CHECK:STDOUT:   %i.var: ref Inner = var "i"
// CHECK:STDOUT:   %i: ref Inner = bind_name "i", %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
