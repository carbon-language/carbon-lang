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

// CHECK:STDOUT: --- nested_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Inner, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.n: i32} [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %.6: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [template = constants.%Outer] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Outer.ref.loc13: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:     %oi.loc13_6.1: Inner = param oi
// CHECK:STDOUT:     @F.%oi: Inner = bind_name oi, %oi.loc13_6.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Outer.ref.loc17: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %o.loc17_6.1: Outer = param o
// CHECK:STDOUT:     @G.%o: Outer = bind_name o, %o.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc9: <unbound element of class Inner> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .n = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%oi: Inner) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %oi.ref: Inner = name_ref oi, %oi
// CHECK:STDOUT:   %n.ref: <unbound element of class Inner> = name_ref n, @Inner.%.loc9 [template = @Inner.%.loc9]
// CHECK:STDOUT:   %.loc14_12.1: ref i32 = class_element_access %oi.ref, element0
// CHECK:STDOUT:   %.loc14_12.2: i32 = bind_value %.loc14_12.1
// CHECK:STDOUT:   return %.loc14_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%o: Outer) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %o.ref: Outer = name_ref o, %o
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref Inner = var i
// CHECK:STDOUT:   %i: ref Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
