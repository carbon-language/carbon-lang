// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class A {
  var a: i32;
}

base class B {
  extend base: A;
  var b: i32;
}

class C {
  extend base: B;
  var b: i32;
}

fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc14_1.1: type = struct_type {.base: A, .b: i32}
// CHECK:STDOUT:   %.loc14_1.2: type = struct_type {.base: {.a: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc14_1.3: type = ptr_type {.base: {.a: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: A, .b: i32}
// CHECK:STDOUT:   %.loc19_1.1: type = struct_type {.base: B, .b: i32}
// CHECK:STDOUT:   %.loc19_1.2: type = struct_type {.base: {.base: A, .b: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc19_1.3: type = ptr_type {.base: {.base: A, .b: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc16: type = ptr_type {.base: B, .b: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A.decl, .B = %B.decl, .C = %C.decl, .ConvertCToB = %ConvertCToB, .ConvertBToA = %ConvertBToA, .ConvertCToA = %ConvertCToA}
// CHECK:STDOUT:   %A.decl = class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %ConvertCToB: <function> = fn_decl @ConvertCToB
// CHECK:STDOUT:   %ConvertBToA: <function> = fn_decl @ConvertBToA
// CHECK:STDOUT:   %ConvertCToA: <function> = fn_decl @ConvertCToA
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type A, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class A> = field_decl a, element0
// CHECK:STDOUT:   %a: <unbound element of class A> = bind_name a, %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A
// CHECK:STDOUT:   %.loc12_17.1: type = unbound_element_type B, A
// CHECK:STDOUT:   %.loc12_17.2: <unbound element of class B> = base_decl A, element0
// CHECK:STDOUT:   %.loc13_8.1: type = unbound_element_type B, i32
// CHECK:STDOUT:   %.loc13_8.2: <unbound element of class B> = field_decl b, element1
// CHECK:STDOUT:   %b: <unbound element of class B> = bind_name b, %.loc13_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12_17.2
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc17_17.1: type = unbound_element_type C, B
// CHECK:STDOUT:   %.loc17_17.2: <unbound element of class C> = base_decl B, element0
// CHECK:STDOUT:   %.loc18_8.1: type = unbound_element_type C, i32
// CHECK:STDOUT:   %.loc18_8.2: <unbound element of class C> = field_decl b, element1
// CHECK:STDOUT:   %b: <unbound element of class C> = bind_name b, %.loc18_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc17_17.2
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p: C*) -> B* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: C* = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: B* = address_of %.loc21_39.2
// CHECK:STDOUT:   %.loc21_39.4: B* = converted %p.ref, %.loc21_39.3
// CHECK:STDOUT:   return %.loc21_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p: B*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: B* = name_ref p, %p
// CHECK:STDOUT:   %.loc22_39.1: ref B = deref %p.ref
// CHECK:STDOUT:   %.loc22_39.2: ref A = class_element_access %.loc22_39.1, element0
// CHECK:STDOUT:   %.loc22_39.3: A* = address_of %.loc22_39.2
// CHECK:STDOUT:   %.loc22_39.4: A* = converted %p.ref, %.loc22_39.3
// CHECK:STDOUT:   return %.loc22_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p: C*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: C* = name_ref p, %p
// CHECK:STDOUT:   %.loc23_39.1: ref C = deref %p.ref
// CHECK:STDOUT:   %.loc23_39.2: ref B = class_element_access %.loc23_39.1, element0
// CHECK:STDOUT:   %.loc23_39.3: ref A = class_element_access %.loc23_39.2, element0
// CHECK:STDOUT:   %.loc23_39.4: A* = address_of %.loc23_39.3
// CHECK:STDOUT:   %.loc23_39.5: A* = converted %p.ref, %.loc23_39.4
// CHECK:STDOUT:   return %.loc23_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
