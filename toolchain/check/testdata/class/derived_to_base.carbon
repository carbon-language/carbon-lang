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
  var c: i32;
}

fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
}

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc14_1.1: type = struct_type {.base: A, .b: i32}
// CHECK:STDOUT:   %.loc14_1.2: type = struct_type {.base: {.a: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc14_1.3: type = ptr_type {.base: {.a: i32}*, .b: i32}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: A, .b: i32}
// CHECK:STDOUT:   %.loc19_1.1: type = struct_type {.base: B, .c: i32}
// CHECK:STDOUT:   %.loc19_1.2: type = struct_type {.base: {.base: A, .b: i32}*, .c: i32}
// CHECK:STDOUT:   %.loc19_1.3: type = ptr_type {.base: {.base: A, .b: i32}*, .c: i32}
// CHECK:STDOUT:   %.loc16: type = ptr_type {.base: B, .c: i32}
// CHECK:STDOUT:   %.loc34_48: type = struct_type {.base: {.a: i32}, .b: i32}
// CHECK:STDOUT:   %.loc34_57: type = struct_type {.base: {.base: {.a: i32}, .b: i32}, .c: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %A.decl, .B = %B.decl, .C = %C.decl, .ConvertCToB = %ConvertCToB, .ConvertBToA = %ConvertBToA, .ConvertCToA = %ConvertCToA, .ConvertValue = %ConvertValue, .ConvertRef = %ConvertRef, .ConvertInit = %ConvertInit}
// CHECK:STDOUT:   %A.decl = class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %ConvertCToB: <function> = fn_decl @ConvertCToB
// CHECK:STDOUT:   %ConvertBToA: <function> = fn_decl @ConvertBToA
// CHECK:STDOUT:   %ConvertCToA: <function> = fn_decl @ConvertCToA
// CHECK:STDOUT:   %ConvertValue: <function> = fn_decl @ConvertValue
// CHECK:STDOUT:   %ConvertRef: <function> = fn_decl @ConvertRef
// CHECK:STDOUT:   %ConvertInit: <function> = fn_decl @ConvertInit
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
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc17_17.1: type = unbound_element_type C, B
// CHECK:STDOUT:   %.loc17_17.2: <unbound element of class C> = base_decl B, element0
// CHECK:STDOUT:   %.loc18_8.1: type = unbound_element_type C, i32
// CHECK:STDOUT:   %.loc18_8.2: <unbound element of class C> = field_decl c, element1
// CHECK:STDOUT:   %c: <unbound element of class C> = bind_name c, %.loc18_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc17_17.2
// CHECK:STDOUT:   .c = %c
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p: C*) -> B* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: C* = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: B* = addr_of %.loc21_39.2
// CHECK:STDOUT:   %.loc21_39.4: B* = converted %p.ref, %.loc21_39.3
// CHECK:STDOUT:   return %.loc21_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p: B*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: B* = name_ref p, %p
// CHECK:STDOUT:   %.loc22_39.1: ref B = deref %p.ref
// CHECK:STDOUT:   %.loc22_39.2: ref A = class_element_access %.loc22_39.1, element0
// CHECK:STDOUT:   %.loc22_39.3: A* = addr_of %.loc22_39.2
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
// CHECK:STDOUT:   %.loc23_39.4: A* = addr_of %.loc23_39.3
// CHECK:STDOUT:   %.loc23_39.5: A* = converted %p.ref, %.loc23_39.4
// CHECK:STDOUT:   return %.loc23_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %.loc26_15.1: ref B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc26_15.2: ref A = class_element_access %.loc26_15.1, element0
// CHECK:STDOUT:   %.loc26_15.3: ref A = converted %c.ref, %.loc26_15.2
// CHECK:STDOUT:   %.loc26_15.4: A = bind_value %.loc26_15.3
// CHECK:STDOUT:   %a: A = bind_name a, %.loc26_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c: C*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: C* = name_ref c, %c
// CHECK:STDOUT:   %.loc30_12: ref C = deref %c.ref
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A
// CHECK:STDOUT:   %.loc30_15.1: ref B = class_element_access %.loc30_12, element0
// CHECK:STDOUT:   %.loc30_15.2: ref A = class_element_access %.loc30_15.1, element0
// CHECK:STDOUT:   %.loc30_15.3: ref A = converted %.loc30_12, %.loc30_15.2
// CHECK:STDOUT:   %.loc30_10: A* = addr_of %.loc30_15.3
// CHECK:STDOUT:   return %.loc30_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A
// CHECK:STDOUT:   %.loc34_38: i32 = int_literal 1
// CHECK:STDOUT:   %.loc34_39.1: {.a: i32} = struct_literal (%.loc34_38)
// CHECK:STDOUT:   %.loc34_47: i32 = int_literal 2
// CHECK:STDOUT:   %.loc34_48.1: {.base: {.a: i32}, .b: i32} = struct_literal (%.loc34_39.1, %.loc34_47)
// CHECK:STDOUT:   %.loc34_56: i32 = int_literal 3
// CHECK:STDOUT:   %.loc34_57.1: {.base: {.base: {.a: i32}, .b: i32}, .c: i32} = struct_literal (%.loc34_48.1, %.loc34_56)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C
// CHECK:STDOUT:   %.loc34_57.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc34_57.3: ref B = class_element_access %.loc34_57.2, element0
// CHECK:STDOUT:   %.loc34_48.2: ref A = class_element_access %.loc34_57.3, element0
// CHECK:STDOUT:   %.loc34_39.2: ref i32 = class_element_access %.loc34_48.2, element0
// CHECK:STDOUT:   %.loc34_39.3: init i32 = initialize_from %.loc34_38 to %.loc34_39.2
// CHECK:STDOUT:   %.loc34_39.4: init A = class_init (%.loc34_39.3), %.loc34_48.2
// CHECK:STDOUT:   %.loc34_39.5: init A = converted %.loc34_39.1, %.loc34_39.4
// CHECK:STDOUT:   %.loc34_48.3: ref i32 = class_element_access %.loc34_57.3, element1
// CHECK:STDOUT:   %.loc34_48.4: init i32 = initialize_from %.loc34_47 to %.loc34_48.3
// CHECK:STDOUT:   %.loc34_48.5: init B = class_init (%.loc34_39.5, %.loc34_48.4), %.loc34_57.3
// CHECK:STDOUT:   %.loc34_48.6: init B = converted %.loc34_48.1, %.loc34_48.5
// CHECK:STDOUT:   %.loc34_57.4: ref i32 = class_element_access %.loc34_57.2, element1
// CHECK:STDOUT:   %.loc34_57.5: init i32 = initialize_from %.loc34_56 to %.loc34_57.4
// CHECK:STDOUT:   %.loc34_57.6: init C = class_init (%.loc34_48.6, %.loc34_57.5), %.loc34_57.2
// CHECK:STDOUT:   %.loc34_57.7: ref C = temporary %.loc34_57.2, %.loc34_57.6
// CHECK:STDOUT:   %.loc34_57.8: ref C = converted %.loc34_57.1, %.loc34_57.7
// CHECK:STDOUT:   %.loc34_63.1: ref B = class_element_access %.loc34_57.8, element0
// CHECK:STDOUT:   %.loc34_63.2: ref A = class_element_access %.loc34_63.1, element0
// CHECK:STDOUT:   %.loc34_63.3: ref A = converted %.loc34_57.8, %.loc34_63.2
// CHECK:STDOUT:   %.loc34_63.4: A = bind_value %.loc34_63.3
// CHECK:STDOUT:   %a: A = bind_name a, %.loc34_63.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
