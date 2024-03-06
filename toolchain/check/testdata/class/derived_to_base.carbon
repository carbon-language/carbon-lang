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
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type B, A [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type B, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: A, .b: i32} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: {.a: i32}*, .b: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.base: {.a: i32}*, .b: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.base: A, .b: i32} [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type C, B [template]
// CHECK:STDOUT:   %.11: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.base: B, .c: i32} [template]
// CHECK:STDOUT:   %.13: type = ptr_type C [template]
// CHECK:STDOUT:   %.14: type = ptr_type B [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: {.base: A, .b: i32}*, .c: i32} [template]
// CHECK:STDOUT:   %.16: type = ptr_type {.base: {.base: A, .b: i32}*, .c: i32} [template]
// CHECK:STDOUT:   %.17: type = ptr_type {.base: B, .c: i32} [template]
// CHECK:STDOUT:   %.18: type = ptr_type A [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.20: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.21: type = struct_type {.base: {.a: i32}, .b: i32} [template]
// CHECK:STDOUT:   %.22: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.23: type = struct_type {.base: {.base: {.a: i32}, .b: i32}, .c: i32} [template]
// CHECK:STDOUT:   %.24: A = struct_value (%.19) [template]
// CHECK:STDOUT:   %.25: B = struct_value (%.24, %.20) [template]
// CHECK:STDOUT:   %.26: C = struct_value (%.25, %.22) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ConvertCToB = %ConvertCToB
// CHECK:STDOUT:     .ConvertBToA = %ConvertBToA
// CHECK:STDOUT:     .ConvertCToA = %ConvertCToA
// CHECK:STDOUT:     .ConvertValue = %ConvertValue
// CHECK:STDOUT:     .ConvertRef = %ConvertRef
// CHECK:STDOUT:     .ConvertInit = %ConvertInit
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %ConvertCToB: <function> = fn_decl @ConvertCToB [template] {
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc21_20: type = ptr_type C [template = constants.%.13]
// CHECK:STDOUT:     %p.loc21_16.1: C* = param p
// CHECK:STDOUT:     @ConvertCToB.%p: C* = bind_name p, %p.loc21_16.1
// CHECK:STDOUT:     %B.ref.loc21: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc21_27: type = ptr_type B [template = constants.%.14]
// CHECK:STDOUT:     %return.var.loc21: ref B* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA: <function> = fn_decl @ConvertBToA [template] {
// CHECK:STDOUT:     %B.ref.loc22: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc22_20: type = ptr_type B [template = constants.%.14]
// CHECK:STDOUT:     %p.loc22_16.1: B* = param p
// CHECK:STDOUT:     @ConvertBToA.%p: B* = bind_name p, %p.loc22_16.1
// CHECK:STDOUT:     %A.ref.loc22: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc22_27: type = ptr_type A [template = constants.%.18]
// CHECK:STDOUT:     %return.var.loc22: ref A* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA: <function> = fn_decl @ConvertCToA [template] {
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_20: type = ptr_type C [template = constants.%.13]
// CHECK:STDOUT:     %p.loc23_16.1: C* = param p
// CHECK:STDOUT:     @ConvertCToA.%p: C* = bind_name p, %p.loc23_16.1
// CHECK:STDOUT:     %A.ref.loc23: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc23_27: type = ptr_type A [template = constants.%.18]
// CHECK:STDOUT:     %return.var.loc23: ref A* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue: <function> = fn_decl @ConvertValue [template] {
// CHECK:STDOUT:     %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc25_17.1: C = param c
// CHECK:STDOUT:     @ConvertValue.%c: C = bind_name c, %c.loc25_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef: <function> = fn_decl @ConvertRef [template] {
// CHECK:STDOUT:     %C.ref.loc29: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc29_19: type = ptr_type C [template = constants.%.13]
// CHECK:STDOUT:     %c.loc29_15.1: C* = param c
// CHECK:STDOUT:     @ConvertRef.%c: C* = bind_name c, %c.loc29_15.1
// CHECK:STDOUT:     %A.ref.loc29: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc29_26: type = ptr_type A [template = constants.%.18]
// CHECK:STDOUT:     %return.var.loc29: ref A* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit: <function> = fn_decl @ConvertInit [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc8: <unbound element of class A> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = base_decl A, element0 [template]
// CHECK:STDOUT:   %.loc13: <unbound element of class B> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .b = %.loc13
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc17: <unbound element of class C> = base_decl B, element0 [template]
// CHECK:STDOUT:   %.loc18: <unbound element of class C> = field_decl c, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .c = %.loc18
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p: C*) -> B* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: C* = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: B* = addr_of %.loc21_39.2
// CHECK:STDOUT:   %.loc21_38: B* = converted %p.ref, %.loc21_39.3
// CHECK:STDOUT:   return %.loc21_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p: B*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: B* = name_ref p, %p
// CHECK:STDOUT:   %.loc22_39.1: ref B = deref %p.ref
// CHECK:STDOUT:   %.loc22_39.2: ref A = class_element_access %.loc22_39.1, element0
// CHECK:STDOUT:   %.loc22_39.3: A* = addr_of %.loc22_39.2
// CHECK:STDOUT:   %.loc22_38: A* = converted %p.ref, %.loc22_39.3
// CHECK:STDOUT:   return %.loc22_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p: C*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: C* = name_ref p, %p
// CHECK:STDOUT:   %.loc23_39.1: ref C = deref %p.ref
// CHECK:STDOUT:   %.loc23_39.2: ref B = class_element_access %.loc23_39.1, element0
// CHECK:STDOUT:   %.loc23_39.3: ref A = class_element_access %.loc23_39.2, element0
// CHECK:STDOUT:   %.loc23_39.4: A* = addr_of %.loc23_39.3
// CHECK:STDOUT:   %.loc23_38: A* = converted %p.ref, %.loc23_39.4
// CHECK:STDOUT:   return %.loc23_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %.loc26_15.1: ref B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc26_15.2: ref A = class_element_access %.loc26_15.1, element0
// CHECK:STDOUT:   %.loc26_14.1: ref A = converted %c.ref, %.loc26_15.2
// CHECK:STDOUT:   %.loc26_14.2: A = bind_value %.loc26_14.1
// CHECK:STDOUT:   %a: A = bind_name a, %.loc26_14.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c: C*) -> A* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: C* = name_ref c, %c
// CHECK:STDOUT:   %.loc30_12.1: ref C = deref %c.ref
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc30_15.1: ref B = class_element_access %.loc30_12.1, element0
// CHECK:STDOUT:   %.loc30_15.2: ref A = class_element_access %.loc30_15.1, element0
// CHECK:STDOUT:   %.loc30_12.2: ref A = converted %.loc30_12.1, %.loc30_15.2
// CHECK:STDOUT:   %.loc30_10: A* = addr_of %.loc30_12.2
// CHECK:STDOUT:   return %.loc30_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc34_38: i32 = int_literal 1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc34_39.1: {.a: i32} = struct_literal (%.loc34_38)
// CHECK:STDOUT:   %.loc34_47: i32 = int_literal 2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc34_48.1: {.base: {.a: i32}, .b: i32} = struct_literal (%.loc34_39.1, %.loc34_47)
// CHECK:STDOUT:   %.loc34_56: i32 = int_literal 3 [template = constants.%.22]
// CHECK:STDOUT:   %.loc34_57.1: {.base: {.base: {.a: i32}, .b: i32}, .c: i32} = struct_literal (%.loc34_48.1, %.loc34_56)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc34_57.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc34_57.3: ref B = class_element_access %.loc34_57.2, element0
// CHECK:STDOUT:   %.loc34_48.2: ref A = class_element_access %.loc34_57.3, element0
// CHECK:STDOUT:   %.loc34_39.2: ref i32 = class_element_access %.loc34_48.2, element0
// CHECK:STDOUT:   %.loc34_39.3: init i32 = initialize_from %.loc34_38 to %.loc34_39.2 [template = constants.%.19]
// CHECK:STDOUT:   %.loc34_39.4: init A = class_init (%.loc34_39.3), %.loc34_48.2 [template = constants.%.24]
// CHECK:STDOUT:   %.loc34_39.5: init A = converted %.loc34_39.1, %.loc34_39.4 [template = constants.%.24]
// CHECK:STDOUT:   %.loc34_48.3: ref i32 = class_element_access %.loc34_57.3, element1
// CHECK:STDOUT:   %.loc34_48.4: init i32 = initialize_from %.loc34_47 to %.loc34_48.3 [template = constants.%.20]
// CHECK:STDOUT:   %.loc34_48.5: init B = class_init (%.loc34_39.5, %.loc34_48.4), %.loc34_57.3 [template = constants.%.25]
// CHECK:STDOUT:   %.loc34_48.6: init B = converted %.loc34_48.1, %.loc34_48.5 [template = constants.%.25]
// CHECK:STDOUT:   %.loc34_57.4: ref i32 = class_element_access %.loc34_57.2, element1
// CHECK:STDOUT:   %.loc34_57.5: init i32 = initialize_from %.loc34_56 to %.loc34_57.4 [template = constants.%.22]
// CHECK:STDOUT:   %.loc34_57.6: init C = class_init (%.loc34_48.6, %.loc34_57.5), %.loc34_57.2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc34_57.7: ref C = temporary %.loc34_57.2, %.loc34_57.6
// CHECK:STDOUT:   %.loc34_57.8: ref C = converted %.loc34_57.1, %.loc34_57.7
// CHECK:STDOUT:   %.loc34_63.1: ref B = class_element_access %.loc34_57.8, element0
// CHECK:STDOUT:   %.loc34_63.2: ref A = class_element_access %.loc34_63.1, element0
// CHECK:STDOUT:   %.loc34_57.9: ref A = converted %.loc34_57.8, %.loc34_63.2
// CHECK:STDOUT:   %.loc34_57.10: A = bind_value %.loc34_57.9
// CHECK:STDOUT:   %a: A = bind_name a, %.loc34_57.10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
