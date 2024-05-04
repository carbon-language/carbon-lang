// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- adapt_class.carbon

library "adapt_class" api;

class A {
  var x: i32;
  var y: i32;

  fn Make() -> A {
    return {.x = 1, .y = 2};
  }
}

class B {
  adapt A;
}

var a_ref: A = {.x = 1, .y = 2};
let a_val: A = a_ref;

// An `as` conversion to an adapter type preserves the expression category.
let b_val: B = a_val as B;
let b_ptr: B* = &(a_ref as B);

var b_factory: B = A.Make() as B;

// --- adapt_i32.carbon

library "adapt_i32" api;

class A {
  adapt i32;
}

let a: A = (1 as i32) as A;
let n: i32 = a as i32;

// --- multi_level_adapt.carbon

library "multi_level_adapt" api;

class A { adapt {}; }
class B { adapt A; }
class C { adapt B; }
class D { adapt C; }

let d: D = {} as D;

// --- fail_init_class.carbon

library "fail_init_class" api;

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

let b_value: B = ({.x = 1, .y = 2} as A) as B;

// TODO: Here, we treat `{.x = 1, .y = 2} as A` as a value expression, not an
// initializing expression, so `(...) as B` is a value expression too, requiring
// a copy to perform initialization. It's not clear whether that is the right
// behavior.

// CHECK:STDERR: fail_init_class.carbon:[[@LINE+4]]:17: ERROR: Cannot copy value of type `B`.
// CHECK:STDERR: var b_init: B = ({.x = 1, .y = 2} as A) as B;
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var b_init: B = ({.x = 1, .y = 2} as A) as B;

// --- fail_adapt_init_from_struct.carbon

library "fail_adapt_init_from_struct" api;

class A {
  var x: i32;
}

class B {
  adapt A;
}

// We do not try to implicitly convert from the first operand of `as` to the
// adapted type of the second operand.

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+3]]:12: ERROR: Cannot convert from `{.x: i32}` to `B` with `as`.
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
var b: B = {.x = 1} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %Make: type = fn_type @Make [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Make = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.2: A = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.7: type = ptr_type B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_factory = %b_factory
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %A.ref.loc17: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %a_ref.var: ref A = var a_ref
// CHECK:STDOUT:   %a_ref: ref A = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %A.ref.loc18: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %a_ref.ref.loc18: ref A = name_ref a_ref, %a_ref
// CHECK:STDOUT:   %.loc18: A = bind_value %a_ref.ref.loc18
// CHECK:STDOUT:   %a_val: A = bind_name a_val, %.loc18
// CHECK:STDOUT:   %B.ref.loc21_12: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %a_val.ref: A = name_ref a_val, %a_val
// CHECK:STDOUT:   %B.ref.loc21_25: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21_22.1: B = as_compatible %a_val.ref
// CHECK:STDOUT:   %.loc21_22.2: B = converted %a_val.ref, %.loc21_22.1
// CHECK:STDOUT:   %b_val: B = bind_name b_val, %.loc21_22.2
// CHECK:STDOUT:   %B.ref.loc22_12: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc22_13: type = ptr_type B [template = constants.%.7]
// CHECK:STDOUT:   %a_ref.ref.loc22: ref A = name_ref a_ref, %a_ref
// CHECK:STDOUT:   %B.ref.loc22_28: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc22_25.1: ref B = as_compatible %a_ref.ref.loc22
// CHECK:STDOUT:   %.loc22_25.2: ref B = converted %a_ref.ref.loc22, %.loc22_25.1
// CHECK:STDOUT:   %.loc22_17: B* = addr_of %.loc22_25.2
// CHECK:STDOUT:   %b_ptr: B* = bind_name b_ptr, %.loc22_17
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b_factory.var: ref B = var b_factory
// CHECK:STDOUT:   %b_factory: ref B = bind_name b_factory, %b_factory.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: <unbound element of class A> = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class A> = field_decl y, element1 [template]
// CHECK:STDOUT:   %Make.decl: Make = fn_decl @Make [template = constants.%struct.1] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %return.var: ref A = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5
// CHECK:STDOUT:   .y = %.loc6
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl A
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> @A.%return.var: A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_26: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_27.1: {.x: i32, .y: i32} = struct_literal (%.loc9_18, %.loc9_26)
// CHECK:STDOUT:   %.loc9_27.2: ref i32 = class_element_access @A.%return.var, element0
// CHECK:STDOUT:   %.loc9_27.3: init i32 = initialize_from %.loc9_18 to %.loc9_27.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_27.4: ref i32 = class_element_access @A.%return.var, element1
// CHECK:STDOUT:   %.loc9_27.5: init i32 = initialize_from %.loc9_26 to %.loc9_27.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_27.6: init A = class_init (%.loc9_27.3, %.loc9_27.5), @A.%return.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_28: init A = converted %.loc9_27.1, %.loc9_27.6 [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc9_28 to @A.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_30: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_31.1: {.x: i32, .y: i32} = struct_literal (%.loc17_22, %.loc17_30)
// CHECK:STDOUT:   %.loc17_31.2: ref i32 = class_element_access file.%a_ref.var, element0
// CHECK:STDOUT:   %.loc17_31.3: init i32 = initialize_from %.loc17_22 to %.loc17_31.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_31.4: ref i32 = class_element_access file.%a_ref.var, element1
// CHECK:STDOUT:   %.loc17_31.5: init i32 = initialize_from %.loc17_30 to %.loc17_31.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_31.6: init A = class_init (%.loc17_31.3, %.loc17_31.5), file.%a_ref.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc17_32: init A = converted %.loc17_31.1, %.loc17_31.6 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc17_32
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %Make.ref: Make = name_ref Make, @A.%Make.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc24_5: ref B = splice_block file.%b_factory.var {}
// CHECK:STDOUT:   %Make.call: init A = call %Make.ref() to %.loc24_5
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc24_29.1: init B = as_compatible %Make.call
// CHECK:STDOUT:   %.loc24_29.2: init B = converted %Make.call, %.loc24_29.1
// CHECK:STDOUT:   assign file.%b_factory.var, %.loc24_29.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %A.ref.loc8_8: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc8_13: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %A.ref.loc8_26: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc8_23.1: A = as_compatible %.loc8_13 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_23.2: A = converted %.loc8_13, %.loc8_23.1 [template = constants.%.1]
// CHECK:STDOUT:   %a: A = bind_name a, %.loc8_23.2
// CHECK:STDOUT:   %a.ref: A = name_ref a, %a
// CHECK:STDOUT:   %.loc9_16.1: i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_16.2: i32 = converted %a.ref, %.loc9_16.1
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc9_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %D.ref.loc9_8: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc9_13: {} = struct_literal ()
// CHECK:STDOUT:   %D.ref.loc9_18: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:   %struct: {} = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_15.1: D = as_compatible %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_15.2: D = converted %.loc9_13, %.loc9_15.1 [template = constants.%struct]
// CHECK:STDOUT:   %d: D = bind_name d, %.loc9_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc4_18: {} = struct_literal ()
// CHECK:STDOUT:   %.loc4_19: type = converted %.loc4_18, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl A
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl C
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_init_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: A = struct_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b_init = %b_init
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %B.ref.loc13_14: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_33: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_34.1: {.x: i32, .y: i32} = struct_literal (%.loc13_25, %.loc13_33)
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc13_34.2: ref A = temporary_storage
// CHECK:STDOUT:   %.loc13_34.3: ref i32 = class_element_access %.loc13_34.2, element0
// CHECK:STDOUT:   %.loc13_34.4: init i32 = initialize_from %.loc13_25 to %.loc13_34.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_34.5: ref i32 = class_element_access %.loc13_34.2, element1
// CHECK:STDOUT:   %.loc13_34.6: init i32 = initialize_from %.loc13_33 to %.loc13_34.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_34.7: init A = class_init (%.loc13_34.4, %.loc13_34.6), %.loc13_34.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_34.8: ref A = temporary %.loc13_34.2, %.loc13_34.7
// CHECK:STDOUT:   %.loc13_36: ref A = converted %.loc13_34.1, %.loc13_34.8
// CHECK:STDOUT:   %B.ref.loc13_45: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13_42.1: ref B = as_compatible %.loc13_36
// CHECK:STDOUT:   %.loc13_42.2: ref B = converted %.loc13_36, %.loc13_42.1
// CHECK:STDOUT:   %.loc13_42.3: B = bind_value %.loc13_42.2
// CHECK:STDOUT:   %b_value: B = bind_name b_value, %.loc13_42.3
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b_init.var: ref B = var b_init
// CHECK:STDOUT:   %b_init: ref B = bind_name b_init, %b_init.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: <unbound element of class A> = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class A> = field_decl y, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5
// CHECK:STDOUT:   .y = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl A
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_24: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_32: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc24_33.1: {.x: i32, .y: i32} = struct_literal (%.loc24_24, %.loc24_32)
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc24_33.2: ref A = temporary_storage
// CHECK:STDOUT:   %.loc24_33.3: ref i32 = class_element_access %.loc24_33.2, element0
// CHECK:STDOUT:   %.loc24_33.4: init i32 = initialize_from %.loc24_24 to %.loc24_33.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_33.5: ref i32 = class_element_access %.loc24_33.2, element1
// CHECK:STDOUT:   %.loc24_33.6: init i32 = initialize_from %.loc24_32 to %.loc24_33.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc24_33.7: init A = class_init (%.loc24_33.4, %.loc24_33.6), %.loc24_33.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc24_33.8: ref A = temporary %.loc24_33.2, %.loc24_33.7
// CHECK:STDOUT:   %.loc24_35: ref A = converted %.loc24_33.1, %.loc24_33.8
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc24_41.1: ref B = as_compatible %.loc24_35
// CHECK:STDOUT:   %.loc24_41.2: ref B = converted %.loc24_35, %.loc24_41.1
// CHECK:STDOUT:   %.loc24_41.3: B = bind_value %.loc24_41.2
// CHECK:STDOUT:   assign file.%b_init.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_init_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.var: ref B = var b
// CHECK:STDOUT:   %b: ref B = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: <unbound element of class A> = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl A
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_18: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_19: {.x: i32} = struct_literal (%.loc18_18)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
