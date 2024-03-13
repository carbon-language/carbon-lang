// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.b: i32} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Derived, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: Base, .d: i32} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: {.b: i32}*, .d: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.base: {.b: i32}*, .d: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.base: Base, .d: i32} [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.base: {.b: i32}, .d: i32} [template]
// CHECK:STDOUT:   %.13: Base = struct_value (%.10) [template]
// CHECK:STDOUT:   %.14: Derived = struct_value (%.13, %.11) [template]
// CHECK:STDOUT:   %.15: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.16: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.17: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make
// CHECK:STDOUT:     .Access = %Access
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template] {
// CHECK:STDOUT:     %Derived.ref.loc17: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     @Make.%return: ref Derived = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access: <function> = fn_decl @Access [template] {
// CHECK:STDOUT:     %Derived.ref.loc21: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc21_11.1: Derived = param d
// CHECK:STDOUT:     @Access.%d: Derived = bind_name d, %d.loc21_11.1
// CHECK:STDOUT:     %.loc21_35.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:     %.loc21_35.2: type = converted %.loc21_35.1, constants.%.16 [template = constants.%.16]
// CHECK:STDOUT:     @Access.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc12: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %.loc14: <unbound element of class Derived> = field_decl d, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .d = %.loc14
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_25: i32 = int_literal 4 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_26.1: {.b: i32} = struct_literal (%.loc18_25)
// CHECK:STDOUT:   %.loc18_34: i32 = int_literal 7 [template = constants.%.11]
// CHECK:STDOUT:   %.loc18_35.1: {.base: {.b: i32}, .d: i32} = struct_literal (%.loc18_26.1, %.loc18_34)
// CHECK:STDOUT:   %.loc18_35.2: ref Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc18_26.2: ref i32 = class_element_access %.loc18_35.2, element0
// CHECK:STDOUT:   %.loc18_26.3: init i32 = initialize_from %.loc18_25 to %.loc18_26.2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_26.4: init Base = class_init (%.loc18_26.3), %.loc18_35.2 [template = constants.%.13]
// CHECK:STDOUT:   %.loc18_26.5: init Base = converted %.loc18_26.1, %.loc18_26.4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc18_35.3: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc18_35.4: init i32 = initialize_from %.loc18_34 to %.loc18_35.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc18_35.5: init Derived = class_init (%.loc18_26.5, %.loc18_35.4), %return [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_35.6: init Derived = converted %.loc18_35.1, %.loc18_35.5 [template = constants.%.14]
// CHECK:STDOUT:   return %.loc18_35.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: Derived) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc22_11: Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc22_12: <unbound element of class Derived> = name_ref d, @Derived.%.loc14 [template = @Derived.%.loc14]
// CHECK:STDOUT:   %.loc22_12.1: ref i32 = class_element_access %d.ref.loc22_11, element1
// CHECK:STDOUT:   %.loc22_12.2: i32 = bind_value %.loc22_12.1
// CHECK:STDOUT:   %d.ref.loc22_16: Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: <unbound element of class Derived> = name_ref base, @Derived.%.loc12 [template = @Derived.%.loc12]
// CHECK:STDOUT:   %.loc22_17.1: ref Base = class_element_access %d.ref.loc22_16, element0
// CHECK:STDOUT:   %.loc22_17.2: Base = bind_value %.loc22_17.1
// CHECK:STDOUT:   %b.ref: <unbound element of class Base> = name_ref b, @Base.%.loc8 [template = @Base.%.loc8]
// CHECK:STDOUT:   %.loc22_22.1: ref i32 = class_element_access %.loc22_17.2, element0
// CHECK:STDOUT:   %.loc22_22.2: i32 = bind_value %.loc22_22.1
// CHECK:STDOUT:   %.loc22_24.1: (i32, i32) = tuple_literal (%.loc22_12.2, %.loc22_22.2)
// CHECK:STDOUT:   %.loc22_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc22_24.3: init i32 = initialize_from %.loc22_12.2 to %.loc22_24.2
// CHECK:STDOUT:   %.loc22_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc22_24.5: init i32 = initialize_from %.loc22_22.2 to %.loc22_24.4
// CHECK:STDOUT:   %.loc22_24.6: init (i32, i32) = tuple_init (%.loc22_24.3, %.loc22_24.5) to %return
// CHECK:STDOUT:   %.loc22_24.7: init (i32, i32) = converted %.loc22_24.1, %.loc22_24.6
// CHECK:STDOUT:   return %.loc22_24.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
