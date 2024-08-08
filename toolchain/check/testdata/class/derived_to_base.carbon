// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/derived_to_base.carbon

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %A, .b: i32} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %.4, .b: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.11: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.12: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: %B, .c: i32} [template]
// CHECK:STDOUT:   %.14: type = ptr_type %C [template]
// CHECK:STDOUT:   %.15: type = ptr_type %B [template]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [template]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [template]
// CHECK:STDOUT:   %.16: type = struct_type {.base: %.10, .c: i32} [template]
// CHECK:STDOUT:   %.17: type = ptr_type %.16 [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.19: type = ptr_type %A [template]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [template]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [template]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [template]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [template]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [template]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [template]
// CHECK:STDOUT:   %.20: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.21: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.22: type = struct_type {.base: %.3, .b: i32} [template]
// CHECK:STDOUT:   %.23: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.24: type = struct_type {.base: %.22, .c: i32} [template]
// CHECK:STDOUT:   %struct.1: %A = struct_value (%.20) [template]
// CHECK:STDOUT:   %struct.2: %B = struct_value (%struct.1, %.21) [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value (%struct.2, %.23) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ConvertCToB = %ConvertCToB.decl
// CHECK:STDOUT:     .ConvertBToA = %ConvertBToA.decl
// CHECK:STDOUT:     .ConvertCToA = %ConvertCToA.decl
// CHECK:STDOUT:     .ConvertValue = %ConvertValue.decl
// CHECK:STDOUT:     .ConvertRef = %ConvertRef.decl
// CHECK:STDOUT:     .ConvertInit = %ConvertInit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [template = constants.%ConvertCToB] {
// CHECK:STDOUT:     %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc25_20: type = ptr_type %C [template = constants.%.14]
// CHECK:STDOUT:     %p.loc25_16.1: %.14 = param p
// CHECK:STDOUT:     @ConvertCToB.%p: %.14 = bind_name p, %p.loc25_16.1
// CHECK:STDOUT:     %B.ref.loc25: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc25_27: type = ptr_type %B [template = constants.%.15]
// CHECK:STDOUT:     @ConvertCToB.%return: ref %.15 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [template = constants.%ConvertBToA] {
// CHECK:STDOUT:     %B.ref.loc26: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc26_20: type = ptr_type %B [template = constants.%.15]
// CHECK:STDOUT:     %p.loc26_16.1: %.15 = param p
// CHECK:STDOUT:     @ConvertBToA.%p: %.15 = bind_name p, %p.loc26_16.1
// CHECK:STDOUT:     %A.ref.loc26: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc26_27: type = ptr_type %A [template = constants.%.19]
// CHECK:STDOUT:     @ConvertBToA.%return: ref %.19 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [template = constants.%ConvertCToA] {
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc27_20: type = ptr_type %C [template = constants.%.14]
// CHECK:STDOUT:     %p.loc27_16.1: %.14 = param p
// CHECK:STDOUT:     @ConvertCToA.%p: %.14 = bind_name p, %p.loc27_16.1
// CHECK:STDOUT:     %A.ref.loc27: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc27_27: type = ptr_type %A [template = constants.%.19]
// CHECK:STDOUT:     @ConvertCToA.%return: ref %.19 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [template = constants.%ConvertValue] {
// CHECK:STDOUT:     %C.ref.loc29: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc29_17.1: %C = param c
// CHECK:STDOUT:     @ConvertValue.%c: %C = bind_name c, %c.loc29_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [template = constants.%ConvertRef] {
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc33_19: type = ptr_type %C [template = constants.%.14]
// CHECK:STDOUT:     %c.loc33_15.1: %.14 = param c
// CHECK:STDOUT:     @ConvertRef.%c: %.14 = bind_name c, %c.loc33_15.1
// CHECK:STDOUT:     %A.ref.loc33: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc33_26: type = ptr_type %A [template = constants.%.19]
// CHECK:STDOUT:     @ConvertRef.%return: ref %.19 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [template = constants.%ConvertInit] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.5 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc17_10.2: type = converted %int.make_type_32, %.loc17_10.1 [template = i32]
// CHECK:STDOUT:   %.loc17_8: %.6 = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .b = %.loc17_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %.11 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc22_10.2: type = converted %int.make_type_32, %.loc22_10.1 [template = i32]
// CHECK:STDOUT:   %.loc22_8: %.12 = field_decl c, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .c = %.loc22_8
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p: %.14) -> %.15 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc25_39.2: ref %B = class_element_access %.loc25_39.1, element0
// CHECK:STDOUT:   %.loc25_39.3: %.15 = addr_of %.loc25_39.2
// CHECK:STDOUT:   %.loc25_39.4: %.15 = converted %p.ref, %.loc25_39.3
// CHECK:STDOUT:   return %.loc25_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p: %.15) -> %.19 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.15 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc26_39.2: ref %A = class_element_access %.loc26_39.1, element0
// CHECK:STDOUT:   %.loc26_39.3: %.19 = addr_of %.loc26_39.2
// CHECK:STDOUT:   %.loc26_39.4: %.19 = converted %p.ref, %.loc26_39.3
// CHECK:STDOUT:   return %.loc26_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p: %.14) -> %.19 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc27_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc27_39.2: ref %B = class_element_access %.loc27_39.1, element0
// CHECK:STDOUT:   %.loc27_39.3: ref %A = class_element_access %.loc27_39.2, element0
// CHECK:STDOUT:   %.loc27_39.4: %.19 = addr_of %.loc27_39.3
// CHECK:STDOUT:   %.loc27_39.5: %.19 = converted %p.ref, %.loc27_39.4
// CHECK:STDOUT:   return %.loc27_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc30_15.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc30_15.2: ref %A = class_element_access %.loc30_15.1, element0
// CHECK:STDOUT:   %.loc30_15.3: ref %A = converted %c.ref, %.loc30_15.2
// CHECK:STDOUT:   %.loc30_15.4: %A = bind_value %.loc30_15.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc30_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c: %.14) -> %.19 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %.14 = name_ref c, %c
// CHECK:STDOUT:   %.loc34_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc34_15.1: ref %B = class_element_access %.loc34_12, element0
// CHECK:STDOUT:   %.loc34_15.2: ref %A = class_element_access %.loc34_15.1, element0
// CHECK:STDOUT:   %.loc34_15.3: ref %A = converted %.loc34_12, %.loc34_15.2
// CHECK:STDOUT:   %.loc34_10: %.19 = addr_of %.loc34_15.3
// CHECK:STDOUT:   return %.loc34_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc38_38: i32 = int_literal 1 [template = constants.%.20]
// CHECK:STDOUT:   %.loc38_39.1: %.3 = struct_literal (%.loc38_38)
// CHECK:STDOUT:   %.loc38_47: i32 = int_literal 2 [template = constants.%.21]
// CHECK:STDOUT:   %.loc38_48.1: %.22 = struct_literal (%.loc38_39.1, %.loc38_47)
// CHECK:STDOUT:   %.loc38_56: i32 = int_literal 3 [template = constants.%.23]
// CHECK:STDOUT:   %.loc38_57.1: %.24 = struct_literal (%.loc38_48.1, %.loc38_56)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc38_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_57.3: ref %B = class_element_access %.loc38_57.2, element0
// CHECK:STDOUT:   %.loc38_48.2: ref %A = class_element_access %.loc38_57.3, element0
// CHECK:STDOUT:   %.loc38_39.2: ref i32 = class_element_access %.loc38_48.2, element0
// CHECK:STDOUT:   %.loc38_39.3: init i32 = initialize_from %.loc38_38 to %.loc38_39.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc38_39.4: init %A = class_init (%.loc38_39.3), %.loc38_48.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc38_48.3: init %A = converted %.loc38_39.1, %.loc38_39.4 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc38_48.4: ref i32 = class_element_access %.loc38_57.3, element1
// CHECK:STDOUT:   %.loc38_48.5: init i32 = initialize_from %.loc38_47 to %.loc38_48.4 [template = constants.%.21]
// CHECK:STDOUT:   %.loc38_48.6: init %B = class_init (%.loc38_48.3, %.loc38_48.5), %.loc38_57.3 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc38_57.4: init %B = converted %.loc38_48.1, %.loc38_48.6 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc38_57.5: ref i32 = class_element_access %.loc38_57.2, element1
// CHECK:STDOUT:   %.loc38_57.6: init i32 = initialize_from %.loc38_56 to %.loc38_57.5 [template = constants.%.23]
// CHECK:STDOUT:   %.loc38_57.7: init %C = class_init (%.loc38_57.4, %.loc38_57.6), %.loc38_57.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc38_57.8: ref %C = temporary %.loc38_57.2, %.loc38_57.7
// CHECK:STDOUT:   %.loc38_59: ref %C = converted %.loc38_57.1, %.loc38_57.8
// CHECK:STDOUT:   %.loc38_63.1: ref %B = class_element_access %.loc38_59, element0
// CHECK:STDOUT:   %.loc38_63.2: ref %A = class_element_access %.loc38_63.1, element0
// CHECK:STDOUT:   %.loc38_63.3: ref %A = converted %.loc38_59, %.loc38_63.2
// CHECK:STDOUT:   %.loc38_63.4: %A = bind_value %.loc38_63.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc38_63.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
