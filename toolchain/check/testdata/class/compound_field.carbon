// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/compound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/compound_field.carbon

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn AccessDerived(d: Derived) -> i32 {
  return d.(Derived.d);
}

fn AccessBase(d: Derived) -> i32 {
  return d.(Base.b);
}

fn AccessDerivedIndirect(p: Derived*) -> i32* {
  return &p->(Derived.d);
}

fn AccessBaseIndirect(p: Derived*) -> i32* {
  return &p->(Base.b);
}

// CHECK:STDOUT: --- compound_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %Base, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [template]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %.4, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [template]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [template]
// CHECK:STDOUT:   %.11: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %.12: type = ptr_type i32 [template]
// CHECK:STDOUT:   %AccessDerivedIndirect.type: type = fn_type @AccessDerivedIndirect [template]
// CHECK:STDOUT:   %AccessDerivedIndirect: %AccessDerivedIndirect.type = struct_value () [template]
// CHECK:STDOUT:   %AccessBaseIndirect.type: type = fn_type @AccessBaseIndirect [template]
// CHECK:STDOUT:   %AccessBaseIndirect: %AccessBaseIndirect.type = struct_value () [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived.decl
// CHECK:STDOUT:     .AccessBase = %AccessBase.decl
// CHECK:STDOUT:     .AccessDerivedIndirect = %AccessDerivedIndirect.decl
// CHECK:STDOUT:     .AccessBaseIndirect = %AccessBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [template = constants.%AccessDerived] {
// CHECK:STDOUT:     %Derived.ref.loc24: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc24_18.1: %Derived = param d, runtime_param0
// CHECK:STDOUT:     @AccessDerived.%d: %Derived = bind_name d, %d.loc24_18.1
// CHECK:STDOUT:     %int.make_type_32.loc24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc24_33.1: type = value_of_initializer %int.make_type_32.loc24 [template = i32]
// CHECK:STDOUT:     %.loc24_33.2: type = converted %int.make_type_32.loc24, %.loc24_33.1 [template = i32]
// CHECK:STDOUT:     @AccessDerived.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [template = constants.%AccessBase] {
// CHECK:STDOUT:     %Derived.ref.loc28: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc28_15.1: %Derived = param d, runtime_param0
// CHECK:STDOUT:     @AccessBase.%d: %Derived = bind_name d, %d.loc28_15.1
// CHECK:STDOUT:     %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc28_30.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:     %.loc28_30.2: type = converted %int.make_type_32.loc28, %.loc28_30.1 [template = i32]
// CHECK:STDOUT:     @AccessBase.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerivedIndirect.decl: %AccessDerivedIndirect.type = fn_decl @AccessDerivedIndirect [template = constants.%AccessDerivedIndirect] {
// CHECK:STDOUT:     %Derived.ref.loc32: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc32_36: type = ptr_type %Derived [template = constants.%.11]
// CHECK:STDOUT:     %p.loc32_26.1: %.11 = param p, runtime_param0
// CHECK:STDOUT:     @AccessDerivedIndirect.%p: %.11 = bind_name p, %p.loc32_26.1
// CHECK:STDOUT:     %int.make_type_32.loc32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc32_45.1: type = value_of_initializer %int.make_type_32.loc32 [template = i32]
// CHECK:STDOUT:     %.loc32_45.2: type = converted %int.make_type_32.loc32, %.loc32_45.1 [template = i32]
// CHECK:STDOUT:     %.loc32_45.3: type = ptr_type i32 [template = constants.%.12]
// CHECK:STDOUT:     @AccessDerivedIndirect.%return: ref %.12 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBaseIndirect.decl: %AccessBaseIndirect.type = fn_decl @AccessBaseIndirect [template = constants.%AccessBaseIndirect] {
// CHECK:STDOUT:     %Derived.ref.loc36: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc36_33: type = ptr_type %Derived [template = constants.%.11]
// CHECK:STDOUT:     %p.loc36_23.1: %.11 = param p, runtime_param0
// CHECK:STDOUT:     @AccessBaseIndirect.%p: %.11 = bind_name p, %p.loc36_23.1
// CHECK:STDOUT:     %int.make_type_32.loc36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc36_42.1: type = value_of_initializer %int.make_type_32.loc36 [template = i32]
// CHECK:STDOUT:     %.loc36_42.2: type = converted %int.make_type_32.loc36, %.loc36_42.1 [template = i32]
// CHECK:STDOUT:     %.loc36_42.3: type = ptr_type i32 [template = constants.%.12]
// CHECK:STDOUT:     @AccessBaseIndirect.%return: ref %.12 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_10.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_10.2: type = converted %int.make_type_32.loc14, %.loc14_10.1 [template = i32]
// CHECK:STDOUT:   %.loc14_8: %.2 = field_decl c, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT:   .c = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc18: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_10.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_10.2: type = converted %int.make_type_32.loc20, %.loc20_10.1 [template = i32]
// CHECK:STDOUT:   %.loc20_8: %.6 = field_decl d, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_32.loc21, %.loc21_10.1 [template = i32]
// CHECK:STDOUT:   %.loc21_8: %.6 = field_decl e, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .d = %.loc20_8
// CHECK:STDOUT:   .e = %.loc21_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d: %Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc25_10: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc25_20: %.6 = name_ref d, @Derived.%.loc20_8 [template = @Derived.%.loc20_8]
// CHECK:STDOUT:   %.loc25_11.1: ref i32 = class_element_access %d.ref.loc25_10, element1
// CHECK:STDOUT:   %.loc25_11.2: i32 = bind_value %.loc25_11.1
// CHECK:STDOUT:   return %.loc25_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d: %Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, @Base.%.loc13_8 [template = @Base.%.loc13_8]
// CHECK:STDOUT:   %.loc29_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc29_11.2: ref %Base = converted %d.ref, %.loc29_11.1
// CHECK:STDOUT:   %.loc29_11.3: ref i32 = class_element_access %.loc29_11.2, element1
// CHECK:STDOUT:   %.loc29_11.4: i32 = bind_value %.loc29_11.3
// CHECK:STDOUT:   return %.loc29_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerivedIndirect(%p: %.11) -> %.12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.11 = name_ref p, %p
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:   %d.ref: %.6 = name_ref d, @Derived.%.loc20_8 [template = @Derived.%.loc20_8]
// CHECK:STDOUT:   %.loc33_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc33_12.2: ref i32 = class_element_access %.loc33_12.1, element1
// CHECK:STDOUT:   %.loc33_10: %.12 = addr_of %.loc33_12.2
// CHECK:STDOUT:   return %.loc33_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBaseIndirect(%p: %.11) -> %.12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.11 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, @Base.%.loc13_8 [template = @Base.%.loc13_8]
// CHECK:STDOUT:   %.loc37_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc37_12.2: ref %Base = class_element_access %.loc37_12.1, element0
// CHECK:STDOUT:   %.loc37_12.3: ref %Base = converted %.loc37_12.1, %.loc37_12.2
// CHECK:STDOUT:   %.loc37_12.4: ref i32 = class_element_access %.loc37_12.3, element1
// CHECK:STDOUT:   %.loc37_10: %.12 = addr_of %.loc37_12.4
// CHECK:STDOUT:   return %.loc37_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
