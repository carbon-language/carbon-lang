// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/class/base.carbon

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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.b: i32} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type Derived, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: Base, .d: i32} [template]
// CHECK:STDOUT:   %Make: type = fn_type @Make [template]
// CHECK:STDOUT:   %struct.2: Make = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: {.b: i32}*, .d: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.base: {.b: i32}*, .d: i32} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.base: Base, .d: i32} [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: {.b: i32}, .d: i32} [template]
// CHECK:STDOUT:   %struct.3: Base = struct_value (%.11) [template]
// CHECK:STDOUT:   %struct.4: Derived = struct_value (%struct.3, %.12) [template]
// CHECK:STDOUT:   %.14: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.15: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %Access: type = fn_type @Access [template]
// CHECK:STDOUT:   %struct.5: Access = struct_value () [template]
// CHECK:STDOUT:   %.16: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Make.decl: Make = fn_decl @Make [template = constants.%struct.2] {
// CHECK:STDOUT:     %Derived.ref.loc21: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     @Make.%return: ref Derived = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Access.decl: Access = fn_decl @Access [template = constants.%struct.5] {
// CHECK:STDOUT:     %Derived.ref.loc25: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc25_11.1: Derived = param d
// CHECK:STDOUT:     @Access.%d: Derived = bind_name d, %d.loc25_11.1
// CHECK:STDOUT:     %int.make_type_32.loc25_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc25_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc25_35.1: (type, type) = tuple_literal (%int.make_type_32.loc25_27, %int.make_type_32.loc25_32)
// CHECK:STDOUT:     %.loc25_35.2: type = value_of_initializer %int.make_type_32.loc25_27 [template = i32]
// CHECK:STDOUT:     %.loc25_35.3: type = converted %int.make_type_32.loc25_27, %.loc25_35.2 [template = i32]
// CHECK:STDOUT:     %.loc25_35.4: type = value_of_initializer %int.make_type_32.loc25_32 [template = i32]
// CHECK:STDOUT:     %.loc25_35.5: type = converted %int.make_type_32.loc25_32, %.loc25_35.4 [template = i32]
// CHECK:STDOUT:     %.loc25_35.6: type = converted %.loc25_35.1, constants.%.15 [template = constants.%.15]
// CHECK:STDOUT:     @Access.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: <unbound element of class Base> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc16: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_32, %.loc18_10.1 [template = i32]
// CHECK:STDOUT:   %.loc18_8: <unbound element of class Derived> = field_decl d, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .d = %.loc18_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_25: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc22_26.1: {.b: i32} = struct_literal (%.loc22_25)
// CHECK:STDOUT:   %.loc22_34: i32 = int_literal 7 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_35.1: {.base: {.b: i32}, .d: i32} = struct_literal (%.loc22_26.1, %.loc22_34)
// CHECK:STDOUT:   %.loc22_35.2: ref Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc22_26.2: ref i32 = class_element_access %.loc22_35.2, element0
// CHECK:STDOUT:   %.loc22_26.3: init i32 = initialize_from %.loc22_25 to %.loc22_26.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc22_26.4: init Base = class_init (%.loc22_26.3), %.loc22_35.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc22_35.3: init Base = converted %.loc22_26.1, %.loc22_26.4 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc22_35.4: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc22_35.5: init i32 = initialize_from %.loc22_34 to %.loc22_35.4 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_35.6: init Derived = class_init (%.loc22_35.3, %.loc22_35.5), %return [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc22_36: init Derived = converted %.loc22_35.1, %.loc22_35.6 [template = constants.%struct.4]
// CHECK:STDOUT:   return %.loc22_36 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: Derived) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc26_11: Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc26_12: <unbound element of class Derived> = name_ref d, @Derived.%.loc18_8 [template = @Derived.%.loc18_8]
// CHECK:STDOUT:   %.loc26_12.1: ref i32 = class_element_access %d.ref.loc26_11, element1
// CHECK:STDOUT:   %.loc26_12.2: i32 = bind_value %.loc26_12.1
// CHECK:STDOUT:   %d.ref.loc26_16: Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: <unbound element of class Derived> = name_ref base, @Derived.%.loc16 [template = @Derived.%.loc16]
// CHECK:STDOUT:   %.loc26_17.1: ref Base = class_element_access %d.ref.loc26_16, element0
// CHECK:STDOUT:   %.loc26_17.2: Base = bind_value %.loc26_17.1
// CHECK:STDOUT:   %b.ref: <unbound element of class Base> = name_ref b, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc26_22.1: ref i32 = class_element_access %.loc26_17.2, element0
// CHECK:STDOUT:   %.loc26_22.2: i32 = bind_value %.loc26_22.1
// CHECK:STDOUT:   %.loc26_24.1: (i32, i32) = tuple_literal (%.loc26_12.2, %.loc26_22.2)
// CHECK:STDOUT:   %.loc26_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc26_24.3: init i32 = initialize_from %.loc26_12.2 to %.loc26_24.2
// CHECK:STDOUT:   %.loc26_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc26_24.5: init i32 = initialize_from %.loc26_22.2 to %.loc26_24.4
// CHECK:STDOUT:   %.loc26_24.6: init (i32, i32) = tuple_init (%.loc26_24.3, %.loc26_24.5) to %return
// CHECK:STDOUT:   %.loc26_25: init (i32, i32) = converted %.loc26_24.1, %.loc26_24.6
// CHECK:STDOUT:   return %.loc26_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
