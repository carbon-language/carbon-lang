// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/base_is_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/base_is_generic.carbon

base class Base(T:! type) {
  var x: T;
}

class Param {
  var y: i32;
}

class Derived {
  extend base: Base(Param);
}

fn DoubleFieldAccess(d: Derived) -> i32 {
  return d.x.y;
}

// CHECK:STDOUT: --- base_is_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Base.1: %Base.type = struct_value () [template]
// CHECK:STDOUT:   %Base.2: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Param, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.y: i32} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base.3: type = class_type @Base, @Base(%Param) [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %Base.3, %Param [template]
// CHECK:STDOUT:   %.9: type = struct_type {.x: %Param} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.x: %.11} [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.14: type = unbound_element_type %Derived, %Base.3 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: %Base.3} [template]
// CHECK:STDOUT:   %.16: <witness> = complete_type_witness %.15 [template]
// CHECK:STDOUT:   %DoubleFieldAccess.type: type = fn_type @DoubleFieldAccess [template]
// CHECK:STDOUT:   %DoubleFieldAccess: %DoubleFieldAccess.type = struct_value () [template]
// CHECK:STDOUT:   %.17: type = struct_type {.base: %.13} [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .DoubleFieldAccess = %DoubleFieldAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [template = constants.%Base.1] {
// CHECK:STDOUT:     %T.patt.loc11_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_17.1, runtime_param<invalid> [symbolic = %T.patt.loc11_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_17.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [template = constants.%Param] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %DoubleFieldAccess.decl: %DoubleFieldAccess.type = fn_decl @DoubleFieldAccess [template = constants.%DoubleFieldAccess] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc23_37.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc23_37.2: type = converted %int.make_type_32, %.loc23_37.1 [template = i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc11_17.1: type) {
// CHECK:STDOUT:   %T.loc11_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc11_17.2) [symbolic = %Base (constants.%Base.2)]
// CHECK:STDOUT:   %.loc12_8.2: type = unbound_element_type @Base.%Base (%Base.2), @Base.%T.loc11_17.2 (%T) [symbolic = %.loc12_8.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc13_1.2: type = struct_type {.x: @Base.%T.loc11_17.2 (%T)} [symbolic = %.loc13_1.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc13_1.3: <witness> = complete_type_witness @Base.%.loc13_1.2 (%.3) [symbolic = %.loc13_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_17.1 [symbolic = %T.loc11_17.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_8.1: @Base.%.loc12_8.2 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:     %.loc13_1.1: <witness> = complete_type_witness %.3 [symbolic = %.loc13_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.2
// CHECK:STDOUT:     .x = %.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_32, %.loc16_10.1 [template = i32]
// CHECK:STDOUT:   %.loc16_8: %.5 = field_decl y, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .y = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [template = constants.%Base.1]
// CHECK:STDOUT:   %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [template = constants.%Base.3]
// CHECK:STDOUT:   %.loc20: %.14 = base_decl %Base.3, element0 [template]
// CHECK:STDOUT:   %.loc21: <witness> = complete_type_witness %.15 [template = constants.%.16]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   extend name_scope2, specific2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoubleFieldAccess(%d.param_patt: %Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %.8 = name_ref x, @Base.%.loc12_8.1 [template = @Base.%.loc12_8.1]
// CHECK:STDOUT:   %.loc24_11.1: ref %Base.3 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc24_11.2: ref %Base.3 = converted %d.ref, %.loc24_11.1
// CHECK:STDOUT:   %.loc24_11.3: ref %Param = class_element_access %.loc24_11.2, element0
// CHECK:STDOUT:   %y.ref: %.5 = name_ref y, @Param.%.loc16_8 [template = @Param.%.loc16_8]
// CHECK:STDOUT:   %.loc24_13.1: ref i32 = class_element_access %.loc24_11.3, element0
// CHECK:STDOUT:   %.loc24_13.2: i32 = bind_value %.loc24_13.1
// CHECK:STDOUT:   return %.loc24_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@Base.%T.loc11_17.2) {
// CHECK:STDOUT:   %T.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T.loc11_17.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc11_17.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base => constants.%Base.3
// CHECK:STDOUT:   %.loc12_8.2 => constants.%.8
// CHECK:STDOUT:   %.loc13_1.2 => constants.%.9
// CHECK:STDOUT:   %.loc13_1.3 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
