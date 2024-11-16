// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var k: T;
}

fn InitFromStructGeneric(T:! type, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

fn InitFromAdaptedGeneric(T:! type, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [template]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.5 [template]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [template]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%i32) [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Class.3, %i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.k: %i32} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .InitFromStructGeneric = %InitFromStructGeneric.decl
// CHECK:STDOUT:     .InitFromStructSpecific = %InitFromStructSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [template = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_26.1, runtime_param<invalid> [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%T.loc8_26.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%T.loc8_26.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%T.loc8_26.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%T.loc8_26.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_39: type = name_ref T, %T.loc8_26.1 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_45: type = name_ref T, %T.loc8_26.1 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.loc8_26.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.loc8_26.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.loc8_26.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.loc8_26.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [template = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.patt: %i32 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %i32 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_30.1: Core.IntLiteral = int_value 32 [template = constants.%.5]
// CHECK:STDOUT:     %int.make_type_signed.loc13_30: init type = call constants.%Int(%.loc13_30.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_30.2: type = value_of_initializer %int.make_type_signed.loc13_30 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_30.3: type = converted %int.make_type_signed.loc13_30, %.loc13_30.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_38.1: Core.IntLiteral = int_value 32 [template = constants.%.5]
// CHECK:STDOUT:     %int.make_type_signed.loc13_38: init type = call constants.%Int(%.loc13_38.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_38.2: type = value_of_initializer %int.make_type_signed.loc13_38 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_38.3: type = converted %int.make_type_signed.loc13_38, %.loc13_38.2 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc5_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc4_13.2 (%T) [symbolic = %.loc5_8.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc6_1.2: type = struct_type {.k: @Class.%T.loc4_13.2 (%T)} [symbolic = %.loc6_1.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc6_1.3: <witness> = complete_type_witness @Class.%.loc6_1.2 (%.2) [symbolic = %.loc6_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc5_8.1: @Class.%.loc5_8.2 (%.1) = field_decl k, element0 [template]
// CHECK:STDOUT:     %.loc6_1.1: <witness> = complete_type_witness %.2 [symbolic = %.loc6_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .k = %.loc5_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc8_26.1: type) {
// CHECK:STDOUT:   %T.loc8_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.loc9_15.2: type = class_type @Class, @Class(%T.loc8_26.2) [symbolic = %Class.loc9_15.2 (constants.%Class.2)]
// CHECK:STDOUT:   %.loc9_28.5: type = struct_type {.k: @InitFromStructGeneric.%T.loc8_26.2 (%T)} [symbolic = %.loc9_28.5 (constants.%.2)]
// CHECK:STDOUT:   %.loc10_11.3: type = unbound_element_type @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2), @InitFromStructGeneric.%T.loc8_26.2 (%T) [symbolic = %.loc10_11.3 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @InitFromStructGeneric.%T.loc8_26.2 (%T)) -> @InitFromStructGeneric.%T.loc8_26.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_26.1 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc9_15.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc9_15.2 (constants.%Class.2)]
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2) = var v
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2) = bind_name v, %v.var
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.loc8_26.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_28.1: @InitFromStructGeneric.%.loc9_28.5 (%.2) = struct_literal (%x.ref)
// CHECK:STDOUT:     %.loc9_28.2: ref @InitFromStructGeneric.%T.loc8_26.2 (%T) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc9_28.3: init @InitFromStructGeneric.%T.loc8_26.2 (%T) = initialize_from %x.ref to %.loc9_28.2
// CHECK:STDOUT:     %.loc9_28.4: init @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2) = class_init (%.loc9_28.3), %v.var
// CHECK:STDOUT:     %.loc9_29: init @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2) = converted %.loc9_28.1, %.loc9_28.4
// CHECK:STDOUT:     assign %v.var, %.loc9_29
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc9_15.2 (%Class.2) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%.loc10_11.3 (%.1) = name_ref k, @Class.%.loc5_8.1 [template = @Class.%.loc5_8.1]
// CHECK:STDOUT:     %.loc10_11.1: ref @InitFromStructGeneric.%T.loc8_26.2 (%T) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc10_11.2: @InitFromStructGeneric.%T.loc8_26.2 (%T) = bind_value %.loc10_11.1
// CHECK:STDOUT:     return %.loc10_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_signed.loc14, %.loc14_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(constants.%i32) [template = constants.%Class.3]
// CHECK:STDOUT:   %v.var: ref %Class.3 = var v
// CHECK:STDOUT:   %v: ref %Class.3 = bind_name v, %v.var
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc14_30.1: %.7 = struct_literal (%x.ref)
// CHECK:STDOUT:   %.loc14_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 = initialize_from %x.ref to %.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.4: init %Class.3 = class_init (%.loc14_30.3), %v.var
// CHECK:STDOUT:   %.loc14_31: init %Class.3 = converted %.loc14_30.1, %.loc14_30.4
// CHECK:STDOUT:   assign %v.var, %.loc14_31
// CHECK:STDOUT:   %v.ref: ref %Class.3 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %.6 = name_ref k, @Class.%.loc5_8.1 [template = @Class.%.loc5_8.1]
// CHECK:STDOUT:   %.loc15_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc15_11.2: %i32 = bind_value %.loc15_11.1
// CHECK:STDOUT:   return %.loc15_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc5_8.2 => constants.%.1
// CHECK:STDOUT:   %.loc6_1.2 => constants.%.2
// CHECK:STDOUT:   %.loc6_1.3 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@InitFromStructGeneric.%T.loc8_26.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.loc5_8.2 => constants.%.6
// CHECK:STDOUT:   %.loc6_1.2 => constants.%.7
// CHECK:STDOUT:   %.loc6_1.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [template]
// CHECK:STDOUT:   %Adapt.1: %Adapt.type = struct_value () [template]
// CHECK:STDOUT:   %Adapt.2: type = class_type @Adapt, @Adapt(%T) [symbolic]
// CHECK:STDOUT:   %.1: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [template]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [template]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %Adapt.3: type = class_type @Adapt, @Adapt(%i32) [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapt = %Adapt.decl
// CHECK:STDOUT:     .InitFromAdaptedGeneric = %InitFromAdaptedGeneric.decl
// CHECK:STDOUT:     .InitFromAdaptedSpecific = %InitFromAdaptedSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapt.decl: %Adapt.type = class_decl @Adapt [template = constants.%Adapt.1] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [template = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_27.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_27.1, runtime_param<invalid> [symbolic = %T.patt.loc8_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_40: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_46: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_27.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [template = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.patt: %i32 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %i32 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_31.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc12_31: init type = call constants.%Int(%.loc12_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_31.2: type = value_of_initializer %int.make_type_signed.loc12_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_31.3: type = converted %int.make_type_signed.loc12_31, %.loc12_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_39.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc12_39: init type = call constants.%Int(%.loc12_39.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_39.2: type = value_of_initializer %int.make_type_signed.loc12_39 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_39.3: type = converted %int.make_type_signed.loc12_39, %.loc12_39.2 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapt(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_1.2: <witness> = complete_type_witness @Adapt.%T.loc4_13.2 (%T) [symbolic = %.loc6_1.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T
// CHECK:STDOUT:     %.loc6_1.1: <witness> = complete_type_witness %T [symbolic = %.loc6_1.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapt.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc8_27.1: type) {
// CHECK:STDOUT:   %T.loc8_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_27.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_27.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Adapt.loc9_21.2: type = class_type @Adapt, @Adapt(%T.loc8_27.2) [symbolic = %Adapt.loc9_21.2 (constants.%Adapt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T)) -> @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [template = constants.%Adapt.1]
// CHECK:STDOUT:     %T.ref.loc9_22: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %Adapt.loc9_21.1: type = class_type @Adapt, @Adapt(constants.%T) [symbolic = %Adapt.loc9_21.2 (constants.%Adapt.2)]
// CHECK:STDOUT:     %.loc9_13.1: @InitFromAdaptedGeneric.%Adapt.loc9_21.2 (%Adapt.2) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc9_13.2: @InitFromAdaptedGeneric.%Adapt.loc9_21.2 (%Adapt.2) = converted %x.ref, %.loc9_13.1
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %.loc9_26.1: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = as_compatible %.loc9_13.2
// CHECK:STDOUT:     %.loc9_26.2: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = converted %.loc9_13.2, %.loc9_26.1
// CHECK:STDOUT:     return %.loc9_26.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [template = constants.%Adapt.1]
// CHECK:STDOUT:   %.loc13_22: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc13_22: init type = call constants.%Int(%.loc13_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_signed.loc13_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_signed.loc13_22, %.loc13_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [template = constants.%Adapt.3]
// CHECK:STDOUT:   %.loc13_13.1: %Adapt.3 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc13_13.2: %Adapt.3 = converted %x.ref, %.loc13_13.1
// CHECK:STDOUT:   %.loc13_31.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc13_31: init type = call constants.%Int(%.loc13_31.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_31.2: type = value_of_initializer %int.make_type_signed.loc13_31 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_31.3: type = converted %int.make_type_signed.loc13_31, %.loc13_31.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_28.1: %i32 = as_compatible %.loc13_13.2
// CHECK:STDOUT:   %.loc13_28.2: %i32 = converted %.loc13_13.2, %.loc13_28.1
// CHECK:STDOUT:   return %.loc13_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_1.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_27.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(@InitFromAdaptedGeneric.%T.loc8_27.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_1.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
