// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon

class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  return c.x;
}

fn G(T:! type, c: Class(T)) -> T {
  return c.x;
}

fn H(U:! type, c: Class(U)) -> U {
  return c.x;
}

// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.elem.1: type = unbound_element_type %Class.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Class.elem.2: type = unbound_element_type %Class.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %Class.elem.3: type = unbound_element_type %Class.3, %U [symbolic]
// CHECK:STDOUT:   %struct_type.x.3: type = struct_type {.x: %U} [symbolic]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.x.3 [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %Class.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc15_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_15: init type = call constants.%Int(%int_32.loc15_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_18.1: type = value_of_initializer %int.make_type_signed.loc15_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_18.2: type = converted %int.make_type_signed.loc15_15, %.loc15_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [template = constants.%Class.2]
// CHECK:STDOUT:     %int_32.loc15_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_24: init type = call constants.%Int(%int_32.loc15_24) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_24.1: type = value_of_initializer %int.make_type_signed.loc15_24 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_24.2: type = converted %int.make_type_signed.loc15_24, %.loc15_24.1 [template = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc19_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc19_6.1, runtime_param<invalid> [symbolic = %T.patt.loc19_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %c.patt: @G.%Class.loc19_26.2 (%Class.1) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @G.%Class.loc19_26.2 (%Class.1) = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @G.%T.loc19_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%T.loc19_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc19_25: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc19_26.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc19_26.2 (constants.%Class.1)]
// CHECK:STDOUT:     %T.ref.loc19_32: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc19_26.2 (%Class.1) = value_param runtime_param0
// CHECK:STDOUT:     %c: @G.%Class.loc19_26.2 (%Class.1) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc19_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc19_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.patt.loc23_6.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc23_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc23_6.1, runtime_param<invalid> [symbolic = %U.patt.loc23_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %c.patt: @H.%Class.loc23_26.2 (%Class.3) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @H.%Class.loc23_26.2 (%Class.3) = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @H.%U.loc23_6.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @H.%U.loc23_6.2 (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %U.ref.loc23_25: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %Class.loc23_26.1: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc23_26.2 (constants.%Class.3)]
// CHECK:STDOUT:     %U.ref.loc23_32: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc23_6.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc23_26.2 (%Class.3) = value_param runtime_param0
// CHECK:STDOUT:     %c: @H.%Class.loc23_26.2 (%Class.3) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.loc23_6.2 (%U) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @H.%U.loc23_6.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class.1), @Class.%T.loc11_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc11_13.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12: @Class.%Class.elem (%Class.elem.1) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.1
// CHECK:STDOUT:     .x = %.loc12
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: %Class.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.2 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.2 = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   return %.loc16_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc19_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Class.loc19_26.2: type = class_type @Class, @Class(%T.loc19_6.2) [symbolic = %Class.loc19_26.2 (constants.%Class.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @G.%Class.loc19_26.2 (%Class.1), @G.%T.loc19_6.2 (%T) [symbolic = %Class.elem (constants.%Class.elem.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %c.param_patt: @G.%Class.loc19_26.2 (%Class.1)) -> @G.%T.loc19_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc19_26.2 (%Class.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.1) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:     %.loc20_11.1: ref @G.%T.loc19_6.2 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc20_11.2: @G.%T.loc19_6.2 (%T) = bind_value %.loc20_11.1
// CHECK:STDOUT:     return %.loc20_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc23_6.1: type) {
// CHECK:STDOUT:   %U.loc23_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc23_6.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc23_6.2 (constants.%U.patt)]
// CHECK:STDOUT:   %Class.loc23_26.2: type = class_type @Class, @Class(%U.loc23_6.2) [symbolic = %Class.loc23_26.2 (constants.%Class.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @H.%Class.loc23_26.2 (%Class.3), @H.%U.loc23_6.2 (%U) [symbolic = %Class.elem (constants.%Class.elem.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: type, %c.param_patt: @H.%Class.loc23_26.2 (%Class.3)) -> @H.%U.loc23_6.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc23_26.2 (%Class.3) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.3) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:     %.loc24_11.1: ref @H.%U.loc23_6.2 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc24_11.2: @H.%U.loc23_6.2 (%U) = bind_value %.loc24_11.1
// CHECK:STDOUT:     return %.loc24_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.1
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.loc19_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc19_6.2 => constants.%T
// CHECK:STDOUT:   %Class.loc19_26.2 => constants.%Class.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.3
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.3
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@H.%U.loc23_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc23_6.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc23_6.2 => constants.%U
// CHECK:STDOUT:   %Class.loc23_26.2 => constants.%Class.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
