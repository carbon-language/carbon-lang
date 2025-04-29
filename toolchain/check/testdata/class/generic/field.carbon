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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1b2d.1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e262de.1: type = unbound_element_type %Class.fe1b2d.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac3f0.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.4339b3.1: <witness> = complete_type_witness %struct_type.x.2ac3f0.1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %Class.247 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %pattern_type.3c18fb.1: type = pattern_type %Class.fe1b2d.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4f8a42.1: <witness> = require_complete_type %Class.fe1b2d.1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %Class.fe1b2d.2: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c18fb.2: type = pattern_type %Class.fe1b2d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4aeca8.2: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %Class.elem.e262de.2: type = unbound_element_type %Class.fe1b2d.2, %U [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac3f0.2: type = struct_type {.x: %U} [symbolic]
// CHECK:STDOUT:   %complete_type.4339b3.2: <witness> = complete_type_witness %struct_type.x.2ac3f0.2 [symbolic]
// CHECK:STDOUT:   %require_complete.4f8a42.2: <witness> = require_complete_type %Class.fe1b2d.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.0fa = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.0fa = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class.247 = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc15_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.247 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @G.%pattern_type.loc19_16 (%pattern_type.3c18fb.1) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @G.%pattern_type.loc19_16 (%pattern_type.3c18fb.1) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc19_29 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc19_29 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19_32: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc19_26.2 (%Class.fe1b2d.1) = value_param call_param0
// CHECK:STDOUT:     %.loc19: type = splice_block %Class.loc19_26.1 [symbolic = %Class.loc19_26.2 (constants.%Class.fe1b2d.1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc19_25: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:       %Class.loc19_26.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc19_26.2 (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @G.%Class.loc19_26.2 (%Class.fe1b2d.1) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc19_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc19_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @H.%pattern_type.loc23_16 (%pattern_type.3c18fb.2) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @H.%pattern_type.loc23_16 (%pattern_type.3c18fb.2) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type.loc23_29 (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type.loc23_29 (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc23_32: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %U.loc23_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc23_26.2 (%Class.fe1b2d.2) = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %Class.loc23_26.1 [symbolic = %Class.loc23_26.2 (constants.%Class.fe1b2d.2)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %U.ref.loc23_25: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:       %Class.loc23_26.1: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc23_26.2 (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @H.%Class.loc23_26.2 (%Class.fe1b2d.2) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.loc23_6.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%U.loc23_6.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc11_13.2 [symbolic = %require_complete (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc11_13.2 [symbolic = %Class.elem (constants.%Class.elem.e262de.1)]
// CHECK:STDOUT:   %struct_type.x.loc13_1.2: type = struct_type {.x: @Class.%T.loc11_13.2 (%T)} [symbolic = %struct_type.x.loc13_1.2 (constants.%struct_type.x.2ac3f0.1)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness %struct_type.x.loc13_1.2 [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.4339b3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12: @Class.%Class.elem (%Class.elem.e262de.1) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %struct_type.x.loc13_1.1: type = struct_type {.x: %T} [symbolic = %struct_type.x.loc13_1.2 (constants.%struct_type.x.2ac3f0.1)]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %struct_type.x.loc13_1.1 [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.4339b3.1)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1b2d.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.247 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.2d8 = name_ref x, @Class.%.loc12 [concrete = @Class.%.loc12]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   return %.loc16_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:   %Class.loc19_26.2: type = class_type @Class, @Class(%T.loc19_6.2) [symbolic = %Class.loc19_26.2 (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %pattern_type.loc19_16: type = pattern_type %Class.loc19_26.2 [symbolic = %pattern_type.loc19_16 (constants.%pattern_type.3c18fb.1)]
// CHECK:STDOUT:   %pattern_type.loc19_29: type = pattern_type %T.loc19_6.2 [symbolic = %pattern_type.loc19_29 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19: <witness> = require_complete_type %Class.loc19_26.2 [symbolic = %require_complete.loc19 (constants.%require_complete.4f8a42.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc19_26.2, %T.loc19_6.2 [symbolic = %Class.elem (constants.%Class.elem.e262de.1)]
// CHECK:STDOUT:   %require_complete.loc20: <witness> = require_complete_type %T.loc19_6.2 [symbolic = %require_complete.loc20 (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc19_26.2 (%Class.fe1b2d.1)) -> @G.%T.loc19_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc19_26.2 (%Class.fe1b2d.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.e262de.1) = name_ref x, @Class.%.loc12 [concrete = @Class.%.loc12]
// CHECK:STDOUT:     %.loc20_11.1: ref @G.%T.loc19_6.2 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc20_11.2: @G.%T.loc19_6.2 (%T) = bind_value %.loc20_11.1
// CHECK:STDOUT:     return %.loc20_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc23_6.1: type) {
// CHECK:STDOUT:   %U.loc23_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:   %Class.loc23_26.2: type = class_type @Class, @Class(%U.loc23_6.2) [symbolic = %Class.loc23_26.2 (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:   %pattern_type.loc23_16: type = pattern_type %Class.loc23_26.2 [symbolic = %pattern_type.loc23_16 (constants.%pattern_type.3c18fb.2)]
// CHECK:STDOUT:   %pattern_type.loc23_29: type = pattern_type %U.loc23_6.2 [symbolic = %pattern_type.loc23_29 (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc23_29: <witness> = require_complete_type %U.loc23_6.2 [symbolic = %require_complete.loc23_29 (constants.%require_complete.4aeca8.2)]
// CHECK:STDOUT:   %require_complete.loc23_17: <witness> = require_complete_type %Class.loc23_26.2 [symbolic = %require_complete.loc23_17 (constants.%require_complete.4f8a42.2)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc23_26.2, %U.loc23_6.2 [symbolic = %Class.elem (constants.%Class.elem.e262de.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc23_26.2 (%Class.fe1b2d.2)) -> @H.%U.loc23_6.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc23_26.2 (%Class.fe1b2d.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.e262de.2) = name_ref x, @Class.%.loc12 [concrete = @Class.%.loc12]
// CHECK:STDOUT:     %.loc24_11.1: ref @H.%U.loc23_6.2 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc24_11.2: @H.%U.loc23_6.2 (%U) = bind_value %.loc24_11.1
// CHECK:STDOUT:     return %.loc24_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.1
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e262de.1
// CHECK:STDOUT:   %struct_type.x.loc13_1.2 => constants.%struct_type.x.2ac3f0.1
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.4339b3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %struct_type.x.loc13_1.2 => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T
// CHECK:STDOUT:   %Class.loc19_26.2 => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %pattern_type.loc19_16 => constants.%pattern_type.3c18fb.1
// CHECK:STDOUT:   %pattern_type.loc19_29 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.loc19_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.2
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e262de.2
// CHECK:STDOUT:   %struct_type.x.loc13_1.2 => constants.%struct_type.x.2ac3f0.2
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.4339b3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc23_6.2 => constants.%U
// CHECK:STDOUT:   %Class.loc23_26.2 => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %pattern_type.loc23_16 => constants.%pattern_type.3c18fb.2
// CHECK:STDOUT:   %pattern_type.loc23_29 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@H.%U.loc23_6.2) {}
// CHECK:STDOUT:
