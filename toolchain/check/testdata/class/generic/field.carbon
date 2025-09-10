// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1b2d.1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e262de.1: type = unbound_element_type %Class.fe1b2d.1, %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1b2d.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac3f0.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.4339b3.1: <witness> = complete_type_witness %struct_type.x.2ac3f0.1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.0fa = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.0fa = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class.247 = value_param call_param0
// CHECK:STDOUT:     %.loc19: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc19_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.247 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @G.%pattern_type.loc23_16 (%pattern_type.3c18fb.1) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @G.%pattern_type.loc23_16 (%pattern_type.3c18fb.1) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc23_29 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc23_29 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc23_32: type = name_ref T, %T.loc23_6.2 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc23_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc23_26.1 (%Class.fe1b2d.1) = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %Class.loc23_26.2 [symbolic = %Class.loc23_26.1 (constants.%Class.fe1b2d.1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc23_25: type = name_ref T, %T.loc23_6.2 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:       %Class.loc23_26.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc23_26.1 (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @G.%Class.loc23_26.1 (%Class.fe1b2d.1) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc23_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc23_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @H.%pattern_type.loc27_16 (%pattern_type.3c18fb.2) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @H.%pattern_type.loc27_16 (%pattern_type.3c18fb.2) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type.loc27_29 (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type.loc27_29 (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc27_32: type = name_ref U, %U.loc27_6.2 [symbolic = %U.loc27_6.1 (constants.%U)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc27_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc27_6.1 (constants.%U)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc27_26.1 (%Class.fe1b2d.2) = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %Class.loc27_26.2 [symbolic = %Class.loc27_26.1 (constants.%Class.fe1b2d.2)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %U.ref.loc27_25: type = name_ref U, %U.loc27_6.2 [symbolic = %U.loc27_6.1 (constants.%U)]
// CHECK:STDOUT:       %Class.loc27_26.2: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc27_26.1 (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @H.%Class.loc27_26.1 (%Class.fe1b2d.2) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.loc27_6.1 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%U.loc27_6.1 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc15_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc15_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1b2d.1)] {
// CHECK:STDOUT:         %.loc15_23.3: type = specific_constant constants.%Class.fe1b2d.1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc15_23.3 [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.loc15_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_13.1 [symbolic = %require_complete (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc15_13.1) [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc15_13.1 [symbolic = %Class.elem (constants.%Class.elem.e262de.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc15_13.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac3f0.1)]
// CHECK:STDOUT:   %complete_type.loc17_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.4339b3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_13.2 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc16: @Class.%Class.elem (%Class.elem.e262de.1) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.fe1b2d.1 [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type.loc17_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac3f0.1 [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.4339b3.1)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc17_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1b2d.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.247 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.2d8 = name_ref x, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:   %.loc20_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc20_11.2: %i32 = bind_value %.loc20_11.1
// CHECK:STDOUT:   return %.loc20_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc23_6.2: type) {
// CHECK:STDOUT:   %T.loc23_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:   %Class.loc23_26.1: type = class_type @Class, @Class(%T.loc23_6.1) [symbolic = %Class.loc23_26.1 (constants.%Class.fe1b2d.1)]
// CHECK:STDOUT:   %pattern_type.loc23_16: type = pattern_type %Class.loc23_26.1 [symbolic = %pattern_type.loc23_16 (constants.%pattern_type.3c18fb.1)]
// CHECK:STDOUT:   %pattern_type.loc23_29: type = pattern_type %T.loc23_6.1 [symbolic = %pattern_type.loc23_29 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc23: <witness> = require_complete_type %Class.loc23_26.1 [symbolic = %require_complete.loc23 (constants.%require_complete.4f8a42.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc23_26.1, %T.loc23_6.1 [symbolic = %Class.elem (constants.%Class.elem.e262de.1)]
// CHECK:STDOUT:   %require_complete.loc24: <witness> = require_complete_type %T.loc23_6.1 [symbolic = %require_complete.loc24 (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc23_26.1 (%Class.fe1b2d.1)) -> @G.%T.loc23_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc23_26.1 (%Class.fe1b2d.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.e262de.1) = name_ref x, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:     %.loc24_11.1: ref @G.%T.loc23_6.1 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc24_11.2: @G.%T.loc23_6.1 (%T) = bind_value %.loc24_11.1
// CHECK:STDOUT:     return %.loc24_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc27_6.2: type) {
// CHECK:STDOUT:   %U.loc27_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc27_6.1 (constants.%U)]
// CHECK:STDOUT:   %Class.loc27_26.1: type = class_type @Class, @Class(%U.loc27_6.1) [symbolic = %Class.loc27_26.1 (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:   %pattern_type.loc27_16: type = pattern_type %Class.loc27_26.1 [symbolic = %pattern_type.loc27_16 (constants.%pattern_type.3c18fb.2)]
// CHECK:STDOUT:   %pattern_type.loc27_29: type = pattern_type %U.loc27_6.1 [symbolic = %pattern_type.loc27_29 (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27_29: <witness> = require_complete_type %U.loc27_6.1 [symbolic = %require_complete.loc27_29 (constants.%require_complete.4aeca8.2)]
// CHECK:STDOUT:   %require_complete.loc27_17: <witness> = require_complete_type %Class.loc27_26.1 [symbolic = %require_complete.loc27_17 (constants.%require_complete.4f8a42.2)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc27_26.1, %U.loc27_6.1 [symbolic = %Class.elem (constants.%Class.elem.e262de.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc27_26.1 (%Class.fe1b2d.2)) -> @H.%U.loc27_6.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc27_26.1 (%Class.fe1b2d.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.e262de.2) = name_ref x, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:     %.loc28_11.1: ref @H.%U.loc27_6.1 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc28_11.2: @H.%U.loc27_6.1 (%U) = bind_value %.loc28_11.1
// CHECK:STDOUT:     return %.loc28_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.1
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e262de.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2ac3f0.1
// CHECK:STDOUT:   %complete_type.loc17_1.2 => constants.%complete_type.4339b3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc17_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc23_6.1 => constants.%T
// CHECK:STDOUT:   %Class.loc23_26.1 => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %pattern_type.loc23_16 => constants.%pattern_type.3c18fb.1
// CHECK:STDOUT:   %pattern_type.loc23_29 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.2
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e262de.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2ac3f0.2
// CHECK:STDOUT:   %complete_type.loc17_1.2 => constants.%complete_type.4339b3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc27_6.1 => constants.%U
// CHECK:STDOUT:   %Class.loc27_26.1 => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %pattern_type.loc27_16 => constants.%pattern_type.3c18fb.2
// CHECK:STDOUT:   %pattern_type.loc27_29 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
