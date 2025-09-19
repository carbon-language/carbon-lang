// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon

// --- field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  return c.x;
}

fn G(T:! Core.Copy, c: Class(T)) -> T {
  return c.x;
}

fn H(U:! Core.Copy, c: Class(U)) -> U {
  return c.x;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T.8b3) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Class.elem.e26: type = unbound_element_type %Class.fe1, %T.8b3 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T.8b3} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %require_complete.1cd77c.1: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15cec.1: <witness> = lookup_impl_witness %T.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427cc3.1: type = fn_type_with_self_type %Copy.Op.type, %T.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168dc4.1: %.427cc3.1 = impl_witness_access %Copy.lookup_impl_witness.e15cec.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cedd6.1: <specific function> = specific_impl_function %impl.elem0.168dc4.1, @Copy.Op(%T.be8) [symbolic]
// CHECK:STDOUT:   %pattern_type.17e4b7.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.09119d.1: type = class_type @Class, @Class(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %pattern_type.4bf21b.1: type = pattern_type %Class.09119d.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem.bb02b4.1: type = unbound_element_type %Class.09119d.1, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.x.bd04be.1: type = struct_type {.x: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.ca9a75.1: <witness> = complete_type_witness %struct_type.x.bd04be.1 [symbolic]
// CHECK:STDOUT:   %require_complete.34f946.1: <witness> = require_complete_type %Class.09119d.1 [symbolic]
// CHECK:STDOUT:   %U.be8: %Copy.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.2a0: type = symbolic_binding_type U, 0, %U.be8 [symbolic]
// CHECK:STDOUT:   %Class.09119d.2: type = class_type @Class, @Class(%U.binding.as_type.2a0) [symbolic]
// CHECK:STDOUT:   %pattern_type.4bf21b.2: type = pattern_type %Class.09119d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.17e4b7.3: type = pattern_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1cd77c.2: <witness> = require_complete_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %Class.elem.bb02b4.2: type = unbound_element_type %Class.09119d.2, %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.bd04be.2: type = struct_type {.x: %U.binding.as_type.2a0} [symbolic]
// CHECK:STDOUT:   %complete_type.ca9a75.2: <witness> = complete_type_witness %struct_type.x.bd04be.2 [symbolic]
// CHECK:STDOUT:   %require_complete.34f946.2: <witness> = require_complete_type %Class.09119d.2 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15cec.2: <witness> = lookup_impl_witness %U.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427cc3.2: type = fn_type_with_self_type %Copy.Op.type, %U.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168dc4.2: %.427cc3.2 = impl_witness_access %Copy.lookup_impl_witness.e15cec.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cedd6.2: <specific function> = specific_impl_function %impl.elem0.168dc4.2, @Copy.Op(%U.be8) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.0fa = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.0fa = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class.247 = value_param call_param0
// CHECK:STDOUT:     %.loc9: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc9_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.247 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.322 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @G.%pattern_type.loc13_21 (%pattern_type.4bf21b.1) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @G.%pattern_type.loc13_21 (%pattern_type.4bf21b.1) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc13_34 (%pattern_type.17e4b7.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc13_34 (%pattern_type.17e4b7.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_37: %Copy.type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc13_37: type = facet_access_type %T.ref.loc13_37 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc13_37: type = converted %T.ref.loc13_37, %T.as_type.loc13_37 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc13_14: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.1 (constants.%T.be8)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc13_31.1 (%Class.09119d.1) = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %Class.loc13_31.2 [symbolic = %Class.loc13_31.1 (constants.%Class.09119d.1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc13_30: %Copy.type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc13_31: type = facet_access_type %T.ref.loc13_30 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc13_31.2: type = converted %T.ref.loc13_30, %T.as_type.loc13_31 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Class.loc13_31.2: type = class_type @Class, @Class(constants.%T.binding.as_type) [symbolic = %Class.loc13_31.1 (constants.%Class.09119d.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @G.%Class.loc13_31.1 (%Class.09119d.1) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %U.patt: %pattern_type.322 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %c.patt: @H.%pattern_type.loc17_21 (%pattern_type.4bf21b.2) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @H.%pattern_type.loc17_21 (%pattern_type.4bf21b.2) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type.loc17_34 (%pattern_type.17e4b7.3) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type.loc17_34 (%pattern_type.17e4b7.3) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc17_37: %Copy.type = name_ref U, %U.loc17_6.2 [symbolic = %U.loc17_6.1 (constants.%U.be8)]
// CHECK:STDOUT:     %U.as_type.loc17_37: type = facet_access_type %U.ref.loc17_37 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %.loc17_37: type = converted %U.ref.loc17_37, %U.as_type.loc17_37 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %.loc17_14: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17_6.2: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc17_6.1 (constants.%U.be8)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc17_31.1 (%Class.09119d.2) = value_param call_param0
// CHECK:STDOUT:     %.loc17_31.1: type = splice_block %Class.loc17_31.2 [symbolic = %Class.loc17_31.1 (constants.%Class.09119d.2)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %U.ref.loc17_30: %Copy.type = name_ref U, %U.loc17_6.2 [symbolic = %U.loc17_6.1 (constants.%U.be8)]
// CHECK:STDOUT:       %U.as_type.loc17_31: type = facet_access_type %U.ref.loc17_30 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:       %.loc17_31.2: type = converted %U.ref.loc17_30, %U.as_type.loc17_31 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:       %Class.loc17_31.2: type = class_type @Class, @Class(constants.%U.binding.as_type.2a0) [symbolic = %Class.loc17_31.1 (constants.%Class.09119d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @H.%Class.loc17_31.1 (%Class.09119d.2) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.binding.as_type (%U.binding.as_type.2a0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%U.binding.as_type (%U.binding.as_type.2a0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc5_13.1 [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc5_13.1 (%T.8b3)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc7_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc6: @Class.%Class.elem (%Class.elem.e26) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.247 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.2d8 = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = bind_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_6.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc13_6.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.1 (constants.%T.be8)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Class.loc13_31.1: type = class_type @Class, @Class(%T.binding.as_type) [symbolic = %Class.loc13_31.1 (constants.%Class.09119d.1)]
// CHECK:STDOUT:   %pattern_type.loc13_21: type = pattern_type %Class.loc13_31.1 [symbolic = %pattern_type.loc13_21 (constants.%pattern_type.4bf21b.1)]
// CHECK:STDOUT:   %pattern_type.loc13_34: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc13_34 (constants.%pattern_type.17e4b7.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_34: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc13_34 (constants.%require_complete.1cd77c.1)]
// CHECK:STDOUT:   %require_complete.loc13_22: <witness> = require_complete_type %Class.loc13_31.1 [symbolic = %require_complete.loc13_22 (constants.%require_complete.34f946.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc13_31.1, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.bb02b4.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_6.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15cec.1)]
// CHECK:STDOUT:   %.loc14_11.4: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc13_6.1 [symbolic = %.loc14_11.4 (constants.%.427cc3.1)]
// CHECK:STDOUT:   %impl.elem0.loc14_11.2: @G.%.loc14_11.4 (%.427cc3.1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_11.2 (constants.%impl.elem0.168dc4.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_11.2: <specific function> = specific_impl_function %impl.elem0.loc14_11.2, @Copy.Op(%T.loc13_6.1) [symbolic = %specific_impl_fn.loc14_11.2 (constants.%specific_impl_fn.2cedd6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc13_31.1 (%Class.09119d.1)) -> %return.param: @G.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc13_31.1 (%Class.09119d.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.bb02b4.1) = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:     %.loc14_11.1: ref @G.%T.binding.as_type (%T.binding.as_type) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc14_11.2: @G.%T.binding.as_type (%T.binding.as_type) = bind_value %.loc14_11.1
// CHECK:STDOUT:     %impl.elem0.loc14_11.1: @G.%.loc14_11.4 (%.427cc3.1) = impl_witness_access constants.%Copy.lookup_impl_witness.e15cec.1, element0 [symbolic = %impl.elem0.loc14_11.2 (constants.%impl.elem0.168dc4.1)]
// CHECK:STDOUT:     %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_11.1: <specific function> = specific_impl_function %impl.elem0.loc14_11.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc14_11.2 (constants.%specific_impl_fn.2cedd6.1)]
// CHECK:STDOUT:     %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_impl_fn.loc14_11.1
// CHECK:STDOUT:     %.loc13_34: ref @G.%T.binding.as_type (%T.binding.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc14_11.3: init @G.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc14_11.2(%.loc14_11.2) to %.loc13_34
// CHECK:STDOUT:     return %.loc14_11.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc17_6.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc17_6.1: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc17_6.1 (constants.%U.be8)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc17_6.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:   %Class.loc17_31.1: type = class_type @Class, @Class(%U.binding.as_type) [symbolic = %Class.loc17_31.1 (constants.%Class.09119d.2)]
// CHECK:STDOUT:   %pattern_type.loc17_21: type = pattern_type %Class.loc17_31.1 [symbolic = %pattern_type.loc17_21 (constants.%pattern_type.4bf21b.2)]
// CHECK:STDOUT:   %pattern_type.loc17_34: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc17_34 (constants.%pattern_type.17e4b7.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_34: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc17_34 (constants.%require_complete.1cd77c.2)]
// CHECK:STDOUT:   %require_complete.loc17_22: <witness> = require_complete_type %Class.loc17_31.1 [symbolic = %require_complete.loc17_22 (constants.%require_complete.34f946.2)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc17_31.1, %U.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.bb02b4.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc17_6.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15cec.2)]
// CHECK:STDOUT:   %.loc18_11.4: type = fn_type_with_self_type constants.%Copy.Op.type, %U.loc17_6.1 [symbolic = %.loc18_11.4 (constants.%.427cc3.2)]
// CHECK:STDOUT:   %impl.elem0.loc18_11.2: @H.%.loc18_11.4 (%.427cc3.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_11.2: <specific function> = specific_impl_function %impl.elem0.loc18_11.2, @Copy.Op(%U.loc17_6.1) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc17_31.1 (%Class.09119d.2)) -> %return.param: @H.%U.binding.as_type (%U.binding.as_type.2a0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc17_31.1 (%Class.09119d.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.bb02b4.2) = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:     %.loc18_11.1: ref @H.%U.binding.as_type (%U.binding.as_type.2a0) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc18_11.2: @H.%U.binding.as_type (%U.binding.as_type.2a0) = bind_value %.loc18_11.1
// CHECK:STDOUT:     %impl.elem0.loc18_11.1: @H.%.loc18_11.4 (%.427cc3.2) = impl_witness_access constants.%Copy.lookup_impl_witness.e15cec.2, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:     %bound_method.loc18_11.1: <bound method> = bound_method %.loc18_11.2, %impl.elem0.loc18_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_11.1: <specific function> = specific_impl_function %impl.elem0.loc18_11.1, @Copy.Op(constants.%U.be8) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:     %bound_method.loc18_11.2: <bound method> = bound_method %.loc18_11.2, %specific_impl_fn.loc18_11.1
// CHECK:STDOUT:     %.loc17_34: ref @H.%U.binding.as_type (%U.binding.as_type.2a0) = splice_block %return {}
// CHECK:STDOUT:     %.loc18_11.3: init @H.%U.binding.as_type (%U.binding.as_type.2a0) = call %bound_method.loc18_11.2(%.loc18_11.2) to %.loc17_34
// CHECK:STDOUT:     return %.loc18_11.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1cd77c.1
// CHECK:STDOUT:   %Class => constants.%Class.09119d.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.bb02b4.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.bd04be.1
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.ca9a75.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T.be8
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Class.loc13_31.1 => constants.%Class.09119d.1
// CHECK:STDOUT:   %pattern_type.loc13_21 => constants.%pattern_type.4bf21b.1
// CHECK:STDOUT:   %pattern_type.loc13_34 => constants.%pattern_type.17e4b7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U.binding.as_type.2a0) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%U.binding.as_type.2a0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1cd77c.2
// CHECK:STDOUT:   %Class => constants.%Class.09119d.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.bb02b4.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.bd04be.2
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.ca9a75.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U.be8) {
// CHECK:STDOUT:   %U.loc17_6.1 => constants.%U.be8
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.2a0
// CHECK:STDOUT:   %Class.loc17_31.1 => constants.%Class.09119d.2
// CHECK:STDOUT:   %pattern_type.loc17_21 => constants.%pattern_type.4bf21b.2
// CHECK:STDOUT:   %pattern_type.loc17_34 => constants.%pattern_type.17e4b7.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
