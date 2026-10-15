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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/member_lookup.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

interface Interface(T: type) {
  let X: T*;
}

fn AccessGeneric[T: type](generic I: Interface(T)) -> T* {
  return I.X;
}

fn AccessConcrete(generic I: Interface(i32)) -> i32* {
  return I.X;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

interface Interface(T: type) {
  let X: T*;
}

fn AccessMissingGeneric[T: type](generic I: Interface(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+2]]:10-19: error: member name `nonesuch` not found in `Interface(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:
  return I.nonesuch;
}

fn AccessMissingConcrete(generic I: Interface(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+2]]:10-19: error: member name `nonesuch` not found in `Interface(i32)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:
  return I.nonesuch;
}


// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.1b1: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.128: type = facet_type <@Interface, @Interface(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.5b9: %Interface.type.128 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.f63: type = assoc_entity_type @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.f45: %Interface.assoc_type.f63 = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.a49: type = pattern_type %Interface.type.128 [symbolic]
// CHECK:STDOUT:   %I.patt.768: %pattern_type.a49 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %I.165: %Interface.type.128 = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.e4d: <witness> = require_complete_type %Interface.type.128 [symbolic]
// CHECK:STDOUT:   %I.as_type.c4b: type = facet_access_type %I.165 [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.e71: <witness> = lookup_impl_witness %I.165, @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e9d: %ptr.e8f = impl_witness_access %Interface.lookup_impl_witness.e71, element0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.66e: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e38: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.3f816: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e38, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.24a: %.3f816 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.7c0: <bound method> = bound_method %impl.elem0.e9d, %impl.elem0.24a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6b3: <specific function> = specific_impl_function %impl.elem0.24a, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %bound_method.104: <bound method> = bound_method %impl.elem0.e9d, %specific_impl_fn.6b3 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.5ec: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.5c6: type = pattern_type %Interface.type.5ec [concrete]
// CHECK:STDOUT:   %I.patt.e71: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %I.eef: %Interface.type.5ec = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.43c: Core.Form = init_form %ptr.a17 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %return.param_patt.6e5: %pattern_type.7dc = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.1b4: %pattern_type.7dc = return_slot_pattern %return.param_patt.6e5, %ptr.a17 [concrete]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.91c: %Interface.type.5ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.2d5: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.8c4: %Interface.assoc_type.2d5 = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.bed: type = facet_access_type %I.eef [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.a30: <witness> = lookup_impl_witness %I.eef, @Interface, @Interface(%i32) [symbolic]
// CHECK:STDOUT:   %impl.elem0.085: %ptr.a17 = impl_witness_access %Interface.lookup_impl_witness.a30, element0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.81f: %ptr.as.Copy.impl.Op.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.652: %Copy.type = facet_value %ptr.a17, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.652) [concrete]
// CHECK:STDOUT:   %.309: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9a2, %Copy.facet.652 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.085, %ptr.as.Copy.impl.Op.81f [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.81f, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.d80: <bound method> = bound_method %impl.elem0.085, %ptr.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessGeneric = %AccessGeneric.decl
// CHECK:STDOUT:     .AccessConcrete = %AccessConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.1b1 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_22.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24.1: type = splice_block %.loc4_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_19.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_19.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:     %I.patt.loc8_36.1: @AccessGeneric.%pattern_type.loc8_36 (%pattern_type.a49) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_36.2 (constants.%I.patt.768)]
// CHECK:STDOUT:     %return.param_patt.loc8_56.1: @AccessGeneric.%pattern_type.loc8_56 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc8_56.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:     %return.patt.loc8_52.1: @AccessGeneric.%pattern_type.loc8_56 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc8_56.1, %ptr.loc8_56.2 [symbolic = %return.patt.loc8_52.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_55: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc8_56.2: type = ptr_type %T.ref.loc8_55 [symbolic = %ptr.loc8_56.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc8_56.2: Core.Form = init_form %ptr.loc8_56.2 [symbolic = %.loc8_56.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc8_21.1: type = splice_block %.loc8_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_19: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc8_49: type = splice_block %Interface.type.loc8_49.2 [symbolic = %Interface.type.loc8_49.1 (constants.%Interface.type.128)] {
// CHECK:STDOUT:       %.Self.frozen.loc8_36: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_48: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:       %Interface.type.loc8_49.2: type = facet_type <@Interface, @Interface(constants.%T.67d)> [symbolic = %Interface.type.loc8_49.1 (constants.%Interface.type.128)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_36.2: @AccessGeneric.%Interface.type.loc8_49.1 (%Interface.type.128) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.165)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt.loc12_28.1: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_28.2 (constants.%I.patt.e71)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7dc = out_param_pattern [concrete = constants.%return.param_patt.6e5]
// CHECK:STDOUT:     %return.patt: %pattern_type.7dc = return_slot_pattern %return.param_patt, %ptr [concrete = constants.%return.patt.1b4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc12_49 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:     %.loc12_52: Core.Form = init_form %ptr [concrete = constants.%.43c]
// CHECK:STDOUT:     %.loc12_43: type = splice_block %Interface.type [concrete = constants.%Interface.type.5ec] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc12_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.5ec]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc12_28.2: %Interface.type.5ec = symbolic_binding I, 0 [symbolic = %I.loc12_28.1 (constants.%I.eef)]
// CHECK:STDOUT:     %return.param: ref %ptr.a17 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.a17 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_22.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.128)]
// CHECK:STDOUT:   %Self.loc4_30.2: @Interface.%Interface.type (%Interface.type.128) = symbolic_binding Self, 1 [symbolic = %Self.loc4_30.2 (constants.%Self.5b9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_30.1: @Interface.%Interface.type (%Interface.type.128) = symbolic_binding Self, 1 [symbolic = %Self.loc4_30.2 (constants.%Self.5b9)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.f63) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.f45)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_30.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Interface.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_19.2: type, %I.loc8_36.2: @AccessGeneric.%Interface.type.loc8_49.1 (%Interface.type.128)) {
// CHECK:STDOUT:   %T.patt.loc8_19.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_19.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc8_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:   %Interface.type.loc8_49.1: type = facet_type <@Interface, @Interface(%T.loc8_19.1)> [symbolic = %Interface.type.loc8_49.1 (constants.%Interface.type.128)]
// CHECK:STDOUT:   %pattern_type.loc8_36: type = pattern_type %Interface.type.loc8_49.1 [symbolic = %pattern_type.loc8_36 (constants.%pattern_type.a49)]
// CHECK:STDOUT:   %I.patt.loc8_36.2: @AccessGeneric.%pattern_type.loc8_36 (%pattern_type.a49) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_36.2 (constants.%I.patt.768)]
// CHECK:STDOUT:   %I.loc8_36.1: @AccessGeneric.%Interface.type.loc8_49.1 (%Interface.type.128) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.165)]
// CHECK:STDOUT:   %ptr.loc8_56.1: type = ptr_type %T.loc8_19.1 [symbolic = %ptr.loc8_56.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc8_56.1: Core.Form = init_form %ptr.loc8_56.1 [symbolic = %.loc8_56.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc8_56: type = pattern_type %ptr.loc8_56.1 [symbolic = %pattern_type.loc8_56 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %return.param_patt.loc8_56.2: @AccessGeneric.%pattern_type.loc8_56 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc8_56.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:   %return.patt.loc8_52.2: @AccessGeneric.%pattern_type.loc8_56 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc8_56.2, %ptr.loc8_56.1 [symbolic = %return.patt.loc8_52.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %ptr.loc8_56.1 [symbolic = %require_complete.loc8 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Interface.type.loc8_49.1 [symbolic = %require_complete.loc9 (constants.%require_complete.e4d)]
// CHECK:STDOUT:   %I.as_type.loc9_11.2: type = facet_access_type %I.loc8_36.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.f63)]
// CHECK:STDOUT:   %assoc0: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.f63) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.f45)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc8_36.1, @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.e71)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.3: @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.e9d)]
// CHECK:STDOUT:   %.loc9_11.5: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc8_19.1) [symbolic = %.loc9_11.5 (constants.%.66e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc8_56.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc9_11.3: %Copy.type = facet_value %ptr.loc8_56.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc9_11.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e38)]
// CHECK:STDOUT:   %.loc9_11.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc9_11.3 [symbolic = %.loc9_11.6 (constants.%.3f816)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.4: @AccessGeneric.%.loc9_11.6 (%.3f816) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.24a)]
// CHECK:STDOUT:   %bound_method.loc9_11.3: <bound method> = bound_method %impl.elem0.loc9_11.3, %impl.elem0.loc9_11.4 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.7c0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.4, @Copy.WithSelf.Op(%Copy.facet.loc9_11.3) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.6b3)]
// CHECK:STDOUT:   %bound_method.loc9_11.4: <bound method> = bound_method %impl.elem0.loc9_11.3, %specific_impl_fn.loc9_11.2 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.104)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_49.1 (%Interface.type.128) = name_ref I, %I.loc8_36.2 [symbolic = %I.loc8_36.1 (constants.%I.165)]
// CHECK:STDOUT:     %I.as_type.loc9_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted %I.ref, %I.as_type.loc9_11.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:     %.loc9_11.2: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.f63) = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%T.67d, constants.%I.165) [symbolic = %assoc0 (constants.%assoc0.f45)]
// CHECK:STDOUT:     %X.ref: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.f63) = name_ref X, %.loc9_11.2 [symbolic = %assoc0 (constants.%assoc0.f45)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) = impl_witness_access constants.%Interface.lookup_impl_witness.e71, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.e9d)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.2: @AccessGeneric.%.loc9_11.6 (%.3f816) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.24a)]
// CHECK:STDOUT:     %bound_method.loc9_11.1: <bound method> = bound_method %impl.elem0.loc9_11.1, %impl.elem0.loc9_11.2 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.7c0)]
// CHECK:STDOUT:     %Copy.facet.loc9_11.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_11.3: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_11.1 [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Copy.facet.loc9_11.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_11.4: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_11.2 [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Copy.WithSelf.Op(constants.%Copy.facet.302) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.6b3)]
// CHECK:STDOUT:     %bound_method.loc9_11.2: <bound method> = bound_method %impl.elem0.loc9_11.1, %specific_impl_fn.loc9_11.1 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.104)]
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @AccessGeneric.%ptr.loc8_56.1 (%ptr.e8f) = call %bound_method.loc9_11.2(%impl.elem0.loc9_11.1)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc12_28.2: %Interface.type.5ec) {
// CHECK:STDOUT:   %I.patt.loc12_28.2: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_28.2 (constants.%I.patt.e71)]
// CHECK:STDOUT:   %I.loc12_28.1: %Interface.type.5ec = symbolic_binding I, 0 [symbolic = %I.loc12_28.1 (constants.%I.eef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc12_28.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc12_28.1, @Interface, @Interface(constants.%i32) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.a30)]
// CHECK:STDOUT:   %impl.elem0.loc13_11.3: %ptr.a17 = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.085)]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.81f [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc13_11.3: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.d80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %ptr.a17 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.5ec = name_ref I, %I.loc12_28.2 [symbolic = %I.loc12_28.1 (constants.%I.eef)]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:     %.loc13_11.1: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:     %.loc13_11.2: %Interface.assoc_type.2d5 = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%i32, constants.%I.eef) [concrete = constants.%assoc0.8c4]
// CHECK:STDOUT:     %X.ref: %Interface.assoc_type.2d5 = name_ref X, %.loc13_11.2 [concrete = constants.%assoc0.8c4]
// CHECK:STDOUT:     %impl.elem0.loc13_11.1: %ptr.a17 = impl_witness_access constants.%Interface.lookup_impl_witness.a30, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.085)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.2: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:     %bound_method.loc13_11.1: <bound method> = bound_method %impl.elem0.loc13_11.1, %impl.elem0.loc13_11.2 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13_11.2, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_11.2: <bound method> = bound_method %impl.elem0.loc13_11.1, %specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.d80)]
// CHECK:STDOUT:     %ptr.as.Copy.impl.Op.call: init %ptr.a17 = call %bound_method.loc13_11.2(%impl.elem0.loc13_11.1) [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.085)]
// CHECK:STDOUT:     return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.128
// CHECK:STDOUT:   %Self.loc4_30.2 => constants.%Self.5b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%Self.5b9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.f63
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T.67d, constants.%I.165) {
// CHECK:STDOUT:   %T.patt.loc8_19.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc8_19.1 => constants.%T.67d
// CHECK:STDOUT:   %Interface.type.loc8_49.1 => constants.%Interface.type.128
// CHECK:STDOUT:   %pattern_type.loc8_36 => constants.%pattern_type.a49
// CHECK:STDOUT:   %I.patt.loc8_36.2 => constants.%I.patt.768
// CHECK:STDOUT:   %I.loc8_36.1 => constants.%I.165
// CHECK:STDOUT:   %ptr.loc8_56.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc8_56.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc8_56 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %return.param_patt.loc8_56.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc8_52.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%I.165) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.f63
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.5ec
// CHECK:STDOUT:   %Self.loc4_30.2 => constants.%Self.91c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.eef) {
// CHECK:STDOUT:   %I.patt.loc12_28.2 => constants.%I.patt.e71
// CHECK:STDOUT:   %I.loc12_28.1 => constants.%I.eef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.5b9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.a17
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.2d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.eef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.a17
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.2d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.1b1: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.128: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.5b9: %Interface.type.128 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.f63: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.f45: %Interface.assoc_type.f63 = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.a49: type = pattern_type %Interface.type.128 [symbolic]
// CHECK:STDOUT:   %I.patt.768: %pattern_type.a49 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %I.165: %Interface.type.128 = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.e4d: <witness> = require_complete_type %Interface.type.128 [symbolic]
// CHECK:STDOUT:   %I.as_type.c4b: type = facet_access_type %I.165 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.5ec: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.5c6: type = pattern_type %Interface.type.5ec [concrete]
// CHECK:STDOUT:   %I.patt.e71: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %I.eef: %Interface.type.5ec = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.91c: %Interface.type.5ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Interface.assoc_type.2d5: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.8c4: %Interface.assoc_type.2d5 = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.bed: type = facet_access_type %I.eef [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessMissingGeneric = %AccessMissingGeneric.decl
// CHECK:STDOUT:     .AccessMissingConcrete = %AccessMissingConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.1b1 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_22.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24.1: type = splice_block %.loc4_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_26.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_43.1: @AccessMissingGeneric.%pattern_type.loc8_43 (%pattern_type.a49) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_43.2 (constants.%I.patt.768)]
// CHECK:STDOUT:     %return.param_patt.loc8_62.1: @AccessMissingGeneric.%pattern_type.loc8_62 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_62.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:     %return.patt.loc8_59.1: @AccessMissingGeneric.%pattern_type.loc8_62 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_62.1, %T.ref.loc8_62 [symbolic = %return.patt.loc8_59.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_62: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_62.2: Core.Form = init_form %T.ref.loc8_62 [symbolic = %.loc8_62.1 (constants.%.184)]
// CHECK:STDOUT:     %.loc8_28.1: type = splice_block %.loc8_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_26: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_26.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_56: type = splice_block %Interface.type.loc8_56.2 [symbolic = %Interface.type.loc8_56.1 (constants.%Interface.type.128)] {
// CHECK:STDOUT:       %.Self.frozen.loc8_43: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_55: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_56.2: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_56.1 (constants.%Interface.type.128)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_43.2: @AccessMissingGeneric.%Interface.type.loc8_56.1 (%Interface.type.128) = symbolic_binding I, 1 [symbolic = %I.loc8_43.1 (constants.%I.165)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt.loc14_35.1: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc14_35.2 (constants.%I.patt.e71)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc14_56 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc14_56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_56: Core.Form = init_form %i32.loc14_56 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %.loc14_50: type = splice_block %Interface.type [concrete = constants.%Interface.type.5ec] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc14_47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.5ec]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc14_35.2: %Interface.type.5ec = symbolic_binding I, 0 [symbolic = %I.loc14_35.1 (constants.%I.eef)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_22.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.128)]
// CHECK:STDOUT:   %Self.loc4_30.2: @Interface.%Interface.type (%Interface.type.128) = symbolic_binding Self, 1 [symbolic = %Self.loc4_30.2 (constants.%Self.5b9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_30.1: @Interface.%Interface.type (%Interface.type.128) = symbolic_binding Self, 1 [symbolic = %Self.loc4_30.2 (constants.%Self.5b9)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.f63) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.f45)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_30.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:     witness = (@Interface.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_26.2: type, %I.loc8_43.2: @AccessMissingGeneric.%Interface.type.loc8_56.1 (%Interface.type.128)) {
// CHECK:STDOUT:   %T.patt.loc8_26.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_26.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:   %Interface.type.loc8_56.1: type = facet_type <@Interface, @Interface(%T.loc8_26.1)> [symbolic = %Interface.type.loc8_56.1 (constants.%Interface.type.128)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %Interface.type.loc8_56.1 [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.a49)]
// CHECK:STDOUT:   %I.patt.loc8_43.2: @AccessMissingGeneric.%pattern_type.loc8_43 (%pattern_type.a49) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_43.2 (constants.%I.patt.768)]
// CHECK:STDOUT:   %I.loc8_43.1: @AccessMissingGeneric.%Interface.type.loc8_56.1 (%Interface.type.128) = symbolic_binding I, 1 [symbolic = %I.loc8_43.1 (constants.%I.165)]
// CHECK:STDOUT:   %.loc8_62.1: Core.Form = init_form %T.loc8_26.1 [symbolic = %.loc8_62.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc8_62: type = pattern_type %T.loc8_26.1 [symbolic = %pattern_type.loc8_62 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc8_62.2: @AccessMissingGeneric.%pattern_type.loc8_62 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_62.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc8_59.2: @AccessMissingGeneric.%pattern_type.loc8_62 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_62.2, %T.loc8_26.1 [symbolic = %return.patt.loc8_59.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_26.1 [symbolic = %require_complete.loc8 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %Interface.type.loc8_56.1 [symbolic = %require_complete.loc11 (constants.%require_complete.e4d)]
// CHECK:STDOUT:   %I.as_type.loc11_11.2: type = facet_access_type %I.loc8_43.1 [symbolic = %I.as_type.loc11_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessMissingGeneric.%T.loc8_26.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_56.1 (%Interface.type.128) = name_ref I, %I.loc8_43.2 [symbolic = %I.loc8_43.1 (constants.%I.165)]
// CHECK:STDOUT:     %I.as_type.loc11_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc11_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:     %.loc11: type = converted %I.ref, %I.as_type.loc11_11.1 [symbolic = %I.as_type.loc11_11.2 (constants.%I.as_type.c4b)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc14_35.2: %Interface.type.5ec) {
// CHECK:STDOUT:   %I.patt.loc14_35.2: %pattern_type.5c6 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc14_35.2 (constants.%I.patt.e71)]
// CHECK:STDOUT:   %I.loc14_35.1: %Interface.type.5ec = symbolic_binding I, 0 [symbolic = %I.loc14_35.1 (constants.%I.eef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc17_11.2: type = facet_access_type %I.loc14_35.1 [symbolic = %I.as_type.loc17_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.5ec = name_ref I, %I.loc14_35.2 [symbolic = %I.loc14_35.1 (constants.%I.eef)]
// CHECK:STDOUT:     %I.as_type.loc17_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc17_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:     %.loc17: type = converted %I.ref, %I.as_type.loc17_11.1 [symbolic = %I.as_type.loc17_11.2 (constants.%I.as_type.bed)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.128
// CHECK:STDOUT:   %Self.loc4_30.2 => constants.%Self.5b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%Self.5b9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.f63
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.165) {
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_26.1 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_56.1 => constants.%Interface.type.128
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.a49
// CHECK:STDOUT:   %I.patt.loc8_43.2 => constants.%I.patt.768
// CHECK:STDOUT:   %I.loc8_43.1 => constants.%I.165
// CHECK:STDOUT:   %.loc8_62.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc8_62 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc8_62.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc8_59.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%I.165) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.f63
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.5ec
// CHECK:STDOUT:   %Self.loc4_30.2 => constants.%Self.91c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.eef) {
// CHECK:STDOUT:   %I.patt.loc14_35.2 => constants.%I.patt.e71
// CHECK:STDOUT:   %I.loc14_35.1 => constants.%I.eef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.5b9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.a17
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.2d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.eef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.a17
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.2d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
