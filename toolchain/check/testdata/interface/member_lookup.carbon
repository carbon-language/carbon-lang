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

interface Interface(T:! type) {
  let X:! T*;
}

fn AccessGeneric[T:! type](I:! Interface(T)) -> T* {
  return I.X;
}

fn AccessConcrete(I:! Interface(i32)) -> i32* {
  return I.X;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

interface Interface(T:! type) {
  let X:! T*;
}

fn AccessMissingGeneric[T:! type](I:! Interface(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Interface(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return I.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return I.nonesuch;
}

fn AccessMissingConcrete(I:! Interface(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Interface(i32)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return I.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return I.nonesuch;
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.82d: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.82d = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.dc0: type = facet_type <@Interface, @Interface(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.ce2: %Interface.type.dc0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.8fd: type = assoc_entity_type @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.2a0: %Interface.assoc_type.8fd = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.ef3: type = pattern_type %Interface.type.dc0 [symbolic]
// CHECK:STDOUT:   %I.fbb: %Interface.type.dc0 = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.8f5: <witness> = require_complete_type %Interface.type.dc0 [symbolic]
// CHECK:STDOUT:   %I.as_type.83f: type = facet_access_type %I.fbb [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.9df: <witness> = lookup_impl_witness %I.fbb, @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.ff1: %ptr.e8f = impl_witness_access %Interface.lookup_impl_witness.9df, element0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.5be: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.546: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.2a3: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.5be) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a8e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.2a3) [symbolic]
// CHECK:STDOUT:   %.86b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a8e, %Copy.facet.2a3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.480: %.86b = impl_witness_access %Copy.lookup_impl_witness.5be, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.5dd: <bound method> = bound_method %impl.elem0.ff1, %impl.elem0.480 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d1f: <specific function> = specific_impl_function %impl.elem0.480, @Copy.WithSelf.Op(%Copy.facet.2a3) [symbolic]
// CHECK:STDOUT:   %bound_method.388: <bound method> = bound_method %impl.elem0.ff1, %specific_impl_fn.d1f [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.f91: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %Interface.type.f91 [concrete]
// CHECK:STDOUT:   %I.788: %Interface.type.f91 = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.793: Core.Form = init_form %ptr.143 [concrete]
// CHECK:STDOUT:   %pattern_type.663: type = pattern_type %ptr.143 [concrete]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.4ef: %Interface.type.f91 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.530: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.4b7: %Interface.assoc_type.530 = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.d66: type = facet_access_type %I.788 [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.01c: <witness> = lookup_impl_witness %I.788, @Interface, @Interface(%i32) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c3f: %ptr.143 = impl_witness_access %Interface.lookup_impl_witness.01c, element0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.45b: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.eac: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e2: %ptr.as.Copy.impl.Op.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.425: %Copy.type = facet_value %ptr.143, (%Copy.impl_witness.45b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.425) [concrete]
// CHECK:STDOUT:   %.948: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72c, %Copy.facet.425 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.c3f, %ptr.as.Copy.impl.Op.6e2 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e2, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.f50: <bound method> = bound_method %impl.elem0.c3f, %ptr.as.Copy.impl.Op.specific_fn [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %Interface.decl: %Interface.type.82d = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %I.patt: @AccessGeneric.%pattern_type.loc8_29 (%pattern_type.ef3) = symbolic_binding_pattern I, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc8_50.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_49: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc8_50.2: type = ptr_type %T.ref.loc8_49 [symbolic = %ptr.loc8_50.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc8_50.2: Core.Form = init_form %ptr.loc8_50.2 [symbolic = %.loc8_50.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc8_22.1: type = splice_block %.loc8_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc8_43: type = splice_block %Interface.type.loc8_43.2 [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.dc0)] {
// CHECK:STDOUT:       %.Self.loc8_29: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.82d = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:       %Interface.type.loc8_43.2: type = facet_type <@Interface, @Interface(constants.%T.67d)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.dc0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_29.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.dc0) = symbolic_binding I, 1 [symbolic = %I.loc8_29.1 (constants.%I.fbb)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt: %pattern_type.54d = symbolic_binding_pattern I, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.663 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.663 = return_slot_pattern %return.param_patt, %ptr [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12_42: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc12_42 [concrete = constants.%ptr.143]
// CHECK:STDOUT:     %.loc12_45: Core.Form = init_form %ptr [concrete = constants.%.793]
// CHECK:STDOUT:     %.loc12_36: type = splice_block %Interface.type [concrete = constants.%Interface.type.f91] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.82d = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc12_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.f91]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc12_20.2: %Interface.type.f91 = symbolic_binding I, 0 [symbolic = %I.loc12_20.1 (constants.%I.788)]
// CHECK:STDOUT:     %return.param: ref %ptr.143 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.143 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.dc0)]
// CHECK:STDOUT:   %Self.loc4_31.2: @Interface.%Interface.type (%Interface.type.dc0) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.ce2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_31.1: @Interface.%Interface.type (%Interface.type.dc0) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.ce2)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.8fd) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.2a0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_31.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Interface.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_19.2: type, %I.loc8_29.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.dc0)) {
// CHECK:STDOUT:   %T.loc8_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:   %Interface.type.loc8_43.1: type = facet_type <@Interface, @Interface(%T.loc8_19.1)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.dc0)]
// CHECK:STDOUT:   %I.loc8_29.1: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.dc0) = symbolic_binding I, 1 [symbolic = %I.loc8_29.1 (constants.%I.fbb)]
// CHECK:STDOUT:   %pattern_type.loc8_29: type = pattern_type %Interface.type.loc8_43.1 [symbolic = %pattern_type.loc8_29 (constants.%pattern_type.ef3)]
// CHECK:STDOUT:   %ptr.loc8_50.1: type = ptr_type %T.loc8_19.1 [symbolic = %ptr.loc8_50.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc8_50.1: Core.Form = init_form %ptr.loc8_50.1 [symbolic = %.loc8_50.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc8_50: type = pattern_type %ptr.loc8_50.1 [symbolic = %pattern_type.loc8_50 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %ptr.loc8_50.1 [symbolic = %require_complete.loc8 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Interface.type.loc8_43.1 [symbolic = %require_complete.loc9 (constants.%require_complete.8f5)]
// CHECK:STDOUT:   %I.as_type.loc9_11.2: type = facet_access_type %I.loc8_29.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.83f)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.8fd)]
// CHECK:STDOUT:   %assoc0: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.8fd) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.2a0)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc8_29.1, @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.9df)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.3: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.ff1)]
// CHECK:STDOUT:   %.loc9_11.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc8_19.1) [symbolic = %.loc9_11.3 (constants.%.546)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc8_50.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.5be)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc8_50.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.2a3)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a8e)]
// CHECK:STDOUT:   %.loc9_11.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc9_11.4 (constants.%.86b)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.4: @AccessGeneric.%.loc9_11.4 (%.86b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.480)]
// CHECK:STDOUT:   %bound_method.loc9_11.3: <bound method> = bound_method %impl.elem0.loc9_11.3, %impl.elem0.loc9_11.4 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.5dd)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.4, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.d1f)]
// CHECK:STDOUT:   %bound_method.loc9_11.4: <bound method> = bound_method %impl.elem0.loc9_11.3, %specific_impl_fn.loc9_11.2 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.388)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.dc0) = name_ref I, %I.loc8_29.2 [symbolic = %I.loc8_29.1 (constants.%I.fbb)]
// CHECK:STDOUT:     %I.as_type.loc9_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.83f)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted %I.ref, %I.as_type.loc9_11.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.83f)]
// CHECK:STDOUT:     %.loc9_11.2: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.8fd) = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%T.67d, constants.%I.fbb) [symbolic = %assoc0 (constants.%assoc0.2a0)]
// CHECK:STDOUT:     %X.ref: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.8fd) = name_ref X, %.loc9_11.2 [symbolic = %assoc0 (constants.%assoc0.2a0)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = impl_witness_access constants.%Interface.lookup_impl_witness.9df, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.ff1)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.2: @AccessGeneric.%.loc9_11.4 (%.86b) = impl_witness_access constants.%Copy.lookup_impl_witness.5be, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.480)]
// CHECK:STDOUT:     %bound_method.loc9_11.1: <bound method> = bound_method %impl.elem0.loc9_11.1, %impl.elem0.loc9_11.2 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.5dd)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Copy.WithSelf.Op(constants.%Copy.facet.2a3) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.d1f)]
// CHECK:STDOUT:     %bound_method.loc9_11.2: <bound method> = bound_method %impl.elem0.loc9_11.1, %specific_impl_fn.loc9_11.1 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.388)]
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = call %bound_method.loc9_11.2(%impl.elem0.loc9_11.1)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc12_20.2: %Interface.type.f91) {
// CHECK:STDOUT:   %I.loc12_20.1: %Interface.type.f91 = symbolic_binding I, 0 [symbolic = %I.loc12_20.1 (constants.%I.788)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc12_20.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.d66)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc12_20.1, @Interface, @Interface(constants.%i32) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.01c)]
// CHECK:STDOUT:   %impl.elem0.loc13_11.3: %ptr.143 = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.c3f)]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.6e2 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc13_11.3: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.f50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %ptr.143 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.f91 = name_ref I, %I.loc12_20.2 [symbolic = %I.loc12_20.1 (constants.%I.788)]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.d66)]
// CHECK:STDOUT:     %.loc13_11.1: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.d66)]
// CHECK:STDOUT:     %.loc13_11.2: %Interface.assoc_type.530 = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%i32, constants.%I.788) [concrete = constants.%assoc0.4b7]
// CHECK:STDOUT:     %X.ref: %Interface.assoc_type.530 = name_ref X, %.loc13_11.2 [concrete = constants.%assoc0.4b7]
// CHECK:STDOUT:     %impl.elem0.loc13_11.1: %ptr.143 = impl_witness_access constants.%Interface.lookup_impl_witness.01c, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.c3f)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.2: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:     %bound_method.loc13_11.1: <bound method> = bound_method %impl.elem0.loc13_11.1, %impl.elem0.loc13_11.2 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13_11.2, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_11.2: <bound method> = bound_method %impl.elem0.loc13_11.1, %specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.f50)]
// CHECK:STDOUT:     %ptr.as.Copy.impl.Op.call: init %ptr.143 = call %bound_method.loc13_11.2(%impl.elem0.loc13_11.1) [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.c3f)]
// CHECK:STDOUT:     return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.dc0
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.ce2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%Self.ce2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.8fd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2a0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T.67d, constants.%I.fbb) {
// CHECK:STDOUT:   %T.loc8_19.1 => constants.%T.67d
// CHECK:STDOUT:   %Interface.type.loc8_43.1 => constants.%Interface.type.dc0
// CHECK:STDOUT:   %I.loc8_29.1 => constants.%I.fbb
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.ef3
// CHECK:STDOUT:   %ptr.loc8_50.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc8_50.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc8_50 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%I.fbb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.8fd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2a0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.f91
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.4ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.788) {
// CHECK:STDOUT:   %I.loc12_20.1 => constants.%I.788
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.ce2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.143
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.530
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.788) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.143
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.530
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.2c5: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.2c5 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.7df: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2bc: %Interface.type.7df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.924: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.981: %Interface.assoc_type.924 = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.76c: type = pattern_type %Interface.type.7df [symbolic]
// CHECK:STDOUT:   %I.9d2: %Interface.type.7df = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.08e: <witness> = require_complete_type %Interface.type.7df [symbolic]
// CHECK:STDOUT:   %I.as_type.90a: type = facet_access_type %I.9d2 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.e02: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.555: type = pattern_type %Interface.type.e02 [concrete]
// CHECK:STDOUT:   %I.d05: %Interface.type.e02 = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.68c: %Interface.type.e02 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Interface.assoc_type.be8: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.039: %Interface.assoc_type.be8 = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.85e: type = facet_access_type %I.d05 [symbolic]
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
// CHECK:STDOUT:   %Interface.decl: %Interface.type.2c5 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %I.patt: @AccessMissingGeneric.%pattern_type.loc8_36 (%pattern_type.76c) = symbolic_binding_pattern I, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc8_56 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_56: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_56.2: Core.Form = init_form %T.ref.loc8_56 [symbolic = %.loc8_56.1 (constants.%.184)]
// CHECK:STDOUT:     %.loc8_29.1: type = splice_block %.loc8_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_26: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_26.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_50: type = splice_block %Interface.type.loc8_50.2 [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.7df)] {
// CHECK:STDOUT:       %.Self.loc8_36: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.2c5 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_49: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_50.2: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.7df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_36.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.7df) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.9d2)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt: %pattern_type.555 = symbolic_binding_pattern I, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc16_49 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_49: Core.Form = init_form %i32.loc16_49 [concrete = constants.%.70c]
// CHECK:STDOUT:     %.loc16_43: type = splice_block %Interface.type [concrete = constants.%Interface.type.e02] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.2c5 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc16_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.e02]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc16_27.2: %Interface.type.e02 = symbolic_binding I, 0 [symbolic = %I.loc16_27.1 (constants.%I.d05)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.7df)]
// CHECK:STDOUT:   %Self.loc4_31.2: @Interface.%Interface.type (%Interface.type.7df) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.2bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_31.1: @Interface.%Interface.type (%Interface.type.7df) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.2bc)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.924) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.981)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_31.1
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
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_26.2: type, %I.loc8_36.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.7df)) {
// CHECK:STDOUT:   %T.loc8_26.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:   %Interface.type.loc8_50.1: type = facet_type <@Interface, @Interface(%T.loc8_26.1)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.7df)]
// CHECK:STDOUT:   %I.loc8_36.1: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.7df) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.9d2)]
// CHECK:STDOUT:   %pattern_type.loc8_36: type = pattern_type %Interface.type.loc8_50.1 [symbolic = %pattern_type.loc8_36 (constants.%pattern_type.76c)]
// CHECK:STDOUT:   %.loc8_56.1: Core.Form = init_form %T.loc8_26.1 [symbolic = %.loc8_56.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc8_56: type = pattern_type %T.loc8_26.1 [symbolic = %pattern_type.loc8_56 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_26.1 [symbolic = %require_complete.loc8 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %Interface.type.loc8_50.1 [symbolic = %require_complete.loc13 (constants.%require_complete.08e)]
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc8_36.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.90a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessMissingGeneric.%T.loc8_26.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.7df) = name_ref I, %I.loc8_36.2 [symbolic = %I.loc8_36.1 (constants.%I.9d2)]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.90a)]
// CHECK:STDOUT:     %.loc13: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.90a)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc16_27.2: %Interface.type.e02) {
// CHECK:STDOUT:   %I.loc16_27.1: %Interface.type.e02 = symbolic_binding I, 0 [symbolic = %I.loc16_27.1 (constants.%I.d05)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc21_11.2: type = facet_access_type %I.loc16_27.1 [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.e02 = name_ref I, %I.loc16_27.2 [symbolic = %I.loc16_27.1 (constants.%I.d05)]
// CHECK:STDOUT:     %I.as_type.loc21_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:     %.loc21: type = converted %I.ref, %I.as_type.loc21_11.1 [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.7df
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.2bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%Self.2bc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.924
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.981
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.9d2) {
// CHECK:STDOUT:   %T.loc8_26.1 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_50.1 => constants.%Interface.type.7df
// CHECK:STDOUT:   %I.loc8_36.1 => constants.%I.9d2
// CHECK:STDOUT:   %pattern_type.loc8_36 => constants.%pattern_type.76c
// CHECK:STDOUT:   %.loc8_56.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc8_56 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%I.9d2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.924
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.981
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.e02
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.68c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.d05) {
// CHECK:STDOUT:   %I.loc16_27.1 => constants.%I.d05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.2bc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.143
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.039
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.d05) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.143
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.039
// CHECK:STDOUT: }
// CHECK:STDOUT:
