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
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.1b1: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.fe8: type = facet_type <@Interface, @Interface(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.45e: %Interface.type.fe8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.61f: type = assoc_entity_type @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.199: %Interface.assoc_type.61f = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.2c6f: type = pattern_type %Interface.type.fe8 [symbolic]
// CHECK:STDOUT:   %I.patt.0cc: %pattern_type.2c6f = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %I.63c: %Interface.type.fe8 = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.08a: <witness> = require_complete_type %Interface.type.fe8 [symbolic]
// CHECK:STDOUT:   %I.as_type.3c3: type = facet_access_type %I.63c [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.ffe: <witness> = lookup_impl_witness %I.63c, @Interface, @Interface(%T.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.602: %ptr.e8f = impl_witness_access %Interface.lookup_impl_witness.ffe, element0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.602: <bound method> = bound_method %impl.elem0.602, %impl.elem0.484 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %bound_method.060: <bound method> = bound_method %impl.elem0.602, %specific_impl_fn.a76 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.c5f: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.3cf: type = pattern_type %Interface.type.c5f [concrete]
// CHECK:STDOUT:   %I.patt.b65: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %I.e05: %Interface.type.c5f = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.952: Core.Form = init_form %ptr.d08 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %return.param_patt.7dd: %pattern_type.f00 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.dbe: %pattern_type.f00 = return_slot_pattern %return.param_patt.7dd, %ptr.d08 [concrete]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.43d: %Interface.type.c5f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.d6a: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.80c: %Interface.assoc_type.d6a = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.307: type = facet_access_type %I.e05 [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.64c: <witness> = lookup_impl_witness %I.e05, @Interface, @Interface(%i32) [symbolic]
// CHECK:STDOUT:   %impl.elem0.3b7: %ptr.d08 = impl_witness_access %Interface.lookup_impl_witness.64c, element0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b49: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b49) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet.b49 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.3b7, %ptr.as.Copy.impl.Op.d1c [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.195: <bound method> = bound_method %impl.elem0.3b7, %ptr.as.Copy.impl.Op.specific_fn [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_22.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_19.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_19.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:     %I.patt.loc8_29.1: @AccessGeneric.%pattern_type.loc8_29 (%pattern_type.2c6f) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_29.2 (constants.%I.patt.0cc)]
// CHECK:STDOUT:     %return.param_patt.loc8_50.1: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc8_50.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:     %return.patt.loc8_46.1: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc8_50.1, %ptr.loc8_50.2 [symbolic = %return.patt.loc8_46.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_49: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc8_50.2: type = ptr_type %T.ref.loc8_49 [symbolic = %ptr.loc8_50.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc8_50.2: Core.Form = init_form %ptr.loc8_50.2 [symbolic = %.loc8_50.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc8_22.1: type = splice_block %.loc8_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc8_43: type = splice_block %Interface.type.loc8_43.2 [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.fe8)] {
// CHECK:STDOUT:       %.Self.loc8_29: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: type = name_ref T, %T.loc8_19.2 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:       %Interface.type.loc8_43.2: type = facet_type <@Interface, @Interface(constants.%T.67d)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.fe8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_29.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.fe8) = symbolic_binding I, 1 [symbolic = %I.loc8_29.1 (constants.%I.63c)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt.loc12_20.1: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_20.2 (constants.%I.patt.b65)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f00 = out_param_pattern [concrete = constants.%return.param_patt.7dd]
// CHECK:STDOUT:     %return.patt: %pattern_type.f00 = return_slot_pattern %return.param_patt, %ptr [concrete = constants.%return.patt.dbe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12_42: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc12_42 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:     %.loc12_45: Core.Form = init_form %ptr [concrete = constants.%.952]
// CHECK:STDOUT:     %.loc12_36: type = splice_block %Interface.type [concrete = constants.%Interface.type.c5f] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc12_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.c5f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc12_20.2: %Interface.type.c5f = symbolic_binding I, 0 [symbolic = %I.loc12_20.1 (constants.%I.e05)]
// CHECK:STDOUT:     %return.param: ref %ptr.d08 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.d08 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_22.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.fe8)]
// CHECK:STDOUT:   %Self.loc4_31.2: @Interface.%Interface.type (%Interface.type.fe8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_31.1: @Interface.%Interface.type (%Interface.type.fe8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.45e)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.61f) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.199)]
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
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_19.2: type, %I.loc8_29.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.fe8)) {
// CHECK:STDOUT:   %T.patt.loc8_19.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_19.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc8_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_19.1 (constants.%T.67d)]
// CHECK:STDOUT:   %Interface.type.loc8_43.1: type = facet_type <@Interface, @Interface(%T.loc8_19.1)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.fe8)]
// CHECK:STDOUT:   %pattern_type.loc8_29: type = pattern_type %Interface.type.loc8_43.1 [symbolic = %pattern_type.loc8_29 (constants.%pattern_type.2c6f)]
// CHECK:STDOUT:   %I.patt.loc8_29.2: @AccessGeneric.%pattern_type.loc8_29 (%pattern_type.2c6f) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_29.2 (constants.%I.patt.0cc)]
// CHECK:STDOUT:   %I.loc8_29.1: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.fe8) = symbolic_binding I, 1 [symbolic = %I.loc8_29.1 (constants.%I.63c)]
// CHECK:STDOUT:   %ptr.loc8_50.1: type = ptr_type %T.loc8_19.1 [symbolic = %ptr.loc8_50.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc8_50.1: Core.Form = init_form %ptr.loc8_50.1 [symbolic = %.loc8_50.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc8_50: type = pattern_type %ptr.loc8_50.1 [symbolic = %pattern_type.loc8_50 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %return.param_patt.loc8_50.2: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc8_50.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:   %return.patt.loc8_46.2: @AccessGeneric.%pattern_type.loc8_50 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc8_50.2, %ptr.loc8_50.1 [symbolic = %return.patt.loc8_46.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %ptr.loc8_50.1 [symbolic = %require_complete.loc8 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Interface.type.loc8_43.1 [symbolic = %require_complete.loc9 (constants.%require_complete.08a)]
// CHECK:STDOUT:   %I.as_type.loc9_11.2: type = facet_access_type %I.loc8_29.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.61f)]
// CHECK:STDOUT:   %assoc0: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.61f) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.199)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc8_29.1, @Interface, @Interface(%T.loc8_19.1) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.ffe)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.3: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.602)]
// CHECK:STDOUT:   %.loc9_11.5: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc8_19.1) [symbolic = %.loc9_11.5 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc8_50.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc9_11.3: %Copy.type = facet_value %ptr.loc8_50.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc9_11.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc9_11.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc9_11.3 [symbolic = %.loc9_11.6 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.4: @AccessGeneric.%.loc9_11.6 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %bound_method.loc9_11.3: <bound method> = bound_method %impl.elem0.loc9_11.3, %impl.elem0.loc9_11.4 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.602)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.4, @Copy.WithSelf.Op(%Copy.facet.loc9_11.3) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:   %bound_method.loc9_11.4: <bound method> = bound_method %impl.elem0.loc9_11.3, %specific_impl_fn.loc9_11.2 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.060)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.fe8) = name_ref I, %I.loc8_29.2 [symbolic = %I.loc8_29.1 (constants.%I.63c)]
// CHECK:STDOUT:     %I.as_type.loc9_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted %I.ref, %I.as_type.loc9_11.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:     %.loc9_11.2: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.61f) = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%T.67d, constants.%I.63c) [symbolic = %assoc0 (constants.%assoc0.199)]
// CHECK:STDOUT:     %X.ref: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.61f) = name_ref X, %.loc9_11.2 [symbolic = %assoc0 (constants.%assoc0.199)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = impl_witness_access constants.%Interface.lookup_impl_witness.ffe, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.602)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.2: @AccessGeneric.%.loc9_11.6 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc9_11.1: <bound method> = bound_method %impl.elem0.loc9_11.1, %impl.elem0.loc9_11.2 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.602)]
// CHECK:STDOUT:     %Copy.facet.loc9_11.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_11.3: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_11.1 [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Copy.facet.loc9_11.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_11.4: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_11.2 [symbolic = %Copy.facet.loc9_11.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Copy.WithSelf.Op(constants.%Copy.facet.302) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc9_11.2: <bound method> = bound_method %impl.elem0.loc9_11.1, %specific_impl_fn.loc9_11.1 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.060)]
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @AccessGeneric.%ptr.loc8_50.1 (%ptr.e8f) = call %bound_method.loc9_11.2(%impl.elem0.loc9_11.1)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc12_20.2: %Interface.type.c5f) {
// CHECK:STDOUT:   %I.patt.loc12_20.2: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_20.2 (constants.%I.patt.b65)]
// CHECK:STDOUT:   %I.loc12_20.1: %Interface.type.c5f = symbolic_binding I, 0 [symbolic = %I.loc12_20.1 (constants.%I.e05)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc12_20.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.307)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc12_20.1, @Interface, @Interface(constants.%i32) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.64c)]
// CHECK:STDOUT:   %impl.elem0.loc13_11.3: %ptr.d08 = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.3b7)]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.d1c [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc13_11.3: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.195)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %ptr.d08 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.c5f = name_ref I, %I.loc12_20.2 [symbolic = %I.loc12_20.1 (constants.%I.e05)]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.307)]
// CHECK:STDOUT:     %.loc13_11.1: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.307)]
// CHECK:STDOUT:     %.loc13_11.2: %Interface.assoc_type.d6a = specific_constant @X.%assoc0, @Interface.WithSelf(constants.%i32, constants.%I.e05) [concrete = constants.%assoc0.80c]
// CHECK:STDOUT:     %X.ref: %Interface.assoc_type.d6a = name_ref X, %.loc13_11.2 [concrete = constants.%assoc0.80c]
// CHECK:STDOUT:     %impl.elem0.loc13_11.1: %ptr.d08 = impl_witness_access constants.%Interface.lookup_impl_witness.64c, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.3b7)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.2: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:     %bound_method.loc13_11.1: <bound method> = bound_method %impl.elem0.loc13_11.1, %impl.elem0.loc13_11.2 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13_11.2, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_11.2: <bound method> = bound_method %impl.elem0.loc13_11.1, %specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.195)]
// CHECK:STDOUT:     %ptr.as.Copy.impl.Op.call: init %ptr.d08 = call %bound_method.loc13_11.2(%impl.elem0.loc13_11.1) [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.3b7)]
// CHECK:STDOUT:     return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.fe8
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.45e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.61f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T.67d, constants.%I.63c) {
// CHECK:STDOUT:   %T.patt.loc8_19.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc8_19.1 => constants.%T.67d
// CHECK:STDOUT:   %Interface.type.loc8_43.1 => constants.%Interface.type.fe8
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.2c6f
// CHECK:STDOUT:   %I.patt.loc8_29.2 => constants.%I.patt.0cc
// CHECK:STDOUT:   %I.loc8_29.1 => constants.%I.63c
// CHECK:STDOUT:   %ptr.loc8_50.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc8_50.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc8_50 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %return.param_patt.loc8_50.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc8_46.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T.67d, constants.%I.63c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.61f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.c5f
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.43d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.e05) {
// CHECK:STDOUT:   %I.patt.loc12_20.2 => constants.%I.patt.b65
// CHECK:STDOUT:   %I.loc12_20.1 => constants.%I.e05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.d08
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.d6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.e05) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.d08
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.d6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.1b1: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.fe8: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.45e: %Interface.type.fe8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.61f: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.199: %Interface.assoc_type.61f = assoc_entity element0, @Interface.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %pattern_type.2c6: type = pattern_type %Interface.type.fe8 [symbolic]
// CHECK:STDOUT:   %I.patt.0cc: %pattern_type.2c6 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %I.63c: %Interface.type.fe8 = symbolic_binding I, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.08a: <witness> = require_complete_type %Interface.type.fe8 [symbolic]
// CHECK:STDOUT:   %I.as_type.3c3: type = facet_access_type %I.63c [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.c5f: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %pattern_type.3cf: type = pattern_type %Interface.type.c5f [concrete]
// CHECK:STDOUT:   %I.patt.b65: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %I.e05: %Interface.type.c5f = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.43d: %Interface.type.c5f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Interface.assoc_type.d6a: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.80c: %Interface.assoc_type.d6a = assoc_entity element0, @Interface.WithSelf.%X [concrete]
// CHECK:STDOUT:   %I.as_type.307: type = facet_access_type %I.e05 [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_22.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_26.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_36.1: @AccessMissingGeneric.%pattern_type.loc8_36 (%pattern_type.2c6) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_36.2 (constants.%I.patt.0cc)]
// CHECK:STDOUT:     %return.param_patt.loc8_56.1: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_56.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:     %return.patt.loc8_53.1: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_56.1, %T.ref.loc8_56 [symbolic = %return.patt.loc8_53.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_56: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_56.2: Core.Form = init_form %T.ref.loc8_56 [symbolic = %.loc8_56.1 (constants.%.184)]
// CHECK:STDOUT:     %.loc8_29.1: type = splice_block %.loc8_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_26: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_26.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_50: type = splice_block %Interface.type.loc8_50.2 [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.fe8)] {
// CHECK:STDOUT:       %.Self.loc8_36: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_49: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_50.2: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.fe8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_36.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.fe8) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.63c)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_26.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt.loc16_27.1: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc16_27.2 (constants.%I.patt.b65)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc16_49 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_49: Core.Form = init_form %i32.loc16_49 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc16_43: type = splice_block %Interface.type [concrete = constants.%Interface.type.c5f] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.1b1 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %i32.loc16_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.c5f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc16_27.2: %Interface.type.c5f = symbolic_binding I, 0 [symbolic = %I.loc16_27.1 (constants.%I.e05)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_22.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_22.1)> [symbolic = %Interface.type (constants.%Interface.type.fe8)]
// CHECK:STDOUT:   %Self.loc4_31.2: @Interface.%Interface.type (%Interface.type.fe8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_31.1: @Interface.%Interface.type (%Interface.type.fe8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_31.2 (constants.%Self.45e)]
// CHECK:STDOUT:     %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: @Interface.WithSelf.%ptr (%ptr.e8f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.WithSelf.%Interface.assoc_type (%Interface.assoc_type.61f) = assoc_entity element0, @Interface.WithSelf.%X [symbolic = @Interface.WithSelf.%assoc0 (constants.%assoc0.199)]
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
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_26.2: type, %I.loc8_36.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.fe8)) {
// CHECK:STDOUT:   %T.patt.loc8_26.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_26.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:   %Interface.type.loc8_50.1: type = facet_type <@Interface, @Interface(%T.loc8_26.1)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.fe8)]
// CHECK:STDOUT:   %pattern_type.loc8_36: type = pattern_type %Interface.type.loc8_50.1 [symbolic = %pattern_type.loc8_36 (constants.%pattern_type.2c6)]
// CHECK:STDOUT:   %I.patt.loc8_36.2: @AccessMissingGeneric.%pattern_type.loc8_36 (%pattern_type.2c6) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_36.2 (constants.%I.patt.0cc)]
// CHECK:STDOUT:   %I.loc8_36.1: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.fe8) = symbolic_binding I, 1 [symbolic = %I.loc8_36.1 (constants.%I.63c)]
// CHECK:STDOUT:   %.loc8_56.1: Core.Form = init_form %T.loc8_26.1 [symbolic = %.loc8_56.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc8_56: type = pattern_type %T.loc8_26.1 [symbolic = %pattern_type.loc8_56 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc8_56.2: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_56.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc8_53.2: @AccessMissingGeneric.%pattern_type.loc8_56 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_56.2, %T.loc8_26.1 [symbolic = %return.patt.loc8_53.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_26.1 [symbolic = %require_complete.loc8 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %Interface.type.loc8_50.1 [symbolic = %require_complete.loc13 (constants.%require_complete.08a)]
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc8_36.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @AccessMissingGeneric.%T.loc8_26.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.fe8) = name_ref I, %I.loc8_36.2 [symbolic = %I.loc8_36.1 (constants.%I.63c)]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:     %.loc13: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.3c3)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc16_27.2: %Interface.type.c5f) {
// CHECK:STDOUT:   %I.patt.loc16_27.2: %pattern_type.3cf = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc16_27.2 (constants.%I.patt.b65)]
// CHECK:STDOUT:   %I.loc16_27.1: %Interface.type.c5f = symbolic_binding I, 0 [symbolic = %I.loc16_27.1 (constants.%I.e05)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc21_11.2: type = facet_access_type %I.loc16_27.1 [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.307)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.c5f = name_ref I, %I.loc16_27.2 [symbolic = %I.loc16_27.1 (constants.%I.e05)]
// CHECK:STDOUT:     %I.as_type.loc21_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.307)]
// CHECK:STDOUT:     %.loc21: type = converted %I.ref, %I.as_type.loc21_11.1 [symbolic = %I.as_type.loc21_11.2 (constants.%I.as_type.307)]
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
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.fe8
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.45e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.61f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.63c) {
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_26.1 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_50.1 => constants.%Interface.type.fe8
// CHECK:STDOUT:   %pattern_type.loc8_36 => constants.%pattern_type.2c6
// CHECK:STDOUT:   %I.patt.loc8_36.2 => constants.%I.patt.0cc
// CHECK:STDOUT:   %I.loc8_36.1 => constants.%I.63c
// CHECK:STDOUT:   %.loc8_56.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc8_56 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc8_56.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc8_53.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%T, constants.%I.63c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.61f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.c5f
// CHECK:STDOUT:   %Self.loc4_31.2 => constants.%Self.43d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.e05) {
// CHECK:STDOUT:   %I.patt.loc16_27.2 => constants.%I.patt.b65
// CHECK:STDOUT:   %I.loc16_27.1 => constants.%I.e05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.d08
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.d6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%i32, constants.%I.e05) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.d08
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.d6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
