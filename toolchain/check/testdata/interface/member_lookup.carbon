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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.1a8: type = facet_type <@Interface, @Interface(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.c05: %Interface.type.1a8 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.0d9: type = assoc_entity_type @Interface, @Interface(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.7e7: %Interface.assoc_type.0d9 = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.c05: %Interface.type.1a8 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.eaa: type = pattern_type %Interface.type.1a8 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.376: <witness> = require_complete_type %Interface.type.1a8 [symbolic]
// CHECK:STDOUT:   %I.as_type.fe1: type = facet_access_type %I.c05 [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.301: <witness> = lookup_impl_witness %I.c05, @Interface, @Interface(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Interface.facet.b19: %Interface.type.1a8 = facet_value %I.as_type.fe1, (%Interface.lookup_impl_witness.301) [symbolic]
// CHECK:STDOUT:   %impl.elem0.aca: %ptr.79f = impl_witness_access %Interface.lookup_impl_witness.301, element0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.def: %Copy.type = facet_value %ptr.79f, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %.fe5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.def [symbolic]
// CHECK:STDOUT:   %impl.elem0.429: %.fe5 = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.313: <bound method> = bound_method %impl.elem0.aca, %impl.elem0.429 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d8a: <specific function> = specific_impl_function %impl.elem0.429, @Copy.Op(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %bound_method.5c7: <bound method> = bound_method %impl.elem0.aca, %specific_impl_fn.d8a [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.bcd: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.910: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.1c1: type = pattern_type %Interface.type.bcd [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.e75: %Interface.type.bcd = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.aaa: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.04e: %Interface.assoc_type.aaa = assoc_entity element0, @Interface.%X [concrete]
// CHECK:STDOUT:   %I.as_type.85e: type = facet_access_type %I.910 [symbolic]
// CHECK:STDOUT:   %Interface.lookup_impl_witness.73f: <witness> = lookup_impl_witness %I.910, @Interface, @Interface(%i32) [symbolic]
// CHECK:STDOUT:   %Interface.facet.c5c: %Interface.type.bcd = facet_value %I.as_type.85e, (%Interface.lookup_impl_witness.73f) [symbolic]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT:   %impl.elem0.5f0: %ptr.235 = impl_witness_access %Interface.lookup_impl_witness.73f, element0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.19c: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.3ea: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.2f4: %ptr.as.Copy.impl.Op.type.3ea = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.bab: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.19c) [concrete]
// CHECK:STDOUT:   %.284: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.bab [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.5f0, %ptr.as.Copy.impl.Op.2f4 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.2f4, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.f5e: <bound method> = bound_method %impl.elem0.5f0, %ptr.as.Copy.impl.Op.specific_fn [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %I.patt: @AccessGeneric.%pattern_type.loc8_28 (%pattern_type.eaa) = symbolic_binding_pattern I, 1 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessGeneric.%pattern_type.loc8_46 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessGeneric.%pattern_type.loc8_46 (%pattern_type.afe) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_49: type = name_ref T, %T.loc8_18.2 [symbolic = %T.loc8_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %ptr.loc8_50.2: type = ptr_type %T.ref.loc8_49 [symbolic = %ptr.loc8_50.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_43.2 [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.1a8)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: type = name_ref T, %T.loc8_18.2 [symbolic = %T.loc8_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:       %Interface.type.loc8_43.2: type = facet_type <@Interface, @Interface(constants.%T.8b3)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.1a8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_28.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.1a8) = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.1 (constants.%I.c05)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt: %pattern_type.1c1 = symbolic_binding_pattern I, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fe8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fe8 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_42: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_42: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc12_42 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %.loc12: type = splice_block %Interface.type [concrete = constants.%Interface.type.bcd] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc12_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.bcd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc12_19.2: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic = %I.loc12_19.1 (constants.%I.910)]
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.2: type) {
// CHECK:STDOUT:   %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.1)> [symbolic = %Interface.type (constants.%Interface.type.1a8)]
// CHECK:STDOUT:   %Self.2: @Interface.%Interface.type (%Interface.type.1a8) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c05)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc4_21.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.1a8) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c05)]
// CHECK:STDOUT:     %X: @X.%ptr (%ptr.79f) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.2: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.1a8)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%ptr (%ptr.79f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_18.2: type, %I.loc8_28.2: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.1a8)) {
// CHECK:STDOUT:   %T.loc8_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:   %Interface.type.loc8_43.1: type = facet_type <@Interface, @Interface(%T.loc8_18.1)> [symbolic = %Interface.type.loc8_43.1 (constants.%Interface.type.1a8)]
// CHECK:STDOUT:   %I.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.1a8) = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.1 (constants.%I.c05)]
// CHECK:STDOUT:   %pattern_type.loc8_28: type = pattern_type %Interface.type.loc8_43.1 [symbolic = %pattern_type.loc8_28 (constants.%pattern_type.eaa)]
// CHECK:STDOUT:   %ptr.loc8_50.1: type = ptr_type %T.loc8_18.1 [symbolic = %ptr.loc8_50.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc8_46: type = pattern_type %ptr.loc8_50.1 [symbolic = %pattern_type.loc8_46 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_10: <witness> = require_complete_type %Interface.type.loc8_43.1 [symbolic = %require_complete.loc9_10 (constants.%require_complete.376)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc8_18.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = %assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:   %I.as_type.loc9_11.2: type = facet_access_type %I.loc8_28.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.fe1)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc8_28.1, @Interface, @Interface(%T.loc8_18.1) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.301)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.3: @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.aca)]
// CHECK:STDOUT:   %require_complete.loc9_13: <witness> = require_complete_type %ptr.loc8_50.1 [symbolic = %require_complete.loc9_13 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc8_50.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc8_50.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.def)]
// CHECK:STDOUT:   %.loc9_11.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc9_11.4 (constants.%.fe5)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.4: @AccessGeneric.%.loc9_11.4 (%.fe5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.429)]
// CHECK:STDOUT:   %bound_method.loc9_11.3: <bound method> = bound_method %impl.elem0.loc9_11.3, %impl.elem0.loc9_11.4 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.313)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.4, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.d8a)]
// CHECK:STDOUT:   %bound_method.loc9_11.4: <bound method> = bound_method %impl.elem0.loc9_11.3, %specific_impl_fn.loc9_11.2 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.5c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_43.1 (%Interface.type.1a8) = name_ref I, %I.loc8_28.2 [symbolic = %I.loc8_28.1 (constants.%I.c05)]
// CHECK:STDOUT:     %.loc9_11.1: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = specific_constant @X.%assoc0, @Interface(constants.%T.8b3) [symbolic = %assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:     %X.ref: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = name_ref X, %.loc9_11.1 [symbolic = %assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:     %I.as_type.loc9_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.fe1)]
// CHECK:STDOUT:     %.loc9_11.2: type = converted %I.ref, %I.as_type.loc9_11.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.fe1)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) = impl_witness_access constants.%Interface.lookup_impl_witness.301, element0 [symbolic = %impl.elem0.loc9_11.3 (constants.%impl.elem0.aca)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.2: @AccessGeneric.%.loc9_11.4 (%.fe5) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [symbolic = %impl.elem0.loc9_11.4 (constants.%impl.elem0.429)]
// CHECK:STDOUT:     %bound_method.loc9_11.1: <bound method> = bound_method %impl.elem0.loc9_11.1, %impl.elem0.loc9_11.2 [symbolic = %bound_method.loc9_11.3 (constants.%bound_method.313)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Copy.Op(constants.%Copy.facet.def) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.d8a)]
// CHECK:STDOUT:     %bound_method.loc9_11.2: <bound method> = bound_method %impl.elem0.loc9_11.1, %specific_impl_fn.loc9_11.1 [symbolic = %bound_method.loc9_11.4 (constants.%bound_method.5c7)]
// CHECK:STDOUT:     %.loc9_11.3: init @AccessGeneric.%ptr.loc8_50.1 (%ptr.79f) = call %bound_method.loc9_11.2(%impl.elem0.loc9_11.1)
// CHECK:STDOUT:     return %.loc9_11.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc12_19.2: %Interface.type.bcd) {
// CHECK:STDOUT:   %I.loc12_19.1: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic = %I.loc12_19.1 (constants.%I.910)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc12_19.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:   %Interface.lookup_impl_witness: <witness> = lookup_impl_witness %I.loc12_19.1, @Interface, @Interface(constants.%i32) [symbolic = %Interface.lookup_impl_witness (constants.%Interface.lookup_impl_witness.73f)]
// CHECK:STDOUT:   %impl.elem0.loc13_11.3: %ptr.235 = impl_witness_access %Interface.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.5f0)]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.2f4 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc13_11.3: <bound method> = bound_method %impl.elem0.loc13_11.3, constants.%ptr.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.f5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %ptr.235 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.bcd = name_ref I, %I.loc12_19.2 [symbolic = %I.loc12_19.1 (constants.%I.910)]
// CHECK:STDOUT:     %.loc13_11.1: %Interface.assoc_type.aaa = specific_constant @X.%assoc0, @Interface(constants.%i32) [concrete = constants.%assoc0.04e]
// CHECK:STDOUT:     %X.ref: %Interface.assoc_type.aaa = name_ref X, %.loc13_11.1 [concrete = constants.%assoc0.04e]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:     %.loc13_11.2: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.85e)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.1: %ptr.235 = impl_witness_access constants.%Interface.lookup_impl_witness.73f, element0 [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.5f0)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.2: %.284 = impl_witness_access constants.%Copy.impl_witness.19c, element0 [concrete = constants.%ptr.as.Copy.impl.Op.2f4]
// CHECK:STDOUT:     %bound_method.loc13_11.1: <bound method> = bound_method %impl.elem0.loc13_11.1, %impl.elem0.loc13_11.2 [symbolic = %ptr.as.Copy.impl.Op.bound (constants.%ptr.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13_11.2, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_11.2: <bound method> = bound_method %impl.elem0.loc13_11.1, %specific_fn [symbolic = %bound_method.loc13_11.3 (constants.%bound_method.f5e)]
// CHECK:STDOUT:     %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc13_11.2(%impl.elem0.loc13_11.1) [symbolic = %impl.elem0.loc13_11.3 (constants.%impl.elem0.5f0)]
// CHECK:STDOUT:     return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_21.1 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.1a8
// CHECK:STDOUT:   %Self.2 => constants.%Self.c05
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.c05) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %ptr => constants.%ptr.79f
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T.8b3, constants.%I.c05) {
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%T.8b3
// CHECK:STDOUT:   %Interface.type.loc8_43.1 => constants.%Interface.type.1a8
// CHECK:STDOUT:   %I.loc8_28.1 => constants.%I.c05
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.eaa
// CHECK:STDOUT:   %ptr.loc8_50.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc8_46 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Interface.facet.b19) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %ptr => constants.%ptr.79f
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.bcd
// CHECK:STDOUT:   %Self.2 => constants.%Self.e75
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.aaa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.910) {
// CHECK:STDOUT:   %I.loc12_19.1 => constants.%I.910
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32, constants.%Interface.facet.c5c) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ptr => constants.%ptr.235
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.1a8: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c05: %Interface.type.1a8 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.0d9: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.7e7: %Interface.assoc_type.0d9 = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.c05: %Interface.type.1a8 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.eaa: type = pattern_type %Interface.type.1a8 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.376: <witness> = require_complete_type %Interface.type.1a8 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.bcd: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.910: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.1c1: type = pattern_type %Interface.type.bcd [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.e75: %Interface.type.bcd = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.aaa: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.04e: %Interface.assoc_type.aaa = assoc_entity element0, @Interface.%X [concrete]
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
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %I.patt: @AccessMissingGeneric.%pattern_type.loc8_35 (%pattern_type.eaa) = symbolic_binding_pattern I, 1 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessMissingGeneric.%pattern_type.loc8_53 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessMissingGeneric.%pattern_type.loc8_53 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_56: type = name_ref T, %T.loc8_25.2 [symbolic = %T.loc8_25.1 (constants.%T)]
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_50.2 [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.1a8)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_49: type = name_ref T, %T.loc8_25.2 [symbolic = %T.loc8_25.1 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_50.2: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.1a8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_35.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.1a8) = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.1 (constants.%I.c05)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_25.1 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_25.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt: %pattern_type.1c1 = symbolic_binding_pattern I, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16: type = splice_block %Interface.type [concrete = constants.%Interface.type.bcd] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc16_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.bcd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc16_26.2: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic = %I.loc16_26.1 (constants.%I.910)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.2: type) {
// CHECK:STDOUT:   %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.1)> [symbolic = %Interface.type (constants.%Interface.type.1a8)]
// CHECK:STDOUT:   %Self.2: @Interface.%Interface.type (%Interface.type.1a8) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c05)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc4_21.1) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.1a8) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c05)]
// CHECK:STDOUT:     %X: @X.%ptr (%ptr) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.7e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.2: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.1a8)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%ptr (%ptr);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_25.2: type, %I.loc8_35.2: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.1a8)) {
// CHECK:STDOUT:   %T.loc8_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.1 (constants.%T)]
// CHECK:STDOUT:   %Interface.type.loc8_50.1: type = facet_type <@Interface, @Interface(%T.loc8_25.1)> [symbolic = %Interface.type.loc8_50.1 (constants.%Interface.type.1a8)]
// CHECK:STDOUT:   %I.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.1a8) = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.1 (constants.%I.c05)]
// CHECK:STDOUT:   %pattern_type.loc8_35: type = pattern_type %Interface.type.loc8_50.1 [symbolic = %pattern_type.loc8_35 (constants.%pattern_type.eaa)]
// CHECK:STDOUT:   %pattern_type.loc8_53: type = pattern_type %T.loc8_25.1 [symbolic = %pattern_type.loc8_53 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_25.1 [symbolic = %require_complete.loc8 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %Interface.type.loc8_50.1 [symbolic = %require_complete.loc13 (constants.%require_complete.376)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @AccessMissingGeneric.%T.loc8_25.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_50.1 (%Interface.type.1a8) = name_ref I, %I.loc8_35.2 [symbolic = %I.loc8_35.1 (constants.%I.c05)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc16_26.2: %Interface.type.bcd) {
// CHECK:STDOUT:   %I.loc16_26.1: %Interface.type.bcd = bind_symbolic_name I, 0 [symbolic = %I.loc16_26.1 (constants.%I.910)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.bcd = name_ref I, %I.loc16_26.2 [symbolic = %I.loc16_26.1 (constants.%I.910)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.1a8
// CHECK:STDOUT:   %Self.2 => constants.%Self.c05
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self.c05) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.c05) {
// CHECK:STDOUT:   %T.loc8_25.1 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_50.1 => constants.%Interface.type.1a8
// CHECK:STDOUT:   %I.loc8_35.1 => constants.%I.c05
// CHECK:STDOUT:   %pattern_type.loc8_35 => constants.%pattern_type.eaa
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.bcd
// CHECK:STDOUT:   %Self.2 => constants.%Self.e75
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.aaa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.910) {
// CHECK:STDOUT:   %I.loc16_26.1 => constants.%I.910
// CHECK:STDOUT: }
// CHECK:STDOUT:
