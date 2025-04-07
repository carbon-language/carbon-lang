// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/member_lookup.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

interface Interface(T:! type) {
  let X:! T;
}

fn AccessGeneric[T:! type](I:! Interface(T)) -> T {
  return I.X;
}

fn AccessConcrete(I:! Interface(i32)) -> i32 {
  return I.X;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

interface Interface(T:! type) {
  let X:! T;
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.d32: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a5f: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.0d9: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.dbe: %Interface.assoc_type.0d9 = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.a5f: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %I.patt.47c: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f3: <witness> = require_complete_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %I.as_type.59e: type = facet_access_type %I.a5f [symbolic]
// CHECK:STDOUT:   %I.as_wit.iface0.82b: <witness> = facet_access_witness %I.a5f, element0 [symbolic]
// CHECK:STDOUT:   %Interface.facet.687: %Interface.type.d32 = facet_value %I.as_type.59e, (%I.as_wit.iface0.82b) [symbolic]
// CHECK:STDOUT:   %impl.elem0.94c: %T = impl_witness_access %I.as_wit.iface0.82b, element0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.981: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.d08: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt.1f8: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Self.62a: %Interface.type.981 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.aaa: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.69f: %Interface.assoc_type.aaa = assoc_entity element0, @Interface.%X [concrete]
// CHECK:STDOUT:   %I.as_type.ee6: type = facet_access_type %I.d08 [symbolic]
// CHECK:STDOUT:   %I.as_wit.iface0.dd7: <witness> = facet_access_witness %I.d08, element0 [symbolic]
// CHECK:STDOUT:   %Interface.facet.124: %Interface.type.981 = facet_value %I.as_type.ee6, (%I.as_wit.iface0.dd7) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e89: %i32 = impl_witness_access %I.as_wit.iface0.dd7, element0 [symbolic]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessGeneric = %AccessGeneric.decl
// CHECK:STDOUT:     .AccessConcrete = %AccessConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_28.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:     %return.patt: @AccessGeneric.%T.loc8_18.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessGeneric.%T.loc8_18.2 (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_49: type = name_ref T, %T.loc8_18.1 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_43.1 [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: type = name_ref T, %T.loc8_18.1 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_43.1: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%T.loc8_18.2 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%T.loc8_18.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt.loc12_19.1: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_19.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_42: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_42: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: type = splice_block %Interface.type [concrete = constants.%Interface.type.981] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc12_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.981]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc12_19.1: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc12_19.2 (constants.%I.d08)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.2)> [symbolic = %Interface.type (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %Self.2: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a5f)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc4_21.2) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.dbe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a5f)]
// CHECK:STDOUT:     %X: @X.%T (%T) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.dbe)]
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
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.1: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.d32)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @X.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_18.1: type, %I.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32)) {
// CHECK:STDOUT:   %T.loc8_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Interface.type.loc8_43.2: type = facet_type <@Interface, @Interface(%T.loc8_18.2)> [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %I.loc8_28.2: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:   %I.patt.loc8_28.2: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_28.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_10: <witness> = require_complete_type @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) [symbolic = %require_complete.loc9_10 (constants.%require_complete.6f3)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc8_18.2) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = %assoc0 (constants.%assoc0.dbe)]
// CHECK:STDOUT:   %I.as_type.loc9_11.2: type = facet_access_type %I.loc8_28.2 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.59e)]
// CHECK:STDOUT:   %I.as_wit.iface0.loc9_11.2: <witness> = facet_access_witness %I.loc8_28.2, element0 [symbolic = %I.as_wit.iface0.loc9_11.2 (constants.%I.as_wit.iface0.82b)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: @AccessGeneric.%T.loc8_18.2 (%T) = impl_witness_access %I.as_wit.iface0.loc9_11.2, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0.94c)]
// CHECK:STDOUT:   %require_complete.loc9_13: <witness> = require_complete_type @AccessGeneric.%T.loc8_18.2 (%T) [symbolic = %require_complete.loc9_13 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_18.1: type](%I.patt.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32)) -> @AccessGeneric.%T.loc8_18.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = name_ref I, %I.loc8_28.1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %.loc9_11.1: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = specific_constant @X.%assoc0, @Interface(constants.%T) [symbolic = %assoc0 (constants.%assoc0.dbe)]
// CHECK:STDOUT:     %X.ref: @AccessGeneric.%Interface.assoc_type (%Interface.assoc_type.0d9) = name_ref X, %.loc9_11.1 [symbolic = %assoc0 (constants.%assoc0.dbe)]
// CHECK:STDOUT:     %I.as_type.loc9_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.59e)]
// CHECK:STDOUT:     %.loc9_11.2: type = converted %I.ref, %I.as_type.loc9_11.1 [symbolic = %I.as_type.loc9_11.2 (constants.%I.as_type.59e)]
// CHECK:STDOUT:     %I.as_wit.iface0.loc9_11.1: <witness> = facet_access_witness %I.ref, element0 [symbolic = %I.as_wit.iface0.loc9_11.2 (constants.%I.as_wit.iface0.82b)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @AccessGeneric.%T.loc8_18.2 (%T) = impl_witness_access %I.as_wit.iface0.loc9_11.1, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0.94c)]
// CHECK:STDOUT:     return %impl.elem0.loc9_11.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc12_19.1: %Interface.type.981) {
// CHECK:STDOUT:   %I.loc12_19.2: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc12_19.2 (constants.%I.d08)]
// CHECK:STDOUT:   %I.patt.loc12_19.2: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc12_19.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.as_type.loc13_11.2: type = facet_access_type %I.loc12_19.2 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.ee6)]
// CHECK:STDOUT:   %I.as_wit.iface0.loc13_11.2: <witness> = facet_access_witness %I.loc12_19.2, element0 [symbolic = %I.as_wit.iface0.loc13_11.2 (constants.%I.as_wit.iface0.dd7)]
// CHECK:STDOUT:   %impl.elem0.loc13_11.2: %i32 = impl_witness_access %I.as_wit.iface0.loc13_11.2, element0 [symbolic = %impl.elem0.loc13_11.2 (constants.%impl.elem0.e89)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.patt.loc12_19.1: %Interface.type.981) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.981 = name_ref I, %I.loc12_19.1 [symbolic = %I.loc12_19.2 (constants.%I.d08)]
// CHECK:STDOUT:     %.loc13_11.1: %Interface.assoc_type.aaa = specific_constant @X.%assoc0, @Interface(constants.%i32) [concrete = constants.%assoc0.69f]
// CHECK:STDOUT:     %X.ref: %Interface.assoc_type.aaa = name_ref X, %.loc13_11.1 [concrete = constants.%assoc0.69f]
// CHECK:STDOUT:     %I.as_type.loc13_11.1: type = facet_access_type %I.ref [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.ee6)]
// CHECK:STDOUT:     %.loc13_11.2: type = converted %I.ref, %I.as_type.loc13_11.1 [symbolic = %I.as_type.loc13_11.2 (constants.%I.as_type.ee6)]
// CHECK:STDOUT:     %I.as_wit.iface0.loc13_11.1: <witness> = facet_access_witness %I.ref, element0 [symbolic = %I.as_wit.iface0.loc13_11.2 (constants.%I.as_wit.iface0.dd7)]
// CHECK:STDOUT:     %impl.elem0.loc13_11.1: %i32 = impl_witness_access %I.as_wit.iface0.loc13_11.1, element0 [symbolic = %impl.elem0.loc13_11.2 (constants.%impl.elem0.e89)]
// CHECK:STDOUT:     return %impl.elem0.loc13_11.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.d32
// CHECK:STDOUT:   %Self.2 => constants.%Self.a5f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dbe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self.a5f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(%T.loc4_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T, constants.%I.a5f) {
// CHECK:STDOUT:   %T.loc8_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_18.2 => constants.%T.patt
// CHECK:STDOUT:   %Interface.type.loc8_43.2 => constants.%Interface.type.d32
// CHECK:STDOUT:   %I.loc8_28.2 => constants.%I.a5f
// CHECK:STDOUT:   %I.patt.loc8_28.2 => constants.%I.patt.47c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(@AccessGeneric.%T.loc8_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Interface.facet.687) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.981
// CHECK:STDOUT:   %Self.2 => constants.%Self.62a
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.aaa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.69f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.d08) {
// CHECK:STDOUT:   %I.loc12_19.2 => constants.%I.d08
// CHECK:STDOUT:   %I.patt.loc12_19.2 => constants.%I.patt.1f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32, constants.%Interface.facet.124) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.d32: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a5f: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.0d9: type = assoc_entity_type @Interface, @Interface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.dbe: %Interface.assoc_type.0d9 = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.a5f: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %I.patt.47c: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f3: <witness> = require_complete_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.981: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.d08: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt.1f8: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.62a: %Interface.type.981 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.aaa: type = assoc_entity_type @Interface, @Interface(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.69f: %Interface.assoc_type.aaa = assoc_entity element0, @Interface.%X [concrete]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessMissingGeneric = %AccessMissingGeneric.decl
// CHECK:STDOUT:     .AccessMissingConcrete = %AccessMissingConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_35.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:     %return.patt: @AccessMissingGeneric.%T.loc8_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingGeneric.%T.loc8_25.2 (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_56: type = name_ref T, %T.loc8_25.1 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_50.1 [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_49: type = name_ref T, %T.loc8_25.1 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_50.1: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_25.2 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt.loc16_26.1: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc16_26.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16: type = splice_block %Interface.type [concrete = constants.%Interface.type.981] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc16_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.981]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc16_26.1: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc16_26.2 (constants.%I.d08)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.2)> [symbolic = %Interface.type (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %Self.2: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a5f)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface, @Interface(%T.loc4_21.2) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.0d9)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.dbe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a5f)]
// CHECK:STDOUT:     %X: @X.%T (%T) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.0d9) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.dbe)]
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
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.1: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.d32)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @X.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_25.1: type, %I.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32)) {
// CHECK:STDOUT:   %T.loc8_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Interface.type.loc8_50.2: type = facet_type <@Interface, @Interface(%T.loc8_25.2)> [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %I.loc8_35.2: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:   %I.patt.loc8_35.2: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_35.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) [symbolic = %require_complete (constants.%require_complete.6f3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_25.1: type](%I.patt.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32)) -> @AccessMissingGeneric.%T.loc8_25.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = name_ref I, %I.loc8_35.1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc16_26.1: %Interface.type.981) {
// CHECK:STDOUT:   %I.loc16_26.2: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc16_26.2 (constants.%I.d08)]
// CHECK:STDOUT:   %I.patt.loc16_26.2: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc16_26.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.patt.loc16_26.1: %Interface.type.981) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.981 = name_ref I, %I.loc16_26.1 [symbolic = %I.loc16_26.2 (constants.%I.d08)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.d32
// CHECK:STDOUT:   %Self.2 => constants.%Self.a5f
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dbe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self.a5f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(%T.loc4_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.a5f) {
// CHECK:STDOUT:   %T.loc8_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_25.2 => constants.%T.patt
// CHECK:STDOUT:   %Interface.type.loc8_50.2 => constants.%Interface.type.d32
// CHECK:STDOUT:   %I.loc8_35.2 => constants.%I.a5f
// CHECK:STDOUT:   %I.patt.loc8_35.2 => constants.%I.patt.47c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(@AccessMissingGeneric.%T.loc8_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.981
// CHECK:STDOUT:   %Self.2 => constants.%Self.62a
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.aaa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.69f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.d08) {
// CHECK:STDOUT:   %I.loc16_26.2 => constants.%I.d08
// CHECK:STDOUT:   %I.patt.loc16_26.2 => constants.%I.patt.1f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
