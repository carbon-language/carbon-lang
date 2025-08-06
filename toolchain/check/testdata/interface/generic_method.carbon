// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic_method.carbon

// --- generic_method_fewer_params.carbon

library "[[@TEST_NAME]]";

// T has index 0, Self has index 1, U has index 2.
interface A(T:! type) {
  fn F(U:! type, u: U) -> (T, Self, U);
}

class X {}
class Y {}
class Z {}

impl Y as A(X) {
  // Here, U has index 0. The specific used for A.F needs to renumber its U from
  // index 2 to index 0.
  fn F(U:! type, u: U) -> (X, Y, U) {
    return ({}, {}, u);
  }
}

fn Call() {
  (Y as A(X)).F(Z, {});
}

fn CallGeneric(T:! A(X)) {
  T.F(Z, {});
}

fn CallIndirect() {
  CallGeneric(Y);
}

// --- generic_method_more_params.carbon

library "[[@TEST_NAME]]";

// T has index 0, Self has index 1, U has index 2.
interface A(T:! type) {
  fn F(U:! type, u: U) -> (T, Self, U);
}

class X {}
class Y1 {}
class Y2 {}
class Z {}

impl forall [V1:! type, V2:! type, W:! type] (V1, V2) as A(W) {
  // Here, U has index 3. The specific used for A.F needs to renumber its U from
  // index 2 to index 3.
  fn F(U:! type, u: U) -> (W, (V1, V2), U) {
    return F(U, u);
  }
}

fn Call() {
  // TODO: The `as type` here should not be necessary.
  (((Y1, Y2) as type) as A(X)).F(Z, {});
}

fn CallGeneric(T:! A(X)) {
  T.F(Z, {});
}

fn CallIndirect() {
  // TODO: The `as type` here should not be necessary.
  CallGeneric((Y1, Y2) as type);
}

// CHECK:STDOUT: --- generic_method_fewer_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.75a: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.753: %A.type.75a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %U.2cb [symbolic]
// CHECK:STDOUT:   %tuple.type.e59: type = tuple_type (type, %A.type.75a, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.81d: type = facet_access_type %Self.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.bd2: type = tuple_type (%T.8b3, %Self.as_type.81d, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.44c: type = pattern_type %tuple.type.bd2 [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.807: <witness> = impl_witness @X.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.621: <witness> = impl_witness @Y.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.455: type = ptr_type %Y [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %ptr.455 [concrete]
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op.type: type = fn_type @Y.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op: %Y.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.2a0: <witness> = impl_witness @Z.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.type: type = fn_type @Z.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op: %Z.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.91f: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.1bb: %A.type.91f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.765: type = tuple_type (%X, %Y, %U.8b3) [symbolic]
// CHECK:STDOUT:   %pattern_type.340: type = pattern_type %tuple.type.765 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.552: %A.type.91f = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.a0c: type = tuple_type (type, %A.type.91f, type) [concrete]
// CHECK:STDOUT:   %require_complete.816: <witness> = require_complete_type %tuple.type.765 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %tuple.type.a41: type = tuple_type (%empty_struct_type, %empty_struct_type, %U.8b3) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a60: type = fn_type_with_self_type %A.F.type.13d, %A.facet.552 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.f6b: type = tuple_type (%X, %Y, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.ef1: type = pattern_type %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.bc9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.46f: %T.as.Destroy.impl.Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.955: <witness> = impl_witness imports.%Destroy.impl_witness_table, @T.as.Destroy.impl(%tuple.type.f6b) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.495: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.f6b) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.51a: %T.as.Destroy.impl.Op.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.959: type = ptr_type %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %complete_type.456: <witness> = complete_type_witness %ptr.959 [concrete]
// CHECK:STDOUT:   %Destroy.facet.1f2: %Destroy.type = facet_value %tuple.type.f6b, (%Destroy.impl_witness.955) [concrete]
// CHECK:STDOUT:   %.a86: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1f2 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.51a, @T.as.Destroy.impl.Op(%tuple.type.f6b) [concrete]
// CHECK:STDOUT:   %T.9c2: %A.type.91f = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.817: type = pattern_type %A.type.91f [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.9c2 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.9c2, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.11f: %A.type.91f = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ffa: type = fn_type_with_self_type %A.F.type.13d, %A.facet.11f [symbolic]
// CHECK:STDOUT:   %impl.elem0.0e5: %.ffa = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.e28: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e7e: <specific function> = specific_impl_function %impl.elem0.0e5, @A.F(%X, %A.facet.11f, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.500: <witness> = require_complete_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %ptr.103: type = ptr_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %require_complete.8d3: <witness> = require_complete_type %ptr.103 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.e28, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1a6: %Destroy.type = facet_value %tuple.type.e28, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.efa: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1a6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.305: %.efa = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.52a: <specific function> = specific_impl_function %impl.elem0.305, @Destroy.Op(%Destroy.facet.1a6) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.552) [concrete]
// CHECK:STDOUT:   %complete_type.a64: <witness> = complete_type_witness %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %tuple.type.9c8: type = tuple_type (%empty_struct_type, %empty_struct_type, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @T.as.Destroy.impl.%T.as.Destroy.impl.Op.type (%T.as.Destroy.impl.Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @T.as.Destroy.impl.%T.as.Destroy.impl.Op (constants.%T.as.Destroy.impl.Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @T.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.817 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %A.type [concrete = constants.%A.type.91f] {
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_16.2: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc25_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.44c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.44c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.75a) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.75a) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.e59) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.bd2 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.Destroy.impl: constants.%X as constants.%Destroy.type {
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.decl: %X.as.Destroy.impl.Op.type = fn_decl @X.as.Destroy.impl.Op [concrete = constants.%X.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1c6 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1c6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %ptr.d17 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @X.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.Destroy.impl: constants.%Y as constants.%Destroy.type {
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op.decl: %Y.as.Destroy.impl.Op.type = fn_decl @Y.as.Destroy.impl.Op [concrete = constants.%Y.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.0ae = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0ae = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc10: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.455 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y [concrete = constants.%Y]
// CHECK:STDOUT:     %self: %ptr.455 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Z.as.Destroy.impl: constants.%Z as constants.%Destroy.type {
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.decl: %Z.as.Destroy.impl.Op.type = fn_decl @Z.as.Destroy.impl.Op [concrete = constants.%Z.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fb6 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:     %self: %ptr.fb6 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Z.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Z.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.A.impl: %Y.ref as %A.type {
// CHECK:STDOUT:   %Y.as.A.impl.F.decl: %Y.as.A.impl.F.type = fn_decl @Y.as.A.impl.F [concrete = constants.%Y.as.A.impl.F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.7dc) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.7dc) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_24 (%pattern_type.340) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_24 (%pattern_type.340) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_34: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %.loc16_35.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %U.ref.loc16_34)
// CHECK:STDOUT:     %.loc16_35.2: type = converted %.loc16_35.1, constants.%tuple.type.765 [symbolic = %tuple.type.loc16 (constants.%tuple.type.765)]
// CHECK:STDOUT:     %U.loc16_8.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3) = value_param call_param0
// CHECK:STDOUT:     %U.ref.loc16_21: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.765) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.765) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .F = %Y.as.A.impl.F.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.807]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   impl_decl @Y.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y.as.Destroy.impl.%Y.as.Destroy.impl.Op.decl), @Y.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.621]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   impl_decl @Z.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Z.as.Destroy.impl.%Z.as.Destroy.impl.Op.decl), @Z.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.2a0]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.75a), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.e59)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.44c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.2cb)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.Destroy.impl.Op(%self.param: %ptr.d17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y.as.Destroy.impl.Op(%self.param: %ptr.455) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Z.as.Destroy.impl.Op(%self.param: %ptr.fb6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %U.loc16_8.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %U.loc16_8.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.765)]
// CHECK:STDOUT:   %pattern_type.loc16_24: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_24 (constants.%pattern_type.340)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_24: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_24 (constants.%require_complete.816)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %U.loc16_8.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %U.loc16_8.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.a41)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3)) -> %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.765) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.a41) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%U.loc16_8.1 (%U.8b3) = initialize_from %u.ref to %tuple.elem2
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.765) = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4) to %return
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.765) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value constants.%Y, (constants.%A.impl_witness) [concrete = constants.%A.facet.552]
// CHECK:STDOUT:   %.loc22_6: %A.type.91f = converted %Y.ref, %A.facet [concrete = constants.%A.facet.552]
// CHECK:STDOUT:   %.loc22_14.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc22_14.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc22_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %.loc22_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.a60 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc22_22.1: ref %tuple.type.f6b = temporary_storage
// CHECK:STDOUT:   %.loc22_21.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc22_21.3: init %Z = class_init (), %.loc22_21.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc22_21.4: ref %Z = temporary %.loc22_21.2, %.loc22_21.3
// CHECK:STDOUT:   %.loc22_21.5: ref %Z = converted %.loc22_21.1, %.loc22_21.4
// CHECK:STDOUT:   %.loc22_21.6: %Z = bind_value %.loc22_21.5
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.f6b = call %specific_fn(%.loc22_21.6) to %.loc22_22.1
// CHECK:STDOUT:   %.loc22_22.2: ref %tuple.type.f6b = temporary %.loc22_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc22_21.2, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22_21: %ptr.fb6 = addr_of %.loc22_21.2
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc22_21)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc22_22.1, constants.%T.as.Destroy.impl.Op.51a
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.51a, @T.as.Destroy.impl.Op(constants.%tuple.type.f6b) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc22_22.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc22_22: %ptr.959 = addr_of %.loc22_22.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc22_22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc25_16.2: %A.type.91f) {
// CHECK:STDOUT:   %T.loc25_16.1: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc25_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc26_4.2: type = facet_access_type %T.loc25_16.1 [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc25_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value %T.as_type.loc26_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.11f)]
// CHECK:STDOUT:   %.loc26_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %A.facet [symbolic = %.loc26_4.3 (constants.%.ffa)]
// CHECK:STDOUT:   %impl.elem0.loc26_4.2: @CallGeneric.%.loc26_4.3 (%.ffa) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0.0e5)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_4.2: <specific function> = specific_impl_function %impl.elem0.loc26_4.2, @A.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn.e7e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc26_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.e28)]
// CHECK:STDOUT:   %require_complete.loc26_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc26_12.1 (constants.%require_complete.500)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1a6)]
// CHECK:STDOUT:   %.loc26_12.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc26_12.5 (constants.%.efa)]
// CHECK:STDOUT:   %impl.elem0.loc26_12.2: @CallGeneric.%.loc26_12.5 (%.efa) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_12.2 (constants.%impl.elem0.305)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_12.2: <specific function> = specific_impl_function %impl.elem0.loc26_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc26_12.2 (constants.%specific_impl_fn.52a)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.103)]
// CHECK:STDOUT:   %require_complete.loc26_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc26_12.2 (constants.%require_complete.8d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.91f = name_ref T, %T.loc25_16.2 [symbolic = %T.loc25_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:     %.loc26_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc26_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc26_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_4.2: type = converted %T.ref, %T.as_type.loc26_4.1 [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc26_4.1: @CallGeneric.%.loc26_4.3 (%.ffa) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0.0e5)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc26_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc26_4.1: <specific function> = specific_impl_function %impl.elem0.loc26_4.1, @A.F(constants.%X, constants.%A.facet.11f, constants.%Z) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn.e7e)]
// CHECK:STDOUT:     %.loc26_12.1: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary_storage
// CHECK:STDOUT:     %.loc26_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc26_11.3: init %Z = class_init (), %.loc26_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc26_11.4: ref %Z = temporary %.loc26_11.2, %.loc26_11.3
// CHECK:STDOUT:     %.loc26_11.5: ref %Z = converted %.loc26_11.1, %.loc26_11.4
// CHECK:STDOUT:     %.loc26_11.6: %Z = bind_value %.loc26_11.5
// CHECK:STDOUT:     %.loc26_12.2: init @CallGeneric.%tuple.type (%tuple.type.e28) = call %specific_impl_fn.loc26_4.1(%.loc26_11.6) to %.loc26_12.1
// CHECK:STDOUT:     %.loc26_12.3: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary %.loc26_12.1, %.loc26_12.2
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc26_11.2, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:     %addr.loc26_11: %ptr.fb6 = addr_of %.loc26_11.2
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc26_11)
// CHECK:STDOUT:     %impl.elem0.loc26_12.1: @CallGeneric.%.loc26_12.5 (%.efa) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_12.2 (constants.%impl.elem0.305)]
// CHECK:STDOUT:     %bound_method.loc26_12.1: <bound method> = bound_method %.loc26_12.1, %impl.elem0.loc26_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_12.1: <specific function> = specific_impl_function %impl.elem0.loc26_12.1, @Destroy.Op(constants.%Destroy.facet.1a6) [symbolic = %specific_impl_fn.loc26_12.2 (constants.%specific_impl_fn.52a)]
// CHECK:STDOUT:     %bound_method.loc26_12.2: <bound method> = bound_method %.loc26_12.1, %specific_impl_fn.loc26_12.1
// CHECK:STDOUT:     %addr.loc26_12: @CallGeneric.%ptr (%ptr.103) = addr_of %.loc26_12.1
// CHECK:STDOUT:     %.loc26_12.4: init %empty_tuple.type = call %bound_method.loc26_12.2(%addr.loc26_12)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value constants.%Y, (constants.%A.impl_witness) [concrete = constants.%A.facet.552]
// CHECK:STDOUT:   %.loc30: %A.type.91f = converted %Y.ref, %A.facet [concrete = constants.%A.facet.552]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.552) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.753, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.75a
// CHECK:STDOUT:   %Self => constants.%Self.753
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e59
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.81d
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bd2
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.44c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self.2 => constants.%Self.1bb
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.765
// CHECK:STDOUT:   %pattern_type.loc16_24 => constants.%pattern_type.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.552, constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.552
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.765
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.f6b
// CHECK:STDOUT:   %pattern_type.loc16_24 => constants.%pattern_type.ef1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_24 => constants.%complete_type.a64
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.9c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.9c2) {
// CHECK:STDOUT:   %T.loc25_16.1 => constants.%T.9c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.11f, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.11f
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.e28
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.f25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.552) {
// CHECK:STDOUT:   %T.loc25_16.1 => constants.%A.facet.552
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc26_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.552
// CHECK:STDOUT:   %.loc26_4.3 => constants.%.a60
// CHECK:STDOUT:   %impl.elem0.loc26_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc26_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.f6b
// CHECK:STDOUT:   %require_complete.loc26_12.1 => constants.%complete_type.a64
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.955
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.1f2
// CHECK:STDOUT:   %.loc26_12.5 => constants.%.a86
// CHECK:STDOUT:   %impl.elem0.loc26_12.2 => constants.%T.as.Destroy.impl.Op.51a
// CHECK:STDOUT:   %specific_impl_fn.loc26_12.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.959
// CHECK:STDOUT:   %require_complete.loc26_12.2 => constants.%complete_type.456
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.552, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.552
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.f6b
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.75a: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.753: %A.type.75a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %U.2cb [symbolic]
// CHECK:STDOUT:   %tuple.type.e59: type = tuple_type (type, %A.type.75a, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.81d: type = facet_access_type %Self.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.bd2: type = tuple_type (%T.8b3, %Self.as_type.81d, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.44c: type = pattern_type %tuple.type.bd2 [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.807: <witness> = impl_witness @X.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.614: <witness> = impl_witness @Y1.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.723: type = ptr_type %Y1 [concrete]
// CHECK:STDOUT:   %pattern_type.273: type = pattern_type %ptr.723 [concrete]
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op.type: type = fn_type @Y1.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op: %Y1.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.7b0: <witness> = impl_witness @Y2.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.23d: type = ptr_type %Y2 [concrete]
// CHECK:STDOUT:   %pattern_type.484: type = pattern_type %ptr.23d [concrete]
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op.type: type = fn_type @Y2.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op: %Y2.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.2a0: <witness> = impl_witness @Z.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.type: type = fn_type @Z.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op: %Z.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.f21: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.4fe: %A.type.f21 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.904: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.1ee: %A.F.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.b05: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.c82: %A.assoc_type.b05 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.796: <witness> = require_complete_type %A.type.f21 [symbolic]
// CHECK:STDOUT:   %A.impl_witness.f15: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.753: type = bind_symbolic_name U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.549: type = pattern_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.8ae: type = tuple_type (%W, %tuple.type.30b, %U.753) [symbolic]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.b3e: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.d79: %tuple.type.as.A.impl.F.type.b3e = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.461: %A.type.f21 = facet_value %tuple.type.30b, (%A.impl_witness.f15) [symbolic]
// CHECK:STDOUT:   %tuple.type.136: type = tuple_type (type, %A.type.f21, type) [symbolic]
// CHECK:STDOUT:   %require_complete.d4d: <witness> = require_complete_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.ed4: <witness> = require_complete_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.7d9: <specific function> = specific_function %tuple.type.as.A.impl.F.d79, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.753) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.bc9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.46f: %T.as.Destroy.impl.Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.ab5: type = ptr_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.293: <witness> = require_complete_type %ptr.ab5 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0d1: <witness> = lookup_impl_witness %tuple.type.8ae, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.3f0: %Destroy.type = facet_value %tuple.type.8ae, (%Destroy.lookup_impl_witness.0d1) [symbolic]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.3f0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.9fc: %.a93 = impl_witness_access %Destroy.lookup_impl_witness.0d1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c6: <specific function> = specific_impl_function %impl.elem0.9fc, @Destroy.Op(%Destroy.facet.3f0) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.91f: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.1bb: %A.type.91f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.d76: <witness> = complete_type_witness %A.type.91f [concrete]
// CHECK:STDOUT:   %A.impl_witness.8ac: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.bf7: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.93b: %tuple.type.as.A.impl.F.type.bf7 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.414: %A.type.91f = facet_value %tuple.type.a46, (%A.impl_witness.8ac) [concrete]
// CHECK:STDOUT:   %.406: type = fn_type_with_self_type %A.F.type.13d, %A.facet.414 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.79b: <specific function> = specific_function %tuple.type.as.A.impl.F.93b, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c80: <witness> = impl_witness imports.%Destroy.impl_witness_table, @T.as.Destroy.impl(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.2a3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.66e: %T.as.Destroy.impl.Op.type.2a3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ad9: type = ptr_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %complete_type.e23: <witness> = complete_type_witness %ptr.ad9 [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b1: %Destroy.type = facet_value %tuple.type.415, (%Destroy.impl_witness.c80) [concrete]
// CHECK:STDOUT:   %.1b7: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.2b1 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.66e, @T.as.Destroy.impl.Op(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.9c2: %A.type.91f = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.817: type = pattern_type %A.type.91f [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.9c2 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.9c2, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.11f: %A.type.91f = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ffa: type = fn_type_with_self_type %A.F.type.13d, %A.facet.11f [symbolic]
// CHECK:STDOUT:   %impl.elem0.0e5: %.ffa = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.a0c: type = tuple_type (type, %A.type.91f, type) [concrete]
// CHECK:STDOUT:   %tuple.type.e28: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e7e: <specific function> = specific_impl_function %impl.elem0.0e5, @A.F(%X, %A.facet.11f, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.500: <witness> = require_complete_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %ptr.103: type = ptr_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %require_complete.8d3: <witness> = require_complete_type %ptr.103 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.1a2: <witness> = lookup_impl_witness %tuple.type.e28, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1a6: %Destroy.type = facet_value %tuple.type.e28, (%Destroy.lookup_impl_witness.1a2) [symbolic]
// CHECK:STDOUT:   %.efa: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1a6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.305: %.efa = impl_witness_access %Destroy.lookup_impl_witness.1a2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.52a: <specific function> = specific_impl_function %impl.elem0.305, @Destroy.Op(%Destroy.facet.1a6) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.414) [concrete]
// CHECK:STDOUT:   %complete_type.aa8: <witness> = complete_type_witness %tuple.type.415 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @T.as.Destroy.impl.%T.as.Destroy.impl.Op.type (%T.as.Destroy.impl.Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @T.as.Destroy.impl.%T.as.Destroy.impl.Op (constants.%T.as.Destroy.impl.Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @T.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y1 = %Y1.decl
// CHECK:STDOUT:     .Y2 = %Y2.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y1.decl: type = class_decl @Y1 [concrete = constants.%Y1] {} {}
// CHECK:STDOUT:   %Y2.decl: type = class_decl @Y2 [concrete = constants.%Y2] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @tuple.type.as.A.impl [concrete] {
// CHECK:STDOUT:     %V1.patt: %pattern_type.98f = symbolic_binding_pattern V1, 0 [concrete]
// CHECK:STDOUT:     %V2.patt: %pattern_type.98f = symbolic_binding_pattern V2, 1 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_14.2 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_25.2 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_36.2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.f21)]
// CHECK:STDOUT:     %V1.loc14_14.2: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.loc14_25.2: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %W.loc14_36.2: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.f15)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.817 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.91f] {
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.44c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.44c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.75a) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.75a) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.e59) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.bd2 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.Destroy.impl: constants.%X as constants.%Destroy.type {
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.decl: %X.as.Destroy.impl.Op.type = fn_decl @X.as.Destroy.impl.Op [concrete = constants.%X.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1c6 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1c6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %ptr.d17 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @X.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y1.as.Destroy.impl: constants.%Y1 as constants.%Destroy.type {
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op.decl: %Y1.as.Destroy.impl.Op.type = fn_decl @Y1.as.Destroy.impl.Op [concrete = constants.%Y1.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.273 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.273 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc10: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.723 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y1 [concrete = constants.%Y1]
// CHECK:STDOUT:     %self: %ptr.723 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y1.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y1.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y2.as.Destroy.impl: constants.%Y2 as constants.%Destroy.type {
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op.decl: %Y2.as.Destroy.impl.Op.type = fn_decl @Y2.as.Destroy.impl.Op [concrete = constants.%Y2.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.484 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.484 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.23d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y2 [concrete = constants.%Y2]
// CHECK:STDOUT:     %self: %ptr.23d = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y2.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y2.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Z.as.Destroy.impl: constants.%Z as constants.%Destroy.type {
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.decl: %Z.as.Destroy.impl.Op.type = fn_decl @Z.as.Destroy.impl.Op [concrete = constants.%Z.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc12: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fb6 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:     %self: %ptr.fb6 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Z.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Z.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_14.2: type, %V2.loc14_25.2: type, %W.loc14_36.2: type) {
// CHECK:STDOUT:   %V1.loc14_14.1: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_25.1: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_36.1: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_36.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.f21)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.796)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.f15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.b3e)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.b3e) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.d79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.b3e) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.d79)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.549) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.549) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.d4a) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.d4a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_36.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_14.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_25.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %.loc17_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41)
// CHECK:STDOUT:       %.loc17_42.2: type = converted %.loc17_38, constants.%tuple.type.30b [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %.loc17_42.3: type = converted %.loc17_42.1, constants.%tuple.type.8ae [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:       %U.loc17_8.2: type = bind_symbolic_name U, 3 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %tuple.type.as.A.impl.F.decl
// CHECK:STDOUT:     witness = file.%A.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.807]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y1 {
// CHECK:STDOUT:   impl_decl @Y1.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y1.as.Destroy.impl.%Y1.as.Destroy.impl.Op.decl), @Y1.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.614]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y2 {
// CHECK:STDOUT:   impl_decl @Y2.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y2.as.Destroy.impl.%Y2.as.Destroy.impl.Op.decl), @Y2.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.7b0]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   impl_decl @Z.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Z.as.Destroy.impl.%Z.as.Destroy.impl.Op.decl), @Z.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.2a0]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.75a), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.e59)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.81d)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.44c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.2cb)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.bd2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.Destroy.impl.Op(%self.param: %ptr.d17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y1.as.Destroy.impl.Op(%self.param: %ptr.723) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y2.as.Destroy.impl.Op(%self.param: %ptr.23d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Z.as.Destroy.impl.Op(%self.param: %ptr.fb6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_14.2: type, @tuple.type.as.A.impl.%V2.loc14_25.2: type, @tuple.type.as.A.impl.%W.loc14_36.2: type, %U.loc17_8.2: type) {
// CHECK:STDOUT:   %U.loc17_8.1: type = bind_symbolic_name U, 3 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:   %pattern_type.loc17_18: type = pattern_type %U.loc17_8.1 [symbolic = %pattern_type.loc17_18 (constants.%pattern_type.549)]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_8.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:   %pattern_type.loc17_24: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_24 (constants.%pattern_type.d4a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24.1: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24.1 (constants.%require_complete.d4d)]
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_8.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.ed4)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.b3e)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.b3e) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.d79)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.7d9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.loc17_42.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.0d1)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.loc17_42.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.3f0)]
// CHECK:STDOUT:   %.loc17_24.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc17_24.3 (constants.%.a93)]
// CHECK:STDOUT:   %impl.elem0.loc17_24.2: @tuple.type.as.A.impl.F.%.loc17_24.3 (%.a93) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_24.2 (constants.%impl.elem0.9fc)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_24.2: <specific function> = specific_impl_function %impl.elem0.loc17_24.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc17_24.2 (constants.%specific_impl_fn.2c6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type.loc17_42.2 [symbolic = %ptr (constants.%ptr.ab5)]
// CHECK:STDOUT:   %require_complete.loc17_24.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc17_24.2 (constants.%require_complete.293)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.b3e) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.d79)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.b3e) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.d79)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.753) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.7d9)]
// CHECK:STDOUT:     %.loc17_24.1: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = splice_block %return {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref) to %.loc17_24.1
// CHECK:STDOUT:     %impl.elem0.loc17_24.1: @tuple.type.as.A.impl.F.%.loc17_24.3 (%.a93) = impl_witness_access constants.%Destroy.lookup_impl_witness.0d1, element0 [symbolic = %impl.elem0.loc17_24.2 (constants.%impl.elem0.9fc)]
// CHECK:STDOUT:     %bound_method.loc17_24.1: <bound method> = bound_method %.loc17_24.1, %impl.elem0.loc17_24.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_24.1: <specific function> = specific_impl_function %impl.elem0.loc17_24.1, @Destroy.Op(constants.%Destroy.facet.3f0) [symbolic = %specific_impl_fn.loc17_24.2 (constants.%specific_impl_fn.2c6)]
// CHECK:STDOUT:     %bound_method.loc17_24.2: <bound method> = bound_method %.loc17_24.1, %specific_impl_fn.loc17_24.1
// CHECK:STDOUT:     %addr: @tuple.type.as.A.impl.F.%ptr (%ptr.ab5) = addr_of %.loc17_24.1
// CHECK:STDOUT:     %.loc17_24.2: init %empty_tuple.type = call %bound_method.loc17_24.2(%addr)
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref)
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value constants.%tuple.type.a46, (constants.%A.impl_witness.8ac) [concrete = constants.%A.facet.414]
// CHECK:STDOUT:   %.loc24_23: %A.type.91f = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.414]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %impl.elem0: %.406 = impl_witness_access constants.%A.impl_witness.8ac, element0 [concrete = constants.%tuple.type.as.A.impl.F.93b]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.79b]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = bind_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.415 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_38.2, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc24_38: %ptr.fb6 = addr_of %.loc24_38.2
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc24_38)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_39.1, constants.%T.as.Destroy.impl.Op.66e
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.66e, @T.as.Destroy.impl.Op(constants.%tuple.type.415) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_39.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc24_39: %ptr.ad9 = addr_of %.loc24_39.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc24_39)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %A.type.91f) {
// CHECK:STDOUT:   %T.loc27_16.1: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_16.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.11f)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %A.facet [symbolic = %.loc28_4.3 (constants.%.ffa)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ffa) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.0e5)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.e7e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.e28)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.500)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.1a2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1a6)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.5 (constants.%.efa)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.5 (%.efa) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.305)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.52a)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.103)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.8d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.91f = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.9c2)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ffa) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.0e5)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%A.facet.11f, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.e7e)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = bind_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.e28) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.2, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:     %addr.loc28_11: %ptr.fb6 = addr_of %.loc28_11.2
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc28_11)
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.5 (%.efa) = impl_witness_access constants.%Destroy.lookup_impl_witness.1a2, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.305)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.1, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.Op(constants.%Destroy.facet.1a6) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.52a)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.1, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.103) = addr_of %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.4: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref)
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value constants.%tuple.type.a46, (constants.%A.impl_witness.8ac) [concrete = constants.%A.facet.414]
// CHECK:STDOUT:   %.loc33_31: %A.type.91f = converted %.loc33_24, %A.facet [concrete = constants.%A.facet.414]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.414) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.753, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.75a
// CHECK:STDOUT:   %Self => constants.%Self.753
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e59
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.81d
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bd2
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.44c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.f21
// CHECK:STDOUT:   %Self.2 => constants.%Self.4fe
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.904
// CHECK:STDOUT:   %A.F => constants.%A.F.1ee
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.b05
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.c82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%W
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.f21
// CHECK:STDOUT:   %require_complete => constants.%require_complete.796
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.f15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.b3e
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.d79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.753) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%U.753
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.549
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.d4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24.1 => constants.%require_complete.d4d
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.ed4
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.b3e
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.d79
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.7d9
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.0d1
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.3f0
// CHECK:STDOUT:   %.loc17_24.3 => constants.%.a93
// CHECK:STDOUT:   %impl.elem0.loc17_24.2 => constants.%impl.elem0.9fc
// CHECK:STDOUT:   %specific_impl_fn.loc17_24.2 => constants.%specific_impl_fn.2c6
// CHECK:STDOUT:   %ptr => constants.%ptr.ab5
// CHECK:STDOUT:   %require_complete.loc17_24.2 => constants.%require_complete.293
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.461, constants.%U.753) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.753
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.549
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.f21
// CHECK:STDOUT:   %Self => constants.%A.facet.461
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.136
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.d4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self.2 => constants.%Self.1bb
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%X
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.91f
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d76
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.8ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.bf7
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.93b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.0b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24.1 => constants.%complete_type.aa8
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.bf7
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.93b
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.79b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.c80
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.2b1
// CHECK:STDOUT:   %.loc17_24.3 => constants.%.1b7
// CHECK:STDOUT:   %impl.elem0.loc17_24.2 => constants.%T.as.Destroy.impl.Op.66e
// CHECK:STDOUT:   %specific_impl_fn.loc17_24.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.ad9
// CHECK:STDOUT:   %require_complete.loc17_24.2 => constants.%complete_type.e23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.9c2) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.9c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.11f, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.11f
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.e28
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.f25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.414) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%A.facet.414
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.8ac
// CHECK:STDOUT:   %A.facet => constants.%A.facet.414
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.406
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.93b
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.79b
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.aa8
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.c80
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.2b1
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.1b7
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%T.as.Destroy.impl.Op.66e
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.ad9
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.e23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.414, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.414
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.0b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
