// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.bd2: type = tuple_type (%T.8b3, %Self.as_type, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.44c: type = pattern_type %tuple.type.bd2 [symbolic]
// CHECK:STDOUT:   %F.type.17a: type = fn_type @F.1, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %F.0d8: %F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.91f: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.1bb: %A.type.91f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.13d: type = fn_type @F.1, @A(%X) [concrete]
// CHECK:STDOUT:   %F.d83: %F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%F.decl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.765: type = tuple_type (%X, %Y, %U.8b3) [symbolic]
// CHECK:STDOUT:   %pattern_type.340: type = pattern_type %tuple.type.765 [symbolic]
// CHECK:STDOUT:   %F.type.436: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.ce9: %F.type.436 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.552: %A.type.91f = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.a0c: type = tuple_type (type, %A.type.91f, type) [concrete]
// CHECK:STDOUT:   %require_complete.816: <witness> = require_complete_type %tuple.type.765 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %tuple.type.a41: type = tuple_type (%empty_struct_type, %empty_struct_type, %U.8b3) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a60: type = fn_type_with_self_type %F.type.13d, %A.facet.552 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.f6b: type = tuple_type (%X, %Y, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.ef1: type = pattern_type %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ce9, @F.2(%Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %T.9c2: %A.type.91f = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.817: type = pattern_type %A.type.91f [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.9c2 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.9c2, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.11f: %A.type.91f = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ffa: type = fn_type_with_self_type %F.type.13d, %A.facet.11f [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ffa = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.e28: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F.1(%X, %A.facet.11f, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.500: <witness> = require_complete_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.552) [concrete]
// CHECK:STDOUT:   %complete_type.a64: <witness> = complete_type_witness %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %tuple.type.9c8: type = tuple_type (%empty_struct_type, %empty_struct_type, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@impl.%F.decl), @impl [concrete]
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
// CHECK:STDOUT:     %T.loc25_16.1: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc25_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.2)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @A(%T.loc5_13.2) [symbolic = %F.type (constants.%F.type.17a)]
// CHECK:STDOUT:   %F: @A.%F.type (%F.type.17a) = struct_value () [symbolic = %F (constants.%F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.2) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:     %F.decl: @A.%F.type (%F.type.17a) = fn_decl @F.1 [symbolic = @A.%F (constants.%F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @F.1.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @F.1.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.1.%pattern_type.loc6_24 (%pattern_type.44c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.1.%pattern_type.loc6_24 (%pattern_type.44c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.1 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @F.1.%A.type (%A.type.75a) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %Self.ref: @F.1.%A.type (%A.type.75a) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @F.1.%tuple.type.loc6_38.1 (%tuple.type.e59) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.bd2 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @F.1.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @F.1.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Y.ref as %A.type {
// CHECK:STDOUT:   %F.decl: %F.type.436 = fn_decl @F.2 [concrete = constants.%F.ce9] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @F.2.%pattern_type.loc16_18 (%pattern_type.7dc) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.2.%pattern_type.loc16_18 (%pattern_type.7dc) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.2.%pattern_type.loc16_24 (%pattern_type.340) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.2.%pattern_type.loc16_24 (%pattern_type.340) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_34: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %.loc16_35.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %U.ref.loc16_34)
// CHECK:STDOUT:     %.loc16_35.2: type = converted %.loc16_35.1, constants.%tuple.type.765 [symbolic = %tuple.type.loc16 (constants.%tuple.type.765)]
// CHECK:STDOUT:     %U.loc16_8.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %u.param: @F.2.%U.loc16_8.1 (%U.8b3) = value_param call_param0
// CHECK:STDOUT:     %U.ref.loc16_21: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %u: @F.2.%U.loc16_8.1 (%U.8b3) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.2.%tuple.type.loc16 (%tuple.type.765) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.2.%tuple.type.loc16 (%tuple.type.765) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@A.%T.loc5_13.1: type, @A.%Self.1: @A.%A.type (%A.type.75a), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self: @F.1.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.e59)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.44c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.1.%U.loc6_8.1 (%U.2cb)) -> @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%U.loc16_8.2: type) {
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
// CHECK:STDOUT:   fn(%u.param: @F.2.%U.loc16_8.1 (%U.8b3)) -> %return.param: @F.2.%tuple.type.loc16 (%tuple.type.765) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %u.ref: @F.2.%U.loc16_8.1 (%U.8b3) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @F.2.%tuple.type.loc17 (%tuple.type.a41) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %tuple.elem2: ref @F.2.%U.loc16_8.1 (%U.8b3) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc17_22.4: init @F.2.%U.loc16_8.1 (%U.8b3) = initialize_from %u.ref to %tuple.elem2
// CHECK:STDOUT:     %.loc17_22.5: init @F.2.%tuple.type.loc16 (%tuple.type.765) = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4) to %return
// CHECK:STDOUT:     %.loc17_23: init @F.2.%tuple.type.loc16 (%tuple.type.765) = converted %.loc17_22.1, %.loc17_22.5
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
// CHECK:STDOUT:   %impl.elem0: %.a60 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%F.ce9]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%Z) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22_22.1: ref %tuple.type.f6b = temporary_storage
// CHECK:STDOUT:   %.loc22_21.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc22_21.3: init %Z = class_init (), %.loc22_21.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc22_21.4: ref %Z = temporary %.loc22_21.2, %.loc22_21.3
// CHECK:STDOUT:   %.loc22_21.5: ref %Z = converted %.loc22_21.1, %.loc22_21.4
// CHECK:STDOUT:   %.loc22_21.6: %Z = bind_value %.loc22_21.5
// CHECK:STDOUT:   %F.call: init %tuple.type.f6b = call %specific_fn(%.loc22_21.6) to %.loc22_22.1
// CHECK:STDOUT:   %.loc22_22.2: ref %tuple.type.f6b = temporary %.loc22_22.1, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc25_16.1: %A.type.91f) {
// CHECK:STDOUT:   %T.loc25_16.2: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc25_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc26_4.2: type = facet_access_type %T.loc25_16.2 [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc25_16.2, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value %T.as_type.loc26_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.11f)]
// CHECK:STDOUT:   %.loc26_4.3: type = fn_type_with_self_type constants.%F.type.13d, %A.facet [symbolic = %.loc26_4.3 (constants.%.ffa)]
// CHECK:STDOUT:   %impl.elem0.loc26_4.2: @CallGeneric.%.loc26_4.3 (%.ffa) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_4.2: <specific function> = specific_impl_function %impl.elem0.loc26_4.2, @F.1(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc26_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.e28)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.500)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.91f = name_ref T, %T.loc25_16.1 [symbolic = %T.loc25_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:     %.loc26_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc26_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc26_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_4.2: type = converted %T.ref, %T.as_type.loc26_4.1 [symbolic = %T.as_type.loc26_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc26_4.1: @CallGeneric.%.loc26_4.3 (%.ffa) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc26_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc26_4.1: <specific function> = specific_impl_function %impl.elem0.loc26_4.1, @F.1(constants.%X, constants.%A.facet.11f, constants.%Z) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc26_12.1: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary_storage
// CHECK:STDOUT:     %.loc26_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc26_11.3: init %Z = class_init (), %.loc26_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc26_11.4: ref %Z = temporary %.loc26_11.2, %.loc26_11.3
// CHECK:STDOUT:     %.loc26_11.5: ref %Z = converted %.loc26_11.1, %.loc26_11.4
// CHECK:STDOUT:     %.loc26_11.6: %Z = bind_value %.loc26_11.5
// CHECK:STDOUT:     %.loc26_12.2: init @CallGeneric.%tuple.type (%tuple.type.e28) = call %specific_impl_fn.loc26_4.1(%.loc26_11.6) to %.loc26_12.1
// CHECK:STDOUT:     %.loc26_12.3: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary %.loc26_12.1, %.loc26_12.2
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
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.8b3, constants.%Self.753, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.75a
// CHECK:STDOUT:   %Self => constants.%Self.753
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e59
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bd2
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.44c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self.2 => constants.%Self.1bb
// CHECK:STDOUT:   %F.type => constants.%F.type.13d
// CHECK:STDOUT:   %F => constants.%F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.765
// CHECK:STDOUT:   %pattern_type.loc16_24 => constants.%pattern_type.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.552, constants.%U.8b3) {
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
// CHECK:STDOUT: specific @F.2(constants.%Z) {
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
// CHECK:STDOUT:   %T.loc25_16.2 => constants.%T.9c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.11f, constants.%Z) {
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
// CHECK:STDOUT:   %T.loc25_16.2 => constants.%A.facet.552
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc26_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.552
// CHECK:STDOUT:   %.loc26_4.3 => constants.%.a60
// CHECK:STDOUT:   %impl.elem0.loc26_4.2 => constants.%F.ce9
// CHECK:STDOUT:   %specific_impl_fn.loc26_4.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.f6b
// CHECK:STDOUT:   %require_complete => constants.%complete_type.a64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.552, constants.%Z) {
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
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.bd2: type = tuple_type (%T.8b3, %Self.as_type, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.44c: type = pattern_type %tuple.type.bd2 [symbolic]
// CHECK:STDOUT:   %F.type.17a: type = fn_type @F.1, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %F.0d8: %F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.f21: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.4fe: %A.type.f21 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.904: type = fn_type @F.1, @A(%W) [symbolic]
// CHECK:STDOUT:   %F.1ee: %F.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.b05: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.c82: %A.assoc_type.b05 = assoc_entity element0, @A.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.796: <witness> = require_complete_type %A.type.f21 [symbolic]
// CHECK:STDOUT:   %A.impl_witness.f15: <witness> = impl_witness file.%A.impl_witness_table, @impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.753: type = bind_symbolic_name U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.549: type = pattern_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.8ae: type = tuple_type (%W, %tuple.type.30b, %U.753) [symbolic]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %F.type.b3e: type = fn_type @F.2, @impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %F.d79: %F.type.b3e = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.461: %A.type.f21 = facet_value %tuple.type.30b, (%A.impl_witness.f15) [symbolic]
// CHECK:STDOUT:   %tuple.type.136: type = tuple_type (type, %A.type.f21, type) [symbolic]
// CHECK:STDOUT:   %require_complete.d4d: <witness> = require_complete_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.ed4: <witness> = require_complete_type %U.753 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.7d9: <specific function> = specific_function %F.d79, @F.2(%V1, %V2, %W, %U.753) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.91f: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.1bb: %A.type.91f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.13d: type = fn_type @F.1, @A(%X) [concrete]
// CHECK:STDOUT:   %F.d83: %F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%F.decl [concrete]
// CHECK:STDOUT:   %complete_type.d76: <witness> = complete_type_witness %A.type.91f [concrete]
// CHECK:STDOUT:   %A.impl_witness.8ac: <witness> = impl_witness file.%A.impl_witness_table, @impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %F.type.bf7: type = fn_type @F.2, @impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %F.93b: %F.type.bf7 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.414: %A.type.91f = facet_value %tuple.type.a46, (%A.impl_witness.8ac) [concrete]
// CHECK:STDOUT:   %.406: type = fn_type_with_self_type %F.type.13d, %A.facet.414 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %F.specific_fn.79b: <specific function> = specific_function %F.93b, @F.2(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %T.9c2: %A.type.91f = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.817: type = pattern_type %A.type.91f [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.9c2 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.9c2, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.11f: %A.type.91f = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ffa: type = fn_type_with_self_type %F.type.13d, %A.facet.11f [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ffa = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.a0c: type = tuple_type (type, %A.type.91f, type) [concrete]
// CHECK:STDOUT:   %tuple.type.e28: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F.1(%X, %A.facet.11f, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.500: <witness> = require_complete_type %tuple.type.e28 [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.414) [concrete]
// CHECK:STDOUT:   %complete_type.aa8: <witness> = complete_type_witness %tuple.type.415 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y1.decl: type = class_decl @Y1 [concrete = constants.%Y1] {} {}
// CHECK:STDOUT:   %Y2.decl: type = class_decl @Y2 [concrete = constants.%Y2] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %V1.patt: %pattern_type.98f = symbolic_binding_pattern V1, 0 [concrete]
// CHECK:STDOUT:     %V2.patt: %pattern_type.98f = symbolic_binding_pattern V2, 1 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_14.1 [symbolic = %V1.loc14_14.2 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_25.1 [symbolic = %V2.loc14_25.2 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_36.1 [symbolic = %W.loc14_36.2 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.1: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.2 (constants.%A.type.f21)]
// CHECK:STDOUT:     %V1.loc14_14.1: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.2 (constants.%V1)]
// CHECK:STDOUT:     %V2.loc14_25.1: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.2 (constants.%V2)]
// CHECK:STDOUT:     %W.loc14_36.1: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@impl.%F.decl), @impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @impl.%A.impl_witness (constants.%A.impl_witness.f15)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.817 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.91f] {
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.1: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.2)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @A(%T.loc5_13.2) [symbolic = %F.type (constants.%F.type.17a)]
// CHECK:STDOUT:   %F: @A.%F.type (%F.type.17a) = struct_value () [symbolic = %F (constants.%F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.2) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:     %F.decl: @A.%F.type (%F.type.17a) = fn_decl @F.1 [symbolic = @A.%F (constants.%F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @F.1.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @F.1.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.1.%pattern_type.loc6_24 (%pattern_type.44c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.1.%pattern_type.loc6_24 (%pattern_type.44c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.1 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @F.1.%A.type (%A.type.75a) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %Self.ref: @F.1.%A.type (%A.type.75a) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @F.1.%tuple.type.loc6_38.1 (%tuple.type.e59) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.bd2 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @F.1.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @F.1.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V1.loc14_14.1: type, %V2.loc14_25.1: type, %W.loc14_36.1: type) {
// CHECK:STDOUT:   %V1.loc14_14.2: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.2 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_25.2: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.2 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_36.2: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.2 (constants.%W)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.2, %V2.loc14_25.2) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %A.type.loc14_61.2: type = facet_type <@A, @A(%W.loc14_36.2)> [symbolic = %A.type.loc14_61.2 (constants.%A.type.f21)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.2 [symbolic = %require_complete (constants.%require_complete.796)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @impl(%V1.loc14_14.2, %V2.loc14_25.2, %W.loc14_36.2) [symbolic = %A.impl_witness (constants.%A.impl_witness.f15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V1.loc14_14.2, %V2.loc14_25.2, %W.loc14_36.2) [symbolic = %F.type (constants.%F.type.b3e)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.b3e) = struct_value () [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.b3e) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.d79)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @F.2.%pattern_type.loc17_18 (%pattern_type.549) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @F.2.%pattern_type.loc17_18 (%pattern_type.549) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.2.%pattern_type.loc17_24 (%pattern_type.d4a) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.2.%pattern_type.loc17_24 (%pattern_type.d4a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @impl.%W.loc14_36.1 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @impl.%V1.loc14_14.1 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @impl.%V2.loc14_25.1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %.loc17_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41)
// CHECK:STDOUT:       %.loc17_42.2: type = converted %.loc17_38, constants.%tuple.type.30b [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %.loc17_42.3: type = converted %.loc17_42.1, constants.%tuple.type.8ae [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:       %U.loc17_8.2: type = bind_symbolic_name U, 3 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u.param: @F.2.%U.loc17_8.1 (%U.753) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u: @F.2.%U.loc17_8.1 (%U.753) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @F.2.%tuple.type.loc17_42.2 (%tuple.type.8ae) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.2.%tuple.type.loc17_42.2 (%tuple.type.8ae) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%A.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y1 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y2 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@A.%T.loc5_13.1: type, @A.%Self.1: @A.%A.type (%A.type.75a), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self: @F.1.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.753)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.e59)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.44c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.1.%U.loc6_8.1 (%U.2cb)) -> @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V1.loc14_14.1: type, @impl.%V2.loc14_25.1: type, @impl.%W.loc14_36.1: type, %U.loc17_8.2: type) {
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
// CHECK:STDOUT:   %require_complete.loc17_24: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24 (constants.%require_complete.d4d)]
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_8.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.ed4)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V1, %V2, %W) [symbolic = %F.type (constants.%F.type.b3e)]
// CHECK:STDOUT:   %F: @F.2.%F.type (%F.type.b3e) = struct_value () [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:   %F.specific_fn.loc18_12.2: <specific function> = specific_function %F, @F.2(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %F.specific_fn.loc18_12.2 (constants.%F.specific_fn.7d9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.2.%U.loc17_8.1 (%U.753)) -> %return.param: @F.2.%tuple.type.loc17_42.2 (%tuple.type.8ae) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @F.2.%F.type (%F.type.b3e) = specific_constant @impl.%F.decl, @impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:     %F.ref: @F.2.%F.type (%F.type.b3e) = name_ref F, %.loc18 [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:     %u.ref: @F.2.%U.loc17_8.1 (%U.753) = name_ref u, %u
// CHECK:STDOUT:     %F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @F.2(constants.%V1, constants.%V2, constants.%W, constants.%U.753) [symbolic = %F.specific_fn.loc18_12.2 (constants.%F.specific_fn.7d9)]
// CHECK:STDOUT:     %.loc17_24: ref @F.2.%tuple.type.loc17_42.2 (%tuple.type.8ae) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.2.%tuple.type.loc17_42.2 (%tuple.type.8ae) = call %F.specific_fn.loc18_12.1(%u.ref) to %.loc17_24
// CHECK:STDOUT:     return %F.call to %return
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
// CHECK:STDOUT:   %impl.elem0: %.406 = impl_witness_access constants.%A.impl_witness.8ac, element0 [concrete = constants.%F.93b]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%F.specific_fn.79b]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = bind_value %.loc24_38.5
// CHECK:STDOUT:   %F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.415 = temporary %.loc24_39.1, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.1: %A.type.91f) {
// CHECK:STDOUT:   %T.loc27_16.2: %A.type.91f = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_16.2 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.2, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.11f)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%F.type.13d, %A.facet [symbolic = %.loc28_4.3 (constants.%.ffa)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ffa) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @F.1(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.e28)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.500)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.91f = name_ref T, %T.loc27_16.1 [symbolic = %T.loc27_16.2 (constants.%T.9c2)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ffa) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @F.1(constants.%X, constants.%A.facet.11f, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = bind_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.e28) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.e28) = temporary %.loc28_12.1, %.loc28_12.2
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
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.8b3, constants.%Self.753, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.75a
// CHECK:STDOUT:   %Self => constants.%Self.753
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e59
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bd2
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.44c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.f21
// CHECK:STDOUT:   %Self.2 => constants.%Self.4fe
// CHECK:STDOUT:   %F.type => constants.%F.type.904
// CHECK:STDOUT:   %F => constants.%F.1ee
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.b05
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.c82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.2 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_25.2 => constants.%V2
// CHECK:STDOUT:   %W.loc14_36.2 => constants.%W
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %A.type.loc14_61.2 => constants.%A.type.f21
// CHECK:STDOUT:   %require_complete => constants.%require_complete.796
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.f15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.b3e
// CHECK:STDOUT:   %F => constants.%F.d79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V1, constants.%V2, constants.%W, constants.%U.753) {
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
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%require_complete.d4d
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.ed4
// CHECK:STDOUT:   %F.type => constants.%F.type.b3e
// CHECK:STDOUT:   %F => constants.%F.d79
// CHECK:STDOUT:   %F.specific_fn.loc18_12.2 => constants.%F.specific_fn.7d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%W, constants.%A.facet.461, constants.%U.753) {
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
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self.2 => constants.%Self.1bb
// CHECK:STDOUT:   %F.type => constants.%F.type.13d
// CHECK:STDOUT:   %F => constants.%F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_14.2 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_25.2 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_36.2 => constants.%X
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.type.loc14_61.2 => constants.%A.type.91f
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d76
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.8ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.bf7
// CHECK:STDOUT:   %F => constants.%F.93b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
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
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%complete_type.aa8
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %F.type => constants.%F.type.bf7
// CHECK:STDOUT:   %F => constants.%F.93b
// CHECK:STDOUT:   %F.specific_fn.loc18_12.2 => constants.%F.specific_fn.79b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.9c2) {
// CHECK:STDOUT:   %T.loc27_16.2 => constants.%T.9c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.11f, constants.%Z) {
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
// CHECK:STDOUT:   %T.loc27_16.2 => constants.%A.facet.414
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.8ac
// CHECK:STDOUT:   %A.facet => constants.%A.facet.414
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.406
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%F.93b
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%F.specific_fn.79b
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.414, constants.%Z) {
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
