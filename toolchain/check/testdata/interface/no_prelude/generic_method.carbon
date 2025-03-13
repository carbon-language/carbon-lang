// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic_method.carbon

// --- generic_method.carbon

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

impl forall [V1:! type, V2:! type, W:! type] (V1, V2) as A(W) {
  // Here, U has index 3. The specific used for A.F needs to renumber its U from
  // index 2 to index 3.
  fn F(U:! type, u: U) -> (W, (V1, V2), U) {
    return F(U, u);
  }
}

// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.75a: type = facet_type <@A, @A(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %A.type.75a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %U.patt.b3a: type = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.e59: type = tuple_type (type, %A.type.75a, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %tuple.type.bd2: type = tuple_type (%T, %Self.as_type, %U.2cb) [symbolic]
// CHECK:STDOUT:   %F.type.17a: type = fn_type @F.1, @A(%T) [symbolic]
// CHECK:STDOUT:   %F.0d8: %F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.d48: type = assoc_entity_type %A.type.75a [symbolic]
// CHECK:STDOUT:   %assoc0.7cb: %A.assoc_type.d48 = assoc_entity element0, @A.%F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.91f: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %F.type.13d: type = fn_type @F.1, @A(%X) [concrete]
// CHECK:STDOUT:   %F.d83: %F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.f41: type = assoc_entity_type %A.type.91f [concrete]
// CHECK:STDOUT:   %assoc0.929: %A.assoc_type.f41 = assoc_entity element0, @A.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness.ad9: <witness> = impl_witness (@impl.5ec.%F.decl) [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.e01: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.765: type = tuple_type (%X, %Y, %U.8b3) [symbolic]
// CHECK:STDOUT:   %F.type.436: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.ce9: %F.type.436 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.166: %A.type.91f = facet_value %Y, (%impl_witness.ad9) [concrete]
// CHECK:STDOUT:   %tuple.type.a0c: type = tuple_type (type, %A.type.91f, type) [concrete]
// CHECK:STDOUT:   %require_complete.816: <witness> = require_complete_type %tuple.type.765 [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %tuple.type.a41: type = tuple_type (%empty_struct_type, %empty_struct_type, %U.8b3) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %.6fe: type = fn_type_with_self_type %F.type.13d, %A.facet.166 [concrete]
// CHECK:STDOUT:   %tuple.type.f6b: type = tuple_type (%X, %Y, %Z) [concrete]
// CHECK:STDOUT:   %F.specific_fn.f76: <specific function> = specific_function %F.ce9, @F.2(%Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic]
// CHECK:STDOUT:   %V1.patt: type = symbolic_binding_pattern V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic]
// CHECK:STDOUT:   %V2.patt: type = symbolic_binding_pattern V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.f21: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %F.type.904: type = fn_type @F.1, @A(%W) [symbolic]
// CHECK:STDOUT:   %F.1ee: %F.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.e6d: type = assoc_entity_type %A.type.f21 [symbolic]
// CHECK:STDOUT:   %assoc0.4a9: %A.assoc_type.e6d = assoc_entity element0, @A.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.796: <witness> = require_complete_type %A.type.f21 [symbolic]
// CHECK:STDOUT:   %impl_witness.0d8: <witness> = impl_witness (@impl.220.%F.decl), @impl.220(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.753: type = bind_symbolic_name U, 3 [symbolic]
// CHECK:STDOUT:   %U.patt.cfd: type = symbolic_binding_pattern U, 3 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.8ae: type = tuple_type (%W, %tuple.type.30b, %U.753) [symbolic]
// CHECK:STDOUT:   %F.type.b3e: type = fn_type @F.3, @impl.220(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %F.d79: %F.type.b3e = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.412: %A.type.f21 = facet_value %tuple.type.30b, (%impl_witness.0d8) [symbolic]
// CHECK:STDOUT:   %tuple.type.136: type = tuple_type (type, %A.type.f21, type) [symbolic]
// CHECK:STDOUT:   %require_complete.d4d: <witness> = require_complete_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.ed4: <witness> = require_complete_type %U.753 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.7d9: <specific function> = specific_function %F.d79, @F.3(%V1, %V2, %W, %U.753) [symbolic]
// CHECK:STDOUT:   %require_complete.f7f: <witness> = require_complete_type %W [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.2: <witness> = require_complete_type %V1 [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %V2 [symbolic]
// CHECK:STDOUT:   %complete_type.a64: <witness> = complete_type_witness %tuple.type.f6b [concrete]
// CHECK:STDOUT:   %tuple.type.9c8: type = tuple_type (%empty_struct_type, %empty_struct_type, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @impl.5ec [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.91f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.5ec.%F.decl) [concrete = constants.%impl_witness.ad9]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   impl_decl @impl.220 [concrete] {
// CHECK:STDOUT:     %V1.patt.loc25_14.1: type = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc25_14.2 (constants.%V1.patt)]
// CHECK:STDOUT:     %V2.patt.loc25_25.1: type = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc25_25.2 (constants.%V2.patt)]
// CHECK:STDOUT:     %W.patt.loc25_36.1: type = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc25_36.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc25_14.1 [symbolic = %V1.loc25_14.2 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc25_25.1 [symbolic = %V2.loc25_25.2 (constants.%V2)]
// CHECK:STDOUT:     %.loc25_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:     %.loc25_53.2: type = converted %.loc25_53.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc25_36.1 [symbolic = %W.loc25_36.2 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc25_61.1: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc25_61.2 (constants.%A.type.f21)]
// CHECK:STDOUT:     %V1.loc25_14.1: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc25_14.2 (constants.%V1)]
// CHECK:STDOUT:     %V2.loc25_25.1: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc25_25.2 (constants.%V2)]
// CHECK:STDOUT:     %W.loc25_36.1: type = bind_symbolic_name W, 2 [symbolic = %W.loc25_36.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (@impl.220.%F.decl), @impl.220(constants.%V1, constants.%V2, constants.%W) [symbolic = @impl.220.%impl_witness (constants.%impl_witness.0d8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.2)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self.2: %A.type.75a = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @A(%T.loc5_13.2) [symbolic = %F.type (constants.%F.type.17a)]
// CHECK:STDOUT:   %F: @A.%F.type (%F.type.17a) = struct_value () [symbolic = %F (constants.%F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A.%A.type (%A.type.75a) [symbolic = %A.assoc_type (constants.%A.assoc_type.d48)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.d48) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.7cb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @A.%F.type (%F.type.17a) = fn_decl @F.1 [symbolic = @A.%F (constants.%F.0d8)] {
// CHECK:STDOUT:       %U.patt.loc6_8.1: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_8.2 (constants.%U.patt.b3a)]
// CHECK:STDOUT:       %u.patt: @F.1.%U.loc6_8.1 (%U.2cb) = binding_pattern u
// CHECK:STDOUT:       %u.param_patt: @F.1.%U.loc6_8.1 (%U.2cb) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_31: @F.1.%A.type (%A.type.75a) = specific_constant @A.%Self.1, @A(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%A.type (%A.type.75a) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self)]
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
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.d48) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.7cb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5ec: %Y.ref as %A.type {
// CHECK:STDOUT:   %F.decl: %F.type.436 = fn_decl @F.2 [concrete = constants.%F.ce9] {
// CHECK:STDOUT:     %U.patt.loc16_8.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_8.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %u.patt: @F.2.%U.loc16_8.1 (%U.8b3) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.2.%U.loc16_8.1 (%U.8b3) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.2.%tuple.type.loc16 (%tuple.type.765) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.2.%tuple.type.loc16 (%tuple.type.765) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.220(%V1.loc25_14.1: type, %V2.loc25_25.1: type, %W.loc25_36.1: type) {
// CHECK:STDOUT:   %V1.loc25_14.2: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc25_14.2 (constants.%V1)]
// CHECK:STDOUT:   %V1.patt.loc25_14.2: type = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc25_14.2 (constants.%V1.patt)]
// CHECK:STDOUT:   %V2.loc25_25.2: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc25_25.2 (constants.%V2)]
// CHECK:STDOUT:   %V2.patt.loc25_25.2: type = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc25_25.2 (constants.%V2.patt)]
// CHECK:STDOUT:   %W.loc25_36.2: type = bind_symbolic_name W, 2 [symbolic = %W.loc25_36.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc25_36.2: type = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc25_36.2 (constants.%W.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@impl.220.%V1.loc25_14.2 (%V1), @impl.220.%V2.loc25_25.2 (%V2)) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %A.type.loc25_61.2: type = facet_type <@A, @A(%W.loc25_36.2)> [symbolic = %A.type.loc25_61.2 (constants.%A.type.f21)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.220.%A.type.loc25_61.2 (%A.type.f21) [symbolic = %require_complete (constants.%require_complete.796)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl.220(%V1.loc25_14.2, %V2.loc25_25.2, %W.loc25_36.2) [symbolic = %impl_witness (constants.%impl_witness.0d8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.3, @impl.220(%V1.loc25_14.2, %V2.loc25_25.2, %W.loc25_36.2) [symbolic = %F.type (constants.%F.type.b3e)]
// CHECK:STDOUT:   %F: @impl.220.%F.type (%F.type.b3e) = struct_value () [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc25_53.2 as %A.type.loc25_61.1 {
// CHECK:STDOUT:     %F.decl: @impl.220.%F.type (%F.type.b3e) = fn_decl @F.3 [symbolic = @impl.220.%F (constants.%F.d79)] {
// CHECK:STDOUT:       %U.patt.loc28_8.1: type = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc28_8.2 (constants.%U.patt.cfd)]
// CHECK:STDOUT:       %u.patt: @F.3.%U.loc28_8.1 (%U.753) = binding_pattern u
// CHECK:STDOUT:       %u.param_patt: @F.3.%U.loc28_8.1 (%U.753) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @impl.220.%W.loc25_36.1 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @impl.220.%V1.loc25_14.1 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @impl.220.%V2.loc25_25.1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc28_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:       %U.ref.loc28_41: type = name_ref U, %U.loc28_8.2 [symbolic = %U.loc28_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %.loc28_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc28_38, %U.ref.loc28_41)
// CHECK:STDOUT:       %.loc28_42.2: type = converted %.loc28_38, constants.%tuple.type.30b [symbolic = %tuple.type.loc28_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %.loc28_42.3: type = converted %.loc28_42.1, constants.%tuple.type.8ae [symbolic = %tuple.type.loc28_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:       %U.loc28_8.2: type = bind_symbolic_name U, 3 [symbolic = %U.loc28_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u.param: @F.3.%U.loc28_8.1 (%U.753) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc28_21: type = name_ref U, %U.loc28_8.2 [symbolic = %U.loc28_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u: @F.3.%U.loc28_8.1 (%U.753) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@A.%T.loc5_13.1: type, @A.%Self.1: @A.%A.type (%A.type.75a), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %U.patt.loc6_8.2: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_8.2 (constants.%U.patt.b3a)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.75a)]
// CHECK:STDOUT:   %Self: %A.type.75a = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, @F.1.%A.type (%A.type.75a), type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.e59)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (@F.1.%T (%T), @F.1.%Self.as_type.loc6_38.1 (%Self.as_type), @F.1.%U.loc6_8.1 (%U.2cb)) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.bd2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc6_8.1: type, %u.param_patt: @F.1.%U.loc6_8.1 (%U.2cb)) -> @F.1.%tuple.type.loc6_38.2 (%tuple.type.bd2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc16_8.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_8.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (%X, %Y, @F.2.%U.loc16_8.1 (%U.8b3)) [symbolic = %tuple.type.loc16 (constants.%tuple.type.765)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_24: <witness> = require_complete_type @F.2.%tuple.type.loc16 (%tuple.type.765) [symbolic = %require_complete.loc16_24 (constants.%require_complete.816)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type @F.2.%U.loc16_8.1 (%U.8b3) [symbolic = %require_complete.loc16_19 (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (%empty_struct_type, %empty_struct_type, @F.2.%U.loc16_8.1 (%U.8b3)) [symbolic = %tuple.type.loc17 (constants.%tuple.type.a41)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc16_8.1: type, %u.param_patt: @F.2.%U.loc16_8.1 (%U.8b3)) -> %return.param_patt: @F.2.%tuple.type.loc16 (%tuple.type.765) {
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
// CHECK:STDOUT:   %A.facet: %A.type.91f = facet_value constants.%Y, (constants.%impl_witness.ad9) [concrete = constants.%A.facet.166]
// CHECK:STDOUT:   %.loc22_6: %A.type.91f = converted %Y.ref, %A.facet [concrete = constants.%A.facet.166]
// CHECK:STDOUT:   %.loc22_14.1: %A.assoc_type.f41 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.929]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.f41 = name_ref F, %.loc22_14.1 [concrete = constants.%assoc0.929]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc22_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %.loc22_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc22_6, element0 [concrete = constants.%impl_witness.ad9]
// CHECK:STDOUT:   %impl.elem0: %.6fe = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%F.ce9]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%Z) [concrete = constants.%F.specific_fn.f76]
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
// CHECK:STDOUT: generic fn @F.3(@impl.220.%V1.loc25_14.1: type, @impl.220.%V2.loc25_25.1: type, @impl.220.%W.loc25_36.1: type, %U.loc28_8.2: type) {
// CHECK:STDOUT:   %U.loc28_8.1: type = bind_symbolic_name U, 3 [symbolic = %U.loc28_8.1 (constants.%U.753)]
// CHECK:STDOUT:   %U.patt.loc28_8.2: type = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc28_8.2 (constants.%U.patt.cfd)]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.type.loc28_42.1: type = tuple_type (@F.3.%V1 (%V1), @F.3.%V2 (%V2)) [symbolic = %tuple.type.loc28_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %tuple.type.loc28_42.2: type = tuple_type (@F.3.%W (%W), @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b), @F.3.%U.loc28_8.1 (%U.753)) [symbolic = %tuple.type.loc28_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_24: <witness> = require_complete_type @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) [symbolic = %require_complete.loc28_24 (constants.%require_complete.d4d)]
// CHECK:STDOUT:   %require_complete.loc28_19: <witness> = require_complete_type @F.3.%U.loc28_8.1 (%U.753) [symbolic = %require_complete.loc28_19 (constants.%require_complete.ed4)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.3, @impl.220(%V1, %V2, %W) [symbolic = %F.type (constants.%F.type.b3e)]
// CHECK:STDOUT:   %F: @F.3.%F.type (%F.type.b3e) = struct_value () [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:   %F.specific_fn.loc29_12.2: <specific function> = specific_function %F, @F.3(%V1, %V2, %W, %U.loc28_8.1) [symbolic = %F.specific_fn.loc29_12.2 (constants.%F.specific_fn.7d9)]
// CHECK:STDOUT:   %require_complete.loc29_18.1: <witness> = require_complete_type @F.3.%W (%W) [symbolic = %require_complete.loc29_18.1 (constants.%require_complete.f7f)]
// CHECK:STDOUT:   %require_complete.loc29_18.2: <witness> = require_complete_type @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b) [symbolic = %require_complete.loc29_18.2 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %require_complete.loc29_18.3: <witness> = require_complete_type @F.3.%V1 (%V1) [symbolic = %require_complete.loc29_18.3 (constants.%require_complete.4aeca8.2)]
// CHECK:STDOUT:   %require_complete.loc29_18.4: <witness> = require_complete_type @F.3.%V2 (%V2) [symbolic = %require_complete.loc29_18.4 (constants.%require_complete.b54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc28_8.1: type, %u.param_patt: @F.3.%U.loc28_8.1 (%U.753)) -> %return.param_patt: @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc29_12: @F.3.%F.type (%F.type.b3e) = specific_constant @impl.220.%F.decl, @impl.220(constants.%V1, constants.%V2, constants.%W) [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:     %F.ref: @F.3.%F.type (%F.type.b3e) = name_ref F, %.loc29_12 [symbolic = %F (constants.%F.d79)]
// CHECK:STDOUT:     %U.ref.loc29: type = name_ref U, %U.loc28_8.2 [symbolic = %U.loc28_8.1 (constants.%U.753)]
// CHECK:STDOUT:     %u.ref: @F.3.%U.loc28_8.1 (%U.753) = name_ref u, %u
// CHECK:STDOUT:     %F.specific_fn.loc29_12.1: <specific function> = specific_function %F.ref, @F.3(constants.%V1, constants.%V2, constants.%W, constants.%U.753) [symbolic = %F.specific_fn.loc29_12.2 (constants.%F.specific_fn.7d9)]
// CHECK:STDOUT:     %.loc29_18.1: ref @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = temporary_storage
// CHECK:STDOUT:     %F.call: init @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = call %F.specific_fn.loc29_12.1(%u.ref) to %.loc29_18.1
// CHECK:STDOUT:     %.loc29_18.2: ref @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = temporary %.loc29_18.1, %F.call
// CHECK:STDOUT:     %tuple.elem0.loc29_18.1: ref @F.3.%W (%W) = tuple_access %.loc29_18.2, element0
// CHECK:STDOUT:     %.loc29_18.3: @F.3.%W (%W) = bind_value %tuple.elem0.loc29_18.1
// CHECK:STDOUT:     %tuple.elem0.loc29_18.2: ref @F.3.%W (%W) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc29_18.4: init @F.3.%W (%W) = initialize_from %.loc29_18.3 to %tuple.elem0.loc29_18.2
// CHECK:STDOUT:     %tuple.elem1.loc29_18.1: ref @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b) = tuple_access %.loc29_18.2, element1
// CHECK:STDOUT:     %tuple.elem0.loc29_18.3: ref @F.3.%V1 (%V1) = tuple_access %tuple.elem1.loc29_18.1, element0
// CHECK:STDOUT:     %.loc29_18.5: @F.3.%V1 (%V1) = bind_value %tuple.elem0.loc29_18.3
// CHECK:STDOUT:     %tuple.elem1.loc29_18.2: ref @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b) = tuple_access %return, element1
// CHECK:STDOUT:     %tuple.elem0.loc29_18.4: ref @F.3.%V1 (%V1) = tuple_access %tuple.elem1.loc29_18.2, element0
// CHECK:STDOUT:     %.loc29_18.6: init @F.3.%V1 (%V1) = initialize_from %.loc29_18.5 to %tuple.elem0.loc29_18.4
// CHECK:STDOUT:     %tuple.elem1.loc29_18.3: ref @F.3.%V2 (%V2) = tuple_access %tuple.elem1.loc29_18.1, element1
// CHECK:STDOUT:     %.loc29_18.7: @F.3.%V2 (%V2) = bind_value %tuple.elem1.loc29_18.3
// CHECK:STDOUT:     %tuple.elem1.loc29_18.4: ref @F.3.%V2 (%V2) = tuple_access %tuple.elem1.loc29_18.2, element1
// CHECK:STDOUT:     %.loc29_18.8: init @F.3.%V2 (%V2) = initialize_from %.loc29_18.7 to %tuple.elem1.loc29_18.4
// CHECK:STDOUT:     %.loc29_18.9: init @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b) = tuple_init (%.loc29_18.6, %.loc29_18.8) to %tuple.elem1.loc29_18.2
// CHECK:STDOUT:     %.loc29_18.10: init @F.3.%tuple.type.loc28_42.1 (%tuple.type.30b) = converted %tuple.elem1.loc29_18.1, %.loc29_18.9
// CHECK:STDOUT:     %tuple.elem2.loc29_18.1: ref @F.3.%U.loc28_8.1 (%U.753) = tuple_access %.loc29_18.2, element2
// CHECK:STDOUT:     %.loc29_18.11: @F.3.%U.loc28_8.1 (%U.753) = bind_value %tuple.elem2.loc29_18.1
// CHECK:STDOUT:     %tuple.elem2.loc29_18.2: ref @F.3.%U.loc28_8.1 (%U.753) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc29_18.12: init @F.3.%U.loc28_8.1 (%U.753) = initialize_from %.loc29_18.11 to %tuple.elem2.loc29_18.2
// CHECK:STDOUT:     %.loc29_18.13: init @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = tuple_init (%.loc29_18.4, %.loc29_18.10, %.loc29_18.12) to %return
// CHECK:STDOUT:     %.loc29_19: init @F.3.%tuple.type.loc28_42.2 (%tuple.type.8ae) = converted %F.call, %.loc29_18.13
// CHECK:STDOUT:     return %.loc29_19 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %U.patt.loc6_8.2 => constants.%U.2cb
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A.type => constants.%A.type.75a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e59
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bd2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%X
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.13d
// CHECK:STDOUT:   %F => constants.%F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.f41
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.929
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc16_8.2 => constants.%U.8b3
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.765
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%A.facet.166, constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc6_8.2 => constants.%U.8b3
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.91f
// CHECK:STDOUT:   %Self => constants.%A.facet.166
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a0c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.765
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%Z
// CHECK:STDOUT:   %U.patt.loc16_8.2 => constants.%Z
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.f6b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_24 => constants.%complete_type.a64
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.9c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.f21
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.904
// CHECK:STDOUT:   %F => constants.%F.1ee
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.e6d
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.4a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.220(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc25_14.2 => constants.%V1
// CHECK:STDOUT:   %V1.patt.loc25_14.2 => constants.%V1
// CHECK:STDOUT:   %V2.loc25_25.2 => constants.%V2
// CHECK:STDOUT:   %V2.patt.loc25_25.2 => constants.%V2
// CHECK:STDOUT:   %W.loc25_36.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc25_36.2 => constants.%W
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %A.type.loc25_61.2 => constants.%A.type.f21
// CHECK:STDOUT:   %require_complete => constants.%require_complete.796
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.0d8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.b3e
// CHECK:STDOUT:   %F => constants.%F.d79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.220.%W.loc25_36.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.220(%V1.loc25_14.2, %V2.loc25_25.2, %W.loc25_36.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(constants.%V1, constants.%V2, constants.%W, constants.%U.753) {
// CHECK:STDOUT:   %U.loc28_8.1 => constants.%U.753
// CHECK:STDOUT:   %U.patt.loc28_8.2 => constants.%U.753
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.type.loc28_42.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc28_42.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_24 => constants.%require_complete.d4d
// CHECK:STDOUT:   %require_complete.loc28_19 => constants.%require_complete.ed4
// CHECK:STDOUT:   %F.type => constants.%F.type.b3e
// CHECK:STDOUT:   %F => constants.%F.d79
// CHECK:STDOUT:   %F.specific_fn.loc29_12.2 => constants.%F.specific_fn.7d9
// CHECK:STDOUT:   %require_complete.loc29_18.1 => constants.%require_complete.f7f
// CHECK:STDOUT:   %require_complete.loc29_18.2 => constants.%require_complete.fe1
// CHECK:STDOUT:   %require_complete.loc29_18.3 => constants.%require_complete.4aeca8.2
// CHECK:STDOUT:   %require_complete.loc29_18.4 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%W, constants.%A.facet.412, constants.%U.753) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.753
// CHECK:STDOUT:   %U.patt.loc6_8.2 => constants.%U.753
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.f21
// CHECK:STDOUT:   %Self => constants.%A.facet.412
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.136
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.8ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.220(@F.3.%V1, @F.3.%V2, @F.3.%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(%V1, %V2, %W, %U.loc28_8.1) {}
// CHECK:STDOUT:
