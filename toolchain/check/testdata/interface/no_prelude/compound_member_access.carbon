// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon

interface A {}
interface B {}
impl forall [T:! A] T as B {}
fn FB[TB:! B](xb: TB) {}
fn FA[TA:! A](xa: TA) {
  FB(xa);
}
fn FC[TC:! A where .Self impls type](xc: TC) {
  FA(xc);
}

interface J {
  let U:! type;
}

// Works.
// fn F(T:! J, S:! T.U) {}

// CHECK:STDERR: compound_member_access.carbon:[[@LINE+4]]:17: error: cannot access member of interface `J` in type `J` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: fn G(V:! J, W:! V.(J.U)) {}
// CHECK:STDERR:                 ^~~~~~~
// CHECK:STDERR:
fn G(V:! J, W:! V.(J.U)) {}

// CHECK:STDOUT: --- compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.783: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %A.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness.94e906.1: <witness> = impl_witness (), @impl(%T) [symbolic]
// CHECK:STDOUT:   %TB: %B.type = bind_symbolic_name TB, 0 [symbolic]
// CHECK:STDOUT:   %TB.patt: %B.type = symbolic_binding_pattern TB, 0 [symbolic]
// CHECK:STDOUT:   %TB.as_type: type = facet_access_type %TB [symbolic]
// CHECK:STDOUT:   %FB.type: type = fn_type @FB [concrete]
// CHECK:STDOUT:   %FB: %FB.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e08: <witness> = require_complete_type %TB.as_type [symbolic]
// CHECK:STDOUT:   %TA: %A.type = bind_symbolic_name TA, 0 [symbolic]
// CHECK:STDOUT:   %TA.patt: %A.type = symbolic_binding_pattern TA, 0 [symbolic]
// CHECK:STDOUT:   %TA.as_type: type = facet_access_type %TA [symbolic]
// CHECK:STDOUT:   %FA.type: type = fn_type @FA [concrete]
// CHECK:STDOUT:   %FA: %FA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf4: <witness> = require_complete_type %TA.as_type [symbolic]
// CHECK:STDOUT:   %impl_witness.94e906.2: <witness> = impl_witness (), @impl(%TA) [symbolic]
// CHECK:STDOUT:   %B.facet.f1b: %B.type = facet_value %TA.as_type, %impl_witness.94e906.2 [symbolic]
// CHECK:STDOUT:   %FB.specific_fn.41c: <specific function> = specific_function %FB, @FB(%B.facet.f1b) [symbolic]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where TODO> [concrete]
// CHECK:STDOUT:   %TC: %A_where.type = bind_symbolic_name TC, 0 [symbolic]
// CHECK:STDOUT:   %TC.patt: %A_where.type = symbolic_binding_pattern TC, 0 [symbolic]
// CHECK:STDOUT:   %TC.as_type: type = facet_access_type %TC [symbolic]
// CHECK:STDOUT:   %FC.type: type = fn_type @FC [concrete]
// CHECK:STDOUT:   %FC: %FC.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.732: <witness> = require_complete_type %TC.as_type [symbolic]
// CHECK:STDOUT:   %TC.as_wit: <witness> = facet_access_witness %TC [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %TC.as_type, %TC.as_wit [symbolic]
// CHECK:STDOUT:   %FA.specific_fn: <specific function> = specific_function %FA, @FA(%A.facet) [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %V: %J.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.1be: <witness> = impl_witness (), @impl(%A.facet) [symbolic]
// CHECK:STDOUT:   %B.facet.d0b: %B.type = facet_value %TC.as_type, %impl_witness.1be [symbolic]
// CHECK:STDOUT:   %FB.specific_fn.328: <specific function> = specific_function %FB, @FB(%B.facet.d0b) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .FB = %FB.decl
// CHECK:STDOUT:     .FA = %FA.decl
// CHECK:STDOUT:     .FC = %FC.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_14.1: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %A.type = value_param_pattern %T.patt.loc13_14.1, runtime_param<none> [symbolic = %T.patt.loc13_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc13_14.1 [symbolic = %T.loc13_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc13_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13: type = converted %T.ref, %T.as_type.loc13_21.1 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %T.param: %A.type = value_param runtime_param<none>
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %T.loc13_14.1: %A.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.94e906.1)]
// CHECK:STDOUT:   %FB.decl: %FB.type = fn_decl @FB [concrete = constants.%FB] {
// CHECK:STDOUT:     %TB.patt.loc14_7.1: %B.type = symbolic_binding_pattern TB, 0 [symbolic = %TB.patt.loc14_7.2 (constants.%TB.patt)]
// CHECK:STDOUT:     %TB.param_patt: %B.type = value_param_pattern %TB.patt.loc14_7.1, runtime_param<none> [symbolic = %TB.patt.loc14_7.2 (constants.%TB.patt)]
// CHECK:STDOUT:     %xb.patt: @FB.%TB.as_type.loc14_19.2 (%TB.as_type) = binding_pattern xb
// CHECK:STDOUT:     %xb.param_patt: @FB.%TB.as_type.loc14_19.2 (%TB.as_type) = value_param_pattern %xb.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %TB.param: %B.type = value_param runtime_param<none>
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %TB.loc14_7.1: %B.type = bind_symbolic_name TB, 0, %TB.param [symbolic = %TB.loc14_7.2 (constants.%TB)]
// CHECK:STDOUT:     %xb.param: @FB.%TB.as_type.loc14_19.2 (%TB.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %TB.as_type.loc14_19.2 (constants.%TB.as_type)] {
// CHECK:STDOUT:       %TB.ref: %B.type = name_ref TB, %TB.loc14_7.1 [symbolic = %TB.loc14_7.2 (constants.%TB)]
// CHECK:STDOUT:       %TB.as_type.loc14_19.1: type = facet_access_type %TB.ref [symbolic = %TB.as_type.loc14_19.2 (constants.%TB.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %TB.ref, %TB.as_type.loc14_19.1 [symbolic = %TB.as_type.loc14_19.2 (constants.%TB.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xb: @FB.%TB.as_type.loc14_19.2 (%TB.as_type) = bind_name xb, %xb.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FA.decl: %FA.type = fn_decl @FA [concrete = constants.%FA] {
// CHECK:STDOUT:     %TA.patt.loc15_7.1: %A.type = symbolic_binding_pattern TA, 0 [symbolic = %TA.patt.loc15_7.2 (constants.%TA.patt)]
// CHECK:STDOUT:     %TA.param_patt: %A.type = value_param_pattern %TA.patt.loc15_7.1, runtime_param<none> [symbolic = %TA.patt.loc15_7.2 (constants.%TA.patt)]
// CHECK:STDOUT:     %xa.patt: @FA.%TA.as_type.loc15_19.2 (%TA.as_type) = binding_pattern xa
// CHECK:STDOUT:     %xa.param_patt: @FA.%TA.as_type.loc15_19.2 (%TA.as_type) = value_param_pattern %xa.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %TA.param: %A.type = value_param runtime_param<none>
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %TA.loc15_7.1: %A.type = bind_symbolic_name TA, 0, %TA.param [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:     %xa.param: @FA.%TA.as_type.loc15_19.2 (%TA.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc15_19.1: type = splice_block %.loc15_19.2 [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)] {
// CHECK:STDOUT:       %TA.ref: %A.type = name_ref TA, %TA.loc15_7.1 [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:       %TA.as_type.loc15_19.1: type = facet_access_type %TA.ref [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:       %.loc15_19.2: type = converted %TA.ref, %TA.as_type.loc15_19.1 [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xa: @FA.%TA.as_type.loc15_19.2 (%TA.as_type) = bind_name xa, %xa.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FC.decl: %FC.type = fn_decl @FC [concrete = constants.%FC] {
// CHECK:STDOUT:     %TC.patt.loc18_7.1: %A_where.type = symbolic_binding_pattern TC, 0 [symbolic = %TC.patt.loc18_7.2 (constants.%TC.patt)]
// CHECK:STDOUT:     %TC.param_patt: %A_where.type = value_param_pattern %TC.patt.loc18_7.1, runtime_param<none> [symbolic = %TC.patt.loc18_7.2 (constants.%TC.patt)]
// CHECK:STDOUT:     %xc.patt: @FC.%TC.as_type.loc18_42.2 (%TC.as_type) = binding_pattern xc
// CHECK:STDOUT:     %xc.param_patt: @FC.%TC.as_type.loc18_42.2 (%TC.as_type) = value_param_pattern %xc.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %TC.param: %A_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc18_14.1: type = splice_block %.loc18_14.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc18_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc18_14.2: type = where_expr %.Self [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc18_20, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %TC.loc18_7.1: %A_where.type = bind_symbolic_name TC, 0, %TC.param [symbolic = %TC.loc18_7.2 (constants.%TC)]
// CHECK:STDOUT:     %xc.param: @FC.%TC.as_type.loc18_42.2 (%TC.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_42.1: type = splice_block %.loc18_42.2 [symbolic = %TC.as_type.loc18_42.2 (constants.%TC.as_type)] {
// CHECK:STDOUT:       %TC.ref: %A_where.type = name_ref TC, %TC.loc18_7.1 [symbolic = %TC.loc18_7.2 (constants.%TC)]
// CHECK:STDOUT:       %TC.as_type.loc18_42.1: type = facet_access_type %TC.ref [symbolic = %TC.as_type.loc18_42.2 (constants.%TC.as_type)]
// CHECK:STDOUT:       %.loc18_42.2: type = converted %TC.ref, %TC.as_type.loc18_42.1 [symbolic = %TC.as_type.loc18_42.2 (constants.%TC.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xc: @FC.%TC.as_type.loc18_42.2 (%TC.as_type) = bind_name xc, %xc.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %V.patt.loc33_6.1: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc33_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %J.type = value_param_pattern %V.patt.loc33_6.1, runtime_param<none> [symbolic = %V.patt.loc33_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc33_13.1: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc33_13.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: <error> = value_param_pattern %W.patt.loc33_13.1, runtime_param<none> [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.param: %J.type = value_param runtime_param<none>
// CHECK:STDOUT:     %J.ref.loc33_10: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %V.loc33_6.1: %J.type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc33_6.2 (constants.%V)]
// CHECK:STDOUT:     %W.param: <error> = value_param runtime_param<none>
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc33_6.1 [symbolic = %V.loc33_6.2 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc33_20: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 1, %W.param [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.783]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc13_14.1: %A.type) {
// CHECK:STDOUT:   %T.loc13_14.2: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_14.2: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc13_21.2: type = facet_access_type %T.loc13_14.2 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc13_14.2) [symbolic = %impl_witness (constants.%impl_witness.94e906.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13 as %B.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FB(%TB.loc14_7.1: %B.type) {
// CHECK:STDOUT:   %TB.loc14_7.2: %B.type = bind_symbolic_name TB, 0 [symbolic = %TB.loc14_7.2 (constants.%TB)]
// CHECK:STDOUT:   %TB.patt.loc14_7.2: %B.type = symbolic_binding_pattern TB, 0 [symbolic = %TB.patt.loc14_7.2 (constants.%TB.patt)]
// CHECK:STDOUT:   %TB.as_type.loc14_19.2: type = facet_access_type %TB.loc14_7.2 [symbolic = %TB.as_type.loc14_19.2 (constants.%TB.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @FB.%TB.as_type.loc14_19.2 (%TB.as_type) [symbolic = %require_complete (constants.%require_complete.e08)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%TB.param_patt: %B.type](%xb.param_patt: @FB.%TB.as_type.loc14_19.2 (%TB.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FA(%TA.loc15_7.1: %A.type) {
// CHECK:STDOUT:   %TA.loc15_7.2: %A.type = bind_symbolic_name TA, 0 [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:   %TA.patt.loc15_7.2: %A.type = symbolic_binding_pattern TA, 0 [symbolic = %TA.patt.loc15_7.2 (constants.%TA.patt)]
// CHECK:STDOUT:   %TA.as_type.loc15_19.2: type = facet_access_type %TA.loc15_7.2 [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @FA.%TA.as_type.loc15_19.2 (%TA.as_type) [symbolic = %require_complete (constants.%require_complete.cf4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%TA.loc15_7.2) [symbolic = %impl_witness (constants.%impl_witness.94e906.2)]
// CHECK:STDOUT:   %B.facet.loc16_8.3: %B.type = facet_value %TA.as_type.loc15_19.2, %impl_witness [symbolic = %B.facet.loc16_8.3 (constants.%B.facet.f1b)]
// CHECK:STDOUT:   %FB.specific_fn.loc16_3.2: <specific function> = specific_function constants.%FB, @FB(%B.facet.loc16_8.3) [symbolic = %FB.specific_fn.loc16_3.2 (constants.%FB.specific_fn.41c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%TA.param_patt: %A.type](%xa.param_patt: @FA.%TA.as_type.loc15_19.2 (%TA.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FB.ref: %FB.type = name_ref FB, file.%FB.decl [concrete = constants.%FB]
// CHECK:STDOUT:     %xa.ref: @FA.%TA.as_type.loc15_19.2 (%TA.as_type) = name_ref xa, %xa
// CHECK:STDOUT:     %TA.as_type.loc16_8.1: type = facet_access_type constants.%TA [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:     %.loc16_8.1: type = converted constants.%TA, %TA.as_type.loc16_8.1 [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:     %.loc16_8.2: %A.type = converted %.loc16_8.1, constants.%TA [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:     %.loc16_8.3: %A.type = converted constants.%TA.as_type, constants.%TA [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:     %B.facet.loc16_8.1: %B.type = facet_value constants.%TA.as_type, constants.%impl_witness.94e906.2 [symbolic = %B.facet.loc16_8.3 (constants.%B.facet.f1b)]
// CHECK:STDOUT:     %.loc16_8.4: %B.type = converted constants.%TA.as_type, %B.facet.loc16_8.1 [symbolic = %B.facet.loc16_8.3 (constants.%B.facet.f1b)]
// CHECK:STDOUT:     %TA.as_type.loc16_8.2: type = facet_access_type constants.%TA [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:     %.loc16_8.5: type = converted constants.%TA, %TA.as_type.loc16_8.2 [symbolic = %TA.as_type.loc15_19.2 (constants.%TA.as_type)]
// CHECK:STDOUT:     %.loc16_8.6: %A.type = converted %.loc16_8.5, constants.%TA [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:     %.loc16_8.7: %A.type = converted constants.%TA.as_type, constants.%TA [symbolic = %TA.loc15_7.2 (constants.%TA)]
// CHECK:STDOUT:     %B.facet.loc16_8.2: %B.type = facet_value constants.%TA.as_type, constants.%impl_witness.94e906.2 [symbolic = %B.facet.loc16_8.3 (constants.%B.facet.f1b)]
// CHECK:STDOUT:     %.loc16_8.8: %B.type = converted constants.%TA.as_type, %B.facet.loc16_8.2 [symbolic = %B.facet.loc16_8.3 (constants.%B.facet.f1b)]
// CHECK:STDOUT:     %FB.specific_fn.loc16_3.1: <specific function> = specific_function %FB.ref, @FB(constants.%B.facet.f1b) [symbolic = %FB.specific_fn.loc16_3.2 (constants.%FB.specific_fn.41c)]
// CHECK:STDOUT:     %FB.call: init %empty_tuple.type = call %FB.specific_fn.loc16_3.1(%xa.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FC(%TC.loc18_7.1: %A_where.type) {
// CHECK:STDOUT:   %TC.loc18_7.2: %A_where.type = bind_symbolic_name TC, 0 [symbolic = %TC.loc18_7.2 (constants.%TC)]
// CHECK:STDOUT:   %TC.patt.loc18_7.2: %A_where.type = symbolic_binding_pattern TC, 0 [symbolic = %TC.patt.loc18_7.2 (constants.%TC.patt)]
// CHECK:STDOUT:   %TC.as_type.loc18_42.2: type = facet_access_type %TC.loc18_7.2 [symbolic = %TC.as_type.loc18_42.2 (constants.%TC.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @FC.%TC.as_type.loc18_42.2 (%TC.as_type) [symbolic = %require_complete (constants.%require_complete.732)]
// CHECK:STDOUT:   %TC.as_wit.loc19_8.3: <witness> = facet_access_witness %TC.loc18_7.2 [symbolic = %TC.as_wit.loc19_8.3 (constants.%TC.as_wit)]
// CHECK:STDOUT:   %A.facet.loc19_8.3: %A.type = facet_value %TC.as_type.loc18_42.2, %TC.as_wit.loc19_8.3 [symbolic = %A.facet.loc19_8.3 (constants.%A.facet)]
// CHECK:STDOUT:   %FA.specific_fn.loc19_3.2: <specific function> = specific_function constants.%FA, @FA(%A.facet.loc19_8.3) [symbolic = %FA.specific_fn.loc19_3.2 (constants.%FA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%TC.param_patt: %A_where.type](%xc.param_patt: @FC.%TC.as_type.loc18_42.2 (%TC.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FA.ref: %FA.type = name_ref FA, file.%FA.decl [concrete = constants.%FA]
// CHECK:STDOUT:     %xc.ref: @FC.%TC.as_type.loc18_42.2 (%TC.as_type) = name_ref xc, %xc
// CHECK:STDOUT:     %TC.as_wit.loc19_8.1: <witness> = facet_access_witness constants.%TC [symbolic = %TC.as_wit.loc19_8.3 (constants.%TC.as_wit)]
// CHECK:STDOUT:     %A.facet.loc19_8.1: %A.type = facet_value constants.%TC.as_type, %TC.as_wit.loc19_8.1 [symbolic = %A.facet.loc19_8.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc19_8.1: %A.type = converted constants.%TC.as_type, %A.facet.loc19_8.1 [symbolic = %A.facet.loc19_8.3 (constants.%A.facet)]
// CHECK:STDOUT:     %TC.as_wit.loc19_8.2: <witness> = facet_access_witness constants.%TC [symbolic = %TC.as_wit.loc19_8.3 (constants.%TC.as_wit)]
// CHECK:STDOUT:     %A.facet.loc19_8.2: %A.type = facet_value constants.%TC.as_type, %TC.as_wit.loc19_8.2 [symbolic = %A.facet.loc19_8.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc19_8.2: %A.type = converted constants.%TC.as_type, %A.facet.loc19_8.2 [symbolic = %A.facet.loc19_8.3 (constants.%A.facet)]
// CHECK:STDOUT:     %FA.specific_fn.loc19_3.1: <specific function> = specific_function %FA.ref, @FA(constants.%A.facet) [symbolic = %FA.specific_fn.loc19_3.2 (constants.%FA.specific_fn)]
// CHECK:STDOUT:     %FA.call: init %empty_tuple.type = call %FA.specific_fn.loc19_3.1(%xc.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%V.loc33_6.1: %J.type, %W: <error>) {
// CHECK:STDOUT:   %V.loc33_6.2: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc33_6.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc33_6.2: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc33_6.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.patt.loc33_13.2: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc33_13.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %J.type, %W.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc13_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.94e906.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc13_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FB(constants.%TB) {
// CHECK:STDOUT:   %TB.loc14_7.2 => constants.%TB
// CHECK:STDOUT:   %TB.patt.loc14_7.2 => constants.%TB
// CHECK:STDOUT:   %TB.as_type.loc14_19.2 => constants.%TB.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FA(constants.%TA) {
// CHECK:STDOUT:   %TA.loc15_7.2 => constants.%TA
// CHECK:STDOUT:   %TA.patt.loc15_7.2 => constants.%TA
// CHECK:STDOUT:   %TA.as_type.loc15_19.2 => constants.%TA.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%TA) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%TA
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%TA
// CHECK:STDOUT:   %T.as_type.loc13_21.2 => constants.%TA.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.94e906.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FB(constants.%B.facet.f1b) {
// CHECK:STDOUT:   %TB.loc14_7.2 => constants.%B.facet.f1b
// CHECK:STDOUT:   %TB.patt.loc14_7.2 => constants.%B.facet.f1b
// CHECK:STDOUT:   %TB.as_type.loc14_19.2 => constants.%TA.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cf4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@FA.%TA.loc15_7.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FB(@FA.%B.facet.loc16_8.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FC(constants.%TC) {
// CHECK:STDOUT:   %TC.loc18_7.2 => constants.%TC
// CHECK:STDOUT:   %TC.patt.loc18_7.2 => constants.%TC
// CHECK:STDOUT:   %TC.as_type.loc18_42.2 => constants.%TC.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FA(constants.%A.facet) {
// CHECK:STDOUT:   %TA.loc15_7.2 => constants.%A.facet
// CHECK:STDOUT:   %TA.patt.loc15_7.2 => constants.%A.facet
// CHECK:STDOUT:   %TA.as_type.loc15_19.2 => constants.%TC.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.732
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1be
// CHECK:STDOUT:   %B.facet.loc16_8.3 => constants.%B.facet.d0b
// CHECK:STDOUT:   %FB.specific_fn.loc16_3.2 => constants.%FB.specific_fn.328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FA(@FC.%A.facet.loc19_8.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%V, <error>) {
// CHECK:STDOUT:   %V.loc33_6.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc33_6.2 => constants.%V
// CHECK:STDOUT:   %W.patt.loc33_13.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%A.facet
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%A.facet
// CHECK:STDOUT:   %T.as_type.loc13_21.2 => constants.%TC.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FB(constants.%B.facet.d0b) {
// CHECK:STDOUT:   %TB.loc14_7.2 => constants.%B.facet.d0b
// CHECK:STDOUT:   %TB.patt.loc14_7.2 => constants.%B.facet.d0b
// CHECK:STDOUT:   %TB.as_type.loc14_19.2 => constants.%TC.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
