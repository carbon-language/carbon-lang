// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon

// FIXME: move to its own test?
// interface A {}
// interface B {}
// impl forall [T:! A] T as B {}
// fn FB[TB:! B](xb: TB) {}
// fn FA[TA:! A](xa: TA) {
//   FB(xa);
// }
// fn FC[TC:! A where .Self impls type](xc: TC) {
//   FA(xc);
// }

interface J {
  let U:! type;
}

// Works.
// fn F(T:! J, S:! T.U) {}

fn G(V:! J, W:! V.(J.U)) {}

// CHECK:STDOUT: --- compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %V: %J.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %V.as_wit: <witness> = facet_access_witness %V [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %V.as_type, %V.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %V.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %W: %impl.elem0 = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: %impl.elem0 = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %V.patt.loc30_6.1: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc30_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %J.type = value_param_pattern %V.patt.loc30_6.1, runtime_param<none> [symbolic = %V.patt.loc30_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc30_13.1: @G.%impl.elem0.loc30_18.2 (%impl.elem0) = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc30_13.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: @G.%impl.elem0.loc30_18.2 (%impl.elem0) = value_param_pattern %W.patt.loc30_13.1, runtime_param<none> [symbolic = %W.patt.loc30_13.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.param: %J.type = value_param runtime_param<none>
// CHECK:STDOUT:     %J.ref.loc30_10: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %V.loc30_6.1: %J.type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc30_6.2 (constants.%V)]
// CHECK:STDOUT:     %W.param: @G.%impl.elem0.loc30_18.2 (%impl.elem0) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc30: type = splice_block %impl.elem0.loc30_18.1 [symbolic = %impl.elem0.loc30_18.2 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc30_6.1 [symbolic = %V.loc30_6.2 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc30_20: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %V.as_wit.loc30_18.1: <witness> = facet_access_witness %V.ref [symbolic = %V.as_wit.loc30_18.2 (constants.%V.as_wit)]
// CHECK:STDOUT:       %impl.elem0.loc30_18.1: type = impl_witness_access %V.as_wit.loc30_18.1, element0 [symbolic = %impl.elem0.loc30_18.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc30_13.1: @G.%impl.elem0.loc30_18.2 (%impl.elem0) = bind_symbolic_name W, 1, %W.param [symbolic = %W.loc30_13.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
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
// CHECK:STDOUT: generic fn @G(%V.loc30_6.1: %J.type, %W.loc30_13.1: @G.%impl.elem0.loc30_18.2 (%impl.elem0)) {
// CHECK:STDOUT:   %V.loc30_6.2: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc30_6.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc30_6.2: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc30_6.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.as_wit.loc30_18.2: <witness> = facet_access_witness %V.loc30_6.2 [symbolic = %V.as_wit.loc30_18.2 (constants.%V.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc30_18.2: type = impl_witness_access %V.as_wit.loc30_18.2, element0 [symbolic = %impl.elem0.loc30_18.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %W.loc30_13.2: %impl.elem0 = bind_symbolic_name W, 1 [symbolic = %W.loc30_13.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc30_13.2: %impl.elem0 = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc30_13.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %J.type, %W.param_patt: @G.%impl.elem0.loc30_18.2 (%impl.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc30_6.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc30_6.2 => constants.%V
// CHECK:STDOUT:   %V.as_wit.loc30_18.2 => constants.%V.as_wit
// CHECK:STDOUT:   %impl.elem0.loc30_18.2 => constants.%impl.elem0
// CHECK:STDOUT:   %W.loc30_13.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc30_13.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
