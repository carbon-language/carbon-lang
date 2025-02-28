// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon

interface J {
  let U:! type;
}

fn F(T:! J, S:! T.U) {}

// CHECK:STDERR: compound_member_access.carbon:[[@LINE+4]]:17: error: cannot access member of interface `J` in type `J` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: fn G(V:! J, W:! V.(J.U)) {}
// CHECK:STDERR:                 ^~~~~~~
// CHECK:STDERR:
fn G(V:! J, W:! V.(J.U)) {}

// CHECK:STDOUT: --- compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %S: %impl.elem0 = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %S.patt: %impl.elem0 = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V: %J.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc15_6.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %J.type = value_param_pattern %T.patt.loc15_6.1, runtime_param<none> [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %S.patt.loc15_13.1: @F.%impl.elem0.loc15_18.2 (%impl.elem0) = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc15_13.2 (constants.%S.patt)]
// CHECK:STDOUT:     %S.param_patt: @F.%impl.elem0.loc15_18.2 (%impl.elem0) = value_param_pattern %S.patt.loc15_13.1, runtime_param<none> [symbolic = %S.patt.loc15_13.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %J.type = value_param runtime_param<none>
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc15_6.1: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:     %S.param: @F.%impl.elem0.loc15_18.2 (%impl.elem0) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %impl.elem0.loc15_18.1 [symbolic = %impl.elem0.loc15_18.2 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %T.ref: %J.type = name_ref T, %T.loc15_6.1 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc15_18.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_18.2: type = converted %T.ref, %T.as_type.loc15_18.1 [symbolic = %T.as_type.loc15_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.loc15_18.1: <witness> = facet_access_witness %T.ref [symbolic = %T.as_wit.loc15_18.2 (constants.%T.as_wit)]
// CHECK:STDOUT:       %impl.elem0.loc15_18.1: type = impl_witness_access %T.as_wit.loc15_18.1, element0 [symbolic = %impl.elem0.loc15_18.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc15_13.1: @F.%impl.elem0.loc15_18.2 (%impl.elem0) = bind_symbolic_name S, 1, %S.param [symbolic = %S.loc15_13.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %V.patt.loc21_6.1: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc21_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %J.type = value_param_pattern %V.patt.loc21_6.1, runtime_param<none> [symbolic = %V.patt.loc21_6.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc21_13.1: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc21_13.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: <error> = value_param_pattern %W.patt.loc21_13.1, runtime_param<none> [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.param: %J.type = value_param runtime_param<none>
// CHECK:STDOUT:     %J.ref.loc21_10: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %V.loc21_6.1: %J.type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc21_6.2 (constants.%V)]
// CHECK:STDOUT:     %W.param: <error> = value_param runtime_param<none>
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc21_6.1 [symbolic = %V.loc21_6.2 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc21_20: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 1, %W.param [concrete = <error>]
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
// CHECK:STDOUT: generic fn @F(%T.loc15_6.1: %J.type, %S.loc15_13.1: @F.%impl.elem0.loc15_18.2 (%impl.elem0)) {
// CHECK:STDOUT:   %T.loc15_6.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_6.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc15_18.2: type = facet_access_type %T.loc15_6.2 [symbolic = %T.as_type.loc15_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc15_18.2: <witness> = facet_access_witness %T.loc15_6.2 [symbolic = %T.as_wit.loc15_18.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc15_18.2: type = impl_witness_access %T.as_wit.loc15_18.2, element0 [symbolic = %impl.elem0.loc15_18.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %S.loc15_13.2: %impl.elem0 = bind_symbolic_name S, 1 [symbolic = %S.loc15_13.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc15_13.2: %impl.elem0 = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc15_13.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %J.type, %S.param_patt: @F.%impl.elem0.loc15_18.2 (%impl.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%V.loc21_6.1: %J.type, %W: <error>) {
// CHECK:STDOUT:   %V.loc21_6.2: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc21_6.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc21_6.2: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc21_6.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.patt.loc21_13.2: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc21_13.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %J.type, %W.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%S) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc15_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.loc15_18.2 => constants.%T.as_wit
// CHECK:STDOUT:   %impl.elem0.loc15_18.2 => constants.%impl.elem0
// CHECK:STDOUT:   %S.loc15_13.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc15_13.2 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%V, <error>) {
// CHECK:STDOUT:   %V.loc21_6.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc21_6.2 => constants.%V
// CHECK:STDOUT:   %W.patt.loc21_13.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
