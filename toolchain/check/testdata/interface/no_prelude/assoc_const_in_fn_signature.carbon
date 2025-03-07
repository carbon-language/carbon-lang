// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_fn_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_fn_signature.carbon

interface I {
  let T:! type;
  fn F() -> T;
}

// CHECK:STDOUT: --- assoc_const_in_fn_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Self.as_wit: <witness> = facet_access_witness %Self [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %Self.as_type, %Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: @F.%impl.elem0.loc13_13.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%impl.elem0.loc13_13.1 (%impl.elem0) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.loc13_13.2: <witness> = facet_access_witness @I.%Self [symbolic = %Self.as_wit.loc13_13.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc13_13.2: type = impl_witness_access %Self.as_wit.loc13_13.2, element0 [symbolic = %impl.elem0.loc13_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %impl.elem0.loc13_13.2 [symbolic = %impl.elem0.loc13_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc13_13.1 (%impl.elem0) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc13_13.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .F = <error>
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_wit.loc13_13.1: <witness> = facet_access_witness %Self [symbolic = %Self.as_wit.loc13_13.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc13_13.1: type = impl_witness_access %Self.as_wit.loc13_13.1, element0 [symbolic = %impl.elem0.loc13_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.%impl.elem0.loc13_13.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_wit.loc13_13.1 => constants.%Self.as_wit
// CHECK:STDOUT:   %impl.elem0.loc13_13.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
