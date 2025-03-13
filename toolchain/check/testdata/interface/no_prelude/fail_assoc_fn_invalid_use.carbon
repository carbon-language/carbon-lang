// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_fn_invalid_use.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_fn_invalid_use.carbon

// --- fail_member_access.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self]();
}

fn Use(T:! I) {
  // CHECK:STDERR: fail_member_access.carbon:[[@LINE+4]]:3: error: type `<type of F in T as I>` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   T.F.member;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  T.F.member;
}

// CHECK:STDOUT: --- fail_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.3d7: type = fn_type_with_self_type %F.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.3d7 = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.3d7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc5_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_8.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc13_4.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc13_4.2: <witness> = facet_access_witness %T.loc8_8.2, element0 [symbolic = %T.as_wit.iface0.loc13_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc13_4.2, (%T.as_wit.iface0.loc13_4.2) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc13_4.2: type = fn_type_with_self_type constants.%F.type, %I.facet [symbolic = %.loc13_4.2 (constants.%.3d7)]
// CHECK:STDOUT:   %impl.elem0.loc13_4.2: @Use.%.loc13_4.2 (%.3d7) = impl_witness_access %T.as_wit.iface0.loc13_4.2, element0 [symbolic = %impl.elem0.loc13_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Use.%.loc13_4.2 (%.3d7) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_8.1: %I.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc13_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_4.1: type = converted %T.ref, %T.as_type.loc13_4.1 [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc13_4.1: <witness> = facet_access_witness %T.ref, element0 [symbolic = %T.as_wit.iface0.loc13_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc13_4.1: @Use.%.loc13_4.2 (%.3d7) = impl_witness_access %T.as_wit.iface0.loc13_4.1, element0 [symbolic = %impl.elem0.loc13_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
