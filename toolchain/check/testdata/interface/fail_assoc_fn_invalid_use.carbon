// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_assoc_fn_invalid_use.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_assoc_fn_invalid_use.carbon

// --- fail_member_access.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self]();
}

fn Use(T:! I) {
  // CHECK:STDERR: fail_member_access.carbon:[[@LINE+4]]:3: error: type `<type of F in T>` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   T.F.member;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  T.F.member;
}

// CHECK:STDOUT: --- fail_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.965: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.a1f: type = fn_type @I.WithSelf.F, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.cc3: %I.WithSelf.F.type.a1f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.7d0: type = fn_type_with_self_type %I.WithSelf.F.type.a1f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.7d0 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.7d0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_9.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.965)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   %T.loc8_9.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc13_4.2: type = facet_access_type %T.loc8_9.1 [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.a1f)]
// CHECK:STDOUT:   %.loc13_4.2: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc8_9.1 [symbolic = %.loc13_4.2 (constants.%.7d0)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_4.2: @Use.%.loc13_4.2 (%.7d0) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc13_4.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc13_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_4.1: type = converted %T.ref, %T.as_type.loc13_4.1 [symbolic = %T.as_type.loc13_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc13_4.1: @Use.%.loc13_4.2 (%.7d0) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.965
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a1f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
