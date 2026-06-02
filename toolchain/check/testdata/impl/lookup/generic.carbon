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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/generic.carbon

// --- deduced_type.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F[self: Self]();
}

impl forall [T:! type] T as HasF {
  fn F[unused self: Self]() {}
}

fn G(x: {}) {
  x.(HasF.F)();
}

// --- deduced_type_subst.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F[self: Self]() -> Self;
}

impl forall [T:! type] T* as HasF {
  fn F[self: Self]() -> T* { return self; }
}

fn G(x: {}*) -> {}* {
  return x.(HasF.F)();
}

// --- deduced_type_argument.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F[self: Self]();
}

class C(T:! type) {}

impl forall [T:! type] C(T) as HasF {
  fn F[unused self: Self]() {}
}

fn G(x: C({})) {
  x.(HasF.F)();
}

// --- deduced_interface_argument.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F[self: Self]();
}

impl forall [T:! type] {} as HasF(T) {
  fn F[unused self: Self]() {}
}

fn G(x: {}) {
  x.(HasF({}).F)();
}

// --- fail_incomplete_deduction.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F[self: Self]();
}

// CHECK:STDERR: fail_incomplete_deduction.carbon:[[@LINE+4]]:13: error: `impl` with unused generic binding [ImplUnusedBinding]
// CHECK:STDERR: impl forall [T:! type, U:! type] T as HasF {
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type, U:! type] T as HasF {
  fn F[unused self: Self]() {}
}

fn G(x: {}) {
  // CHECK:STDERR: fail_incomplete_deduction.carbon:[[@LINE+4]]:3: error: cannot access member of interface `HasF` in type `{}` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   x.(HasF.F)();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  x.(HasF.F)();
}

// --- fail_inconsistent_deduction.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F[self: Self]();
}

impl forall [T:! type] T as HasF(T) {
  fn F[unused self: Self]() {}
}

class A {}
class B {}

fn G(x: A) {
  // TODO: It'd be nice to include a note here saying that deduction failed because
  // we deduced two different values for `T`.
  // CHECK:STDERR: fail_inconsistent_deduction.carbon:[[@LINE+4]]:3: error: cannot access member of interface `HasF(B)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   x.(HasF(B).F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  x.(HasF(B).F)();
}

// CHECK:STDOUT: --- deduced_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.2da: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.4d3: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.8d9: %HasF.WithSelf.F.type.4d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.55f: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.4df: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.796: %T.as.HasF.impl.F.type.4df = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.e89: %HasF.type = facet_value %T, (%HasF.impl_witness.55f) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.dff: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.e89) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.f04: %HasF.WithSelf.F.type.dff = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.c1a: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.a34: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.8b8: %T.as.HasF.impl.F.type.a34 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.340: %HasF.type = facet_value %empty_struct_type, (%HasF.impl_witness.c1a) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.16a: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.340) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.cb4: %HasF.WithSelf.F.type.16a = struct_value () [concrete]
// CHECK:STDOUT:   %.275: type = fn_type_with_self_type %HasF.WithSelf.F.type.16a, %HasF.facet.340 [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.specific_fn: <specific function> = specific_function %T.as.HasF.impl.F.8b8, @T.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.4d3) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.8d9)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @HasF.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.55f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type.4df)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.4df) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.796)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.4df) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.796)] {
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_34.1: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.55f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.ref
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.2da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @T.as.HasF.impl.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.275 = impl_witness_access constants.%HasF.impl_witness.c1a, element0 [concrete = constants.%T.as.HasF.impl.F.8b8]
// CHECK:STDOUT:   %bound_method.loc13_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%T.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_14: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %T.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc13_14(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.4d3
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.8d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.55f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.4df
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.796
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.e89) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.e89
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.dff
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.f04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.e89) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.e89
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.c1a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.a34
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.8b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.340) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.340
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.16a
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.cb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_subst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.62e: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.9df: type = facet_access_type %Self.62e [symbolic]
// CHECK:STDOUT:   %pattern_type.2da: type = pattern_type %Self.as_type.9df [symbolic]
// CHECK:STDOUT:   %.87b: Core.Form = init_form %Self.as_type.9df [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.4d3: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self.62e) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.8d9: %HasF.WithSelf.F.type.4d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.3b4: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.6f2: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.19a: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.8af: %ptr.as.HasF.impl.F.type.19a = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.6f4: %HasF.type = facet_value %ptr.e8f, (%HasF.impl_witness.6f2) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.f1b: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.6f4) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.db9: %HasF.WithSelf.F.type.f1b = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2bf: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.b6a: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.44d: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2bf) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %.39b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72b, %Copy.facet.44d [symbolic]
// CHECK:STDOUT:   %impl.elem0.8a3: %.39b = impl_witness_access %Copy.lookup_impl_witness.2bf, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a13: <specific function> = specific_impl_function %impl.elem0.8a3, @Copy.WithSelf.Op(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %.9ba: Core.Form = init_form %ptr.c28 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.639: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.eca: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.70c: %ptr.as.HasF.impl.F.type.eca = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.3ef: %HasF.type = facet_value %ptr.c28, (%HasF.impl_witness.639) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.a75: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.3ef) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.847: %HasF.WithSelf.F.type.a75 = struct_value () [concrete]
// CHECK:STDOUT:   %.353: type = fn_type_with_self_type %HasF.WithSelf.F.type.a75, %HasF.facet.3ef [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.specific_fn: <specific function> = specific_function %ptr.as.HasF.impl.F.70c, @ptr.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1c0: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.91f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d7b: %ptr.as.Copy.impl.Op.type.91f = struct_value () [concrete]
// CHECK:STDOUT:   %.41e: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Copy.facet.e29: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.1c0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.980: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e29) [concrete]
// CHECK:STDOUT:   %.f8c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.980, %Copy.facet.e29 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d7b, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @ptr.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc8_25.2: type = ptr_type %T.ref [symbolic = %ptr.loc8_25.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1cc = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.1cc = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1cc = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1cc = return_slot_pattern %return.param_patt, %ptr.loc12_19 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_18: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc12_19.1: type = converted %.loc12_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc12_19: type = ptr_type %.loc12_19.1 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %.loc12_19.2: Core.Form = init_form %ptr.loc12_19 [concrete = constants.%.9ba]
// CHECK:STDOUT:     %x.param: %ptr.c28 = value_param call_param0
// CHECK:STDOUT:     %.loc12_11.1: type = splice_block %ptr.loc12_11 [concrete = constants.%ptr.c28] {
// CHECK:STDOUT:       %.loc12_10: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc12_11.2: type = converted %.loc12_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %ptr.loc12_11: type = ptr_type %.loc12_11.2 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr.c28 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %ptr.c28 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62e]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.4d3) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.8d9)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = return_slot_pattern %return.param_patt, %.loc5_25.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_25: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.62e)]
// CHECK:STDOUT:     %Self.as_type.loc5_25: type = facet_access_type %Self.ref.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)]
// CHECK:STDOUT:     %.loc5_25.2: type = converted %Self.ref.loc5_25, %Self.as_type.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)]
// CHECK:STDOUT:     %.loc5_25.3: Core.Form = init_form %.loc5_25.2 [symbolic = %.loc5_25.1 (constants.%.87b)]
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)] {
// CHECK:STDOUT:       %Self.ref.loc5_14: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.62e)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref.loc5_14 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref.loc5_14, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df) = out_param call_param1
// CHECK:STDOUT:     %return: ref @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0.3b4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @HasF.WithSelf.%assoc0
// CHECK:STDOUT:   .T = <poisoned>
// CHECK:STDOUT:   witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @ptr.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T.67d)]
// CHECK:STDOUT:   %ptr.loc8_25.1: type = ptr_type %T.loc8_15.1 [symbolic = %ptr.loc8_25.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.6f2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %ptr.as.HasF.impl.F.type (constants.%ptr.as.HasF.impl.F.type.19a)]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.19a) = struct_value () [symbolic = %ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.8af)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %ptr.loc8_25.2 as %HasF.ref {
// CHECK:STDOUT:     %ptr.as.HasF.impl.F.decl: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.19a) = fn_decl @ptr.as.HasF.impl.F [symbolic = @ptr.as.HasF.impl.%ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.8af)] {
// CHECK:STDOUT:       %self.param_patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc9_26 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ptr.as.HasF.impl.%T.loc8_15.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %ptr.loc9_26: type = ptr_type %T.ref [symbolic = %ptr.loc9_14 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %.loc9_26.2: Core.Form = init_form %ptr.loc9_26 [symbolic = %.loc9_26.1 (constants.%.cb6)]
// CHECK:STDOUT:       %self.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @ptr.as.HasF.impl.%ptr.loc8_25.2 [symbolic = %ptr.loc9_14 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %self: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%ptr.as.HasF.impl.F.decl), @ptr.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_35.1: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(constants.%T.67d) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.6f2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %ptr.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.ref
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_35.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.62e)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.9df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.2da)]
// CHECK:STDOUT:   %.loc5_25.1: Core.Form = init_form %Self.as_type.loc5_14.1 [symbolic = %.loc5_25.1 (constants.%.87b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df)) -> out %return.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.9df);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ptr.as.HasF.impl.F(@ptr.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %ptr.loc9_14: type = ptr_type %T [symbolic = %ptr.loc9_14 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc9_14 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %.loc9_26.1: Core.Form = init_form %ptr.loc9_14 [symbolic = %.loc9_26.1 (constants.%.cb6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc9_14 [symbolic = %require_complete (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc9_37.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T) [symbolic = %.loc9_37.1 (constants.%.b6a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc9_14, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2bf)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc9_14, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.44d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.72b)]
// CHECK:STDOUT:   %.loc9_37.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc9_37.2 (constants.%.39b)]
// CHECK:STDOUT:   %impl.elem0.loc9_37.2: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.39b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2: <specific function> = specific_impl_function %impl.elem0.loc9_37.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f)) -> out %return.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc9_37.1: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.39b) = impl_witness_access constants.%Copy.lookup_impl_witness.2bf, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:     %bound_method.loc9_37.1: <bound method> = bound_method %self.ref, %impl.elem0.loc9_37.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_37.1: <specific function> = specific_impl_function %impl.elem0.loc9_37.1, @Copy.WithSelf.Op(constants.%Copy.facet.44d) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:     %bound_method.loc9_37.2: <bound method> = bound_method %self.ref, %specific_impl_fn.loc9_37.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = call %bound_method.loc9_37.2(%self.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %ptr.c28) -> out %return.param: %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %ptr.c28 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0.3b4]
// CHECK:STDOUT:   %impl.elem0: %.353 = impl_witness_access constants.%HasF.impl_witness.639, element0 [concrete = constants.%ptr.as.HasF.impl.F.70c]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%ptr.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.call: init %ptr.c28 = call %bound_method.loc13_21(%x.ref)
// CHECK:STDOUT:   return %ptr.as.HasF.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self.62e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.62e
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.4d3
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.8d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self.62e) {
// CHECK:STDOUT:   %Self => constants.%Self.62e
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.9df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2da
// CHECK:STDOUT:   %.loc5_25.1 => constants.%.87b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.6f2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.19a
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.8af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc9_14 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc9_26.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.6f4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.6f4
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.f1b
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.db9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.6f4) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.6f4
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5_25.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.c28
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.639
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.eca
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.70c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.3ef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.3ef
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.a75
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.847
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc9_14 => constants.%ptr.c28
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1cc
// CHECK:STDOUT:   %.loc9_26.1 => constants.%.9ba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %.loc9_37.1 => constants.%.41e
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1c0
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.e29
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.980
// CHECK:STDOUT:   %.loc9_37.2 => constants.%.f8c
// CHECK:STDOUT:   %impl.elem0.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.d7b
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.2da: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.4d3: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.8d9: %HasF.WithSelf.F.type.4d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1f9: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.02d: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.69e: type = pattern_type %C.1f9 [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.d51: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.4fd: %C.as.HasF.impl.F.type.d51 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.081: %HasF.type = facet_value %C.1f9, (%HasF.impl_witness.02d) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.a83: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.081) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.917: %HasF.WithSelf.F.type.a83 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.1f9 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.ea5: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.4ac: type = pattern_type %C.ea5 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.994: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.613: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.880: %C.as.HasF.impl.F.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.5ef: %HasF.type = facet_value %C.ea5, (%HasF.impl_witness.994) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.1c2: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.5ef) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.7ab: %HasF.WithSelf.F.type.1c2 = struct_value () [concrete]
// CHECK:STDOUT:   %.fc0: type = fn_type_with_self_type %HasF.WithSelf.F.type.1c2, %HasF.facet.5ef [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.specific_fn: <specific function> = specific_function %C.as.HasF.impl.F.880, @C.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.2 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.1 (constants.%C.1f9)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.4ac = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.4ac = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.ea5 = value_param call_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %C [concrete = constants.%C.ea5] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %.loc14_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.ea5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.ea5 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.4d3) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.8d9)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @HasF.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.HasF.impl(%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc10_27.1: type = class_type @C, @C(%T.loc10_15.1) [symbolic = %C.loc10_27.1 (constants.%C.1f9)]
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.02d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %C.as.HasF.impl.F.type (constants.%C.as.HasF.impl.F.type.d51)]
// CHECK:STDOUT:   %C.as.HasF.impl.F: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.d51) = struct_value () [symbolic = %C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.4fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.2 as %HasF.ref {
// CHECK:STDOUT:     %C.as.HasF.impl.F.decl: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.d51) = fn_decl @C.as.HasF.impl.F [symbolic = @C.as.HasF.impl.%C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.4fd)] {
// CHECK:STDOUT:       %self.param_patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.69e) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.69e) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.HasF.impl.F.%C (%C.1f9) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.loc10_27.2 [symbolic = %C (constants.%C.1f9)]
// CHECK:STDOUT:       %self: @C.as.HasF.impl.F.%C (%C.1f9) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc10_37.1: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.02d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.ref
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc10_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_10.2: type) {
// CHECK:STDOUT:   %T.loc8_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1f9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.2da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.HasF.impl.F(@C.as.HasF.impl.%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.1f9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.69e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.HasF.impl.F.%C (%C.1f9)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %C.ea5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.ea5 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.fc0 = impl_witness_access constants.%HasF.impl_witness.994, element0 [concrete = constants.%C.as.HasF.impl.F.880]
// CHECK:STDOUT:   %bound_method.loc15_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%C.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %C.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc15_14(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.4d3
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.8d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.1f9
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.02d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.d51
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.4fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.1f9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.69e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.081) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.081
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.a83
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.917
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.081) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.081
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C.1f9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.69e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.ea5
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.994
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.613
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.880
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.5ef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.5ef
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.1c2
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.7ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.ea5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_interface_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.040: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.040 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.91c: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.067: %HasF.type.91c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.067 [symbolic]
// CHECK:STDOUT:   %pattern_type.c8d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.d19: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.067) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.b0e: %HasF.WithSelf.F.type.d19 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.93b: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95a: %HasF.assoc_type.93b = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.b64: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.91c [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.35a: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.827: %empty_struct_type.as.HasF.impl.F.type.35a = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.6fb: %HasF.type.91c = facet_value %empty_struct_type, (%HasF.impl_witness.b64) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.265: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet.6fb) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.991: %HasF.WithSelf.F.type.265 = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.a43: type = facet_type <@HasF, @HasF(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.f8a: %HasF.type.a43 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.f80: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %Self.067) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.df2: %HasF.WithSelf.F.type.f80 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.5bf: type = assoc_entity_type @HasF, @HasF(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.d84: %HasF.assoc_type.5bf = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.caa: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %HasF.type.a43 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.873: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.6be: %empty_struct_type.as.HasF.impl.F.type.873 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.891: %HasF.type.a43 = facet_value %empty_struct_type, (%HasF.impl_witness.caa) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.0a1: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %HasF.facet.891) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.765: %HasF.WithSelf.F.type.0a1 = struct_value () [concrete]
// CHECK:STDOUT:   %.c1c: type = fn_type_with_self_type %HasF.WithSelf.F.type.0a1, %HasF.facet.891 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.specific_fn: <specific function> = specific_function %empty_struct_type.as.HasF.impl.F.6be, @empty_struct_type.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.040 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_25.2: type = converted %.loc8_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.040 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_36.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.91c)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.067)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.067)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.d19) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.b0e)] {
// CHECK:STDOUT:       %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.c8d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.c8d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.93b) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = @HasF.WithSelf.%assoc0.loc5_21.1
// CHECK:STDOUT:     witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @empty_struct_type.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_36.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.b64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_36.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %empty_struct_type.as.HasF.impl.F.type (constants.%empty_struct_type.as.HasF.impl.F.type.35a)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.35a) = struct_value () [symbolic = %empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.827)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_36.2 {
// CHECK:STDOUT:     %empty_struct_type.as.HasF.impl.F.decl: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.35a) = fn_decl @empty_struct_type.as.HasF.impl.F [symbolic = @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.827)] {
// CHECK:STDOUT:       %self.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: %pattern_type.a96 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @empty_struct_type.as.HasF.impl.%.loc8_25.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %self: %empty_struct_type = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%empty_struct_type.as.HasF.impl.F.decl), @empty_struct_type.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_38.1: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.b64)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %empty_struct_type.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.type.loc8_36.2
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_38.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.91c)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.c8d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_struct_type.as.HasF.impl.F(@empty_struct_type.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_struct_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: %HasF.type.040 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %.loc13_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_13: type = converted %.loc13_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%empty_struct_type)> [concrete = constants.%HasF.type.a43]
// CHECK:STDOUT:   %.loc13_14: %HasF.assoc_type.5bf = specific_constant @HasF.WithSelf.%assoc0.loc5_21.1, @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.067) [concrete = constants.%assoc0.d84]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.5bf = name_ref F, %.loc13_14 [concrete = constants.%assoc0.d84]
// CHECK:STDOUT:   %impl.elem0: %.c1c = impl_witness_access constants.%HasF.impl_witness.caa, element0 [concrete = constants.%empty_struct_type.as.HasF.impl.F.6be]
// CHECK:STDOUT:   %bound_method.loc13_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @empty_struct_type.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%empty_struct_type.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc13_18(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.067
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.067) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.d19
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b0e
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.93b
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.95a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.067) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.91c
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.b64
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.35a
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.827
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%HasF.facet.6fb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%HasF.facet.6fb
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.265
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.991
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.93b
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.95a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet.6fb) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%HasF.facet.6fb
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.a43
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.067) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.a43
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.f80
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.df2
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.5bf
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.d84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.a43
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.caa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.873
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.6be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%HasF.facet.891) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.a43
// CHECK:STDOUT:   %Self => constants.%HasF.facet.891
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.0a1
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.765
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.5bf
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.d84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.2da: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F: %HasF.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_15.2 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc12_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc12_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_25.2: type = symbolic_binding U, 1 [symbolic = %U.loc12_25.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc16_10.1: type = splice_block %.loc16_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc16_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc16_10.3: type = converted %.loc16_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.2da) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @HasF.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc12_15.2: type, %U.loc12_25.2: type) {
// CHECK:STDOUT:   %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc12_25.1: type = symbolic_binding U, 1 [symbolic = %U.loc12_25.1 (constants.%U)]
// CHECK:STDOUT:   %HasF.impl_witness.loc12_44.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc12_15.1, %U.loc12_25.1) [symbolic = %HasF.impl_witness.loc12_44.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc12_15.1, %U.loc12_25.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc12_44.1: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T, constants.%U) [symbolic = %HasF.impl_witness.loc12_44.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.ref
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.2da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc12_15.2: type, @T.as.HasF.impl.%U.loc12_25.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @T.as.HasF.impl.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%T
// CHECK:STDOUT:   %U.loc12_25.1 => constants.%U
// CHECK:STDOUT:   %HasF.impl_witness.loc12_44.2 => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.040: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.040 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.91c: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.067: %HasF.type.91c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.067 [symbolic]
// CHECK:STDOUT:   %pattern_type.c8d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.d19: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.067) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.b0e: %HasF.WithSelf.F.type.d19 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.93b: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95a: %HasF.assoc_type.93b = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.b6b: <witness> = require_complete_type %HasF.type.91c [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.91c = facet_value %T, (%HasF.impl_witness) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.b46: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.56a: %HasF.WithSelf.F.type.b46 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.ccb: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.90e: type = facet_type <@HasF, @HasF(%B)> [concrete]
// CHECK:STDOUT:   %Self.baa: %HasF.type.90e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.534: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%B, %Self.067) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.f7e: %HasF.WithSelf.F.type.534 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.e5f: type = assoc_entity_type @HasF, @HasF(%B) [concrete]
// CHECK:STDOUT:   %assoc0.4f9: %HasF.assoc_type.e5f = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.040 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_24: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.040 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_35.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.91c)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.ccb = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.ccb = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %x: %A = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.067)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.067)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.d19) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.b0e)] {
// CHECK:STDOUT:       %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.c8d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.c8d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.93b) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = @HasF.WithSelf.%assoc0.loc5_21.1
// CHECK:STDOUT:     witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_35.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_37.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_35.1 [symbolic = %require_complete (constants.%require_complete.b6b)]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc8_24 as %HasF.type.loc8_35.2 {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref.loc8_24 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_37.1: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_37.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.type.loc8_35.2
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.91c)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.91c)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.91c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.067)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.c8d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @T.as.HasF.impl.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %A = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: %HasF.type.040 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [concrete = constants.%HasF.type.90e]
// CHECK:STDOUT:   %.loc22: %HasF.assoc_type.e5f = specific_constant @HasF.WithSelf.%assoc0.loc5_21.1, @HasF.WithSelf(constants.%B, constants.%Self.067) [concrete = constants.%assoc0.4f9]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.e5f = name_ref F, %.loc22 [concrete = constants.%assoc0.4f9]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.067
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.067) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.d19
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b0e
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.93b
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.95a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.067) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_35.1 => constants.%HasF.type.91c
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2 => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b6b
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.b46
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.56a
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.93b
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.95a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.91c
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.90e
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.baa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%B, constants.%Self.067) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.90e
// CHECK:STDOUT:   %Self => constants.%Self.067
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.534
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.f7e
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.e5f
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.4f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
