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
  fn F[self: Self]() {}
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
  fn F[self: Self]() {}
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
  fn F[self: Self]() {}
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
  fn F[self: Self]() {}
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
  fn F[self: Self]() {}
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
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.e682: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.bdb: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e683: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.92f: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.ed9: %T.as.HasF.impl.F.type.92f = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.aaf: %HasF.type = facet_value %T, (%HasF.impl_witness.bdb) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.22a: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.ce2: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.80a: %T.as.HasF.impl.F.type.ce2 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.040: %HasF.type = facet_value %empty_struct_type, (%HasF.impl_witness.22a) [concrete]
// CHECK:STDOUT:   %.f99: type = fn_type_with_self_type %HasF.F.type, %HasF.facet.040 [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.specific_fn: <specific function> = specific_function %T.as.HasF.impl.F.80a, @T.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@T.as.HasF.impl.%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = @T.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness.bdb)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.e682) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.e682) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_14.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness.bdb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_14.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type.92f)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.92f) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.ed9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.92f) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.ed9)] {
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e683) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e683) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = file.%HasF.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.e682)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.e683)]
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
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.f99 = impl_witness_access constants.%HasF.impl_witness.22a, element0 [concrete = constants.%T.as.HasF.impl.F.80a]
// CHECK:STDOUT:   %bound_method.loc13_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%T.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_14: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %T.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc13_14(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e682
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.bdb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.92f
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.ed9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet.aaf) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.aaf
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.22a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.ce2
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.80a
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
// CHECK:STDOUT:   %Self.9a7: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.07b: type = symbolic_binding_type Self, 0, %Self.9a7 [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %Self.binding.as_type.07b [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.97a: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.c1a: <witness> = impl_witness file.%HasF.impl_witness_table, @ptr.as.HasF.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %pattern_type.a60: type = pattern_type %ptr.4f0 [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.dd9: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.d3c: %ptr.as.HasF.impl.F.type.dd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.a17: %HasF.type = facet_value %ptr.4f0, (%HasF.impl_witness.c1a) [symbolic]
// CHECK:STDOUT:   %require_complete.482: <witness> = require_complete_type %ptr.4f0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692: %ptr.as.Copy.impl.Op.type.75b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr.4f0, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.747: %Copy.type = facet_value %ptr.4f0, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.747 [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet.747) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.825: <witness> = impl_witness file.%HasF.impl_witness_table, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.4d2: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.b12: %ptr.as.HasF.impl.F.type.4d2 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.10f: %HasF.type = facet_value %ptr.c28, (%HasF.impl_witness.825) [concrete]
// CHECK:STDOUT:   %.22b: type = fn_type_with_self_type %HasF.F.type, %HasF.facet.10f [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.specific_fn: <specific function> = specific_function %ptr.as.HasF.impl.F.b12, @ptr.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.433: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.424: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.15e: %ptr.as.Copy.impl.Op.type.424 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.48c: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.433) [concrete]
// CHECK:STDOUT:   %.fb0: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.48c [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.15e, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75b) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T.d9f)]
// CHECK:STDOUT:     %ptr.loc8_25.2: type = ptr_type %T.ref [symbolic = %ptr.loc8_25.1 (constants.%ptr.4f0)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@ptr.as.HasF.impl.%ptr.as.HasF.impl.F.decl), @ptr.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(constants.%T.d9f) [symbolic = @ptr.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness.c1a)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.1cc = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1cc = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1cc = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1cc = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_19: type = converted %.loc12_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc12_19: type = ptr_type %.loc12_19 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %x.param: %ptr.c28 = value_param call_param0
// CHECK:STDOUT:     %.loc12_11.1: type = splice_block %ptr.loc12_11 [concrete = constants.%ptr.c28] {
// CHECK:STDOUT:       %.loc12_10: %empty_struct_type = struct_literal ()
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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9a7]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.e68) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.e68) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @HasF.F.%pattern_type (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @HasF.F.%pattern_type (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_25: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.9a7)]
// CHECK:STDOUT:     %Self.as_type.loc5_25: type = facet_access_type %Self.ref.loc5_25 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:     %.loc5_25: type = converted %Self.ref.loc5_25, %Self.as_type.loc5_25 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)] {
// CHECK:STDOUT:       %Self.ref.loc5_14: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.9a7)]
// CHECK:STDOUT:       %Self.as_type.loc5_14: type = facet_access_type %Self.ref.loc5_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref.loc5_14, %Self.as_type.loc5_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @ptr.as.HasF.impl(%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T.d9f)]
// CHECK:STDOUT:   %ptr.loc8_25.1: type = ptr_type %T.loc8_14.1 [symbolic = %ptr.loc8_25.1 (constants.%ptr.4f0)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @ptr.as.HasF.impl(%T.loc8_14.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness.c1a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.loc8_14.1) [symbolic = %ptr.as.HasF.impl.F.type (constants.%ptr.as.HasF.impl.F.type.dd9)]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.dd9) = struct_value () [symbolic = %ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.d3c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %ptr.loc8_25.2 as %HasF.ref {
// CHECK:STDOUT:     %ptr.as.HasF.impl.F.decl: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.dd9) = fn_decl @ptr.as.HasF.impl.F [symbolic = @ptr.as.HasF.impl.%ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.d3c)] {
// CHECK:STDOUT:       %self.patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.a60) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.a60) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.a60) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.a60) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ptr.as.HasF.impl.%T.loc8_14.2 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:       %ptr.loc9_26: type = ptr_type %T.ref [symbolic = %ptr.loc9_14 (constants.%ptr.4f0)]
// CHECK:STDOUT:       %self.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @ptr.as.HasF.impl.%ptr.loc8_25.2 [symbolic = %ptr.loc9_14 (constants.%ptr.4f0)]
// CHECK:STDOUT:       %self: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %ptr.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = file.%HasF.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9a7)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b)) -> @HasF.F.%Self.binding.as_type (%Self.binding.as_type.07b);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ptr.as.HasF.impl.F(@ptr.as.HasF.impl.%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:   %ptr.loc9_14: type = ptr_type %T [symbolic = %ptr.loc9_14 (constants.%ptr.4f0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc9_14 [symbolic = %pattern_type (constants.%pattern_type.a60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc9_14 [symbolic = %require_complete (constants.%require_complete.482)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc9_14, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc9_14, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.747)]
// CHECK:STDOUT:   %.loc9_37.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc9_37.2 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc9_37.2: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2: <specific function> = specific_impl_function %impl.elem0.loc9_37.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0)) -> @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc9_37.1: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:     %bound_method.loc9_37.1: <bound method> = bound_method %self.ref, %impl.elem0.loc9_37.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_37.1: <specific function> = specific_impl_function %impl.elem0.loc9_37.1, @Copy.Op(constants.%Copy.facet.747) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:     %bound_method.loc9_37.2: <bound method> = bound_method %self.ref, %specific_impl_fn.loc9_37.1
// CHECK:STDOUT:     %.loc9_37.1: init @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.4f0) = call %bound_method.loc9_37.2(%self.ref)
// CHECK:STDOUT:     return %.loc9_37.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %ptr.c28) -> %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %ptr.c28 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:   %impl.elem0: %.22b = impl_witness_access constants.%HasF.impl_witness.825, element0 [concrete = constants.%ptr.as.HasF.impl.F.b12]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%ptr.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.call: init %ptr.c28 = call %bound_method.loc13_21(%x.ref)
// CHECK:STDOUT:   return %ptr.as.HasF.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self.9a7) {
// CHECK:STDOUT:   %Self => constants.%Self.9a7
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.07b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T.d9f
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.4f0
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.c1a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.dd9
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.d3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%T.d9f) {
// CHECK:STDOUT:   %T => constants.%T.d9f
// CHECK:STDOUT:   %ptr.loc9_14 => constants.%ptr.4f0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet.a17) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.a17
// CHECK:STDOUT:   %Self.binding.as_type => constants.%ptr.4f0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.c28
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.825
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.4d2
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.b12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc9_14 => constants.%ptr.c28
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.433
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.48c
// CHECK:STDOUT:   %.loc9_37.2 => constants.%.fb0
// CHECK:STDOUT:   %impl.elem0.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.15e
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.94c: <witness> = impl_witness file.%HasF.impl_witness_table, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.06f: type = pattern_type %C.3f0 [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.ce1: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.019: %C.as.HasF.impl.F.type.ce1 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.e4a: %HasF.type = facet_value %C.3f0, (%HasF.impl_witness.94c) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.3f0 [symbolic]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.cb0: <witness> = impl_witness file.%HasF.impl_witness_table, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.3eb: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.6c4: %C.as.HasF.impl.F.type.3eb = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.a36: %HasF.type = facet_value %C.7a7, (%HasF.impl_witness.cb0) [concrete]
// CHECK:STDOUT:   %.b41: type = fn_type_with_self_type %HasF.F.type, %HasF.facet.a36 [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.specific_fn: <specific function> = specific_function %C.as.HasF.impl.F.6c4, @C.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.2 [symbolic = %T.loc10_14.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.1 (constants.%C.3f0)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc10_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@C.as.HasF.impl.%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(constants.%T) [symbolic = @C.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness.94c)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.99a = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.99a = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.7a7 = value_param call_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %C [concrete = constants.%C.7a7] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc14_13.2: type = converted %.loc14_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.7a7 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.e68) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.e68) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.HasF.impl(%T.loc10_14.2: type) {
// CHECK:STDOUT:   %T.loc10_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc10_27.1: type = class_type @C, @C(%T.loc10_14.1) [symbolic = %C.loc10_27.1 (constants.%C.3f0)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @C.as.HasF.impl(%T.loc10_14.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness.94c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T.loc10_14.1) [symbolic = %C.as.HasF.impl.F.type (constants.%C.as.HasF.impl.F.type.ce1)]
// CHECK:STDOUT:   %C.as.HasF.impl.F: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.ce1) = struct_value () [symbolic = %C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.019)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.2 as %HasF.ref {
// CHECK:STDOUT:     %C.as.HasF.impl.F.decl: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.ce1) = fn_decl @C.as.HasF.impl.F [symbolic = @C.as.HasF.impl.%C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.019)] {
// CHECK:STDOUT:       %self.patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.06f) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.06f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.HasF.impl.F.%C (%C.3f0) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.loc10_27.2 [symbolic = %C (constants.%C.3f0)]
// CHECK:STDOUT:       %self: @C.as.HasF.impl.F.%C (%C.3f0) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = file.%HasF.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.HasF.impl.F(@C.as.HasF.impl.%T.loc10_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.3f0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.HasF.impl.F.%C (%C.3f0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %C.7a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.7a7 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.b41 = impl_witness_access constants.%HasF.impl_witness.cb0, element0 [concrete = constants.%C.as.HasF.impl.F.6c4]
// CHECK:STDOUT:   %bound_method.loc15_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%C.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %C.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc15_14(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.1 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.3f0
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.94c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.ce1
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.019
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.3f0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.06f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet.e4a) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.e4a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.3f0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.06f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc10_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.7a7
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.cb0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.3eb
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.6c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.7a7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %HasF.type.fe3: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.59b: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.69f: %HasF.type.59b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.69f [symbolic]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type.940: type = fn_type @HasF.F, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %HasF.F.86b: %HasF.F.type.940 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.1aa: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.bd3: %HasF.assoc_type.1aa = assoc_entity element0, @HasF.%HasF.F.decl [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.59b [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.350: <witness> = impl_witness file.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.83a: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.1e9: %empty_struct_type.as.HasF.impl.F.type.83a = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.67e: %HasF.type.59b = facet_value %empty_struct_type, (%HasF.impl_witness.350) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.48b: type = facet_type <@HasF, @HasF(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.82e: %HasF.type.48b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.F.type.b0b: type = fn_type @HasF.F, @HasF(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %HasF.F.418: %HasF.F.type.b0b = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.9e4: type = assoc_entity_type @HasF, @HasF(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.95d: %HasF.assoc_type.9e4 = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %HasF.type.48b [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.d67: <witness> = impl_witness file.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.587: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.eab: %empty_struct_type.as.HasF.impl.F.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.f01: %HasF.type.48b = facet_value %empty_struct_type, (%HasF.impl_witness.d67) [concrete]
// CHECK:STDOUT:   %.24f: type = fn_type_with_self_type %HasF.F.type.b0b, %HasF.facet.f01 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.specific_fn: <specific function> = specific_function %empty_struct_type.as.HasF.impl.F.eab, @empty_struct_type.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe3 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_25.2: type = converted %.loc8_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_36.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.59b)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.decl), @empty_struct_type.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(constants.%T) [symbolic = @empty_struct_type.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness.350)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.1)> [symbolic = %HasF.type (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.69f)]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F, @HasF(%T.loc4_16.1) [symbolic = %HasF.F.type (constants.%HasF.F.type.940)]
// CHECK:STDOUT:   %HasF.F: @HasF.%HasF.F.type (%HasF.F.type.940) = struct_value () [symbolic = %HasF.F (constants.%HasF.F.86b)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF, @HasF(%T.loc4_16.1) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.1aa)]
// CHECK:STDOUT:   %assoc0.loc5_21.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.1aa) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.bd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.69f)]
// CHECK:STDOUT:     %HasF.F.decl: @HasF.%HasF.F.type (%HasF.F.type.940) = fn_decl @HasF.F [symbolic = @HasF.%HasF.F (constants.%HasF.F.86b)] {
// CHECK:STDOUT:       %self.patt: @HasF.F.%pattern_type (%pattern_type.429) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @HasF.F.%pattern_type (%pattern_type.429) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.F.%HasF.type (%HasF.type.59b) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:         %Self.ref: @HasF.F.%HasF.type (%HasF.type.59b) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.1aa) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.bd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = %assoc0.loc5_21.1
// CHECK:STDOUT:     witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @empty_struct_type.as.HasF.impl(%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_36.1: type = facet_type <@HasF, @HasF(%T.loc8_14.1)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_36.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T.loc8_14.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness.350)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T.loc8_14.1) [symbolic = %empty_struct_type.as.HasF.impl.F.type (constants.%empty_struct_type.as.HasF.impl.F.type.83a)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.83a) = struct_value () [symbolic = %empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.1e9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_36.2 {
// CHECK:STDOUT:     %empty_struct_type.as.HasF.impl.F.decl: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.83a) = fn_decl @empty_struct_type.as.HasF.impl.F [symbolic = @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.1e9)] {
// CHECK:STDOUT:       %self.patt: %pattern_type.a96 = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: %pattern_type.a96 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @empty_struct_type.as.HasF.impl.%.loc8_25.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %self: %empty_struct_type = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %empty_struct_type.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = file.%HasF.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%T.loc4_16.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.59b)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %Self: @HasF.F.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.429)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_struct_type.as.HasF.impl.F(@empty_struct_type.as.HasF.impl.%T.loc8_14.2: type) {
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %.loc13_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_13: type = converted %.loc13_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%empty_struct_type)> [concrete = constants.%HasF.type.48b]
// CHECK:STDOUT:   %.loc13_14: %HasF.assoc_type.9e4 = specific_constant @HasF.%assoc0.loc5_21.1, @HasF(constants.%empty_struct_type) [concrete = constants.%assoc0.95d]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.9e4 = name_ref F, %.loc13_14 [concrete = constants.%assoc0.95d]
// CHECK:STDOUT:   %impl.elem0: %.24f = impl_witness_access constants.%HasF.impl_witness.d67, element0 [concrete = constants.%empty_struct_type.as.HasF.impl.F.eab]
// CHECK:STDOUT:   %bound_method.loc13_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @empty_struct_type.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%empty_struct_type.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc13_18(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.69f
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.940
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.86b
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.1aa
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.bd3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T, constants.%Self.69f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self => constants.%Self.69f
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.59b
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.350
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.83a
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.1e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T, constants.%HasF.facet.67e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self => constants.%HasF.facet.67e
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.48b
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.82e
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.b0b
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.418
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.9e4
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.95d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.48b
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness.d67
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.587
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.eab
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
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.e682: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.e683: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.2 [symbolic = %T.loc12_14.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc12_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.1 (constants.%T)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc12_24.2: type = symbolic_binding U, 1 [symbolic = %U.loc12_24.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (<error>), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T, constants.%U) [symbolic = @T.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc16_10.1: type = splice_block %.loc16_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc16_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc16_10.3: type = converted %.loc16_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.e682) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.e682) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc12_14.2: type, %U.loc12_24.2: type) {
// CHECK:STDOUT:   %T.loc12_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc12_24.1: type = symbolic_binding U, 1 [symbolic = %U.loc12_24.1 (constants.%U)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T.loc12_14.1, %U.loc12_24.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc12_14.1, %U.loc12_24.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e683) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e683) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.e682)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc12_14.2: type, @T.as.HasF.impl.%U.loc12_24.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.e683)]
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
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e682
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc12_14.1 => constants.%T
// CHECK:STDOUT:   %U.loc12_24.1 => constants.%U
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %HasF.type.fe3: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.59b: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.69f: %HasF.type.59b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.69f [symbolic]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type.940: type = fn_type @HasF.F, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %HasF.F.86b: %HasF.F.type.940 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.1aa: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.bd3: %HasF.assoc_type.1aa = assoc_entity element0, @HasF.%HasF.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.b92: <witness> = require_complete_type %HasF.type.59b [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.59b = facet_value %T, (%HasF.impl_witness) [symbolic]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.296: type = facet_type <@HasF, @HasF(%B)> [concrete]
// CHECK:STDOUT:   %Self.3f1: %HasF.type.296 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.F.type.1c6: type = fn_type @HasF.F, @HasF(%B) [concrete]
// CHECK:STDOUT:   %HasF.F.7cf: %HasF.F.type.1c6 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.1af: type = assoc_entity_type @HasF, @HasF(%B) [concrete]
// CHECK:STDOUT:   %assoc0.4fe: %HasF.assoc_type.1af = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe3 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_24: type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_35.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.59b)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@T.as.HasF.impl.%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = @T.as.HasF.impl.%HasF.impl_witness (constants.%HasF.impl_witness)]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type.c10 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c10 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %x: %A = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.1)> [symbolic = %HasF.type (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.69f)]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F, @HasF(%T.loc4_16.1) [symbolic = %HasF.F.type (constants.%HasF.F.type.940)]
// CHECK:STDOUT:   %HasF.F: @HasF.%HasF.F.type (%HasF.F.type.940) = struct_value () [symbolic = %HasF.F (constants.%HasF.F.86b)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF, @HasF(%T.loc4_16.1) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.1aa)]
// CHECK:STDOUT:   %assoc0.loc5_21.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.1aa) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.bd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.69f)]
// CHECK:STDOUT:     %HasF.F.decl: @HasF.%HasF.F.type (%HasF.F.type.940) = fn_decl @HasF.F [symbolic = @HasF.%HasF.F (constants.%HasF.F.86b)] {
// CHECK:STDOUT:       %self.patt: @HasF.F.%pattern_type (%pattern_type.429) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @HasF.F.%pattern_type (%pattern_type.429) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.F.%HasF.type (%HasF.type.59b) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:         %Self.ref: @HasF.F.%HasF.type (%HasF.type.59b) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.1aa) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.bd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = %assoc0.loc5_21.1
// CHECK:STDOUT:     witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_35.1: type = facet_type <@HasF, @HasF(%T.loc8_14.1)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_35.1 [symbolic = %require_complete (constants.%require_complete.b92)]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_14.1) [symbolic = %HasF.impl_witness (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_14.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc8_24 as %HasF.type.loc8_35.2 {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e68) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.e68) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref.loc8_24 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.HasF.impl.F.decl
// CHECK:STDOUT:     witness = file.%HasF.impl_witness
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
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%T.loc4_16.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.59b)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.59b)]
// CHECK:STDOUT:   %Self: @HasF.F.%HasF.type (%HasF.type.59b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.69f)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.429)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4b7)]
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [concrete = constants.%HasF.type.296]
// CHECK:STDOUT:   %.loc22: %HasF.assoc_type.1af = specific_constant @HasF.%assoc0.loc5_21.1, @HasF(constants.%B) [concrete = constants.%assoc0.4fe]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.1af = name_ref F, %.loc22 [concrete = constants.%assoc0.4fe]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.69f
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.940
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.86b
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.1aa
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.bd3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T, constants.%Self.69f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self => constants.%Self.69f
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_35.1 => constants.%HasF.type.59b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b92
// CHECK:STDOUT:   %HasF.impl_witness => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.59b
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.296
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.3f1
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.1c6
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.7cf
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.1af
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.4fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
