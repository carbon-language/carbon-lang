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
// CHECK:STDOUT:   %pattern_type.180: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.ae7: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.923: %HasF.WithSelf.F.type.ae7 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.760: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.644: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.641: %T.as.HasF.impl.F.type.644 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.230: %HasF.type = facet_value %T, (%HasF.impl_witness.760) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.8e6: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.230) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.d73: %HasF.WithSelf.F.type.8e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.94f: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.3e7: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.d35: %T.as.HasF.impl.F.type.3e7 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.9a7: %HasF.type = facet_value %empty_struct_type, (%HasF.impl_witness.94f) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.140: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.9a7) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.db5: %HasF.WithSelf.F.type.140 = struct_value () [concrete]
// CHECK:STDOUT:   %.af1: type = fn_type_with_self_type %HasF.WithSelf.F.type.140, %HasF.facet.9a7 [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.specific_fn: <specific function> = specific_function %T.as.HasF.impl.F.d35, @T.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.ae7) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.923)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.180) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.180) = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.760)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type.644)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.644) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.641)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.644) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.641)] {
// CHECK:STDOUT:       %self.param_patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_34.1: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.760)]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.180)]
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
// CHECK:STDOUT:   %impl.elem0: %.af1 = impl_witness_access constants.%HasF.impl_witness.94f, element0 [concrete = constants.%T.as.HasF.impl.F.d35]
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
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.ae7
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.923
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.760
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.644
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.641
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.230) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.230
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.8e6
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.d73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.230) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.230
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.3e7
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.d35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.9a7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.9a7
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.140
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.db5
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
// CHECK:STDOUT:   %Self.2b1: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.38c: type = facet_access_type %Self.2b1 [symbolic]
// CHECK:STDOUT:   %pattern_type.17f: type = pattern_type %Self.as_type.38c [symbolic]
// CHECK:STDOUT:   %.2c7: Core.Form = init_form %Self.as_type.38c [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.5da: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self.2b1) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.b3a: %HasF.WithSelf.F.type.5da = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.a20: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.75e: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.c09: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.c05: %ptr.as.HasF.impl.F.type.c09 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.376: %HasF.type = facet_value %ptr.e8f, (%HasF.impl_witness.75e) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.d51: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.376) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.518: %HasF.WithSelf.F.type.d51 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.5be: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.546: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.2a3: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.5be) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a8e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.2a3) [symbolic]
// CHECK:STDOUT:   %.86b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a8e, %Copy.facet.2a3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.480: %.86b = impl_witness_access %Copy.lookup_impl_witness.5be, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d1f: <specific function> = specific_impl_function %impl.elem0.480, @Copy.WithSelf.Op(%Copy.facet.2a3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %.9ba: Core.Form = init_form %ptr.c28 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.3f2: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.5ae: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.a15: %ptr.as.HasF.impl.F.type.5ae = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.0d1: %HasF.type = facet_value %ptr.c28, (%HasF.impl_witness.3f2) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.362: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.0d1) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.b2b: %HasF.WithSelf.F.type.362 = struct_value () [concrete]
// CHECK:STDOUT:   %.0b0: type = fn_type_with_self_type %HasF.WithSelf.F.type.362, %HasF.facet.0d1 [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.specific_fn: <specific function> = specific_function %ptr.as.HasF.impl.F.a15, @ptr.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.561: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.1db: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.503: %ptr.as.Copy.impl.Op.type.1db = struct_value () [concrete]
// CHECK:STDOUT:   %.74f: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Copy.facet.695: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.561) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.1fd: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.695) [concrete]
// CHECK:STDOUT:   %.18e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.1fd, %Copy.facet.695 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.503, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self.2b1]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.5da) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.b3a)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.17f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.17f) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.17f) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.17f) = return_slot_pattern %return.param_patt, %.loc5_25.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_25: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.2b1)]
// CHECK:STDOUT:     %Self.as_type.loc5_25: type = facet_access_type %Self.ref.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)]
// CHECK:STDOUT:     %.loc5_25.2: type = converted %Self.ref.loc5_25, %Self.as_type.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)]
// CHECK:STDOUT:     %.loc5_25.3: Core.Form = init_form %.loc5_25.2 [symbolic = %.loc5_25.1 (constants.%.2c7)]
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)] {
// CHECK:STDOUT:       %Self.ref.loc5_14: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.2b1)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref.loc5_14 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref.loc5_14, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0.a20]
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
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.75e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %ptr.as.HasF.impl.F.type (constants.%ptr.as.HasF.impl.F.type.c09)]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.c09) = struct_value () [symbolic = %ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %ptr.loc8_25.2 as %HasF.ref {
// CHECK:STDOUT:     %ptr.as.HasF.impl.F.decl: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.c09) = fn_decl @ptr.as.HasF.impl.F [symbolic = @ptr.as.HasF.impl.%ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.c05)] {
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
// CHECK:STDOUT:     %HasF.impl_witness.loc8_35.1: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(constants.%T.67d) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.75e)]
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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.2b1)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.38c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.17f)]
// CHECK:STDOUT:   %.loc5_25.1: Core.Form = init_form %Self.as_type.loc5_14.1 [symbolic = %.loc5_25.1 (constants.%.2c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c)) -> out %return.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type.38c);
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
// CHECK:STDOUT:   %.loc9_37.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T) [symbolic = %.loc9_37.1 (constants.%.546)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc9_14, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.5be)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc9_14, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.2a3)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a8e)]
// CHECK:STDOUT:   %.loc9_37.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc9_37.2 (constants.%.86b)]
// CHECK:STDOUT:   %impl.elem0.loc9_37.2: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.86b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.480)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2: <specific function> = specific_impl_function %impl.elem0.loc9_37.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.d1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f)) -> out %return.param: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @ptr.as.HasF.impl.F.%ptr.loc9_14 (%ptr.e8f) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc9_37.1: @ptr.as.HasF.impl.F.%.loc9_37.2 (%.86b) = impl_witness_access constants.%Copy.lookup_impl_witness.5be, element0 [symbolic = %impl.elem0.loc9_37.2 (constants.%impl.elem0.480)]
// CHECK:STDOUT:     %bound_method.loc9_37.1: <bound method> = bound_method %self.ref, %impl.elem0.loc9_37.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_37.1: <specific function> = specific_impl_function %impl.elem0.loc9_37.1, @Copy.WithSelf.Op(constants.%Copy.facet.2a3) [symbolic = %specific_impl_fn.loc9_37.2 (constants.%specific_impl_fn.d1f)]
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
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0.a20]
// CHECK:STDOUT:   %impl.elem0: %.0b0 = impl_witness_access constants.%HasF.impl_witness.3f2, element0 [concrete = constants.%ptr.as.HasF.impl.F.a15]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%ptr.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.call: init %ptr.c28 = call %bound_method.loc13_21(%x.ref)
// CHECK:STDOUT:   return %ptr.as.HasF.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self.2b1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.2b1
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.5da
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self.2b1) {
// CHECK:STDOUT:   %Self => constants.%Self.2b1
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.38c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17f
// CHECK:STDOUT:   %.loc5_25.1 => constants.%.2c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.75e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.c09
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.c05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc9_14 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc9_26.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.376) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.376
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.d51
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.518
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.376) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.376
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5_25.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.c28
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.3f2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.5ae
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.a15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.0d1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.0d1
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.362
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b2b
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
// CHECK:STDOUT:   %.loc9_37.1 => constants.%.74f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.561
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.695
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.1fd
// CHECK:STDOUT:   %.loc9_37.2 => constants.%.18e
// CHECK:STDOUT:   %impl.elem0.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.503
// CHECK:STDOUT:   %specific_impl_fn.loc9_37.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.b27: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.ebe: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.5fe: %HasF.WithSelf.F.type.ebe = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.14d: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.62c: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.a4d: type = pattern_type %C.14d [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.9c6: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.0a6: %C.as.HasF.impl.F.type.9c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.81b: %HasF.type = facet_value %C.14d, (%HasF.impl_witness.62c) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.bd3: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.81b) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.8cd: %HasF.WithSelf.F.type.bd3 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.14d [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.d02: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.e33: type = pattern_type %C.d02 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.f4c: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.e82: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.788: %C.as.HasF.impl.F.type.e82 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.7ab: %HasF.type = facet_value %C.d02, (%HasF.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.ba5: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.7ab) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.ad9: %HasF.WithSelf.F.type.ba5 = struct_value () [concrete]
// CHECK:STDOUT:   %.927: type = fn_type_with_self_type %HasF.WithSelf.F.type.ba5, %HasF.facet.7ab [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.specific_fn: <specific function> = specific_function %C.as.HasF.impl.F.788, @C.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:     %C.loc10_27.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.1 (constants.%C.14d)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.e33 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.e33 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.d02 = value_param call_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %C [concrete = constants.%C.d02] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %.loc14_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.d02]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.d02 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.ebe) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.5fe)] {
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.b27) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.b27) = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %C.loc10_27.1: type = class_type @C, @C(%T.loc10_15.1) [symbolic = %C.loc10_27.1 (constants.%C.14d)]
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.62c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %C.as.HasF.impl.F.type (constants.%C.as.HasF.impl.F.type.9c6)]
// CHECK:STDOUT:   %C.as.HasF.impl.F: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.9c6) = struct_value () [symbolic = %C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.0a6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.2 as %HasF.ref {
// CHECK:STDOUT:     %C.as.HasF.impl.F.decl: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.9c6) = fn_decl @C.as.HasF.impl.F [symbolic = @C.as.HasF.impl.%C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.0a6)] {
// CHECK:STDOUT:       %self.param_patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.a4d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.as.HasF.impl.F.%pattern_type (%pattern_type.a4d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.HasF.impl.F.%C (%C.14d) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.loc10_27.2 [symbolic = %C (constants.%C.14d)]
// CHECK:STDOUT:       %self: @C.as.HasF.impl.F.%C (%C.14d) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc10_37.1: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.62c)]
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
// CHECK:STDOUT:     .Self = constants.%C.14d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.b27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.HasF.impl.F(@C.as.HasF.impl.%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.14d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.a4d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.HasF.impl.F.%C (%C.14d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %C.d02) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.d02 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.927 = impl_witness_access constants.%HasF.impl_witness.f4c, element0 [concrete = constants.%C.as.HasF.impl.F.788]
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
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.ebe
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.5fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b27
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
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.14d
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.62c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.9c6
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.14d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.81b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.81b
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.bd3
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.8cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.81b) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.81b
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C.14d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4d
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
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.d02
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.f4c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.e82
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.788
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.7ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.7ab
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.ba5
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.ad9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.d02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e33
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
// CHECK:STDOUT:   %HasF.type.38c: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.38c = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.e32: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.965: %HasF.type.e32 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.965 [symbolic]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.d3a: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.965) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.53d: %HasF.WithSelf.F.type.d3a = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.a6e: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.b35: %HasF.assoc_type.a6e = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.fe5: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.e32 [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.012: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.80e: %empty_struct_type.as.HasF.impl.F.type.012 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.059: %HasF.type.e32 = facet_value %empty_struct_type, (%HasF.impl_witness.fe5) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.577: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet.059) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.fbd: %HasF.WithSelf.F.type.577 = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.60f: type = facet_type <@HasF, @HasF(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.510: %HasF.type.60f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.666: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %Self.965) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.14a: %HasF.WithSelf.F.type.666 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.824: type = assoc_entity_type @HasF, @HasF(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.101: %HasF.assoc_type.824 = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.7f7: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %HasF.type.60f [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.31f: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.0c2: %empty_struct_type.as.HasF.impl.F.type.31f = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.c55: %HasF.type.60f = facet_value %empty_struct_type, (%HasF.impl_witness.7f7) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.bc9: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %HasF.facet.c55) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.03b: %HasF.WithSelf.F.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %.dd1: type = fn_type_with_self_type %HasF.WithSelf.F.type.bc9, %HasF.facet.c55 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.specific_fn: <specific function> = specific_function %empty_struct_type.as.HasF.impl.F.0c2, @empty_struct_type.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.38c = interface_decl @HasF [concrete = constants.%HasF.generic] {
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
// CHECK:STDOUT:     %HasF.ref: %HasF.type.38c = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_36.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.e32)]
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
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.e32)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.e32) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.965)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.e32) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.965)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.d3a) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.53d)] {
// CHECK:STDOUT:       %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.1c6) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.1c6) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.e32) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.965)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.e32) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.965)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.a6e) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.b35)]
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
// CHECK:STDOUT:   %HasF.type.loc8_36.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.e32)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.fe5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_36.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %empty_struct_type.as.HasF.impl.F.type (constants.%empty_struct_type.as.HasF.impl.F.type.012)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.012) = struct_value () [symbolic = %empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.80e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_36.2 {
// CHECK:STDOUT:     %empty_struct_type.as.HasF.impl.F.decl: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.012) = fn_decl @empty_struct_type.as.HasF.impl.F [symbolic = @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.80e)] {
// CHECK:STDOUT:       %self.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: %pattern_type.a96 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @empty_struct_type.as.HasF.impl.%.loc8_25.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %self: %empty_struct_type = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%empty_struct_type.as.HasF.impl.F.decl), @empty_struct_type.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_38.1: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.fe5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %empty_struct_type.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.type.loc8_36.2
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_38.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.e32)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.e32)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.e32) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.965)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.1c6)]
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.38c = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %.loc13_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_13: type = converted %.loc13_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%empty_struct_type)> [concrete = constants.%HasF.type.60f]
// CHECK:STDOUT:   %.loc13_14: %HasF.assoc_type.824 = specific_constant @HasF.WithSelf.%assoc0.loc5_21.1, @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.965) [concrete = constants.%assoc0.101]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.824 = name_ref F, %.loc13_14 [concrete = constants.%assoc0.101]
// CHECK:STDOUT:   %impl.elem0: %.dd1 = impl_witness_access constants.%HasF.impl_witness.7f7, element0 [concrete = constants.%empty_struct_type.as.HasF.impl.F.0c2]
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
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.e32
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.965
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.965) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.e32
// CHECK:STDOUT:   %Self => constants.%Self.965
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.d3a
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.53d
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.a6e
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.b35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.965) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.e32
// CHECK:STDOUT:   %Self => constants.%Self.965
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.e32
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.fe5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.012
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.80e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%HasF.facet.059) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.e32
// CHECK:STDOUT:   %Self => constants.%HasF.facet.059
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.577
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.fbd
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.a6e
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.b35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet.059) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.e32
// CHECK:STDOUT:   %Self => constants.%HasF.facet.059
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.60f
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.965) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.60f
// CHECK:STDOUT:   %Self => constants.%Self.965
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.666
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.14a
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.824
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.60f
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.7f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.31f
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.0c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%HasF.facet.c55) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.60f
// CHECK:STDOUT:   %Self => constants.%HasF.facet.c55
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.bc9
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.03b
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.824
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.101
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
// CHECK:STDOUT:   %pattern_type.171: type = pattern_type %Self.as_type [symbolic]
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
// CHECK:STDOUT:     %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.171) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.171) = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.171)]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.171
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
// CHECK:STDOUT:   %HasF.type.993: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.993 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.250: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.548: %HasF.type.250 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.548 [symbolic]
// CHECK:STDOUT:   %pattern_type.338: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.645: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.548) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.b30: %HasF.WithSelf.F.type.645 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.e56: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.3cd: %HasF.assoc_type.e56 = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.54c: <witness> = require_complete_type %HasF.type.250 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.250 = facet_value %T, (%HasF.impl_witness) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.d4a: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.78b: %HasF.WithSelf.F.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.e0a: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.b83: type = facet_type <@HasF, @HasF(%B)> [concrete]
// CHECK:STDOUT:   %Self.fc2: %HasF.type.b83 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.3c1: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%B, %Self.548) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.b80: %HasF.WithSelf.F.type.3c1 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.a81: type = assoc_entity_type @HasF, @HasF(%B) [concrete]
// CHECK:STDOUT:   %assoc0.314: %HasF.assoc_type.a81 = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.993 = interface_decl @HasF [concrete = constants.%HasF.generic] {
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
// CHECK:STDOUT:     %HasF.ref: %HasF.type.993 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_35.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.250)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.e0a = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.e0a = at_binding_pattern x, %x.param_patt [concrete]
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
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.250)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.250) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.548)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.250) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.548)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.645) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.b30)] {
// CHECK:STDOUT:       %self.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.338) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.338) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.250) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.548)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.250) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.548)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_21.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.e56) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_21.2 (constants.%assoc0.3cd)]
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
// CHECK:STDOUT:   %HasF.type.loc8_35.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.250)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_37.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_35.1 [symbolic = %require_complete (constants.%require_complete.54c)]
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
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.250)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.250)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.250) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.548)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.338)]
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.993 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [concrete = constants.%HasF.type.b83]
// CHECK:STDOUT:   %.loc22: %HasF.assoc_type.a81 = specific_constant @HasF.WithSelf.%assoc0.loc5_21.1, @HasF.WithSelf(constants.%B, constants.%Self.548) [concrete = constants.%assoc0.314]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.a81 = name_ref F, %.loc22 [concrete = constants.%assoc0.314]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.250
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.548
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.548) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.250
// CHECK:STDOUT:   %Self => constants.%Self.548
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.645
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b30
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.e56
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.3cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.548) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.250
// CHECK:STDOUT:   %Self => constants.%Self.548
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_35.1 => constants.%HasF.type.250
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2 => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.54c
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
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.250
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.d4a
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.78b
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.e56
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.3cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.250
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b83
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.fc2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%B, constants.%Self.548) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b83
// CHECK:STDOUT:   %Self => constants.%Self.548
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.3c1
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.b80
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.a81
// CHECK:STDOUT:   %assoc0.loc5_21.2 => constants.%assoc0.314
// CHECK:STDOUT: }
// CHECK:STDOUT:
