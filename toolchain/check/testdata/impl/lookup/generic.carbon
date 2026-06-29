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
  fn F(self);
}

impl forall [T:! type] T as HasF {
  fn F(unused self) {}
}

fn G(x: {}) {
  x.(HasF.F)();
}

// --- deduced_type_subst.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F(self) -> Self;
}

impl forall [T:! type] T* as HasF {
  fn F(self) -> T* { return self; }
}

fn G(x: {}*) -> {}* {
  return x.(HasF.F)();
}

// --- deduced_type_argument.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F(self);
}

class C(T:! type) {}

impl forall [T:! type] C(T) as HasF {
  fn F(unused self) {}
}

fn G(x: C({})) {
  x.(HasF.F)();
}

// --- deduced_interface_argument.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F(self);
}

impl forall [T:! type] {} as HasF(T) {
  fn F(unused self) {}
}

fn G(x: {}) {
  x.(HasF({}).F)();
}

// --- fail_incomplete_deduction.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F(self);
}

// CHECK:STDERR: fail_incomplete_deduction.carbon:[[@LINE+4]]:13: error: `impl` with unused generic binding [ImplUnusedBinding]
// CHECK:STDERR: impl forall [T:! type, U:! type] T as HasF {
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type, U:! type] T as HasF {
  fn F(unused self) {}
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
  fn F(self);
}

impl forall [T:! type] T as HasF(T) {
  fn F(unused self) {}
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
// CHECK:STDOUT:   %pattern_type.c9f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.74d: %pattern_type.c9f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b33: %pattern_type.c9f = at_binding_pattern self, %self.param_patt.74d [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.a73: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.851: %HasF.WithSelf.F.type.a73 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.ebe: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %self.param_patt.b7c: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.7e4: %pattern_type.51d = at_binding_pattern self, %self.param_patt.b7c [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.195: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.863: %T.as.HasF.impl.F.type.195 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.805: %HasF.type = facet_value %T, (%HasF.impl_witness.ebe) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.b00: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.805) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.33f: %HasF.WithSelf.F.type.b00 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.aaf: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type.d06: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.e31: %T.as.HasF.impl.F.type.d06 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.b0f: %HasF.type = facet_value %empty_struct_type, (%HasF.impl_witness.aaf) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.476: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.b0f) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.141: %HasF.WithSelf.F.type.476 = struct_value () [concrete]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %HasF.WithSelf.F.type.476, %HasF.facet.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.482: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b63: %pattern_type.a96 = at_binding_pattern self, %self.param_patt.482 [concrete]
// CHECK:STDOUT:   %T.as.HasF.impl.F.specific_fn: <specific function> = specific_function %T.as.HasF.impl.F.e31, @T.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.a73) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.851)] {
// CHECK:STDOUT:     %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:     %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_8.1: type = splice_block %.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_8.2: type = converted %Self.ref, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.ebe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type.195)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.195) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.863)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type.195) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F.863)] {
// CHECK:STDOUT:       %self.param_patt.loc9_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc9_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:       %self.patt.loc9_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc9_15.1 [symbolic = %self.patt.loc9_15.2 (constants.%self.patt.7e4)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%T.as.HasF.impl.F.decl), @T.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_34.1: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_34.2 (constants.%HasF.impl_witness.ebe)]
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
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.c9f)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %self.param_patt.loc9_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc9_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:   %self.patt.loc9_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc9_15.2 [symbolic = %self.patt.loc9_15.2 (constants.%self.patt.7e4)]
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
// CHECK:STDOUT:   %impl.elem0: %.be5 = impl_witness_access constants.%HasF.impl_witness.aaf, element0 [concrete = constants.%T.as.HasF.impl.F.e31]
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
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.a73
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.851
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c9f
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.74d
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.b33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.ebe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.195
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.863
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %self.param_patt.loc9_15.2 => constants.%self.param_patt.b7c
// CHECK:STDOUT:   %self.patt.loc9_15.2 => constants.%self.patt.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.805) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.805
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.b00
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.33f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.805) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.805
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.b7c
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.impl_witness.loc8_34.2 => constants.%HasF.impl_witness.aaf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type.d06
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F.e31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.b0f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.b0f
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.476
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT:   %self.param_patt.loc9_15.2 => constants.%self.param_patt.482
// CHECK:STDOUT:   %self.patt.loc9_15.2 => constants.%self.patt.b63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_subst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.c62: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.5c2: type = facet_access_type %Self.c62 [symbolic]
// CHECK:STDOUT:   %pattern_type.c9f: type = pattern_type %Self.as_type.5c2 [symbolic]
// CHECK:STDOUT:   %self.param_patt.74d: %pattern_type.c9f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b33: %pattern_type.c9f = at_binding_pattern self, %self.param_patt.74d [symbolic]
// CHECK:STDOUT:   %.376: Core.Form = init_form %Self.as_type.5c2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.78b: %pattern_type.c9f = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c00: %pattern_type.c9f = return_slot_pattern %return.param_patt.78b, %Self.as_type.5c2 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.a73: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self.c62) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.851: %HasF.WithSelf.F.type.a73 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.a0c: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness.672: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %self.param_patt.921: %pattern_type.4f4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.99b: %pattern_type.4f4 = at_binding_pattern self, %self.param_patt.921 [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.d5f: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.be3: %ptr.as.HasF.impl.F.type.d5f = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.3fb: %HasF.type = facet_value %ptr.e8f, (%HasF.impl_witness.672) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.7fc: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.3fb) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.cbb: %HasF.WithSelf.F.type.7fc = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.1cc = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.1cc = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %.9ba: Core.Form = init_form %ptr.c28 [concrete]
// CHECK:STDOUT:   %return.param_patt.07a: %pattern_type.1cc = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.da5: %pattern_type.1cc = return_slot_pattern %return.param_patt.07a, %ptr.c28 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.e43: <witness> = impl_witness @ptr.as.HasF.impl.%HasF.impl_witness_table, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type.03e: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.156: %ptr.as.HasF.impl.F.type.03e = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.7a9: %HasF.type = facet_value %ptr.c28, (%HasF.impl_witness.e43) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.c4f: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.7a9) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.814: %HasF.WithSelf.F.type.c4f = struct_value () [concrete]
// CHECK:STDOUT:   %.2ad: type = fn_type_with_self_type %HasF.WithSelf.F.type.c4f, %HasF.facet.7a9 [concrete]
// CHECK:STDOUT:   %self.param_patt.4b0: %pattern_type.1cc = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.254: %pattern_type.1cc = at_binding_pattern self, %self.param_patt.4b0 [concrete]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.specific_fn: <specific function> = specific_function %ptr.as.HasF.impl.F.156, @ptr.as.HasF.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1c2: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.6c5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.9ed: %ptr.as.Copy.impl.Op.type.6c5 = struct_value () [concrete]
// CHECK:STDOUT:   %.a81: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Copy.facet.80f: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.1c2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.8a4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.80f) [concrete]
// CHECK:STDOUT:   %.523: type = fn_type_with_self_type %Copy.WithSelf.Op.type.8a4, %Copy.facet.80f [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.9ed, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt.d47)]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1cc = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.1cc = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1cc = out_param_pattern [concrete = constants.%return.param_patt.07a]
// CHECK:STDOUT:     %return.patt: %pattern_type.1cc = return_slot_pattern %return.param_patt, %ptr.loc12_19 [concrete = constants.%return.patt.da5]
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
// CHECK:STDOUT:     %x: %ptr.c28 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %ptr.c28 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c62]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.a73) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.851)] {
// CHECK:STDOUT:     %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:     %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:     %return.param_patt.loc5_17.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = out_param_pattern [symbolic = %return.param_patt.loc5_17.2 (constants.%return.param_patt.78b)]
// CHECK:STDOUT:     %return.patt.loc5_14.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = return_slot_pattern %return.param_patt.loc5_17.1, %.loc5_17.2 [symbolic = %return.patt.loc5_14.2 (constants.%return.patt.c00)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_17: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.c62)]
// CHECK:STDOUT:     %Self.as_type.loc5_17: type = facet_access_type %Self.ref.loc5_17 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)]
// CHECK:STDOUT:     %.loc5_17.2: type = converted %Self.ref.loc5_17, %Self.as_type.loc5_17 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)]
// CHECK:STDOUT:     %.loc5_17.3: Core.Form = init_form %.loc5_17.2 [symbolic = %.loc5_17.1 (constants.%.376)]
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2) = value_param call_param0
// CHECK:STDOUT:     %.loc5_8.1: type = splice_block %.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)] {
// CHECK:STDOUT:       %Self.ref.loc5_8: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self.c62)]
// CHECK:STDOUT:       %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref.loc5_8 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)]
// CHECK:STDOUT:       %.loc5_8.2: type = converted %Self.ref.loc5_8, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0.a0c]
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
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T.67d)]
// CHECK:STDOUT:   %ptr.loc8_25.1: type = ptr_type %T.loc8_15.1 [symbolic = %ptr.loc8_25.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.672)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type: type = fn_type @ptr.as.HasF.impl.F, @ptr.as.HasF.impl(%T.loc8_15.1) [symbolic = %ptr.as.HasF.impl.F.type (constants.%ptr.as.HasF.impl.F.type.d5f)]
// CHECK:STDOUT:   %ptr.as.HasF.impl.F: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.d5f) = struct_value () [symbolic = %ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.be3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %ptr.loc8_25.2 as %HasF.ref {
// CHECK:STDOUT:     %ptr.as.HasF.impl.F.decl: @ptr.as.HasF.impl.%ptr.as.HasF.impl.F.type (%ptr.as.HasF.impl.F.type.d5f) = fn_decl @ptr.as.HasF.impl.F [symbolic = @ptr.as.HasF.impl.%ptr.as.HasF.impl.F (constants.%ptr.as.HasF.impl.F.be3)] {
// CHECK:STDOUT:       %self.param_patt.loc9_8.1: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = value_param_pattern [symbolic = %self.param_patt.loc9_8.2 (constants.%self.param_patt.921)]
// CHECK:STDOUT:       %self.patt.loc9_8.1: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = at_binding_pattern self, %self.param_patt.loc9_8.1 [symbolic = %self.patt.loc9_8.2 (constants.%self.patt.99b)]
// CHECK:STDOUT:       %return.param_patt.loc9_18.1: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc9_18.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:       %return.patt.loc9_14.1: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc9_18.1, %ptr.loc9_18 [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ptr.as.HasF.impl.%T.loc8_15.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %ptr.loc9_18: type = ptr_type %T.ref [symbolic = %ptr.loc9_8 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %.loc9_18.2: Core.Form = init_form %ptr.loc9_18 [symbolic = %.loc9_18.1 (constants.%.cb6)]
// CHECK:STDOUT:       %self.param: @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @ptr.as.HasF.impl.%ptr.loc8_25.2 [symbolic = %ptr.loc9_8 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %self: @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%ptr.as.HasF.impl.F.decl), @ptr.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_35.1: <witness> = impl_witness %HasF.impl_witness_table, @ptr.as.HasF.impl(constants.%T.67d) [symbolic = %HasF.impl_witness.loc8_35.2 (constants.%HasF.impl_witness.672)]
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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c62)]
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type.5c2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.c9f)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:   %.loc5_17.1: Core.Form = init_form %Self.as_type.loc5_8.1 [symbolic = %.loc5_17.1 (constants.%.376)]
// CHECK:STDOUT:   %return.param_patt.loc5_17.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = out_param_pattern [symbolic = %return.param_patt.loc5_17.2 (constants.%return.param_patt.78b)]
// CHECK:STDOUT:   %return.patt.loc5_14.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = return_slot_pattern %return.param_patt.loc5_17.2, %Self.as_type.loc5_8.1 [symbolic = %return.patt.loc5_14.2 (constants.%return.patt.c00)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2)) -> out %return.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type.5c2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ptr.as.HasF.impl.F(@ptr.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %ptr.loc9_8: type = ptr_type %T [symbolic = %ptr.loc9_8 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc9_8 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %self.param_patt.loc9_8.2: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = value_param_pattern [symbolic = %self.param_patt.loc9_8.2 (constants.%self.param_patt.921)]
// CHECK:STDOUT:   %self.patt.loc9_8.2: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = at_binding_pattern self, %self.param_patt.loc9_8.2 [symbolic = %self.patt.loc9_8.2 (constants.%self.patt.99b)]
// CHECK:STDOUT:   %.loc9_18.1: Core.Form = init_form %ptr.loc9_8 [symbolic = %.loc9_18.1 (constants.%.cb6)]
// CHECK:STDOUT:   %return.param_patt.loc9_18.2: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc9_18.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:   %return.patt.loc9_14.2: @ptr.as.HasF.impl.F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc9_18.2, %ptr.loc9_8 [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc9_8 [symbolic = %require_complete (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc9_29.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T) [symbolic = %.loc9_29.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc9_8, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc9_29.3: %Copy.type = facet_value %ptr.loc9_8, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc9_29.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc9_29.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc9_29.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc9_29.3 [symbolic = %.loc9_29.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc9_29.2: @ptr.as.HasF.impl.F.%.loc9_29.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_29.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_29.2: <specific function> = specific_impl_function %impl.elem0.loc9_29.2, @Copy.WithSelf.Op(%Copy.facet.loc9_29.3) [symbolic = %specific_impl_fn.loc9_29.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f)) -> out %return.param: @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc9_29.1: @ptr.as.HasF.impl.F.%.loc9_29.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc9_29.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc9_29.1: <bound method> = bound_method %self.ref, %impl.elem0.loc9_29.1
// CHECK:STDOUT:     %Copy.facet.loc9_29.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_29.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_29.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_29.1 [symbolic = %Copy.facet.loc9_29.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Copy.facet.loc9_29.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc9_29.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc9_29.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc9_29.2 [symbolic = %Copy.facet.loc9_29.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_29.1: <specific function> = specific_impl_function %impl.elem0.loc9_29.1, @Copy.WithSelf.Op(constants.%Copy.facet.302) [symbolic = %specific_impl_fn.loc9_29.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc9_29.2: <bound method> = bound_method %self.ref, %specific_impl_fn.loc9_29.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @ptr.as.HasF.impl.F.%ptr.loc9_8 (%ptr.e8f) = call %bound_method.loc9_29.2(%self.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %ptr.c28) -> out %return.param: %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %ptr.c28 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0.a0c]
// CHECK:STDOUT:   %impl.elem0: %.2ad = impl_witness_access constants.%HasF.impl_witness.e43, element0 [concrete = constants.%ptr.as.HasF.impl.F.156]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%ptr.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.call: init %ptr.c28 = call %bound_method.loc13_21(%x.ref)
// CHECK:STDOUT:   return %ptr.as.HasF.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self.c62) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c62
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.a73
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.851
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self.c62) {
// CHECK:STDOUT:   %Self => constants.%Self.c62
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type.5c2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c9f
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.74d
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.b33
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.376
// CHECK:STDOUT:   %return.param_patt.loc5_17.2 => constants.%return.param_patt.78b
// CHECK:STDOUT:   %return.patt.loc5_14.2 => constants.%return.patt.c00
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.672
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.d5f
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.be3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc9_8 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %self.param_patt.loc9_8.2 => constants.%self.param_patt.921
// CHECK:STDOUT:   %self.patt.loc9_8.2 => constants.%self.patt.99b
// CHECK:STDOUT:   %.loc9_18.1 => constants.%.cb6
// CHECK:STDOUT:   %return.param_patt.loc9_18.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.3fb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.3fb
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.7fc
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.3fb) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.3fb
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.921
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.99b
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.cb6
// CHECK:STDOUT:   %return.param_patt.loc5_17.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc5_14.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc8_25.1 => constants.%ptr.c28
// CHECK:STDOUT:   %HasF.impl_witness.loc8_35.2 => constants.%HasF.impl_witness.e43
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.as.HasF.impl.F.type => constants.%ptr.as.HasF.impl.F.type.03e
// CHECK:STDOUT:   %ptr.as.HasF.impl.F => constants.%ptr.as.HasF.impl.F.156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.7a9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.7a9
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.c4f
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.814
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ptr.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc9_8 => constants.%ptr.c28
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1cc
// CHECK:STDOUT:   %self.param_patt.loc9_8.2 => constants.%self.param_patt.4b0
// CHECK:STDOUT:   %self.patt.loc9_8.2 => constants.%self.patt.254
// CHECK:STDOUT:   %.loc9_18.1 => constants.%.9ba
// CHECK:STDOUT:   %return.param_patt.loc9_18.2 => constants.%return.param_patt.07a
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.da5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %.loc9_29.3 => constants.%.a81
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1c2
// CHECK:STDOUT:   %Copy.facet.loc9_29.3 => constants.%Copy.facet.80f
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.8a4
// CHECK:STDOUT:   %.loc9_29.4 => constants.%.523
// CHECK:STDOUT:   %impl.elem0.loc9_29.2 => constants.%ptr.as.Copy.impl.Op.9ed
// CHECK:STDOUT:   %specific_impl_fn.loc9_29.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.c9f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.74d: %pattern_type.c9f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b33: %pattern_type.c9f = at_binding_pattern self, %self.param_patt.74d [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.a73: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.851: %HasF.WithSelf.F.type.a73 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1d0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.4e6: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.ebe: type = pattern_type %C.1d0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.5bc: %pattern_type.ebe = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.6f1: %pattern_type.ebe = at_binding_pattern self, %self.param_patt.5bc [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.67f: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.HasF.impl.F.7ba: %C.as.HasF.impl.F.type.67f = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.0c3: %HasF.type = facet_value %C.1d0, (%HasF.impl_witness.4e6) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.2db: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.0c3) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.5ef: %HasF.WithSelf.F.type.2db = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.1d0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.d8e: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.2fd: type = pattern_type %C.d8e [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.2fd = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.2fd = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.f0d: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type.edc: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.612: %C.as.HasF.impl.F.type.edc = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.132: %HasF.type = facet_value %C.d8e, (%HasF.impl_witness.f0d) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.938: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet.132) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.787: %HasF.WithSelf.F.type.938 = struct_value () [concrete]
// CHECK:STDOUT:   %.bf7: type = fn_type_with_self_type %HasF.WithSelf.F.type.938, %HasF.facet.132 [concrete]
// CHECK:STDOUT:   %self.param_patt.12a: %pattern_type.2fd = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ac5: %pattern_type.2fd = at_binding_pattern self, %self.param_patt.12a [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.specific_fn: <specific function> = specific_function %C.as.HasF.impl.F.612, @C.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:     %T.patt.loc8_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.2 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.1 (constants.%C.1d0)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.2fd = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.2fd = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.d8e = value_param call_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %C [concrete = constants.%C.d8e] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %.loc14_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.d8e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.d8e = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.a73) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.851)] {
// CHECK:STDOUT:     %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:     %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_8.1: type = splice_block %.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_8.2: type = converted %Self.ref, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %T.patt.loc10_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc10_27.1: type = class_type @C, @C(%T.loc10_15.1) [symbolic = %C.loc10_27.1 (constants.%C.1d0)]
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.4e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F, @C.as.HasF.impl(%T.loc10_15.1) [symbolic = %C.as.HasF.impl.F.type (constants.%C.as.HasF.impl.F.type.67f)]
// CHECK:STDOUT:   %C.as.HasF.impl.F: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.67f) = struct_value () [symbolic = %C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.7ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.2 as %HasF.ref {
// CHECK:STDOUT:     %C.as.HasF.impl.F.decl: @C.as.HasF.impl.%C.as.HasF.impl.F.type (%C.as.HasF.impl.F.type.67f) = fn_decl @C.as.HasF.impl.F [symbolic = @C.as.HasF.impl.%C.as.HasF.impl.F (constants.%C.as.HasF.impl.F.7ba)] {
// CHECK:STDOUT:       %self.param_patt.loc11_15.1: @C.as.HasF.impl.F.%pattern_type (%pattern_type.ebe) = value_param_pattern [symbolic = %self.param_patt.loc11_15.2 (constants.%self.param_patt.5bc)]
// CHECK:STDOUT:       %self.patt.loc11_15.1: @C.as.HasF.impl.F.%pattern_type (%pattern_type.ebe) = at_binding_pattern self, %self.param_patt.loc11_15.1 [symbolic = %self.patt.loc11_15.2 (constants.%self.patt.6f1)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.HasF.impl.F.%C (%C.1d0) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.loc10_27.2 [symbolic = %C (constants.%C.1d0)]
// CHECK:STDOUT:       %self: @C.as.HasF.impl.F.%C (%C.1d0) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc10_37.1: <witness> = impl_witness %HasF.impl_witness_table, @C.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc10_37.2 (constants.%HasF.impl_witness.4e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.ref
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc10_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1d0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.c9f)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.HasF.impl.F(@C.as.HasF.impl.%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.1d0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.ebe)]
// CHECK:STDOUT:   %self.param_patt.loc11_15.2: @C.as.HasF.impl.F.%pattern_type (%pattern_type.ebe) = value_param_pattern [symbolic = %self.param_patt.loc11_15.2 (constants.%self.param_patt.5bc)]
// CHECK:STDOUT:   %self.patt.loc11_15.2: @C.as.HasF.impl.F.%pattern_type (%pattern_type.ebe) = at_binding_pattern self, %self.param_patt.loc11_15.2 [symbolic = %self.patt.loc11_15.2 (constants.%self.patt.6f1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.HasF.impl.F.%C (%C.1d0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %C.d8e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.d8e = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.bf7 = impl_witness_access constants.%HasF.impl_witness.f0d, element0 [concrete = constants.%C.as.HasF.impl.F.612]
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
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.a73
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.851
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c9f
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.74d
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.b33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.1d0
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.67f
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.7ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.1d0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ebe
// CHECK:STDOUT:   %self.param_patt.loc11_15.2 => constants.%self.param_patt.5bc
// CHECK:STDOUT:   %self.patt.loc11_15.2 => constants.%self.patt.6f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.0c3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.0c3
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.2db
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.5ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet.0c3) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet.0c3
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%C.1d0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ebe
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.5bc
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.6f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc8_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C.d8e
// CHECK:STDOUT:   %HasF.impl_witness.loc10_37.2 => constants.%HasF.impl_witness.f0d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.HasF.impl.F.type => constants.%C.as.HasF.impl.F.type.edc
// CHECK:STDOUT:   %C.as.HasF.impl.F => constants.%C.as.HasF.impl.F.612
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet.132) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet.132
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.938
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.787
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.HasF.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.d8e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2fd
// CHECK:STDOUT:   %self.param_patt.loc11_15.2 => constants.%self.param_patt.12a
// CHECK:STDOUT:   %self.patt.loc11_15.2 => constants.%self.patt.ac5
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
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.c96: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.c96 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.daf: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.822: %HasF.type.daf = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.822 [symbolic]
// CHECK:STDOUT:   %pattern_type.a2b: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.d2c: %pattern_type.a2b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.763: %pattern_type.a2b = at_binding_pattern self, %self.param_patt.d2c [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.41d: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.822) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.ff7: %HasF.WithSelf.F.type.41d = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.ddf: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.e9a: %HasF.assoc_type.ddf = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.932: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.daf [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.482: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b63: %pattern_type.a96 = at_binding_pattern self, %self.param_patt.482 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.5ee: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.b79: %empty_struct_type.as.HasF.impl.F.type.5ee = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet.9f2: %HasF.type.daf = facet_value %empty_struct_type, (%HasF.impl_witness.932) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.dd7: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet.9f2) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.7b3: %HasF.WithSelf.F.type.dd7 = struct_value () [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.b91: type = facet_type <@HasF, @HasF(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.aa9: %HasF.type.b91 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.b94: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %Self.822) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.0d9: %HasF.WithSelf.F.type.b94 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.1d0: type = assoc_entity_type @HasF, @HasF(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.3f7: %HasF.assoc_type.1d0 = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness.7f1: <witness> = impl_witness @empty_struct_type.as.HasF.impl.%HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %HasF.type.b91 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type.6cf: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.0aa: %empty_struct_type.as.HasF.impl.F.type.6cf = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet.b45: %HasF.type.b91 = facet_value %empty_struct_type, (%HasF.impl_witness.7f1) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.1ff: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%empty_struct_type, %HasF.facet.b45) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.260: %HasF.WithSelf.F.type.1ff = struct_value () [concrete]
// CHECK:STDOUT:   %.abf: type = fn_type_with_self_type %HasF.WithSelf.F.type.1ff, %HasF.facet.b45 [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.specific_fn: <specific function> = specific_function %empty_struct_type.as.HasF.impl.F.0aa, @empty_struct_type.as.HasF.impl.F(%empty_struct_type) [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.c96 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt.loc4_17.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_25.2: type = converted %.loc8_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.c96 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_36.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.daf)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_17.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.822)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.822)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.41d) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.ff7)] {
// CHECK:STDOUT:       %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.d2c)]
// CHECK:STDOUT:       %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.763)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_8.1: type = splice_block %.loc5_8.3 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_8.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = name_ref Self, %.loc5_8.2 [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:         %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_8.3: type = converted %Self.ref, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_13.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.ddf) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_13.2 (constants.%assoc0.e9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = @HasF.WithSelf.%assoc0.loc5_13.1
// CHECK:STDOUT:     witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @empty_struct_type.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_36.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_36.1 (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.932)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_36.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type: type = fn_type @empty_struct_type.as.HasF.impl.F, @empty_struct_type.as.HasF.impl(%T.loc8_15.1) [symbolic = %empty_struct_type.as.HasF.impl.F.type (constants.%empty_struct_type.as.HasF.impl.F.type.5ee)]
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.5ee) = struct_value () [symbolic = %empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.b79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_36.2 {
// CHECK:STDOUT:     %empty_struct_type.as.HasF.impl.F.decl: @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F.type (%empty_struct_type.as.HasF.impl.F.type.5ee) = fn_decl @empty_struct_type.as.HasF.impl.F [symbolic = @empty_struct_type.as.HasF.impl.%empty_struct_type.as.HasF.impl.F (constants.%empty_struct_type.as.HasF.impl.F.b79)] {
// CHECK:STDOUT:       %self.param_patt: %pattern_type.a96 = value_param_pattern [concrete = constants.%self.param_patt.482]
// CHECK:STDOUT:       %self.patt: %pattern_type.a96 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b63]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @empty_struct_type.as.HasF.impl.%.loc8_25.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %self: %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %HasF.impl_witness_table = impl_witness_table (%empty_struct_type.as.HasF.impl.F.decl), @empty_struct_type.as.HasF.impl [concrete]
// CHECK:STDOUT:     %HasF.impl_witness.loc8_38.1: <witness> = impl_witness %HasF.impl_witness_table, @empty_struct_type.as.HasF.impl(constants.%T) [symbolic = %HasF.impl_witness.loc8_38.2 (constants.%HasF.impl_witness.932)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %empty_struct_type.as.HasF.impl.F.decl
// CHECK:STDOUT:     extend %HasF.type.loc8_36.2
// CHECK:STDOUT:     witness = %HasF.impl_witness.loc8_38.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.daf)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.a2b)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.d2c)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.763)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type));
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.c96 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %.loc13_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_13: type = converted %.loc13_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%empty_struct_type)> [concrete = constants.%HasF.type.b91]
// CHECK:STDOUT:   %.loc13_14: %HasF.assoc_type.1d0 = specific_constant @HasF.WithSelf.%assoc0.loc5_13.1, @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.822) [concrete = constants.%assoc0.3f7]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.1d0 = name_ref F, %.loc13_14 [concrete = constants.%assoc0.3f7]
// CHECK:STDOUT:   %impl.elem0: %.abf = impl_witness_access constants.%HasF.impl_witness.7f1, element0 [concrete = constants.%empty_struct_type.as.HasF.impl.F.0aa]
// CHECK:STDOUT:   %bound_method.loc13_4: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @empty_struct_type.as.HasF.impl.F(constants.%empty_struct_type) [concrete = constants.%empty_struct_type.as.HasF.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method.loc13_18(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.41d
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.ff7
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.ddf
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.e9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.822) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a2b
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.d2c
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.763
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.daf
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.932
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.5ee
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.b79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%HasF.facet.9f2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%HasF.facet.9f2
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.dd7
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.7b3
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.ddf
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.e9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet.9f2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%HasF.facet.9f2
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a96
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.482
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.b63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b91
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.aa9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b91
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.b94
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.0d9
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.1d0
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.3f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_struct_type.as.HasF.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type.loc8_36.1 => constants.%HasF.type.b91
// CHECK:STDOUT:   %HasF.impl_witness.loc8_38.2 => constants.%HasF.impl_witness.7f1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F.type => constants.%empty_struct_type.as.HasF.impl.F.type.6cf
// CHECK:STDOUT:   %empty_struct_type.as.HasF.impl.F => constants.%empty_struct_type.as.HasF.impl.F.0aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%empty_struct_type, constants.%HasF.facet.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b91
// CHECK:STDOUT:   %Self => constants.%HasF.facet.b45
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.1ff
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.260
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.1d0
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.3f7
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
// CHECK:STDOUT:   %pattern_type.c9f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.74d: %pattern_type.c9f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b33: %pattern_type.c9f = at_binding_pattern self, %self.param_patt.74d [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F: %HasF.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %self.param_patt.b7c: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.7e4: %pattern_type.51d = at_binding_pattern self, %self.param_patt.b7c [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
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
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc12_25.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_25.2 (constants.%U.patt)]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc16_10.1: type = splice_block %.loc16_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc16_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc16_10.3: type = converted %.loc16_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:     %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_8.1: type = splice_block %.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_8.2: type = converted %Self.ref, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc12_25.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_25.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc12_25.1: type = symbolic_binding U, 1 [symbolic = %U.loc12_25.1 (constants.%U)]
// CHECK:STDOUT:   %HasF.impl_witness.loc12_44.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc12_15.1, %U.loc12_25.1) [symbolic = %HasF.impl_witness.loc12_44.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc12_15.1, %U.loc12_25.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.param_patt.loc13_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc13_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:       %self.patt.loc13_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc13_15.1 [symbolic = %self.patt.loc13_15.2 (constants.%self.patt.7e4)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.c9f)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.74d)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.c9f) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.b33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc12_15.2: type, @T.as.HasF.impl.%U.loc12_25.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %self.param_patt.loc13_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc13_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:   %self.patt.loc13_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc13_15.2 [symbolic = %self.patt.loc13_15.2 (constants.%self.patt.7e4)]
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
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c9f
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.74d
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.b33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc12_25.2 => constants.%U.patt
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
// CHECK:STDOUT:   %self.param_patt.loc13_15.2 => constants.%self.param_patt.b7c
// CHECK:STDOUT:   %self.patt.loc13_15.2 => constants.%self.patt.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.c96: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.c96 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.daf: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.822: %HasF.type.daf = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.822 [symbolic]
// CHECK:STDOUT:   %pattern_type.a2b: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.d2c: %pattern_type.a2b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.763: %pattern_type.a2b = at_binding_pattern self, %self.param_patt.d2c [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.41d: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %Self.822) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.ff7: %HasF.WithSelf.F.type.41d = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.ddf: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.e9a: %HasF.assoc_type.ddf = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @T.as.HasF.impl.%HasF.impl_witness_table, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.513: <witness> = require_complete_type %HasF.type.daf [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %self.param_patt.b7c: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.7e4: %pattern_type.51d = at_binding_pattern self, %self.param_patt.b7c [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.HasF.impl.F: %T.as.HasF.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.daf = facet_value %T, (%HasF.impl_witness) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.f9f: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T, %HasF.facet) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.3d5: %HasF.WithSelf.F.type.f9f = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9ef = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.5a7: type = facet_type <@HasF, @HasF(%B)> [concrete]
// CHECK:STDOUT:   %Self.ad0: %HasF.type.5a7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.2a0: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%B, %Self.822) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.836: %HasF.WithSelf.F.type.2a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.43e: type = assoc_entity_type @HasF, @HasF(%B) [concrete]
// CHECK:STDOUT:   %assoc0.613: %HasF.assoc_type.43e = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.c96 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt.loc4_17.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.HasF.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_24: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.c96 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_35.2: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.daf)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.9ef = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %x: %A = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_17.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.822)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.822)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.41d) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.ff7)] {
// CHECK:STDOUT:       %self.param_patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.d2c)]
// CHECK:STDOUT:       %self.patt.loc5_8.1: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = at_binding_pattern self, %self.param_patt.loc5_8.1 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.763)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_8.1: type = splice_block %.loc5_8.3 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_8.2: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = specific_constant @HasF.%Self.loc4_26.1, @HasF(constants.%T) [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:         %Self.ref: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = name_ref Self, %.loc5_8.2 [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:         %Self.as_type.loc5_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_8.3: type = converted %Self.ref, %Self.as_type.loc5_8.2 [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_13.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.ddf) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_13.2 (constants.%assoc0.e9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .F = @HasF.WithSelf.%assoc0.loc5_13.1
// CHECK:STDOUT:     witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.HasF.impl(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   %HasF.type.loc8_35.1: type = facet_type <@HasF, @HasF(%T.loc8_15.1)> [symbolic = %HasF.type.loc8_35.1 (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2: <witness> = impl_witness %HasF.impl_witness_table, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %HasF.impl_witness.loc8_37.2 (constants.%HasF.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.loc8_35.1 [symbolic = %require_complete (constants.%require_complete.513)]
// CHECK:STDOUT:   %T.as.HasF.impl.F.type: type = fn_type @T.as.HasF.impl.F, @T.as.HasF.impl(%T.loc8_15.1) [symbolic = %T.as.HasF.impl.F.type (constants.%T.as.HasF.impl.F.type)]
// CHECK:STDOUT:   %T.as.HasF.impl.F: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = struct_value () [symbolic = %T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc8_24 as %HasF.type.loc8_35.2 {
// CHECK:STDOUT:     %T.as.HasF.impl.F.decl: @T.as.HasF.impl.%T.as.HasF.impl.F.type (%T.as.HasF.impl.F.type) = fn_decl @T.as.HasF.impl.F [symbolic = @T.as.HasF.impl.%T.as.HasF.impl.F (constants.%T.as.HasF.impl.F)] {
// CHECK:STDOUT:       %self.param_patt.loc9_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc9_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:       %self.patt.loc9_15.1: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc9_15.1 [symbolic = %self.patt.loc9_15.2 (constants.%self.patt.7e4)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T.as.HasF.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @T.as.HasF.impl.%T.ref.loc8_24 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @T.as.HasF.impl.F.%T (%T) = wrapper_binding self, %self.param
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
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.daf)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T)> [symbolic = %HasF.type (constants.%HasF.type.daf)]
// CHECK:STDOUT:   %Self: @HasF.WithSelf.F.%HasF.type (%HasF.type.daf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:   %Self.as_type.loc5_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_8.1 [symbolic = %pattern_type (constants.%pattern_type.a2b)]
// CHECK:STDOUT:   %self.param_patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = value_param_pattern [symbolic = %self.param_patt.loc5_8.2 (constants.%self.param_patt.d2c)]
// CHECK:STDOUT:   %self.patt.loc5_8.2: @HasF.WithSelf.F.%pattern_type (%pattern_type.a2b) = at_binding_pattern self, %self.param_patt.loc5_8.2 [symbolic = %self.patt.loc5_8.2 (constants.%self.patt.763)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.WithSelf.F.%Self.as_type.loc5_8.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.HasF.impl.F(@T.as.HasF.impl.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %self.param_patt.loc9_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %self.param_patt.loc9_15.2 (constants.%self.param_patt.b7c)]
// CHECK:STDOUT:   %self.patt.loc9_15.2: @T.as.HasF.impl.F.%pattern_type (%pattern_type.51d) = at_binding_pattern self, %self.param_patt.loc9_15.2 [symbolic = %self.patt.loc9_15.2 (constants.%self.patt.7e4)]
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.c96 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [concrete = constants.%HasF.type.5a7]
// CHECK:STDOUT:   %.loc22: %HasF.assoc_type.43e = specific_constant @HasF.WithSelf.%assoc0.loc5_13.1, @HasF.WithSelf(constants.%B, constants.%Self.822) [concrete = constants.%assoc0.613]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type.43e = name_ref F, %.loc22 [concrete = constants.%assoc0.613]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.41d
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.ff7
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.ddf
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.e9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%Self.822) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a2b
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.d2c
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.763
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_35.1 => constants.%HasF.type.daf
// CHECK:STDOUT:   %HasF.impl_witness.loc8_37.2 => constants.%HasF.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.513
// CHECK:STDOUT:   %T.as.HasF.impl.F.type => constants.%T.as.HasF.impl.F.type
// CHECK:STDOUT:   %T.as.HasF.impl.F => constants.%T.as.HasF.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.HasF.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %self.param_patt.loc9_15.2 => constants.%self.param_patt.b7c
// CHECK:STDOUT:   %self.patt.loc9_15.2 => constants.%self.patt.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.f9f
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.3d5
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.ddf
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.e9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.daf
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_8.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %self.param_patt.loc5_8.2 => constants.%self.param_patt.b7c
// CHECK:STDOUT:   %self.patt.loc5_8.2 => constants.%self.patt.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.5a7
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.ad0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%B, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.5a7
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.2a0
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.836
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.43e
// CHECK:STDOUT:   %assoc0.loc5_13.2 => constants.%assoc0.613
// CHECK:STDOUT: }
// CHECK:STDOUT:
