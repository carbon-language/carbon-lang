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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_compound.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

interface NonInstance {
  fn F();
}

impl {.i: ()} as NonInstance {
  fn F() {}
}

interface Instance {
  fn G[self: Self]();
}

impl {.i: ()} as Instance {
  fn G[unused self: Self]() {}
}

// --- import_non-instance_success.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn NonInstanceCallImport() {
  {.i: ()}.(NonInstance.F)();
}

// --- fail_import_non-instance.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn NonInstanceCallImportFail(n: {.i: ()}) {
  // CHECK:STDERR: fail_import_non-instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `{.i: ()}` into type implementing `NonInstance` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   n.(NonInstance.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non-instance.carbon:[[@LINE+4]]:3: note: type `{.i: ()}` does not implement interface `Core.ImplicitAs(NonInstance)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   n.(NonInstance.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  n.(NonInstance.F)();
}

// --- fail_import_non-instance_indirect.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn NonInstanceCallIndirectImport(p: {.i: ()}*) {
  // CHECK:STDERR: fail_import_non-instance_indirect.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `{.i: ()}` into type implementing `NonInstance` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   p->(NonInstance.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non-instance_indirect.carbon:[[@LINE+4]]:3: note: type `{.i: ()}` does not implement interface `Core.ImplicitAs(NonInstance)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   p->(NonInstance.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(NonInstance.F)();
}

// --- import_instance_success.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn InstanceCallImport(n: {.i: ()}) {
  n.(Instance.G)();
}

fn InstanceCallIndirectImport(p: {.i: ()}*) {
  p->(Instance.G)();
}

// --- fail_import_instance.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn InstanceCallImportFail() {
  // CHECK:STDERR: fail_import_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Instance` in type `type` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   {.i: ()}.(Instance.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.i: ()}.(Instance.G)();
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.a14: %NonInstance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type.0be: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%Self.a14) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.0dc: %NonInstance.WithSelf.F.type.0be = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.e8a: %NonInstance.assoc_type = assoc_entity element0, @NonInstance.WithSelf.%NonInstance.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness @struct_type.i.as.NonInstance.impl.%NonInstance.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.type: type = fn_type @struct_type.i.as.NonInstance.impl.F [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F: %struct_type.i.as.NonInstance.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%NonInstance.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type.0ea: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%NonInstance.facet) [concrete]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.c3f: %NonInstance.WithSelf.F.type.0ea = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.bb5: %Instance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.bb5 [symbolic]
// CHECK:STDOUT:   %pattern_type.362: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G.type.f64: type = fn_type @Instance.WithSelf.G, @Instance.WithSelf(%Self.bb5) [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G.395: %Instance.WithSelf.G.type.f64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0.94f: %Instance.assoc_type = assoc_entity element0, @Instance.WithSelf.%Instance.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness @struct_type.i.as.Instance.impl.%Instance.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.type: type = fn_type @struct_type.i.as.Instance.impl.G [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G: %struct_type.i.as.Instance.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%Instance.impl_witness) [concrete]
// CHECK:STDOUT:   %Instance.WithSelf.G.type.9b8: type = fn_type @Instance.WithSelf.G, @Instance.WithSelf(%Instance.facet) [concrete]
// CHECK:STDOUT:   %Instance.WithSelf.G.c6f: %Instance.WithSelf.G.type.9b8 = struct_value () [concrete]
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
// CHECK:STDOUT:     .NonInstance = %NonInstance.decl
// CHECK:STDOUT:     .Instance = %Instance.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonInstance.decl: type = interface_decl @NonInstance [concrete = constants.%NonInstance.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.i.as.NonInstance.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %NonInstance.ref: type = name_ref NonInstance, file.%NonInstance.decl [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance.decl: type = interface_decl @Instance [concrete = constants.%Instance.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.i.as.Instance.impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %Instance.ref: type = name_ref Instance, file.%Instance.decl [concrete = constants.%Instance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance {
// CHECK:STDOUT:   %Self: %NonInstance.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a14]
// CHECK:STDOUT:   %NonInstance.WithSelf.decl = interface_with_self_decl @NonInstance [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %NonInstance.WithSelf.F.decl: @NonInstance.WithSelf.%NonInstance.WithSelf.F.type (%NonInstance.WithSelf.F.type.0be) = fn_decl @NonInstance.WithSelf.F [symbolic = @NonInstance.WithSelf.%NonInstance.WithSelf.F (constants.%NonInstance.WithSelf.F.0dc)] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, %NonInstance.WithSelf.F.decl [concrete = constants.%assoc0.e8a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @NonInstance.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@NonInstance.WithSelf.%NonInstance.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance {
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bb5]
// CHECK:STDOUT:   %Instance.WithSelf.decl = interface_with_self_decl @Instance [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Instance.WithSelf.G.decl: @Instance.WithSelf.%Instance.WithSelf.G.type (%Instance.WithSelf.G.type.f64) = fn_decl @Instance.WithSelf.G [symbolic = @Instance.WithSelf.%Instance.WithSelf.G (constants.%Instance.WithSelf.G.395)] {
// CHECK:STDOUT:     %self.param_patt: @Instance.WithSelf.G.%pattern_type (%pattern_type.362) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Instance.WithSelf.G.%pattern_type (%pattern_type.362) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance.WithSelf.G.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance.type = name_ref Self, @Instance.%Self [symbolic = %Self (constants.%Self.bb5)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance.WithSelf.G.%Self.as_type.loc12_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, %Instance.WithSelf.G.decl [concrete = constants.%assoc0.94f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @Instance.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Instance.WithSelf.%Instance.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: %struct_type.i as %NonInstance.ref {
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.decl: %struct_type.i.as.NonInstance.impl.F.type = fn_decl @struct_type.i.as.NonInstance.impl.F [concrete = constants.%struct_type.i.as.NonInstance.impl.F] {} {}
// CHECK:STDOUT:   %NonInstance.impl_witness_table = impl_witness_table (%struct_type.i.as.NonInstance.impl.F.decl), @struct_type.i.as.NonInstance.impl [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness %NonInstance.impl_witness_table [concrete = constants.%NonInstance.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %struct_type.i.as.NonInstance.impl.F.decl
// CHECK:STDOUT:   extend %NonInstance.ref
// CHECK:STDOUT:   witness = %NonInstance.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: %struct_type.i as %Instance.ref {
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.decl: %struct_type.i.as.Instance.impl.G.type = fn_decl @struct_type.i.as.Instance.impl.G [concrete = constants.%struct_type.i.as.Instance.impl.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9b2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9b2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.i.as.Instance.impl.%struct_type.i [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %self: %struct_type.i = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance.impl_witness_table = impl_witness_table (%struct_type.i.as.Instance.impl.G.decl), @struct_type.i.as.Instance.impl [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness %Instance.impl_witness_table [concrete = constants.%Instance.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %struct_type.i.as.Instance.impl.G.decl
// CHECK:STDOUT:   extend %Instance.ref
// CHECK:STDOUT:   witness = %Instance.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.WithSelf.F(@NonInstance.%Self: %NonInstance.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.NonInstance.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.WithSelf.G(@Instance.%Self: %Instance.type) {
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bb5)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc12_14.1 [symbolic = %pattern_type (constants.%pattern_type.362)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance.WithSelf.G.%Self.as_type.loc12_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.Instance.impl.G(%self.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%Self.a14) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.a14
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type.0be
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F.0dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf.F(constants.%Self.a14) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%NonInstance.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%NonInstance.facet
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type.0ea
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F.c3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf.F(constants.%NonInstance.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf(constants.%Self.bb5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.bb5
// CHECK:STDOUT:   %Instance.WithSelf.G.type => constants.%Instance.WithSelf.G.type.f64
// CHECK:STDOUT:   %Instance.WithSelf.G => constants.%Instance.WithSelf.G.395
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf.G(constants.%Self.bb5) {
// CHECK:STDOUT:   %Self => constants.%Self.bb5
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.362
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf(constants.%Instance.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Instance.facet
// CHECK:STDOUT:   %Instance.WithSelf.G.type => constants.%Instance.WithSelf.G.type.9b8
// CHECK:STDOUT:   %Instance.WithSelf.G => constants.%Instance.WithSelf.G.c6f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf.G(constants.%Instance.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance.facet
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%struct_type.i
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non-instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstanceCallImport.type: type = fn_type @NonInstanceCallImport [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstanceCallImport: %NonInstanceCallImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self: %NonInstance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type.df3: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.62f: %NonInstance.WithSelf.F.type.df3 = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.255 [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness imports.%NonInstance.impl_witness_table [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%NonInstance.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type.f88: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%NonInstance.facet) [concrete]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.b35: %NonInstance.WithSelf.F.type.f88 = struct_value () [concrete]
// CHECK:STDOUT:   %.a34: type = fn_type_with_self_type %NonInstance.WithSelf.F.type.f88, %NonInstance.facet [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.type: type = fn_type @struct_type.i.as.NonInstance.impl.F [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F: %struct_type.i.as.NonInstance.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.a5f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.9ad11e.2: %NonInstance.type = import_ref Main//lib, loc3_23, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.61e = import_ref Main//lib, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.255: @NonInstance.WithSelf.%NonInstance.WithSelf.F.type (%NonInstance.WithSelf.F.type.df3) = import_ref Main//lib, loc4_9, loaded [symbolic = @NonInstance.WithSelf.%NonInstance.WithSelf.F (constants.%NonInstance.WithSelf.F.62f)]
// CHECK:STDOUT:   %Main.import_ref.80b: <witness> = import_ref Main//lib, loc7_30, loaded [concrete = constants.%NonInstance.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.c9a: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.282: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.0d0: %struct_type.i.as.NonInstance.impl.F.type = import_ref Main//lib, loc8_10, loaded [concrete = constants.%struct_type.i.as.NonInstance.impl.F]
// CHECK:STDOUT:   %NonInstance.impl_witness_table = impl_witness_table (%Main.import_ref.0d0), @struct_type.i.as.NonInstance.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonInstanceCallImport = %NonInstanceCallImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NonInstanceCallImport.decl: %NonInstanceCallImport.type = fn_decl @NonInstanceCallImport [concrete = constants.%NonInstanceCallImport] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.61e
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a5f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a as imports.%Main.import_ref.282 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.80b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImport() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.a5f [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (constants.%NonInstance.impl_witness) [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %.loc6_11: %NonInstance.type = converted %struct_type.i, %NonInstance.facet [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %impl.elem0: %.a34 = impl_witness_access constants.%NonInstance.impl_witness, element0 [concrete = constants.%struct_type.i.as.NonInstance.impl.F]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.WithSelf.F(imports.%Main.import_ref.9ad11e.2: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.NonInstance.impl.F [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type.df3
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F.62f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%NonInstance.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%NonInstance.facet
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type.f88
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F.b35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail.type: type = fn_type @NonInstanceCallImportFail [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail: %NonInstanceCallImportFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.002: %NonInstance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%Self.002) [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F: %NonInstance.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.8c6: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.255 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.a5f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.8c6]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.9ad11e.2: %NonInstance.type = import_ref Main//lib, loc3_23, loaded [symbolic = constants.%Self.002]
// CHECK:STDOUT:   %Main.import_ref.61e = import_ref Main//lib, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.255: @NonInstance.WithSelf.%NonInstance.WithSelf.F.type (%NonInstance.WithSelf.F.type) = import_ref Main//lib, loc4_9, loaded [symbolic = @NonInstance.WithSelf.%NonInstance.WithSelf.F (constants.%NonInstance.WithSelf.F)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonInstanceCallImportFail = %NonInstanceCallImportFail.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NonInstanceCallImportFail.decl: %NonInstanceCallImportFail.type = fn_decl @NonInstanceCallImportFail [concrete = constants.%NonInstanceCallImportFail] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.9b2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.9b2 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %.loc5_40: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc5_39.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc5_39.2: type = converted %.loc5_39.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.61e
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a5f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImportFail(%n.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.a5f [concrete = constants.%assoc0.8c6]
// CHECK:STDOUT:   %.loc13: %NonInstance.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.WithSelf.F(imports.%Main.import_ref.9ad11e.2: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%Self.002) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.002
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf.F(constants.%Self.002) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %pattern_type.72c: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport.type: type = fn_type @NonInstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport: %NonInstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.002: %NonInstance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type: type = fn_type @NonInstance.WithSelf.F, @NonInstance.WithSelf(%Self.002) [symbolic]
// CHECK:STDOUT:   %NonInstance.WithSelf.F: %NonInstance.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.8c6: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.255 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.a5f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.8c6]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.9ad11e.2: %NonInstance.type = import_ref Main//lib, loc3_23, loaded [symbolic = constants.%Self.002]
// CHECK:STDOUT:   %Main.import_ref.61e = import_ref Main//lib, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.255: @NonInstance.WithSelf.%NonInstance.WithSelf.F.type (%NonInstance.WithSelf.F.type) = import_ref Main//lib, loc4_9, loaded [symbolic = @NonInstance.WithSelf.%NonInstance.WithSelf.F (constants.%NonInstance.WithSelf.F)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonInstanceCallIndirectImport = %NonInstanceCallIndirectImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NonInstanceCallIndirectImport.decl: %NonInstanceCallIndirectImport.type = fn_decl @NonInstanceCallIndirectImport [concrete = constants.%NonInstanceCallIndirectImport] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.72c = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.72c = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc5_45: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc5_43.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc5_43.2: type = converted %.loc5_43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.i [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.61e
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a5f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirectImport(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.a5f [concrete = constants.%assoc0.8c6]
// CHECK:STDOUT:   %.loc13_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %.loc13_4.2: %NonInstance.type = converted %.loc13_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.WithSelf.F(imports.%Main.import_ref.9ad11e.2: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf(constants.%Self.002) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.002
// CHECK:STDOUT:   %NonInstance.WithSelf.F.type => constants.%NonInstance.WithSelf.F.type
// CHECK:STDOUT:   %NonInstance.WithSelf.F => constants.%NonInstance.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.WithSelf.F(constants.%Self.002) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %InstanceCallImport.type: type = fn_type @InstanceCallImport [concrete]
// CHECK:STDOUT:   %InstanceCallImport: %InstanceCallImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G.type.8ba: type = fn_type @Instance.WithSelf.G, @Instance.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G.114: %Instance.WithSelf.G.type.8ba = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.77c: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b3c [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness imports.%Instance.impl_witness_table [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%Instance.impl_witness) [concrete]
// CHECK:STDOUT:   %Instance.WithSelf.G.type.551: type = fn_type @Instance.WithSelf.G, @Instance.WithSelf(%Instance.facet) [concrete]
// CHECK:STDOUT:   %Instance.WithSelf.G.fe9: %Instance.WithSelf.G.type.551 = struct_value () [concrete]
// CHECK:STDOUT:   %.2e3: type = fn_type_with_self_type %Instance.WithSelf.G.type.551, %Instance.facet [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.type: type = fn_type @struct_type.i.as.Instance.impl.G [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G: %struct_type.i.as.Instance.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %pattern_type.72c: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport.type: type = fn_type @InstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport: %InstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.i = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance = import_ref Main//lib, NonInstance, unloaded
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.756: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.a85b8b.2: %Instance.type = import_ref Main//lib, loc11_20, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.2cd = import_ref Main//lib, loc11_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3c: @Instance.WithSelf.%Instance.WithSelf.G.type (%Instance.WithSelf.G.type.8ba) = import_ref Main//lib, loc12_21, loaded [symbolic = @Instance.WithSelf.%Instance.WithSelf.G (constants.%Instance.WithSelf.G.114)]
// CHECK:STDOUT:   %Main.import_ref.b09: <witness> = import_ref Main//lib, loc15_27, loaded [concrete = constants.%Instance.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.c9a: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.fd3: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.7e2: %struct_type.i.as.Instance.impl.G.type = import_ref Main//lib, loc16_29, loaded [concrete = constants.%struct_type.i.as.Instance.impl.G]
// CHECK:STDOUT:   %Instance.impl_witness_table = impl_witness_table (%Main.import_ref.7e2), @struct_type.i.as.Instance.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .InstanceCallImport = %InstanceCallImport.decl
// CHECK:STDOUT:     .InstanceCallIndirectImport = %InstanceCallIndirectImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InstanceCallImport.decl: %InstanceCallImport.type = fn_decl @InstanceCallImport [concrete = constants.%InstanceCallImport] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.9b2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.9b2 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %.loc5_33: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc5_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc5_32.2: type = converted %.loc5_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirectImport.decl: %InstanceCallIndirectImport.type = fn_decl @InstanceCallIndirectImport [concrete = constants.%InstanceCallIndirectImport] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.72c = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.72c = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc9_42: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc9_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_40.2: type = converted %.loc9_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.i [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2cd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.756
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a as imports.%Main.import_ref.fd3 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.b09
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImport(%n.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.756 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.2e3 = impl_witness_access constants.%Instance.impl_witness, element0 [concrete = constants.%struct_type.i.as.Instance.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.WithSelf.G(imports.%Main.import_ref.a85b8b.2: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.77c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.Instance.impl.G [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirectImport(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.756 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc10_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.2e3 = impl_witness_access constants.%Instance.impl_witness, element0 [concrete = constants.%struct_type.i.as.Instance.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_4.1, %impl.elem0
// CHECK:STDOUT:   %.loc10_4.2: ref %empty_tuple.type = struct_access %.loc10_4.1, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_4.3: %empty_tuple.type = converted %.loc10_4.2, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.i = struct_value (%.loc10_4.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_4.4: %struct_type.i = converted %.loc10_4.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.call: init %empty_tuple.type = call %bound_method(%.loc10_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Instance.WithSelf.G.type => constants.%Instance.WithSelf.G.type.8ba
// CHECK:STDOUT:   %Instance.WithSelf.G => constants.%Instance.WithSelf.G.114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.77c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf(constants.%Instance.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Instance.facet
// CHECK:STDOUT:   %Instance.WithSelf.G.type => constants.%Instance.WithSelf.G.type.551
// CHECK:STDOUT:   %Instance.WithSelf.G => constants.%Instance.WithSelf.G.fe9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InstanceCallImportFail.type: type = fn_type @InstanceCallImportFail [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InstanceCallImportFail: %InstanceCallImportFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G.type: type = fn_type @Instance.WithSelf.G, @Instance.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Instance.WithSelf.G: %Instance.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b3c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance = import_ref Main//lib, NonInstance, unloaded
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.756: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.a85b8b.2: %Instance.type = import_ref Main//lib, loc11_20, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.2cd = import_ref Main//lib, loc11_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3c: @Instance.WithSelf.%Instance.WithSelf.G.type (%Instance.WithSelf.G.type) = import_ref Main//lib, loc12_21, loaded [symbolic = @Instance.WithSelf.%Instance.WithSelf.G (constants.%Instance.WithSelf.G)]
// CHECK:STDOUT:   %Main.import_ref.e22 = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.fd3: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .InstanceCallImportFail = %InstanceCallImportFail.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InstanceCallImportFail.decl: %InstanceCallImportFail.type = fn_decl @InstanceCallImportFail [concrete = constants.%InstanceCallImportFail] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2cd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.756
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a as imports.%Main.import_ref.fd3 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.e22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImportFail() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.756 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.WithSelf.G(imports.%Main.import_ref.a85b8b.2: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Instance.WithSelf.G.type => constants.%Instance.WithSelf.G.type
// CHECK:STDOUT:   %Instance.WithSelf.G => constants.%Instance.WithSelf.G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
