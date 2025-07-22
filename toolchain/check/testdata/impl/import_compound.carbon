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
  fn G[self: Self]() {}
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
  // CHECK:STDERR: fail_import_non-instance.carbon:[[@LINE+4]]:3: note: type `{.i: ()}` does not implement interface `Core.ImplicitAs(NonInstance)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_import_non-instance_indirect.carbon:[[@LINE+4]]:3: note: type `{.i: ()}` does not implement interface `Core.ImplicitAs(NonInstance)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.F.type: type = fn_type @NonInstance.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance.F: %NonInstance.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.525: %NonInstance.assoc_type = assoc_entity element0, @NonInstance.%NonInstance.F.decl [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness file.%NonInstance.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.type: type = fn_type @struct_type.i.as.NonInstance.impl.F [concrete]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F: %struct_type.i.as.NonInstance.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%NonInstance.impl_witness) [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.cf8: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %pattern_type.4e4: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Instance.G.type: type = fn_type @Instance.G [concrete]
// CHECK:STDOUT:   %Instance.G: %Instance.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0.ab8: %Instance.assoc_type = assoc_entity element0, @Instance.%Instance.G.decl [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness file.%Instance.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.type: type = fn_type @struct_type.i.as.Instance.impl.G [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G: %struct_type.i.as.Instance.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%Instance.impl_witness) [concrete]
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
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %NonInstance.ref: type = name_ref NonInstance, file.%NonInstance.decl [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance.impl_witness_table = impl_witness_table (@struct_type.i.as.NonInstance.impl.%struct_type.i.as.NonInstance.impl.F.decl), @struct_type.i.as.NonInstance.impl [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness %NonInstance.impl_witness_table [concrete = constants.%NonInstance.impl_witness]
// CHECK:STDOUT:   %Instance.decl: type = interface_decl @Instance [concrete = constants.%Instance.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.i.as.Instance.impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %Instance.ref: type = name_ref Instance, file.%Instance.decl [concrete = constants.%Instance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance.impl_witness_table = impl_witness_table (@struct_type.i.as.Instance.impl.%struct_type.i.as.Instance.impl.G.decl), @struct_type.i.as.Instance.impl [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness %Instance.impl_witness_table [concrete = constants.%Instance.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance {
// CHECK:STDOUT:   %Self: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %NonInstance.F.decl: %NonInstance.F.type = fn_decl @NonInstance.F [concrete = constants.%NonInstance.F] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, %NonInstance.F.decl [concrete = constants.%assoc0.525]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%NonInstance.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %Instance.G.decl: %Instance.G.type = fn_decl @Instance.G [concrete = constants.%Instance.G] {
// CHECK:STDOUT:     %self.patt: @Instance.G.%pattern_type (%pattern_type.4e4) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Instance.G.%pattern_type (%pattern_type.4e4) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance.G.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance.type = name_ref Self, @Instance.%Self [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance.G.%Self.as_type.loc12_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, %Instance.G.decl [concrete = constants.%assoc0.ab8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%Instance.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: %struct_type.i as %NonInstance.ref {
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.decl: %struct_type.i.as.NonInstance.impl.F.type = fn_decl @struct_type.i.as.NonInstance.impl.F [concrete = constants.%struct_type.i.as.NonInstance.impl.F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %struct_type.i.as.NonInstance.impl.F.decl
// CHECK:STDOUT:   witness = file.%NonInstance.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: %struct_type.i as %Instance.ref {
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.decl: %struct_type.i.as.Instance.impl.G.type = fn_decl @struct_type.i.as.Instance.impl.G [concrete = constants.%struct_type.i.as.Instance.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.9b2 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9b2 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.i.as.Instance.impl.%struct_type.i [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %self: %struct_type.i = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %struct_type.i.as.Instance.impl.G.decl
// CHECK:STDOUT:   witness = file.%Instance.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.F(@NonInstance.%Self: %NonInstance.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.NonInstance.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.G(@Instance.%Self: %Instance.type) {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc12_14.1 [symbolic = %pattern_type (constants.%pattern_type.4e4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance.G.%Self.as_type.loc12_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.Instance.impl.G(%self.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.F(constants.%NonInstance.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.G(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.G(constants.%Instance.facet) {
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
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %NonInstance.F.type: type = fn_type @NonInstance.F [concrete]
// CHECK:STDOUT:   %NonInstance.F: %NonInstance.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %NonInstance.impl_witness: <witness> = impl_witness imports.%NonInstance.impl_witness_table [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%NonInstance.impl_witness) [concrete]
// CHECK:STDOUT:   %.923: type = fn_type_with_self_type %NonInstance.F.type, %NonInstance.facet [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e3c: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.474: %NonInstance.F.type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%NonInstance.F]
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst19 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Main.import_ref.a26: <witness> = import_ref Main//lib, loc7_30, loaded [concrete = constants.%NonInstance.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst45 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f09 = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.a9f = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.2d7: %struct_type.i.as.NonInstance.impl.F.type = import_ref Main//lib, loc8_10, loaded [concrete = constants.%struct_type.i.as.NonInstance.impl.F]
// CHECK:STDOUT:   %NonInstance.impl_witness_table = impl_witness_table (%Main.import_ref.2d7), @struct_type.i.as.NonInstance.impl [concrete]
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e3c
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.f09
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a3b6.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a3b6.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImport() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.e3c [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value constants.%struct_type.i, (constants.%NonInstance.impl_witness) [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %.loc6_11: %NonInstance.type = converted %struct_type.i, %NonInstance.facet [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %impl.elem0: %.923 = impl_witness_access constants.%NonInstance.impl_witness, element0 [concrete = constants.%struct_type.i.as.NonInstance.impl.F]
// CHECK:STDOUT:   %struct_type.i.as.NonInstance.impl.F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.F(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.i.as.NonInstance.impl.F [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail.type: type = fn_type @NonInstanceCallImportFail [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail: %NonInstanceCallImportFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.8ce: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %NonInstance.F.type: type = fn_type @NonInstance.F [concrete]
// CHECK:STDOUT:   %NonInstance.F: %NonInstance.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e3c: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.8ce]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.474: %NonInstance.F.type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%NonInstance.F]
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst19 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Main.import_ref.eb8 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst45 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f09 = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.a9f = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.9b2 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.9b2 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %.loc5_40: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc5_39.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_39.2: type = converted %.loc5_39.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e3c
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.f09
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a3b6.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.eb8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a3b6.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImportFail(%n.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.e3c [concrete = constants.%assoc0.8ce]
// CHECK:STDOUT:   %.loc13: %NonInstance.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.F(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %pattern_type.72c: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport.type: type = fn_type @NonInstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport: %NonInstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type @NonInstance [concrete]
// CHECK:STDOUT:   %assoc0.8ce: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %NonInstance.F.type: type = fn_type @NonInstance.F [concrete]
// CHECK:STDOUT:   %NonInstance.F: %NonInstance.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e3c: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.8ce]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.474: %NonInstance.F.type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%NonInstance.F]
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst19 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Main.import_ref.eb8 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst45 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f09 = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.a9f = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.72c = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.72c = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc5_45: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc5_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_43.2: type = converted %.loc5_43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.i [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e3c
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.f09
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a3b6.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.eb8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a3b6.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirectImport(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.e3c [concrete = constants.%assoc0.8ce]
// CHECK:STDOUT:   %.loc13_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %.loc13_4.2: %NonInstance.type = converted %.loc13_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance.F(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance.F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.9b2: type = pattern_type %struct_type.i [concrete]
// CHECK:STDOUT:   %InstanceCallImport.type: type = fn_type @InstanceCallImport [concrete]
// CHECK:STDOUT:   %InstanceCallImport: %InstanceCallImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.cf8: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b4d [concrete]
// CHECK:STDOUT:   %Instance.G.type: type = fn_type @Instance.G [concrete]
// CHECK:STDOUT:   %Instance.G: %Instance.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %pattern_type.4e4: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Instance.impl_witness: <witness> = impl_witness imports.%Instance.impl_witness_table [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%Instance.impl_witness) [concrete]
// CHECK:STDOUT:   %.efa: type = fn_type_with_self_type %Instance.G.type, %Instance.facet [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.type: type = fn_type @struct_type.i.as.Instance.impl.G [concrete]
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G: %struct_type.i.as.Instance.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %pattern_type.72c: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport.type: type = fn_type @InstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport: %InstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst45 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.124: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.b4d: %Instance.G.type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%Instance.G]
// CHECK:STDOUT:   %Main.import_ref.0d8: %Instance.type = import_ref Main//lib, inst45 [no loc], loaded [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.49c = import_ref Main//lib, loc4_9, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.eb8 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.97f: <witness> = import_ref Main//lib, loc15_27, loaded [concrete = constants.%Instance.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.373: %struct_type.i.as.Instance.impl.G.type = import_ref Main//lib, loc16_22, loaded [concrete = constants.%struct_type.i.as.Instance.impl.G]
// CHECK:STDOUT:   %Instance.impl_witness_table = impl_witness_table (%Main.import_ref.373), @struct_type.i.as.Instance.impl [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.9b2 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.9b2 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %.loc5_33: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc5_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_32.2: type = converted %.loc5_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirectImport.decl: %InstanceCallIndirectImport.type = fn_decl @InstanceCallIndirectImport [concrete = constants.%InstanceCallIndirectImport] {
// CHECK:STDOUT:     %p.patt: %pattern_type.72c = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.72c = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc9_42: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc9_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_40.2: type = converted %.loc9_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.i [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.124
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.49c
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a3b6.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.eb8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a3b6.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.97f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImport(%n.param: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.124 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.efa = impl_witness_access constants.%Instance.impl_witness, element0 [concrete = constants.%struct_type.i.as.Instance.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %struct_type.i.as.Instance.impl.G.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.G(imports.%Main.import_ref.0d8: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.4e4)]
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
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.124 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc10_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.efa = impl_witness_access constants.%Instance.impl_witness, element0 [concrete = constants.%struct_type.i.as.Instance.impl.G]
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
// CHECK:STDOUT: specific @Instance.G(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InstanceCallImportFail.type: type = fn_type @InstanceCallImportFail [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InstanceCallImportFail: %InstanceCallImportFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.cf8: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type @Instance [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b4d [concrete]
// CHECK:STDOUT:   %Instance.G.type: type = fn_type @Instance.G [concrete]
// CHECK:STDOUT:   %Instance.G: %Instance.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance = import_ref Main//lib, NonInstance, unloaded
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst45 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.124: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.b4d: %Instance.G.type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%Instance.G]
// CHECK:STDOUT:   %Main.import_ref.0d8: %Instance.type = import_ref Main//lib, inst45 [no loc], loaded [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.49c = import_ref Main//lib, loc4_9, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.eb8 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.a9f = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.c9a3b6.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.124
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.49c
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.NonInstance.impl: imports.%Main.import_ref.c9a3b6.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.eb8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.i.as.Instance.impl: imports.%Main.import_ref.c9a3b6.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImportFail() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.124 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance.G(imports.%Main.import_ref.0d8: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance.G(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
