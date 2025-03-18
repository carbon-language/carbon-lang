// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_compound.carbon

// --- core.carbon
package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

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
import Core;

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
import Core;

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
import Core;

fn InstanceCallImportFail() {
  // CHECK:STDERR: fail_import_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Instance` in type `type` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   {.i: ()}.(Instance.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.i: ()}.(Instance.G)();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc3_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.1: type) {
// CHECK:STDOUT:   %Dest.loc3_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc3_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest.loc3_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%Dest.loc3_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc3_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc3_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.9e5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.4e4: %F.type.9e5 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type %NonInstance.type [concrete]
// CHECK:STDOUT:   %assoc0.b89: %NonInstance.assoc_type = assoc_entity element0, @NonInstance.%F.decl [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness.e97: <witness> = impl_witness (@impl.c58.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.042: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.db4: %F.type.042 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%impl_witness.e97) [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.cf8: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %G.type.1e1: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.449: %G.type.1e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type %Instance.type [concrete]
// CHECK:STDOUT:   %assoc0.b0a: %Instance.assoc_type = assoc_entity element0, @Instance.%G.decl [concrete]
// CHECK:STDOUT:   %impl_witness.114: <witness> = impl_witness (@impl.477.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.0c0: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.484: %G.type.0c0 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%impl_witness.114) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = %NonInstance.decl
// CHECK:STDOUT:     .Instance = %Instance.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance.decl: type = interface_decl @NonInstance [concrete = constants.%NonInstance.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.c58 [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %NonInstance.ref: type = name_ref NonInstance, file.%NonInstance.decl [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc7: <witness> = impl_witness (@impl.c58.%F.decl) [concrete = constants.%impl_witness.e97]
// CHECK:STDOUT:   %Instance.decl: type = interface_decl @Instance [concrete = constants.%Instance.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.477 [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %Instance.ref: type = name_ref Instance, file.%Instance.decl [concrete = constants.%Instance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.477.%G.decl) [concrete = constants.%impl_witness.114]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance {
// CHECK:STDOUT:   %Self: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %F.decl: %F.type.9e5 = fn_decl @F.1 [concrete = constants.%F.4e4] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.b89]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %G.decl: %G.type.1e1 = fn_decl @G.1 [concrete = constants.%G.449] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance.type = name_ref Self, @Instance.%Self [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.b0a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: %struct_type.i as %NonInstance.ref {
// CHECK:STDOUT:   %F.decl: %F.type.042 = fn_decl @F.2 [concrete = constants.%F.db4] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: %struct_type.i as %Instance.ref {
// CHECK:STDOUT:   %G.decl: %G.type.0c0 = fn_decl @G.2 [concrete = constants.%G.484] {
// CHECK:STDOUT:     %self.patt: %struct_type.i = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %struct_type.i = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.477.%struct_type.i [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %self: %struct_type.i = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@NonInstance.%Self: %NonInstance.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@Instance.%Self: %Instance.type) {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %struct_type.i]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%NonInstance.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Instance.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance.facet
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%struct_type.i
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
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type %NonInstance.type [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %F.type.9e5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.2d7) [concrete]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value %struct_type.i, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.1bc: type = fn_type_with_self_type %F.type.9e5, %NonInstance.facet [concrete]
// CHECK:STDOUT:   %F.type.042: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.db4: %F.type.042 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.69f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst15 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Main.import_ref.826: <witness> = import_ref Main//lib, loc7_30, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c1d = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.ab2 = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .NonInstanceCallImport = %NonInstanceCallImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NonInstanceCallImport.decl: %NonInstanceCallImport.type = fn_decl @NonInstanceCallImport [concrete = constants.%NonInstanceCallImport] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.69f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.c1d
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.826
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ab2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImport() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.69f [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance.facet: %NonInstance.type = facet_value constants.%struct_type.i, (constants.%impl_witness) [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %.loc6_11: %NonInstance.type = converted %struct_type.i, %NonInstance.facet [concrete = constants.%NonInstance.facet]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc6_11, element0 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.1bc = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%F.db4]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail.type: type = fn_type @NonInstanceCallImportFail [concrete]
// CHECK:STDOUT:   %NonInstanceCallImportFail: %NonInstanceCallImportFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type %NonInstance.type [concrete]
// CHECK:STDOUT:   %assoc0.b0b: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.a4a: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.0a7: type = fn_type @Convert, @ImplicitAs(%NonInstance.type) [concrete]
// CHECK:STDOUT:   %Convert.c2d: %Convert.type.0a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ffd: type = assoc_entity_type %ImplicitAs.type.a4a [concrete]
// CHECK:STDOUT:   %assoc0.46f: %ImplicitAs.assoc_type.ffd = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.69f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.b0b]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst15 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Main.import_ref.112 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c1d = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.ab2 = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonInstanceCallImportFail.decl: %NonInstanceCallImportFail.type = fn_decl @NonInstanceCallImportFail [concrete = constants.%NonInstanceCallImportFail] {
// CHECK:STDOUT:     %n.patt: %struct_type.i = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.i = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param call_param0
// CHECK:STDOUT:     %.loc6_40: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc6_39.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_39.2: type = converted %.loc6_39.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.69f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.c1d
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ab2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallImportFail(%n.param_patt: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.69f [concrete = constants.%assoc0.b0b]
// CHECK:STDOUT:   %.loc14: %NonInstance.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance.type
// CHECK:STDOUT:   %Dest.patt => constants.%NonInstance.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a4a
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.0a7
// CHECK:STDOUT:   %Convert => constants.%Convert.c2d
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.ffd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.46f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport.type: type = fn_type @NonInstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirectImport: %NonInstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %Self.73c: %NonInstance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance.assoc_type: type = assoc_entity_type %NonInstance.type [concrete]
// CHECK:STDOUT:   %assoc0.b0b: %NonInstance.assoc_type = assoc_entity element0, imports.%Main.import_ref.474 [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.a4a: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.0a7: type = fn_type @Convert, @ImplicitAs(%NonInstance.type) [concrete]
// CHECK:STDOUT:   %Convert.c2d: %Convert.type.0a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ffd: type = assoc_entity_type %ImplicitAs.type.a4a [concrete]
// CHECK:STDOUT:   %assoc0.46f: %ImplicitAs.assoc_type.ffd = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance: type = import_ref Main//lib, NonInstance, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.Instance = import_ref Main//lib, Instance, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.69f: %NonInstance.assoc_type = import_ref Main//lib, loc4_9, loaded [concrete = constants.%assoc0.b0b]
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.f85: %NonInstance.type = import_ref Main//lib, inst15 [no loc], loaded [symbolic = constants.%Self.73c]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Main.import_ref.112 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c1d = import_ref Main//lib, loc12_21, unloaded
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.ab2 = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonInstanceCallIndirectImport.decl: %NonInstanceCallIndirectImport.type = fn_decl @NonInstanceCallIndirectImport [concrete = constants.%NonInstanceCallIndirectImport] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc6_45: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc6_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_43.2: type = converted %.loc6_43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
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
// CHECK:STDOUT:   .F = imports.%Main.import_ref.69f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.c1d
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ab2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirectImport(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance.ref: type = name_ref NonInstance, imports.%Main.NonInstance [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %F.ref: %NonInstance.assoc_type = name_ref F, imports.%Main.import_ref.69f [concrete = constants.%assoc0.b0b]
// CHECK:STDOUT:   %.loc14_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %.loc14_4.2: %NonInstance.type = converted %.loc14_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(imports.%Main.import_ref.f85: %NonInstance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.73c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance.type
// CHECK:STDOUT:   %Dest.patt => constants.%NonInstance.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a4a
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.0a7
// CHECK:STDOUT:   %Convert => constants.%Convert.c2d
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.ffd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.46f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %InstanceCallImport.type: type = fn_type @InstanceCallImport [concrete]
// CHECK:STDOUT:   %InstanceCallImport: %InstanceCallImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self.cf8: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type %Instance.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b4d [concrete]
// CHECK:STDOUT:   %G.type.1e1: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.373) [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.59b: type = fn_type_with_self_type %G.type.1e1, %Instance.facet [concrete]
// CHECK:STDOUT:   %G.type.0c0: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.484: %G.type.0c0 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.i [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport.type: type = fn_type @InstanceCallIndirectImport [concrete]
// CHECK:STDOUT:   %InstanceCallIndirectImport: %InstanceCallIndirectImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.i = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance = import_ref Main//lib, NonInstance, unloaded
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.7cb: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.0d8: %Instance.type = import_ref Main//lib, inst39 [no loc], loaded [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93f = import_ref Main//lib, loc4_9, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.112 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.97c: <witness> = import_ref Main//lib, loc15_27, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance = imports.%Main.NonInstance
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .InstanceCallImport = %InstanceCallImport.decl
// CHECK:STDOUT:     .InstanceCallIndirectImport = %InstanceCallIndirectImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InstanceCallImport.decl: %InstanceCallImport.type = fn_decl @InstanceCallImport [concrete = constants.%InstanceCallImport] {
// CHECK:STDOUT:     %n.patt: %struct_type.i = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.i = value_param_pattern %n.patt, call_param0
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
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
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
// CHECK:STDOUT:   .G = imports.%Main.import_ref.7cb
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.93f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.97c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImport(%n.param_patt: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.7cb [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.59b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G.484]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(imports.%Main.import_ref.0d8: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %struct_type.i]() [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirectImport(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.7cb [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc10_4.1: ref %struct_type.i = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.59b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G.484]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_4.1, %impl.elem0
// CHECK:STDOUT:   %.loc10_4.2: ref %empty_tuple.type = struct_access %.loc10_4.1, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_4.3: %empty_tuple.type = converted %.loc10_4.2, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.i = struct_value (%.loc10_4.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_4.4: %struct_type.i = converted %.loc10_4.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%.loc10_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
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
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type %Instance.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b4d [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf8 [symbolic]
// CHECK:STDOUT:   %NonInstance.type: type = facet_type <@NonInstance> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonInstance = import_ref Main//lib, NonInstance, unloaded
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.7cb: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.0d8: %Instance.type = import_ref Main//lib, inst39 [no loc], loaded [symbolic = constants.%Self.cf8]
// CHECK:STDOUT:   %Main.import_ref.c55 = import_ref Main//lib, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93f = import_ref Main//lib, loc4_9, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//lib, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.112 = import_ref Main//lib, loc7_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.ef5: type = import_ref Main//lib, loc7_18, loaded [concrete = constants.%NonInstance.type]
// CHECK:STDOUT:   %Main.import_ref.ab2 = import_ref Main//lib, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %InstanceCallImportFail.decl: %InstanceCallImportFail.type = fn_decl @InstanceCallImportFail [concrete = constants.%InstanceCallImportFail] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.7cb
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c55
// CHECK:STDOUT:   .F = imports.%Main.import_ref.93f
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c58: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.ef5 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ab2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImportFail() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.7cb [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(imports.%Main.import_ref.0d8: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf8)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.cf8) {
// CHECK:STDOUT:   %Self => constants.%Self.cf8
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
