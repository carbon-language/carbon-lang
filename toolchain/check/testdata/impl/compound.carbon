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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/compound.carbon

// --- core.carbon
package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- non-instance_success.carbon
library "[[@TEST_NAME]]";

interface NonInstance1 {
  fn F1();
}

impl {.a: ()} as NonInstance1 {
  fn F1() {}
}

fn NonInstanceCall1() {
  {.a: ()}.(NonInstance1.F1)();
}

// --- fail_non-instance.carbon
library "[[@TEST_NAME]]";

import Core;

interface NonInstance2 {
  fn F2();
}

impl {.b: ()} as NonInstance2 {
  fn F2() {}
}

fn NonInstanceCall2(n: {.b: ()}) {
  // CHECK:STDERR: fail_non-instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `{.b: ()}` into type implementing `NonInstance2` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   n.(NonInstance2.F2)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non-instance.carbon:[[@LINE+4]]:3: note: type `{.b: ()}` does not implement interface `Core.ImplicitAs(NonInstance2)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   n.(NonInstance2.F2)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  n.(NonInstance2.F2)();
}

// --- fail_non-instance_indirect.carbon
library "[[@TEST_NAME]]";

import Core;

interface NonInstance3 {
  fn F3();
}

impl {.c: ()} as NonInstance3 {
  fn F3() {}
}

fn NonInstanceCallIndirect(p: {.c: ()}*) {
  // CHECK:STDERR: fail_non-instance_indirect.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `{.c: ()}` into type implementing `NonInstance3` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   p->(NonInstance3.F3)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non-instance_indirect.carbon:[[@LINE+4]]:3: note: type `{.c: ()}` does not implement interface `Core.ImplicitAs(NonInstance3)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   p->(NonInstance3.F3)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(NonInstance3.F3)();
}

// --- instance_success.carbon
library "[[@TEST_NAME]]";

interface Instance1 {
  fn G1[self: Self]();
}

impl {.d: ()} as Instance1 {
  fn G1[self: Self]() {}
}

fn InstanceCall(n: {.d: ()}) {
  n.(Instance1.G1)();
}

fn InstanceCallIndirect(p: {.d: ()}*) {
  p->(Instance1.G1)();
}

// --- fail_instance.carbon
library "[[@TEST_NAME]]";

interface Instance2 {
  fn G2[self: Self]();
}

impl {.e: ()} as Instance2 {
  fn G2[self: Self]() {}
}

fn InstanceCallFail() {
  // CHECK:STDERR: fail_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Instance2` in type `type` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   {.e: ()}.(Instance2.G2)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.e: ()}.(Instance2.G2)();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3ac: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.3ac = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert: %ImplicitAs.Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Dest.loc3_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.2: type) {
// CHECK:STDOUT:   %Dest.loc3_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3ac)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3ac) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3ac) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.f6d) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.f6d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.3ac) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.3ac) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%ImplicitAs.Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3ac)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3ac)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.3ac) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.f6d)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type)) -> @ImplicitAs.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3ac
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.f6d
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance1.type: type = facet_type <@NonInstance1> [concrete]
// CHECK:STDOUT:   %Self: %NonInstance1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance1.F1.type: type = fn_type @NonInstance1.F1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance1.F1: %NonInstance1.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance1.assoc_type: type = assoc_entity_type @NonInstance1 [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, @NonInstance1.%NonInstance1.F1.decl [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance1.impl_witness: <witness> = impl_witness file.%NonInstance1.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.type: type = fn_type @struct_type.a.as.NonInstance1.impl.F1 [concrete]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1: %struct_type.a.as.NonInstance1.impl.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, (%NonInstance1.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstanceCall1.type: type = fn_type @NonInstanceCall1 [concrete]
// CHECK:STDOUT:   %NonInstanceCall1: %NonInstanceCall1.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ced: type = fn_type_with_self_type %NonInstance1.F1.type, %NonInstance1.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance1 = %NonInstance1.decl
// CHECK:STDOUT:     .NonInstanceCall1 = %NonInstanceCall1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance1.decl: type = interface_decl @NonInstance1 [concrete = constants.%NonInstance1.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.a.as.NonInstance1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance1.impl_witness_table = impl_witness_table (@struct_type.a.as.NonInstance1.impl.%struct_type.a.as.NonInstance1.impl.F1.decl), @struct_type.a.as.NonInstance1.impl [concrete]
// CHECK:STDOUT:   %NonInstance1.impl_witness: <witness> = impl_witness %NonInstance1.impl_witness_table [concrete = constants.%NonInstance1.impl_witness]
// CHECK:STDOUT:   %NonInstanceCall1.decl: %NonInstanceCall1.type = fn_decl @NonInstanceCall1 [concrete = constants.%NonInstanceCall1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance1 {
// CHECK:STDOUT:   %Self: %NonInstance1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %NonInstance1.F1.decl: %NonInstance1.F1.type = fn_decl @NonInstance1.F1 [concrete = constants.%NonInstance1.F1] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, %NonInstance1.F1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F1 = %assoc0
// CHECK:STDOUT:   witness = (%NonInstance1.F1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.a.as.NonInstance1.impl: %struct_type.a as %NonInstance1.ref {
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.decl: %struct_type.a.as.NonInstance1.impl.F1.type = fn_decl @struct_type.a.as.NonInstance1.impl.F1 [concrete = constants.%struct_type.a.as.NonInstance1.impl.F1] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F1 = %struct_type.a.as.NonInstance1.impl.F1.decl
// CHECK:STDOUT:   witness = file.%NonInstance1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance1.F1(@NonInstance1.%Self: %NonInstance1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.a.as.NonInstance1.impl.F1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   %F1.ref: %NonInstance1.assoc_type = name_ref F1, @NonInstance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, (constants.%NonInstance1.impl_witness) [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %.loc12_11: %NonInstance1.type = converted %struct_type.a, %NonInstance1.facet [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %impl.elem0: %.ced = impl_witness_access constants.%NonInstance1.impl_witness, element0 [concrete = constants.%struct_type.a.as.NonInstance1.impl.F1]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.F1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.F1(constants.%NonInstance1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance2.type: type = facet_type <@NonInstance2> [concrete]
// CHECK:STDOUT:   %Self.cdb: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance2.F2.type: type = fn_type @NonInstance2.F2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance2.F2: %NonInstance2.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.assoc_type: type = assoc_entity_type @NonInstance2 [concrete]
// CHECK:STDOUT:   %assoc0.92e: %NonInstance2.assoc_type = assoc_entity element0, @NonInstance2.%NonInstance2.F2.decl [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance2.impl_witness: <witness> = impl_witness file.%NonInstance2.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2.type: type = fn_type @struct_type.b.as.NonInstance2.impl.F2 [concrete]
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2: %struct_type.b.as.NonInstance2.impl.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.facet: %NonInstance2.type = facet_value %struct_type.b, (%NonInstance2.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.d5c: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %NonInstanceCall2.type: type = fn_type @NonInstanceCall2 [concrete]
// CHECK:STDOUT:   %NonInstanceCall2: %NonInstanceCall2.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.edb: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.152: %ImplicitAs.type.edb = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.152 [symbolic]
// CHECK:STDOUT:   %pattern_type.519: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d3c: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance2.type)> [concrete]
// CHECK:STDOUT:   %Self.010: %ImplicitAs.type.d3c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.de0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%NonInstance2.type) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.a1a: %ImplicitAs.Convert.type.de0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.926: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%NonInstance2.type) [concrete]
// CHECK:STDOUT:   %assoc0.7c8: %ImplicitAs.assoc_type.926 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.e2c = import_ref Core//default, inst{{\d+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb) = import_ref Core//default, inst{{\d+}} [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.152)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonInstance2 = %NonInstance2.decl
// CHECK:STDOUT:     .NonInstanceCall2 = %NonInstanceCall2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonInstance2.decl: type = interface_decl @NonInstance2 [concrete = constants.%NonInstance2.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.b.as.NonInstance2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance2.impl_witness_table = impl_witness_table (@struct_type.b.as.NonInstance2.impl.%struct_type.b.as.NonInstance2.impl.F2.decl), @struct_type.b.as.NonInstance2.impl [concrete]
// CHECK:STDOUT:   %NonInstance2.impl_witness: <witness> = impl_witness %NonInstance2.impl_witness_table [concrete = constants.%NonInstance2.impl_witness]
// CHECK:STDOUT:   %NonInstanceCall2.decl: %NonInstanceCall2.type = fn_decl @NonInstanceCall2 [concrete = constants.%NonInstanceCall2] {
// CHECK:STDOUT:     %n.patt: %pattern_type.d5c = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d5c = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.b = value_param call_param0
// CHECK:STDOUT:     %.loc13_31: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:       %.loc13_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc13_30.2: type = converted %.loc13_30.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.b = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance2 {
// CHECK:STDOUT:   %Self: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.cdb]
// CHECK:STDOUT:   %NonInstance2.F2.decl: %NonInstance2.F2.type = fn_decl @NonInstance2.F2 [concrete = constants.%NonInstance2.F2] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance2.assoc_type = assoc_entity element0, %NonInstance2.F2.decl [concrete = constants.%assoc0.92e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F2 = %assoc0
// CHECK:STDOUT:   witness = (%NonInstance2.F2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.edb)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.152)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.275)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.e2c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.b.as.NonInstance2.impl: %struct_type.b as %NonInstance2.ref {
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2.decl: %struct_type.b.as.NonInstance2.impl.F2.type = fn_decl @struct_type.b.as.NonInstance2.impl.F2 [concrete = constants.%struct_type.b.as.NonInstance2.impl.F2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F2 = %struct_type.b.as.NonInstance2.impl.F2.decl
// CHECK:STDOUT:   witness = file.%NonInstance2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance2.F2(@NonInstance2.%Self: %NonInstance2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.b.as.NonInstance2.impl.F2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall2(%n.param: %struct_type.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.b = name_ref n, %n
// CHECK:STDOUT:   %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   %F2.ref: %NonInstance2.assoc_type = name_ref F2, @NonInstance2.%assoc0 [concrete = constants.%assoc0.92e]
// CHECK:STDOUT:   %.loc21: %NonInstance2.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.edb)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.edb) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.152)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.519)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Dest [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.F2(constants.%Self.cdb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.F2(constants.%NonInstance2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%Dest, constants.%Self.152) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.edb
// CHECK:STDOUT:   %Self => constants.%Self.152
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.519
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance2.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance2.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d3c
// CHECK:STDOUT:   %Self => constants.%Self.010
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.de0
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.a1a
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.926
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance3.type: type = facet_type <@NonInstance3> [concrete]
// CHECK:STDOUT:   %Self.919: %NonInstance3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance3.F3.type: type = fn_type @NonInstance3.F3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance3.F3: %NonInstance3.F3.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance3.assoc_type: type = assoc_entity_type @NonInstance3 [concrete]
// CHECK:STDOUT:   %assoc0.2f5: %NonInstance3.assoc_type = assoc_entity element0, @NonInstance3.%NonInstance3.F3.decl [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance3.impl_witness: <witness> = impl_witness file.%NonInstance3.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3.type: type = fn_type @struct_type.c.as.NonInstance3.impl.F3 [concrete]
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3: %struct_type.c.as.NonInstance3.impl.F3.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance3.facet: %NonInstance3.type = facet_value %struct_type.c, (%NonInstance3.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.5f5: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.edb: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.152: %ImplicitAs.type.edb = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.152 [symbolic]
// CHECK:STDOUT:   %pattern_type.519: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2b7: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance3.type)> [concrete]
// CHECK:STDOUT:   %Self.d7a: %ImplicitAs.type.2b7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.162: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%NonInstance3.type) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.af3: %ImplicitAs.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.f78: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%NonInstance3.type) [concrete]
// CHECK:STDOUT:   %assoc0.78e: %ImplicitAs.assoc_type.f78 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.e2c = import_ref Core//default, inst{{\d+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb) = import_ref Core//default, inst{{\d+}} [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.152)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonInstance3 = %NonInstance3.decl
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonInstance3.decl: type = interface_decl @NonInstance3 [concrete = constants.%NonInstance3.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.c.as.NonInstance3.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %NonInstance3.ref: type = name_ref NonInstance3, file.%NonInstance3.decl [concrete = constants.%NonInstance3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance3.impl_witness_table = impl_witness_table (@struct_type.c.as.NonInstance3.impl.%struct_type.c.as.NonInstance3.impl.F3.decl), @struct_type.c.as.NonInstance3.impl [concrete]
// CHECK:STDOUT:   %NonInstance3.impl_witness: <witness> = impl_witness %NonInstance3.impl_witness_table [concrete = constants.%NonInstance3.impl_witness]
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [concrete = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.5f5 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.5f5 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc13_39: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc13_37.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc13_37.2: type = converted %.loc13_37.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.c [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance3 {
// CHECK:STDOUT:   %Self: %NonInstance3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.919]
// CHECK:STDOUT:   %NonInstance3.F3.decl: %NonInstance3.F3.type = fn_decl @NonInstance3.F3 [concrete = constants.%NonInstance3.F3] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance3.assoc_type = assoc_entity element0, %NonInstance3.F3.decl [concrete = constants.%assoc0.2f5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F3 = %assoc0
// CHECK:STDOUT:   witness = (%NonInstance3.F3.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.edb)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.152)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.275)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.e2c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.c.as.NonInstance3.impl: %struct_type.c as %NonInstance3.ref {
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3.decl: %struct_type.c.as.NonInstance3.impl.F3.type = fn_decl @struct_type.c.as.NonInstance3.impl.F3 [concrete = constants.%struct_type.c.as.NonInstance3.impl.F3] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F3 = %struct_type.c.as.NonInstance3.impl.F3.decl
// CHECK:STDOUT:   witness = file.%NonInstance3.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance3.F3(@NonInstance3.%Self: %NonInstance3.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.c.as.NonInstance3.impl.F3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance3.ref: type = name_ref NonInstance3, file.%NonInstance3.decl [concrete = constants.%NonInstance3.type]
// CHECK:STDOUT:   %F3.ref: %NonInstance3.assoc_type = name_ref F3, @NonInstance3.%assoc0 [concrete = constants.%assoc0.2f5]
// CHECK:STDOUT:   %.loc21_4.1: ref %struct_type.c = deref %p.ref
// CHECK:STDOUT:   %.loc21_4.2: %NonInstance3.type = converted %.loc21_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.edb)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.edb)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.edb) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.152)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.519)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Dest [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.F3(constants.%Self.919) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.F3(constants.%NonInstance3.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%Dest, constants.%Self.152) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.edb
// CHECK:STDOUT:   %Self => constants.%Self.152
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.519
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance3.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance3.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2b7
// CHECK:STDOUT:   %Self => constants.%Self.d7a
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.162
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.af3
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.f78
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.78e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance1.type: type = facet_type <@Instance1> [concrete]
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.ac5: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Instance1.G1.type: type = fn_type @Instance1.G1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Instance1.G1: %Instance1.G1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.assoc_type: type = assoc_entity_type @Instance1 [concrete]
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, @Instance1.%Instance1.G1.decl [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance1.impl_witness: <witness> = impl_witness file.%Instance1.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.515: type = pattern_type %struct_type.d [concrete]
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.type: type = fn_type @struct_type.d.as.Instance1.impl.G1 [concrete]
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1: %struct_type.d.as.Instance1.impl.G1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.facet: %Instance1.type = facet_value %struct_type.d, (%Instance1.impl_witness) [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.e61: type = fn_type_with_self_type %Instance1.G1.type, %Instance1.facet [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.d [concrete]
// CHECK:STDOUT:   %pattern_type.8d6: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect.type: type = fn_type @InstanceCallIndirect [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect: %InstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.d = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Instance1 = %Instance1.decl
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:     .InstanceCallIndirect = %InstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance1.decl: type = interface_decl @Instance1 [concrete = constants.%Instance1.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.d.as.Instance1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance1.impl_witness_table = impl_witness_table (@struct_type.d.as.Instance1.impl.%struct_type.d.as.Instance1.impl.G1.decl), @struct_type.d.as.Instance1.impl [concrete]
// CHECK:STDOUT:   %Instance1.impl_witness: <witness> = impl_witness %Instance1.impl_witness_table [concrete = constants.%Instance1.impl_witness]
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %pattern_type.515 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.515 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.d = value_param call_param0
// CHECK:STDOUT:     %.loc11_27: type = splice_block %struct_type.d [concrete = constants.%struct_type.d] {
// CHECK:STDOUT:       %.loc11_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_26.2: type = converted %.loc11_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.d = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect.decl: %InstanceCallIndirect.type = fn_decl @InstanceCallIndirect [concrete = constants.%InstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.8d6 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.8d6 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc15_36: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc15_34.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_34.2: type = converted %.loc15_34.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.d [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance1 {
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Instance1.G1.decl: %Instance1.G1.type = fn_decl @Instance1.G1 [concrete = constants.%Instance1.G1] {
// CHECK:STDOUT:     %self.patt: @Instance1.G1.%pattern_type (%pattern_type.ac5) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Instance1.G1.%pattern_type (%pattern_type.ac5) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance1.G1.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance1.type = name_ref Self, @Instance1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance1.G1.%Self.binding.as_type (%Self.binding.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, %Instance1.G1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G1 = %assoc0
// CHECK:STDOUT:   witness = (%Instance1.G1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.d.as.Instance1.impl: %struct_type.d as %Instance1.ref {
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.decl: %struct_type.d.as.Instance1.impl.G1.type = fn_decl @struct_type.d.as.Instance1.impl.G1 [concrete = constants.%struct_type.d.as.Instance1.impl.G1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.515 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.515 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.d.as.Instance1.impl.%struct_type.d [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %self: %struct_type.d = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G1 = %struct_type.d.as.Instance1.impl.G1.decl
// CHECK:STDOUT:   witness = file.%Instance1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance1.G1(@Instance1.%Self: %Instance1.type) {
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance1.G1.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.d.as.Instance1.impl.G1(%self.param: %struct_type.d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param: %struct_type.d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.d = name_ref n, %n
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.e61 = impl_witness_access constants.%Instance1.impl_witness, element0 [concrete = constants.%struct_type.d.as.Instance1.impl.G1]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc16_4.1: ref %struct_type.d = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.e61 = impl_witness_access constants.%Instance1.impl_witness, element0 [concrete = constants.%struct_type.d.as.Instance1.impl.G1]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc16_4.1, %impl.elem0
// CHECK:STDOUT:   %.loc16_4.2: ref %empty_tuple.type = struct_access %.loc16_4.1, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_4.3: %empty_tuple.type = converted %.loc16_4.2, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.d = struct_value (%.loc16_4.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_4.4: %struct_type.d = converted %.loc16_4.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.call: init %empty_tuple.type = call %bound_method(%.loc16_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance1.G1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ac5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance1.G1(constants.%Instance1.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance1.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%struct_type.d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.515
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance2.type: type = facet_type <@Instance2> [concrete]
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.a70: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Instance2.G2.type: type = fn_type @Instance2.G2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Instance2.G2: %Instance2.G2.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance2.assoc_type: type = assoc_entity_type @Instance2 [concrete]
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, @Instance2.%Instance2.G2.decl [concrete]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance2.impl_witness: <witness> = impl_witness file.%Instance2.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.efd: type = pattern_type %struct_type.e [concrete]
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2.type: type = fn_type @struct_type.e.as.Instance2.impl.G2 [concrete]
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2: %struct_type.e.as.Instance2.impl.G2.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance2.facet: %Instance2.type = facet_value %struct_type.e, (%Instance2.impl_witness) [concrete]
// CHECK:STDOUT:   %InstanceCallFail.type: type = fn_type @InstanceCallFail [concrete]
// CHECK:STDOUT:   %InstanceCallFail: %InstanceCallFail.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Instance2 = %Instance2.decl
// CHECK:STDOUT:     .InstanceCallFail = %InstanceCallFail.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance2.decl: type = interface_decl @Instance2 [concrete = constants.%Instance2.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.e.as.Instance2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %Instance2.ref: type = name_ref Instance2, file.%Instance2.decl [concrete = constants.%Instance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance2.impl_witness_table = impl_witness_table (@struct_type.e.as.Instance2.impl.%struct_type.e.as.Instance2.impl.G2.decl), @struct_type.e.as.Instance2.impl [concrete]
// CHECK:STDOUT:   %Instance2.impl_witness: <witness> = impl_witness %Instance2.impl_witness_table [concrete = constants.%Instance2.impl_witness]
// CHECK:STDOUT:   %InstanceCallFail.decl: %InstanceCallFail.type = fn_decl @InstanceCallFail [concrete = constants.%InstanceCallFail] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance2 {
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Instance2.G2.decl: %Instance2.G2.type = fn_decl @Instance2.G2 [concrete = constants.%Instance2.G2] {
// CHECK:STDOUT:     %self.patt: @Instance2.G2.%pattern_type (%pattern_type.a70) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Instance2.G2.%pattern_type (%pattern_type.a70) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance2.G2.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance2.type = name_ref Self, @Instance2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance2.G2.%Self.binding.as_type (%Self.binding.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, %Instance2.G2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G2 = %assoc0
// CHECK:STDOUT:   witness = (%Instance2.G2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.e.as.Instance2.impl: %struct_type.e as %Instance2.ref {
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2.decl: %struct_type.e.as.Instance2.impl.G2.type = fn_decl @struct_type.e.as.Instance2.impl.G2 [concrete = constants.%struct_type.e.as.Instance2.impl.G2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.efd = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.efd = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.e.as.Instance2.impl.%struct_type.e [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %self: %struct_type.e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G2 = %struct_type.e.as.Instance2.impl.G2.decl
// CHECK:STDOUT:   witness = file.%Instance2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance2.G2(@Instance2.%Self: %Instance2.type) {
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.a70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance2.G2.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.e.as.Instance2.impl.G2(%self.param: %struct_type.e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallFail() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete = constants.%struct_type.e]
// CHECK:STDOUT:   %Instance2.ref: type = name_ref Instance2, file.%Instance2.decl [concrete = constants.%Instance2.type]
// CHECK:STDOUT:   %G2.ref: %Instance2.assoc_type = name_ref G2, @Instance2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.G2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.G2(constants.%Instance2.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance2.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%struct_type.e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.efd
// CHECK:STDOUT: }
// CHECK:STDOUT:
