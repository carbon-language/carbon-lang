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
  // CHECK:STDERR: fail_non-instance.carbon:[[@LINE+4]]:3: note: type `{.b: ()}` does not implement interface `Core.ImplicitAs(NonInstance2)` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_non-instance_indirect.carbon:[[@LINE+4]]:3: note: type `{.c: ()}` does not implement interface `Core.ImplicitAs(NonInstance3)` [MissingImplInMemberAccessInContext]
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
  fn G1[unused self: Self]() {}
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
  fn G2[unused self: Self]() {}
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fe: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.9fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.8de: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %Dest [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert: %ImplicitAs.WithSelf.Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.595 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_29.1: type = splice_block %.loc3_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Dest.loc3_22.2: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.2: type) {
// CHECK:STDOUT:   %Dest.loc3_22.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.9fe)]
// CHECK:STDOUT:   %Self.loc3_35.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %.loc4_31.2: Core.Form = init_form %Dest.ref [symbolic = %.loc4_31.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_35.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.9fe)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.8de)]
// CHECK:STDOUT:   %.loc4_31.1: Core.Form = init_form %Dest [symbolic = %.loc4_31.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fe
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.8de
// CHECK:STDOUT:   %.loc4_31.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance1.type: type = facet_type <@NonInstance1> [concrete]
// CHECK:STDOUT:   %Self: %NonInstance1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.type.3a6: type = fn_type @NonInstance1.WithSelf.F1, @NonInstance1.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.898: %NonInstance1.WithSelf.F1.type.3a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance1.assoc_type: type = assoc_entity_type @NonInstance1 [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, @NonInstance1.WithSelf.%NonInstance1.WithSelf.F1.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance1.impl_witness: <witness> = impl_witness @struct_type.a.as.NonInstance1.impl.%NonInstance1.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.type: type = fn_type @struct_type.a.as.NonInstance1.impl.F1 [concrete]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1: %struct_type.a.as.NonInstance1.impl.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, (%NonInstance1.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.type.a2c: type = fn_type @NonInstance1.WithSelf.F1, @NonInstance1.WithSelf(%NonInstance1.facet) [concrete]
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.e33: %NonInstance1.WithSelf.F1.type.a2c = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstanceCall1.type: type = fn_type @NonInstanceCall1 [concrete]
// CHECK:STDOUT:   %NonInstanceCall1: %NonInstanceCall1.type = struct_value () [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %NonInstance1.WithSelf.F1.type.a2c, %NonInstance1.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance1 = %NonInstance1.decl
// CHECK:STDOUT:     .NonInstanceCall1 = %NonInstanceCall1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance1.decl: type = interface_decl @NonInstance1 [concrete = constants.%NonInstance1.type] {} {}
// CHECK:STDOUT:   impl_decl @struct_type.a.as.NonInstance1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCall1.decl: %NonInstanceCall1.type = fn_decl @NonInstanceCall1 [concrete = constants.%NonInstanceCall1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance1 {
// CHECK:STDOUT:   %Self: %NonInstance1.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %NonInstance1.WithSelf.decl = interface_with_self_decl @NonInstance1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.decl: @NonInstance1.WithSelf.%NonInstance1.WithSelf.F1.type (%NonInstance1.WithSelf.F1.type.3a6) = fn_decl @NonInstance1.WithSelf.F1 [symbolic = @NonInstance1.WithSelf.%NonInstance1.WithSelf.F1 (constants.%NonInstance1.WithSelf.F1.898)] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, %NonInstance1.WithSelf.F1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F1 = @NonInstance1.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@NonInstance1.WithSelf.%NonInstance1.WithSelf.F1.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.a.as.NonInstance1.impl: %struct_type.a as %NonInstance1.ref {
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.decl: %struct_type.a.as.NonInstance1.impl.F1.type = fn_decl @struct_type.a.as.NonInstance1.impl.F1 [concrete = constants.%struct_type.a.as.NonInstance1.impl.F1] {} {}
// CHECK:STDOUT:   %NonInstance1.impl_witness_table = impl_witness_table (%struct_type.a.as.NonInstance1.impl.F1.decl), @struct_type.a.as.NonInstance1.impl [concrete]
// CHECK:STDOUT:   %NonInstance1.impl_witness: <witness> = impl_witness %NonInstance1.impl_witness_table [concrete = constants.%NonInstance1.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F1 = %struct_type.a.as.NonInstance1.impl.F1.decl
// CHECK:STDOUT:   witness = %NonInstance1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance1.WithSelf.F1(@NonInstance1.%Self: %NonInstance1.type) {
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
// CHECK:STDOUT:   %.loc12_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   %F1.ref: %NonInstance1.assoc_type = name_ref F1, @NonInstance1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, (constants.%NonInstance1.impl_witness) [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %.loc12_11: %NonInstance1.type = converted %struct_type.a, %NonInstance1.facet [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%NonInstance1.impl_witness, element0 [concrete = constants.%struct_type.a.as.NonInstance1.impl.F1]
// CHECK:STDOUT:   %struct_type.a.as.NonInstance1.impl.F1.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.type => constants.%NonInstance1.WithSelf.F1.type.3a6
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1 => constants.%NonInstance1.WithSelf.F1.898
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.WithSelf.F1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.WithSelf(constants.%NonInstance1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%NonInstance1.facet
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1.type => constants.%NonInstance1.WithSelf.F1.type.a2c
// CHECK:STDOUT:   %NonInstance1.WithSelf.F1 => constants.%NonInstance1.WithSelf.F1.e33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance1.WithSelf.F1(constants.%NonInstance1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance2.type: type = facet_type <@NonInstance2> [concrete]
// CHECK:STDOUT:   %Self.e6c: %NonInstance2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.type.824: type = fn_type @NonInstance2.WithSelf.F2, @NonInstance2.WithSelf(%Self.e6c) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.9da: %NonInstance2.WithSelf.F2.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance2.assoc_type: type = assoc_entity_type @NonInstance2 [concrete]
// CHECK:STDOUT:   %assoc0.56f: %NonInstance2.assoc_type = assoc_entity element0, @NonInstance2.WithSelf.%NonInstance2.WithSelf.F2.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance2.impl_witness: <witness> = impl_witness @struct_type.b.as.NonInstance2.impl.%NonInstance2.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2.type: type = fn_type @struct_type.b.as.NonInstance2.impl.F2 [concrete]
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2: %struct_type.b.as.NonInstance2.impl.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.facet: %NonInstance2.type = facet_value %struct_type.b, (%NonInstance2.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.type.041: type = fn_type @NonInstance2.WithSelf.F2, @NonInstance2.WithSelf(%NonInstance2.facet) [concrete]
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.fc4: %NonInstance2.WithSelf.F2.type.041 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d5c: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %NonInstanceCall2.type: type = fn_type @NonInstanceCall2 [concrete]
// CHECK:STDOUT:   %NonInstanceCall2: %NonInstanceCall2.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.031: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.738: %ImplicitAs.type.031 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b3a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.1de: %ImplicitAs.WithSelf.Convert.type.b3a = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.738 [symbolic]
// CHECK:STDOUT:   %pattern_type.e9a: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.282: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance2.type)> [concrete]
// CHECK:STDOUT:   %Self.df1: %ImplicitAs.type.282 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.03b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%NonInstance2.type, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.4fe: %ImplicitAs.WithSelf.Convert.type.03b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.cd6: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%NonInstance2.type) [concrete]
// CHECK:STDOUT:   %assoc0.af4: %ImplicitAs.assoc_type.cd6 = assoc_entity element0, imports.%Core.import_ref.201 [concrete]
// CHECK:STDOUT:   %assoc0.843: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.cc1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.178: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.843)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.201: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.b3a) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.1de)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ac4dc5.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.738)]
// CHECK:STDOUT:   %Core.import_ref.9ec = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.cc1 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCall2.decl: %NonInstanceCall2.type = fn_decl @NonInstanceCall2 [concrete = constants.%NonInstanceCall2] {
// CHECK:STDOUT:     %n.patt: %pattern_type.d5c = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d5c = value_param_pattern %n.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.b = value_param call_param0
// CHECK:STDOUT:     %.loc13_31: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:       %.loc13_30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc13_30.2: type = converted %.loc13_30.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.b = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance2 {
// CHECK:STDOUT:   %Self: %NonInstance2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e6c]
// CHECK:STDOUT:   %NonInstance2.WithSelf.decl = interface_with_self_decl @NonInstance2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.decl: @NonInstance2.WithSelf.%NonInstance2.WithSelf.F2.type (%NonInstance2.WithSelf.F2.type.824) = fn_decl @NonInstance2.WithSelf.F2 [symbolic = @NonInstance2.WithSelf.%NonInstance2.WithSelf.F2 (constants.%NonInstance2.WithSelf.F2.9da)] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance2.assoc_type = assoc_entity element0, %NonInstance2.WithSelf.F2.decl [concrete = constants.%assoc0.56f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F2 = @NonInstance2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@NonInstance2.WithSelf.%NonInstance2.WithSelf.F2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.031)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.738)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.9ec
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.178
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.b.as.NonInstance2.impl: %struct_type.b as %NonInstance2.ref {
// CHECK:STDOUT:   %struct_type.b.as.NonInstance2.impl.F2.decl: %struct_type.b.as.NonInstance2.impl.F2.type = fn_decl @struct_type.b.as.NonInstance2.impl.F2 [concrete = constants.%struct_type.b.as.NonInstance2.impl.F2] {} {}
// CHECK:STDOUT:   %NonInstance2.impl_witness_table = impl_witness_table (%struct_type.b.as.NonInstance2.impl.F2.decl), @struct_type.b.as.NonInstance2.impl [concrete]
// CHECK:STDOUT:   %NonInstance2.impl_witness: <witness> = impl_witness %NonInstance2.impl_witness_table [concrete = constants.%NonInstance2.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F2 = %struct_type.b.as.NonInstance2.impl.F2.decl
// CHECK:STDOUT:   witness = %NonInstance2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance2.WithSelf.F2(@NonInstance2.%Self: %NonInstance2.type) {
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
// CHECK:STDOUT:   %F2.ref: %NonInstance2.assoc_type = name_ref F2, @NonInstance2.WithSelf.%assoc0 [concrete = constants.%assoc0.56f]
// CHECK:STDOUT:   %.loc21: %NonInstance2.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.ac4dc5.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.031)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.031) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.738)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.e9a)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Dest [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Dest [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.WithSelf(constants.%Self.e6c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e6c
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.type => constants.%NonInstance2.WithSelf.F2.type.824
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2 => constants.%NonInstance2.WithSelf.F2.9da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.WithSelf.F2(constants.%Self.e6c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.WithSelf(constants.%NonInstance2.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%NonInstance2.facet
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2.type => constants.%NonInstance2.WithSelf.F2.type.041
// CHECK:STDOUT:   %NonInstance2.WithSelf.F2 => constants.%NonInstance2.WithSelf.F2.fc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance2.WithSelf.F2(constants.%NonInstance2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self.738) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self.738) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.031
// CHECK:STDOUT:   %Self => constants.%Self.738
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e9a
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance2.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance2.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.282
// CHECK:STDOUT:   %Self => constants.%Self.df1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%NonInstance2.type, constants.%Self.738) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Dest => constants.%NonInstance2.type
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.282
// CHECK:STDOUT:   %Self => constants.%Self.738
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.03b
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.4fe
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.cd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.af4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance3.type: type = facet_type <@NonInstance3> [concrete]
// CHECK:STDOUT:   %Self.0da: %NonInstance3.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.type.c21: type = fn_type @NonInstance3.WithSelf.F3, @NonInstance3.WithSelf(%Self.0da) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.b62: %NonInstance3.WithSelf.F3.type.c21 = struct_value () [symbolic]
// CHECK:STDOUT:   %NonInstance3.assoc_type: type = assoc_entity_type @NonInstance3 [concrete]
// CHECK:STDOUT:   %assoc0.342: %NonInstance3.assoc_type = assoc_entity element0, @NonInstance3.WithSelf.%NonInstance3.WithSelf.F3.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %NonInstance3.impl_witness: <witness> = impl_witness @struct_type.c.as.NonInstance3.impl.%NonInstance3.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3.type: type = fn_type @struct_type.c.as.NonInstance3.impl.F3 [concrete]
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3: %struct_type.c.as.NonInstance3.impl.F3.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance3.facet: %NonInstance3.type = facet_value %struct_type.c, (%NonInstance3.impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.type.a3f: type = fn_type @NonInstance3.WithSelf.F3, @NonInstance3.WithSelf(%NonInstance3.facet) [concrete]
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.04b: %NonInstance3.WithSelf.F3.type.a3f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.5f5: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.031: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.738: %ImplicitAs.type.031 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b3a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.1de: %ImplicitAs.WithSelf.Convert.type.b3a = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.738 [symbolic]
// CHECK:STDOUT:   %pattern_type.e9a: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.deb: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance3.type)> [concrete]
// CHECK:STDOUT:   %Self.9cf: %ImplicitAs.type.deb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8ec: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%NonInstance3.type, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.0e4: %ImplicitAs.WithSelf.Convert.type.8ec = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.eea: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%NonInstance3.type) [concrete]
// CHECK:STDOUT:   %assoc0.605: %ImplicitAs.assoc_type.eea = assoc_entity element0, imports.%Core.import_ref.201 [concrete]
// CHECK:STDOUT:   %assoc0.843: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.cc1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.178: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.843)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.201: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.b3a) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.1de)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ac4dc5.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.738)]
// CHECK:STDOUT:   %Core.import_ref.9ec = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.cc1 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %NonInstance3.ref: type = name_ref NonInstance3, file.%NonInstance3.decl [concrete = constants.%NonInstance3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [concrete = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.5f5 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.5f5 = value_param_pattern %p.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc13_39: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc13_37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc13_37.2: type = converted %.loc13_37.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.c [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance3 {
// CHECK:STDOUT:   %Self: %NonInstance3.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0da]
// CHECK:STDOUT:   %NonInstance3.WithSelf.decl = interface_with_self_decl @NonInstance3 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.decl: @NonInstance3.WithSelf.%NonInstance3.WithSelf.F3.type (%NonInstance3.WithSelf.F3.type.c21) = fn_decl @NonInstance3.WithSelf.F3 [symbolic = @NonInstance3.WithSelf.%NonInstance3.WithSelf.F3 (constants.%NonInstance3.WithSelf.F3.b62)] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance3.assoc_type = assoc_entity element0, %NonInstance3.WithSelf.F3.decl [concrete = constants.%assoc0.342]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F3 = @NonInstance3.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@NonInstance3.WithSelf.%NonInstance3.WithSelf.F3.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.031)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.738)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.9ec
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.178
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.c.as.NonInstance3.impl: %struct_type.c as %NonInstance3.ref {
// CHECK:STDOUT:   %struct_type.c.as.NonInstance3.impl.F3.decl: %struct_type.c.as.NonInstance3.impl.F3.type = fn_decl @struct_type.c.as.NonInstance3.impl.F3 [concrete = constants.%struct_type.c.as.NonInstance3.impl.F3] {} {}
// CHECK:STDOUT:   %NonInstance3.impl_witness_table = impl_witness_table (%struct_type.c.as.NonInstance3.impl.F3.decl), @struct_type.c.as.NonInstance3.impl [concrete]
// CHECK:STDOUT:   %NonInstance3.impl_witness: <witness> = impl_witness %NonInstance3.impl_witness_table [concrete = constants.%NonInstance3.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F3 = %struct_type.c.as.NonInstance3.impl.F3.decl
// CHECK:STDOUT:   witness = %NonInstance3.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonInstance3.WithSelf.F3(@NonInstance3.%Self: %NonInstance3.type) {
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
// CHECK:STDOUT:   %F3.ref: %NonInstance3.assoc_type = name_ref F3, @NonInstance3.WithSelf.%assoc0 [concrete = constants.%assoc0.342]
// CHECK:STDOUT:   %.loc21_4.1: ref %struct_type.c = deref %p.ref
// CHECK:STDOUT:   %.loc21_4.2: %NonInstance3.type = converted %.loc21_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.ac4dc5.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.031)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.031)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.031) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.738)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.e9a)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Dest [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Dest [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.WithSelf(constants.%Self.0da) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.0da
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.type => constants.%NonInstance3.WithSelf.F3.type.c21
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3 => constants.%NonInstance3.WithSelf.F3.b62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.WithSelf.F3(constants.%Self.0da) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.WithSelf(constants.%NonInstance3.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%NonInstance3.facet
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3.type => constants.%NonInstance3.WithSelf.F3.type.a3f
// CHECK:STDOUT:   %NonInstance3.WithSelf.F3 => constants.%NonInstance3.WithSelf.F3.04b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonInstance3.WithSelf.F3(constants.%NonInstance3.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self.738) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self.738) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.031
// CHECK:STDOUT:   %Self => constants.%Self.738
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e9a
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance3.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance3.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.deb
// CHECK:STDOUT:   %Self => constants.%Self.9cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%NonInstance3.type, constants.%Self.738) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Dest => constants.%NonInstance3.type
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.deb
// CHECK:STDOUT:   %Self => constants.%Self.738
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.8ec
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.0e4
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.eea
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.605
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance1.type: type = facet_type <@Instance1> [concrete]
// CHECK:STDOUT:   %Self: %Instance1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.97b: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Instance1.WithSelf.G1.type.291: type = fn_type @Instance1.WithSelf.G1, @Instance1.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Instance1.WithSelf.G1.db7: %Instance1.WithSelf.G1.type.291 = struct_value () [symbolic]
// CHECK:STDOUT:   %Instance1.assoc_type: type = assoc_entity_type @Instance1 [concrete]
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, @Instance1.WithSelf.%Instance1.WithSelf.G1.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance1.impl_witness: <witness> = impl_witness @struct_type.d.as.Instance1.impl.%Instance1.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.515: type = pattern_type %struct_type.d [concrete]
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.type: type = fn_type @struct_type.d.as.Instance1.impl.G1 [concrete]
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1: %struct_type.d.as.Instance1.impl.G1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.facet: %Instance1.type = facet_value %struct_type.d, (%Instance1.impl_witness) [concrete]
// CHECK:STDOUT:   %Instance1.WithSelf.G1.type.ba6: type = fn_type @Instance1.WithSelf.G1, @Instance1.WithSelf(%Instance1.facet) [concrete]
// CHECK:STDOUT:   %Instance1.WithSelf.G1.ceb: %Instance1.WithSelf.G1.type.ba6 = struct_value () [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.276: type = fn_type_with_self_type %Instance1.WithSelf.G1.type.ba6, %Instance1.facet [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.d [concrete]
// CHECK:STDOUT:   %pattern_type.8d6: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect.type: type = fn_type @InstanceCallIndirect [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect: %InstanceCallIndirect.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %pattern_type.515 = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.515 = value_param_pattern %n.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.d = value_param call_param0
// CHECK:STDOUT:     %.loc11_27: type = splice_block %struct_type.d [concrete = constants.%struct_type.d] {
// CHECK:STDOUT:       %.loc11_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_26.2: type = converted %.loc11_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.d = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect.decl: %InstanceCallIndirect.type = fn_decl @InstanceCallIndirect [concrete = constants.%InstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.8d6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.8d6 = value_param_pattern %p.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc15_36: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc15_34.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc15_34.2: type = converted %.loc15_34.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.d [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance1 {
// CHECK:STDOUT:   %Self: %Instance1.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Instance1.WithSelf.decl = interface_with_self_decl @Instance1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Instance1.WithSelf.G1.decl: @Instance1.WithSelf.%Instance1.WithSelf.G1.type (%Instance1.WithSelf.G1.type.291) = fn_decl @Instance1.WithSelf.G1 [symbolic = @Instance1.WithSelf.%Instance1.WithSelf.G1 (constants.%Instance1.WithSelf.G1.db7)] {
// CHECK:STDOUT:     %self.patt: @Instance1.WithSelf.G1.%pattern_type (%pattern_type.97b) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Instance1.WithSelf.G1.%pattern_type (%pattern_type.97b) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance1.WithSelf.G1.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance1.type = name_ref Self, @Instance1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance1.WithSelf.G1.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, %Instance1.WithSelf.G1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G1 = @Instance1.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Instance1.WithSelf.%Instance1.WithSelf.G1.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.d.as.Instance1.impl: %struct_type.d as %Instance1.ref {
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.decl: %struct_type.d.as.Instance1.impl.G1.type = fn_decl @struct_type.d.as.Instance1.impl.G1 [concrete = constants.%struct_type.d.as.Instance1.impl.G1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.515 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.515 = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.d.as.Instance1.impl.%struct_type.d [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %self: %struct_type.d = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance1.impl_witness_table = impl_witness_table (%struct_type.d.as.Instance1.impl.G1.decl), @struct_type.d.as.Instance1.impl [concrete]
// CHECK:STDOUT:   %Instance1.impl_witness: <witness> = impl_witness %Instance1.impl_witness_table [concrete = constants.%Instance1.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G1 = %struct_type.d.as.Instance1.impl.G1.decl
// CHECK:STDOUT:   witness = %Instance1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance1.WithSelf.G1(@Instance1.%Self: %Instance1.type) {
// CHECK:STDOUT:   %Self: %Instance1.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.97b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance1.WithSelf.G1.%Self.binding.as_type (%Self.binding.as_type));
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
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.276 = impl_witness_access constants.%Instance1.impl_witness, element0 [concrete = constants.%struct_type.d.as.Instance1.impl.G1]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %struct_type.d.as.Instance1.impl.G1.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc16_4.1: ref %struct_type.d = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.276 = impl_witness_access constants.%Instance1.impl_witness, element0 [concrete = constants.%struct_type.d.as.Instance1.impl.G1]
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
// CHECK:STDOUT: specific @Instance1.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Instance1.WithSelf.G1.type => constants.%Instance1.WithSelf.G1.type.291
// CHECK:STDOUT:   %Instance1.WithSelf.G1 => constants.%Instance1.WithSelf.G1.db7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance1.WithSelf.G1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.97b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance1.WithSelf(constants.%Instance1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Instance1.facet
// CHECK:STDOUT:   %Instance1.WithSelf.G1.type => constants.%Instance1.WithSelf.G1.type.ba6
// CHECK:STDOUT:   %Instance1.WithSelf.G1 => constants.%Instance1.WithSelf.G1.ceb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance1.WithSelf.G1(constants.%Instance1.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance1.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%struct_type.d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.515
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance2.type: type = facet_type <@Instance2> [concrete]
// CHECK:STDOUT:   %Self: %Instance2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.a35: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Instance2.WithSelf.G2.type.e15: type = fn_type @Instance2.WithSelf.G2, @Instance2.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Instance2.WithSelf.G2.6c8: %Instance2.WithSelf.G2.type.e15 = struct_value () [symbolic]
// CHECK:STDOUT:   %Instance2.assoc_type: type = assoc_entity_type @Instance2 [concrete]
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, @Instance2.WithSelf.%Instance2.WithSelf.G2.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Instance2.impl_witness: <witness> = impl_witness @struct_type.e.as.Instance2.impl.%Instance2.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.efd: type = pattern_type %struct_type.e [concrete]
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2.type: type = fn_type @struct_type.e.as.Instance2.impl.G2 [concrete]
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2: %struct_type.e.as.Instance2.impl.G2.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance2.facet: %Instance2.type = facet_value %struct_type.e, (%Instance2.impl_witness) [concrete]
// CHECK:STDOUT:   %Instance2.WithSelf.G2.type.008: type = fn_type @Instance2.WithSelf.G2, @Instance2.WithSelf(%Instance2.facet) [concrete]
// CHECK:STDOUT:   %Instance2.WithSelf.G2.f23: %Instance2.WithSelf.G2.type.008 = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %Instance2.ref: type = name_ref Instance2, file.%Instance2.decl [concrete = constants.%Instance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallFail.decl: %InstanceCallFail.type = fn_decl @InstanceCallFail [concrete = constants.%InstanceCallFail] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance2 {
// CHECK:STDOUT:   %Self: %Instance2.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Instance2.WithSelf.decl = interface_with_self_decl @Instance2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Instance2.WithSelf.G2.decl: @Instance2.WithSelf.%Instance2.WithSelf.G2.type (%Instance2.WithSelf.G2.type.e15) = fn_decl @Instance2.WithSelf.G2 [symbolic = @Instance2.WithSelf.%Instance2.WithSelf.G2 (constants.%Instance2.WithSelf.G2.6c8)] {
// CHECK:STDOUT:     %self.patt: @Instance2.WithSelf.G2.%pattern_type (%pattern_type.a35) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Instance2.WithSelf.G2.%pattern_type (%pattern_type.a35) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Instance2.WithSelf.G2.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance2.type = name_ref Self, @Instance2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Instance2.WithSelf.G2.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, %Instance2.WithSelf.G2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G2 = @Instance2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Instance2.WithSelf.%Instance2.WithSelf.G2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @struct_type.e.as.Instance2.impl: %struct_type.e as %Instance2.ref {
// CHECK:STDOUT:   %struct_type.e.as.Instance2.impl.G2.decl: %struct_type.e.as.Instance2.impl.G2.type = fn_decl @struct_type.e.as.Instance2.impl.G2 [concrete = constants.%struct_type.e.as.Instance2.impl.G2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.efd = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.efd = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @struct_type.e.as.Instance2.impl.%struct_type.e [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %self: %struct_type.e = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance2.impl_witness_table = impl_witness_table (%struct_type.e.as.Instance2.impl.G2.decl), @struct_type.e.as.Instance2.impl [concrete]
// CHECK:STDOUT:   %Instance2.impl_witness: <witness> = impl_witness %Instance2.impl_witness_table [concrete = constants.%Instance2.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G2 = %struct_type.e.as.Instance2.impl.G2.decl
// CHECK:STDOUT:   witness = %Instance2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Instance2.WithSelf.G2(@Instance2.%Self: %Instance2.type) {
// CHECK:STDOUT:   %Self: %Instance2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.a35)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Instance2.WithSelf.G2.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @struct_type.e.as.Instance2.impl.G2(%self.param: %struct_type.e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallFail() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete = constants.%struct_type.e]
// CHECK:STDOUT:   %Instance2.ref: type = name_ref Instance2, file.%Instance2.decl [concrete = constants.%Instance2.type]
// CHECK:STDOUT:   %G2.ref: %Instance2.assoc_type = name_ref G2, @Instance2.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Instance2.WithSelf.G2.type => constants.%Instance2.WithSelf.G2.type.e15
// CHECK:STDOUT:   %Instance2.WithSelf.G2 => constants.%Instance2.WithSelf.G2.6c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.WithSelf.G2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.WithSelf(constants.%Instance2.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Instance2.facet
// CHECK:STDOUT:   %Instance2.WithSelf.G2.type => constants.%Instance2.WithSelf.G2.type.008
// CHECK:STDOUT:   %Instance2.WithSelf.G2 => constants.%Instance2.WithSelf.G2.f23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Instance2.WithSelf.G2(constants.%Instance2.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance2.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%struct_type.e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.efd
// CHECK:STDOUT: }
// CHECK:STDOUT:
