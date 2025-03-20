// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/compound.carbon

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
// CHECK:STDOUT: --- non-instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance1.type: type = facet_type <@NonInstance1> [concrete]
// CHECK:STDOUT:   %Self: %NonInstance1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F1.type.163: type = fn_type @F1.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F1.b1f: %F1.type.163 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance1.assoc_type: type = assoc_entity_type %NonInstance1.type [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, @NonInstance1.%F1.decl [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F1.decl) [concrete]
// CHECK:STDOUT:   %F1.type.d4a: type = fn_type @F1.2 [concrete]
// CHECK:STDOUT:   %F1.c75: %F1.type.d4a = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, (%impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstanceCall1.type: type = fn_type @NonInstanceCall1 [concrete]
// CHECK:STDOUT:   %NonInstanceCall1: %NonInstanceCall1.type = struct_value () [concrete]
// CHECK:STDOUT:   %.971: type = fn_type_with_self_type %F1.type.163, %NonInstance1.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance1 = %NonInstance1.decl
// CHECK:STDOUT:     .NonInstanceCall1 = %NonInstanceCall1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance1.decl: type = interface_decl @NonInstance1 [concrete = constants.%NonInstance1.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F1.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %NonInstanceCall1.decl: %NonInstanceCall1.type = fn_decl @NonInstanceCall1 [concrete = constants.%NonInstanceCall1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance1 {
// CHECK:STDOUT:   %Self: %NonInstance1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F1.decl: %F1.type.163 = fn_decl @F1.1 [concrete = constants.%F1.b1f] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance1.assoc_type = assoc_entity element0, %F1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F1 = %assoc0
// CHECK:STDOUT:   witness = (%F1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.a as %NonInstance1.ref {
// CHECK:STDOUT:   %F1.decl: %F1.type.d4a = fn_decl @F1.2 [concrete = constants.%F1.c75] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F1.1(@NonInstance1.%Self: %NonInstance1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1.2() {
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
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value constants.%struct_type.a, (constants.%impl_witness) [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %.loc12_11: %NonInstance1.type = converted %struct_type.a, %NonInstance1.facet [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc12_11, element0 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.971 = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%F1.c75]
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1.1(constants.%NonInstance1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance2.type: type = facet_type <@NonInstance2> [concrete]
// CHECK:STDOUT:   %Self.5a9: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F2.type.608: type = fn_type @F2.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F2.86a: %F2.type.608 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.assoc_type: type = assoc_entity_type %NonInstance2.type [concrete]
// CHECK:STDOUT:   %assoc0.97a: %NonInstance2.assoc_type = assoc_entity element0, @NonInstance2.%F2.decl [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F2.decl) [concrete]
// CHECK:STDOUT:   %F2.type.1ec: type = fn_type @F2.2 [concrete]
// CHECK:STDOUT:   %F2.941: %F2.type.1ec = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.facet: %NonInstance2.type = facet_value %struct_type.b, (%impl_witness) [concrete]
// CHECK:STDOUT:   %NonInstanceCall2.type: type = fn_type @NonInstanceCall2 [concrete]
// CHECK:STDOUT:   %NonInstanceCall2: %NonInstanceCall2.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c38: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance2.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.d1f: type = fn_type @Convert, @ImplicitAs(%NonInstance2.type) [concrete]
// CHECK:STDOUT:   %Convert.079: %Convert.type.d1f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.5d3: type = assoc_entity_type %ImplicitAs.type.c38 [concrete]
// CHECK:STDOUT:   %assoc0.c84: %ImplicitAs.assoc_type.5d3 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst25 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F2.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %NonInstanceCall2.decl: %NonInstanceCall2.type = fn_decl @NonInstanceCall2 [concrete = constants.%NonInstanceCall2] {
// CHECK:STDOUT:     %n.patt: %struct_type.b = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.b = value_param_pattern %n.patt, call_param0
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
// CHECK:STDOUT:   %Self: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.5a9]
// CHECK:STDOUT:   %F2.decl: %F2.type.608 = fn_decl @F2.1 [concrete = constants.%F2.86a] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance2.assoc_type = assoc_entity element0, %F2.decl [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F2 = %assoc0
// CHECK:STDOUT:   witness = (%F2.decl)
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
// CHECK:STDOUT: impl @impl: %struct_type.b as %NonInstance2.ref {
// CHECK:STDOUT:   %F2.decl: %F2.type.1ec = fn_decl @F2.2 [concrete = constants.%F2.941] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F2 = %F2.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F2.1(@NonInstance2.%Self: %NonInstance2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall2(%n.param_patt: %struct_type.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.b = name_ref n, %n
// CHECK:STDOUT:   %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   %F2.ref: %NonInstance2.assoc_type = name_ref F2, @NonInstance2.%assoc0 [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:   %.loc21: %NonInstance2.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @F2.1(constants.%Self.5a9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.1(constants.%NonInstance2.facet) {}
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
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance2.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance2.type
// CHECK:STDOUT:   %Dest.patt => constants.%NonInstance2.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c38
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.d1f
// CHECK:STDOUT:   %Convert => constants.%Convert.079
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.5d3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non-instance_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonInstance3.type: type = facet_type <@NonInstance3> [concrete]
// CHECK:STDOUT:   %Self.dd1: %NonInstance3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F3.type.ee4: type = fn_type @F3.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F3.752: %F3.type.ee4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance3.assoc_type: type = assoc_entity_type %NonInstance3.type [concrete]
// CHECK:STDOUT:   %assoc0.140: %NonInstance3.assoc_type = assoc_entity element0, @NonInstance3.%F3.decl [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F3.decl) [concrete]
// CHECK:STDOUT:   %F3.type.1c2: type = fn_type @F3.2 [concrete]
// CHECK:STDOUT:   %F3.1d3: %F3.type.1c2 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance3.facet: %NonInstance3.type = facet_value %struct_type.c, (%impl_witness) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.c [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c69: type = facet_type <@ImplicitAs, @ImplicitAs(%NonInstance3.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.0f1: type = fn_type @Convert, @ImplicitAs(%NonInstance3.type) [concrete]
// CHECK:STDOUT:   %Convert.bb3: %Convert.type.0f1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.43b: type = assoc_entity_type %ImplicitAs.type.c69 [concrete]
// CHECK:STDOUT:   %assoc0.085: %ImplicitAs.assoc_type.43b = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst25 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %NonInstance3.ref: type = name_ref NonInstance3, file.%NonInstance3.decl [concrete = constants.%NonInstance3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F3.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [concrete = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
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
// CHECK:STDOUT:   %Self: %NonInstance3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.dd1]
// CHECK:STDOUT:   %F3.decl: %F3.type.ee4 = fn_decl @F3.1 [concrete = constants.%F3.752] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance3.assoc_type = assoc_entity element0, %F3.decl [concrete = constants.%assoc0.140]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F3 = %assoc0
// CHECK:STDOUT:   witness = (%F3.decl)
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
// CHECK:STDOUT: impl @impl: %struct_type.c as %NonInstance3.ref {
// CHECK:STDOUT:   %F3.decl: %F3.type.1c2 = fn_decl @F3.2 [concrete = constants.%F3.1d3] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F3 = %F3.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F3.1(@NonInstance3.%Self: %NonInstance3.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F3.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance3.ref: type = name_ref NonInstance3, file.%NonInstance3.decl [concrete = constants.%NonInstance3.type]
// CHECK:STDOUT:   %F3.ref: %NonInstance3.assoc_type = name_ref F3, @NonInstance3.%assoc0 [concrete = constants.%assoc0.140]
// CHECK:STDOUT:   %.loc21_4.1: ref %struct_type.c = deref %p.ref
// CHECK:STDOUT:   %.loc21_4.2: %NonInstance3.type = converted %.loc21_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @F3.1(constants.%Self.dd1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F3.1(constants.%NonInstance3.facet) {}
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
// CHECK:STDOUT: specific @ImplicitAs(constants.%NonInstance3.type) {
// CHECK:STDOUT:   %Dest => constants.%NonInstance3.type
// CHECK:STDOUT:   %Dest.patt => constants.%NonInstance3.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c69
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.0f1
// CHECK:STDOUT:   %Convert => constants.%Convert.bb3
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.43b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.085
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance1.type: type = facet_type <@Instance1> [concrete]
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %G1.type.cde: type = fn_type @G1.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G1.79a: %G1.type.cde = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.assoc_type: type = assoc_entity_type %Instance1.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, @Instance1.%G1.decl [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G1.decl) [concrete]
// CHECK:STDOUT:   %G1.type.c50: type = fn_type @G1.2 [concrete]
// CHECK:STDOUT:   %G1.fee: %G1.type.c50 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.facet: %Instance1.type = facet_value %struct_type.d, (%impl_witness) [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dfe: type = fn_type_with_self_type %G1.type.cde, %Instance1.facet [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.d [concrete]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G1.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %struct_type.d = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.d = value_param_pattern %n.patt, call_param0
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
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
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
// CHECK:STDOUT:   %G1.decl: %G1.type.cde = fn_decl @G1.1 [concrete = constants.%G1.79a] {
// CHECK:STDOUT:     %self.patt: @G1.1.%Self.as_type.loc4_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G1.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G1.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance1.type = name_ref Self, @Instance1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G1.1.%Self.as_type.loc4_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, %G1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G1 = %assoc0
// CHECK:STDOUT:   witness = (%G1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.d as %Instance1.ref {
// CHECK:STDOUT:   %G1.decl: %G1.type.c50 = fn_decl @G1.2 [concrete = constants.%G1.fee] {
// CHECK:STDOUT:     %self.patt: %struct_type.d = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %struct_type.d = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%struct_type.d [concrete = constants.%struct_type.d]
// CHECK:STDOUT:     %self: %struct_type.d = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G1 = %G1.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G1.1(@Instance1.%Self: %Instance1.type) {
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G1.1.%Self.as_type.loc4_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G1.2[%self.param_patt: %struct_type.d]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param_patt: %struct_type.d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.d = name_ref n, %n
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.dfe = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G1.fee]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %G1.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc16_4.1: ref %struct_type.d = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.dfe = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G1.fee]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc16_4.1, %impl.elem0
// CHECK:STDOUT:   %.loc16_4.2: ref %empty_tuple.type = struct_access %.loc16_4.1, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_4.3: %empty_tuple.type = converted %.loc16_4.2, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.d = struct_value (%.loc16_4.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_4.4: %struct_type.d = converted %.loc16_4.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %G1.call: init %empty_tuple.type = call %bound_method(%.loc16_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G1.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G1.1(constants.%Instance1.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance1.facet
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%struct_type.d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Instance2.type: type = facet_type <@Instance2> [concrete]
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %G2.type.aa7: type = fn_type @G2.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G2.588: %G2.type.aa7 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance2.assoc_type: type = assoc_entity_type %Instance2.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, @Instance2.%G2.decl [concrete]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G2.decl) [concrete]
// CHECK:STDOUT:   %G2.type.50c: type = fn_type @G2.2 [concrete]
// CHECK:STDOUT:   %G2.707: %G2.type.50c = struct_value () [concrete]
// CHECK:STDOUT:   %Instance2.facet: %Instance2.type = facet_value %struct_type.e, (%impl_witness) [concrete]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.e: type = struct_type {.e: %empty_tuple.type} [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %Instance2.ref: type = name_ref Instance2, file.%Instance2.decl [concrete = constants.%Instance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G2.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %InstanceCallFail.decl: %InstanceCallFail.type = fn_decl @InstanceCallFail [concrete = constants.%InstanceCallFail] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance2 {
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G2.decl: %G2.type.aa7 = fn_decl @G2.1 [concrete = constants.%G2.588] {
// CHECK:STDOUT:     %self.patt: @G2.1.%Self.as_type.loc4_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G2.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G2.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance2.type = name_ref Self, @Instance2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G2.1.%Self.as_type.loc4_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance2.assoc_type = assoc_entity element0, %G2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G2 = %assoc0
// CHECK:STDOUT:   witness = (%G2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.e as %Instance2.ref {
// CHECK:STDOUT:   %G2.decl: %G2.type.50c = fn_decl @G2.2 [concrete = constants.%G2.707] {
// CHECK:STDOUT:     %self.patt: %struct_type.e = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %struct_type.e = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%struct_type.e [concrete = constants.%struct_type.e]
// CHECK:STDOUT:     %self: %struct_type.e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G2 = %G2.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G2.1(@Instance2.%Self: %Instance2.type) {
// CHECK:STDOUT:   %Self: %Instance2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G2.1.%Self.as_type.loc4_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G2.2[%self.param_patt: %struct_type.e]() {
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
// CHECK:STDOUT: specific @G2.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G2.1(constants.%Instance2.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance2.facet
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%struct_type.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
