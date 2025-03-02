// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/compound.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

interface NonInstance1 {
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

// --- core.carbon
package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- fail_import_non-instance.carbon
library "[[@TEST_NAME]]";

import library "lib";
import Core

fn NonInstanceCallImportFail(n: {.i: ()}) {
  // CHECK:STDERR: fail_non-instance.carbon:[[@LINE+4]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   n.(NonInstance2.F2)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  n.(NonInstance.F)();
}

// --- fail_import_non-instance_indirect.carbon
library "[[@TEST_NAME]]";

import library "lib";
import Core

fn NonInstanceCallIndirectImport(p: {.i: ()}*) {
  // CHECK:STDERR: fail_non-instance.carbon:[[@LINE+4]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   p->(NonInstance2.F2)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
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
import Core

fn InstanceCallImportFail() {
  {.i: ()}.(Instance.G)();
}

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
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value %struct_type.a, %impl_witness [concrete]
// CHECK:STDOUT:   %NonInstanceCall1.type: type = fn_type @NonInstanceCall1 [concrete]
// CHECK:STDOUT:   %NonInstanceCall1: %NonInstanceCall1.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %F1.type.163, %NonInstance1.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance1 = %NonInstance1.decl
// CHECK:STDOUT:     .NonInstanceCall1 = %NonInstanceCall1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance1.decl: type = interface_decl @NonInstance1 [concrete = constants.%NonInstance1.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
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
// CHECK:STDOUT:   %.loc24_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_9.2: type = converted %.loc24_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %NonInstance1.ref: type = name_ref NonInstance1, file.%NonInstance1.decl [concrete = constants.%NonInstance1.type]
// CHECK:STDOUT:   %F1.ref: %NonInstance1.assoc_type = name_ref F1, @NonInstance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NonInstance1.facet: %NonInstance1.type = facet_value constants.%struct_type.a, constants.%impl_witness [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %.loc24_11: %NonInstance1.type = converted %struct_type.a, %NonInstance1.facet [concrete = constants.%NonInstance1.facet]
// CHECK:STDOUT:   %as_wit: <witness> = facet_access_witness %.loc24_11 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.a93 = impl_witness_access %as_wit, element0 [concrete = constants.%F1.c75]
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
// CHECK:STDOUT:   %Self: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F2.type.608: type = fn_type @F2.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F2.86a: %F2.type.608 = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.assoc_type: type = assoc_entity_type %NonInstance2.type [concrete]
// CHECK:STDOUT:   %assoc0: %NonInstance2.assoc_type = assoc_entity element0, @NonInstance2.%F2.decl [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F2.decl) [concrete]
// CHECK:STDOUT:   %F2.type.1ec: type = fn_type @F2.2 [concrete]
// CHECK:STDOUT:   %F2.941: %F2.type.1ec = struct_value () [concrete]
// CHECK:STDOUT:   %NonInstance2.facet: %NonInstance2.type = facet_value %struct_type.b, %impl_witness [concrete]
// CHECK:STDOUT:   %NonInstanceCall2.type: type = fn_type @NonInstanceCall2 [concrete]
// CHECK:STDOUT:   %NonInstanceCall2: %NonInstanceCall2.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.b [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NonInstance2 = %NonInstance2.decl
// CHECK:STDOUT:     .NonInstanceCall2 = %NonInstanceCall2.decl
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstance2.decl: type = interface_decl @NonInstance2 [concrete = constants.%NonInstance2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F2.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %NonInstanceCall2.decl: %NonInstanceCall2.type = fn_decl @NonInstanceCall2 [concrete = constants.%NonInstanceCall2] {
// CHECK:STDOUT:     %n.patt: %struct_type.b = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.b = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.b = value_param runtime_param0
// CHECK:STDOUT:     %.loc23_31: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:       %.loc23_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc23_30.2: type = converted %.loc23_30.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.b = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [concrete = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc31_39: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc31_37.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc31_37.2: type = converted %.loc31_37.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.b [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonInstance2 {
// CHECK:STDOUT:   %Self: %NonInstance2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F2.decl: %F2.type.608 = fn_decl @F2.1 [concrete = constants.%F2.86a] {} {}
// CHECK:STDOUT:   %assoc0: %NonInstance2.assoc_type = assoc_entity element0, %F2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F2 = %assoc0
// CHECK:STDOUT:   witness = (%F2.decl)
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
// CHECK:STDOUT:   %F2.ref: %NonInstance2.assoc_type = name_ref F2, @NonInstance2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc24: %NonInstance2.type = converted %n.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %NonInstance2.ref: type = name_ref NonInstance2, file.%NonInstance2.decl [concrete = constants.%NonInstance2.type]
// CHECK:STDOUT:   %F2.ref: %NonInstance2.assoc_type = name_ref F2, @NonInstance2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %.loc32_4.1: ref %struct_type.b = deref %p.ref
// CHECK:STDOUT:   %.loc32_4.2: %NonInstance2.type = converted %.loc32_4.1, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.1(constants.%NonInstance2.facet) {}
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
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G1.decl) [concrete]
// CHECK:STDOUT:   %G1.type.997: type = fn_type @G1.2 [concrete]
// CHECK:STDOUT:   %G1.5e0: %G1.type.997 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance1.facet: %Instance1.type = facet_value %struct_type.c, %impl_witness [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.422: type = fn_type_with_self_type %G1.type.cde, %Instance1.facet [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %struct_type.c [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect.type: type = fn_type @InstanceCallIndirect [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect: %InstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c = struct_value (%empty_tuple) [concrete]
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
// CHECK:STDOUT:     %.loc18_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_12.2: type = converted %.loc18_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G1.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %struct_type.c = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.c = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.c = value_param runtime_param0
// CHECK:STDOUT:     %.loc26_27: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:       %.loc26_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc26_26.2: type = converted %.loc26_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.c = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect.decl: %InstanceCallIndirect.type = fn_decl @InstanceCallIndirect [concrete = constants.%InstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc34_36: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %.loc34_34.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc34_34.2: type = converted %.loc34_34.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:       %ptr: type = ptr_type %struct_type.c [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance1 {
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G1.decl: %G1.type.cde = fn_decl @G1.1 [concrete = constants.%G1.79a] {
// CHECK:STDOUT:     %self.patt: @G1.1.%Self.as_type.loc11_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G1.1.%Self.as_type.loc11_15.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G1.1.%Self.as_type.loc11_15.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_15.1: type = splice_block %.loc11_15.2 [symbolic = %Self.as_type.loc11_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance1.type = name_ref Self, @Instance1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc11_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc11_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc11_15.2: type = converted %Self.ref, %Self.as_type.loc11_15.2 [symbolic = %Self.as_type.loc11_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G1.1.%Self.as_type.loc11_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance1.assoc_type = assoc_entity element0, %G1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G1 = %assoc0
// CHECK:STDOUT:   witness = (%G1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.c as %Instance1.ref {
// CHECK:STDOUT:   %G1.decl: %G1.type.997 = fn_decl @G1.2 [concrete = constants.%G1.5e0] {
// CHECK:STDOUT:     %self.patt: %struct_type.c = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %struct_type.c = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.c = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%struct_type.c [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %self: %struct_type.c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G1 = %G1.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G1.1(@Instance1.%Self: %Instance1.type) {
// CHECK:STDOUT:   %Self: %Instance1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc11_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc11_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G1.1.%Self.as_type.loc11_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G1.2[%self.param_patt: %struct_type.c]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param_patt: %struct_type.c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.c = name_ref n, %n
// CHECK:STDOUT:   %Instance1.ref: type = name_ref Instance1, file.%Instance1.decl [concrete = constants.%Instance1.type]
// CHECK:STDOUT:   %G1.ref: %Instance1.assoc_type = name_ref G1, @Instance1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.422 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G1.5e0]
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
// CHECK:STDOUT:   %.loc35_4.1: ref %struct_type.c = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.422 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G1.5e0]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc35_4.1, %impl.elem0
// CHECK:STDOUT:   %.loc35_4.2: ref %empty_tuple.type = struct_access %.loc35_4.1, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc35_4.3: %empty_tuple.type = converted %.loc35_4.2, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.c = struct_value (%.loc35_4.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc35_4.4: %struct_type.c = converted %.loc35_4.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %G1.call: init %empty_tuple.type = call %bound_method(%.loc35_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G1.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc11_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G1.1(constants.%Instance1.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance1.facet
// CHECK:STDOUT:   %Self.as_type.loc11_15.1 => constants.%struct_type.c
// CHECK:STDOUT: }
// CHECK:STDOUT:
