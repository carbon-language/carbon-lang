// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/compound_member_access.carbon

// --- core.carbon
package Core;

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

impl forall [T:! type] T as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}

// --- fail_todo_associated_constant.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
}

// Simple member access.
fn Simple1(T:! J, S:! T.U) {}

// This should be equivalent to `Simple1` above, but using compound member access.
// CHECK:STDERR: fail_todo_associated_constant.carbon:[[@LINE+4]]:25: error: cannot access member of interface `J` in type `J` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: fn Compound1(V:! J, W:! V.(J.U)) {}
// CHECK:STDERR:                         ^~~~~~~
// CHECK:STDERR:
fn Compound1(V:! J, W:! V.(J.U)) {}

// --- fail_todo_non_instance.carbon
library "[[@TEST_NAME]]";

interface K1 {
  fn Q1();
}

// Simple member access.
fn Simple2(T:! K1) {
  T.Q1();
}

// This should be equivalent to `Simple2` above, but using compound member access.
fn Compound2(V:! K1) {
  // CHECK:STDERR: fail_todo_non_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `K1` in type `K1` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   V.(K1.Q1)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  V.(K1.Q1)();
}

// --- fail_caller_instance_interface_not.carbon
library "[[@TEST_NAME]]";

import Core;

interface K2 {
  fn Q2();
}

// Simple member access allows this.
// TODO: This is not expected to fail.
fn Simple3(T:! K2, x: T) {
  x.Q2();
}

// Compound member access does not. It tries to convert `y` to `K2`, but only
// its type `V` can.
fn Compound3(V:! K2, y: V) {
  // CHECK:STDERR: fail_caller_instance_interface_not.carbon:[[@LINE+4]]:3: error: cannot access member of interface `K2` in type `V` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   y.(K2.Q2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  y.(K2.Q2)();
}

// --- fail_todo_instance.carbon
library "[[@TEST_NAME]]";

interface L1 {
  fn R1[self: Self]();
  fn S1[addr self: Self*]();
}

// Simple member access.
// TODO: This is not expected to fail.
fn Simple4(T:! L1, x: T) {
  x.R1();
  var xcopy: T = x;
  xcopy.S1();
}

// This should be equivalent to `Simple4` above, but using compound member access.
fn Compound4(V:! L1, y: V) {
  // CHECK:STDERR: fail_todo_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L1` in type `V` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   y.(L1.R1)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  y.(L1.R1)();
  var ycopy: V = y;
  // CHECK:STDERR: fail_todo_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L1` in type `V` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   ycopy.(L1.S1)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  ycopy.(L1.S1)();
}

// --- fail_interface_instance_caller_not.carbon

library "[[@TEST_NAME]]";

interface L2 {
  fn R2[self: Self]();
  fn S2[addr self: Self*]();
}

// Simple member access. Fails since calling an instance method without an object.
fn Simple5(T:! L2) {
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   T.R2();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE-9]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn R2[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.R2();
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   T.S2();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE-16]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn S2[addr self: Self*]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.S2();
}

// TODO: Expected to fail in the same way as `Simple5`.
fn Compound5(V:! L2) {
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L2` in type `L2` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   V.(L2.R2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  V.(L2.R2)();
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L2` in type `L2` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   V.(L2.S2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  V.(L2.S2)();
}

// --- todo_fail_combine_non_instance.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {
  fn G();
}

class C {}
impl C as A {
  fn G() {}
}

// Since `A.G` is a non-instance method, compound member access may only be
// used with a type, not an instance.
fn Fails() {
  ({} as C).((A & A).G)();
  (({} as C) as (C as (A & A))).((A & A).G)();
  (({} as C) as (C as (A & A))).(A.G)();
}

// --- fail_todo_allowed_combine_non_instance.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {
  fn G();
}

class C {}
impl C as A {
  fn G() {}
}

fn Works() {
  // CHECK:STDERR: fail_todo_allowed_combine_non_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `A` in type `type` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   C.((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C.((A & A).G)();
  // CHECK:STDERR: fail_todo_allowed_combine_non_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `A` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   (C as (A & A)).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (C as (A & A)).((A & A).G)();
  // CHECK:STDERR: fail_todo_allowed_combine_non_instance.carbon:[[@LINE+4]]:3: error: cannot access member of interface `A` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   (C as (A & A)).(A.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (C as (A & A)).(A.G)();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc3_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc7_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc7_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc15_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc3_14.1: type) {
// CHECK:STDOUT:   %Dest.loc3_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc3_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc3_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc3_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc3_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc4_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc7_22.1: type) {
// CHECK:STDOUT:   %Dest.loc7_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc7_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc7_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc7_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc8_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc7_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc8_20.1: type = splice_block %.loc8_20.3 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc8_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc8_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc8_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc8_20.3: type = converted %Self.ref, %Self.as_type.loc8_20.2 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc8_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc12_37: type = facet_access_type %Self.ref.loc12_37 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc12_37: type = converted %Self.ref.loc12_37, %Self.as_type.loc12_37 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc12_15.1: type = splice_block %.loc12_15.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc12_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc12_15.2: type = facet_access_type %Self.ref.loc12_15 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc12_15.2: type = converted %Self.ref.loc12_15, %Self.as_type.loc12_15.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc12_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc12_28: type = facet_access_type %Self.ref.loc12_28 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc12_28.2: type = converted %Self.ref.loc12_28, %Self.as_type.loc12_28 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T.loc15_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc15_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc15_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc15_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc16_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc16_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc16_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc3_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc7_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc8_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc12_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc3_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc3_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc7_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc7_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc8_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc7_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc12_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc12_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %S: %impl.elem0 = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %S.patt: %impl.elem0 = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %Simple1.type: type = fn_type @Simple1 [concrete]
// CHECK:STDOUT:   %Simple1: %Simple1.type = struct_value () [concrete]
// CHECK:STDOUT:   %V: %J.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %Compound1.type: type = fn_type @Compound1 [concrete]
// CHECK:STDOUT:   %Compound1: %Compound1.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Simple1 = %Simple1.decl
// CHECK:STDOUT:     .Compound1 = %Compound1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Simple1.decl: %Simple1.type = fn_decl @Simple1 [concrete = constants.%Simple1] {
// CHECK:STDOUT:     %T.patt.loc8_12.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %S.patt.loc8_19.1: @Simple1.%impl.elem0.loc8_24.2 (%impl.elem0) = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc8_19.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc8_12.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %impl.elem0.loc8_24.1 [symbolic = %impl.elem0.loc8_24.2 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %T.ref: %J.type = name_ref T, %T.loc8_12.1 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc8_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_24.2: type = converted %T.ref, %T.as_type.loc8_24.1 [symbolic = %T.as_type.loc8_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.loc8_24.1: <witness> = facet_access_witness %T.ref [symbolic = %T.as_wit.loc8_24.2 (constants.%T.as_wit)]
// CHECK:STDOUT:       %impl.elem0.loc8_24.1: type = impl_witness_access %T.as_wit.loc8_24.1, element0 [symbolic = %impl.elem0.loc8_24.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc8_19.1: @Simple1.%impl.elem0.loc8_24.2 (%impl.elem0) = bind_symbolic_name S, 1 [symbolic = %S.loc8_19.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound1.decl: %Compound1.type = fn_decl @Compound1 [concrete = constants.%Compound1] {
// CHECK:STDOUT:     %V.patt.loc15_14.1: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc15_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc15_21.1: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc15_21.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref.loc15_18: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %V.loc15_14.1: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc15_14.2 (constants.%V)]
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc15_14.1 [symbolic = %V.loc15_14.2 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc15_28: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 1 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple1(%T.loc8_12.1: %J.type, %S.loc8_19.1: @Simple1.%impl.elem0.loc8_24.2 (%impl.elem0)) {
// CHECK:STDOUT:   %T.loc8_12.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_12.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_24.2: type = facet_access_type %T.loc8_12.2 [symbolic = %T.as_type.loc8_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc8_24.2: <witness> = facet_access_witness %T.loc8_12.2 [symbolic = %T.as_wit.loc8_24.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc8_24.2: type = impl_witness_access %T.as_wit.loc8_24.2, element0 [symbolic = %impl.elem0.loc8_24.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %S.loc8_19.2: %impl.elem0 = bind_symbolic_name S, 1 [symbolic = %S.loc8_19.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc8_19.2: %impl.elem0 = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc8_19.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_12.1: %J.type, %S.patt.loc8_19.1: @Simple1.%impl.elem0.loc8_24.2 (%impl.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound1(%V.loc15_14.1: %J.type, %W: <error>) {
// CHECK:STDOUT:   %V.loc15_14.2: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc15_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc15_14.2: %J.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc15_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.patt.loc15_21.2: <error> = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc15_21.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc15_14.1: %J.type, %W.patt.loc15_21.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple1(constants.%T, constants.%S) {
// CHECK:STDOUT:   %T.loc8_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_12.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_24.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.loc8_24.2 => constants.%T.as_wit
// CHECK:STDOUT:   %impl.elem0.loc8_24.2 => constants.%impl.elem0
// CHECK:STDOUT:   %S.loc8_19.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc8_19.2 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound1(constants.%V, <error>) {
// CHECK:STDOUT:   %V.loc15_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc15_14.2 => constants.%V
// CHECK:STDOUT:   %W.patt.loc15_21.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K1.type: type = facet_type <@K1> [concrete]
// CHECK:STDOUT:   %Self: %K1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Q1.type: type = fn_type @Q1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Q1: %Q1.type = struct_value () [concrete]
// CHECK:STDOUT:   %K1.assoc_type: type = assoc_entity_type %K1.type [concrete]
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, @K1.%Q1.decl [concrete]
// CHECK:STDOUT:   %T: %K1.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %K1.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Simple2.type: type = fn_type @Simple2 [concrete]
// CHECK:STDOUT:   %Simple2: %Simple2.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %K1.facet: %K1.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.aaf: type = fn_type_with_self_type %Q1.type, %K1.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.aaf = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Q1(%K1.facet) [symbolic]
// CHECK:STDOUT:   %V: %K1.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %K1.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %Compound2.type: type = fn_type @Compound2 [concrete]
// CHECK:STDOUT:   %Compound2: %Compound2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .K1 = %K1.decl
// CHECK:STDOUT:     .Simple2 = %Simple2.decl
// CHECK:STDOUT:     .Compound2 = %Compound2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K1.decl: type = interface_decl @K1 [concrete = constants.%K1.type] {} {}
// CHECK:STDOUT:   %Simple2.decl: %Simple2.type = fn_decl @Simple2 [concrete = constants.%Simple2] {
// CHECK:STDOUT:     %T.patt.loc8_12.1: %K1.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K1.ref: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     %T.loc8_12.1: %K1.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound2.decl: %Compound2.type = fn_decl @Compound2 [concrete = constants.%Compound2] {
// CHECK:STDOUT:     %V.patt.loc13_14.1: %K1.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K1.ref.loc13: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     %V.loc13_14.1: %K1.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K1 {
// CHECK:STDOUT:   %Self: %K1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Q1.decl: %Q1.type = fn_decl @Q1 [concrete = constants.%Q1] {} {}
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, %Q1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q1 = %assoc0
// CHECK:STDOUT:   witness = (%Q1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q1(@K1.%Self: %K1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple2(%T.loc8_12.1: %K1.type) {
// CHECK:STDOUT:   %T.loc8_12.2: %K1.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_12.2: %K1.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_4.2: type = facet_access_type %T.loc8_12.2 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc9_4.2: <witness> = facet_access_witness %T.loc8_12.2 [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %K1.facet: %K1.type = facet_value %T.as_type.loc9_4.2, %T.as_wit.loc9_4.2 [symbolic = %K1.facet (constants.%K1.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%Q1.type, %K1.facet [symbolic = %.loc9_4.2 (constants.%.aaf)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Simple2.%.loc9_4.2 (%.aaf) = impl_witness_access %T.as_wit.loc9_4.2, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_4.2: <specific function> = specific_function %impl.elem0.loc9_4.2, @Q1(%K1.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_12.1: %K1.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %K1.type = name_ref T, %T.loc8_12.1 [symbolic = %T.loc8_12.2 (constants.%T)]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted %T.ref, %T.as_type.loc9_4.1 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_4.1: <witness> = facet_access_witness %T.ref [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Simple2.%.loc9_4.2 (%.aaf) = impl_witness_access %T.as_wit.loc9_4.1, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc9_4.1: <specific function> = specific_function %impl.elem0.loc9_4.1, @Q1(constants.%K1.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %Q1.call: init %empty_tuple.type = call %specific_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound2(%V.loc13_14.1: %K1.type) {
// CHECK:STDOUT:   %V.loc13_14.2: %K1.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc13_14.2: %K1.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_14.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc13_14.1: %K1.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref: %K1.type = name_ref V, %V.loc13_14.1 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:     %K1.ref.loc18: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q1(constants.%K1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q1(@Simple2.%K1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound2(constants.%V) {
// CHECK:STDOUT:   %V.loc13_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc13_14.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_caller_instance_interface_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K2.type: type = facet_type <@K2> [concrete]
// CHECK:STDOUT:   %Self: %K2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Q2.type: type = fn_type @Q2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Q2: %Q2.type = struct_value () [concrete]
// CHECK:STDOUT:   %K2.assoc_type: type = assoc_entity_type %K2.type [concrete]
// CHECK:STDOUT:   %assoc0: %K2.assoc_type = assoc_entity element0, @K2.%Q2.decl [concrete]
// CHECK:STDOUT:   %T: %K2.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %K2.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Simple3.type: type = fn_type @Simple3 [concrete]
// CHECK:STDOUT:   %Simple3: %Simple3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.022e81.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %K2.facet: %K2.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.813: type = fn_type_with_self_type %Q2.type, %K2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.813 = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Q2(%K2.facet) [symbolic]
// CHECK:STDOUT:   %V: %K2.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %K2.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %Compound3.type: type = fn_type @Compound3 [concrete]
// CHECK:STDOUT:   %Compound3: %Compound3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.022e81.2: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .K2 = %K2.decl
// CHECK:STDOUT:     .Simple3 = %Simple3.decl
// CHECK:STDOUT:     .Compound3 = %Compound3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K2.decl: type = interface_decl @K2 [concrete = constants.%K2.type] {} {}
// CHECK:STDOUT:   %Simple3.decl: %Simple3.type = fn_decl @Simple3 [concrete = constants.%Simple3] {
// CHECK:STDOUT:     %T.patt.loc11_12.1: %K2.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Simple3.%T.as_type.loc11_23.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Simple3.%T.as_type.loc11_23.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K2.ref: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     %T.loc11_12.1: %K2.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Simple3.%T.as_type.loc11_23.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %.loc11_23.2 [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %K2.type = name_ref T, %T.loc11_12.1 [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc11_23.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_23.2: type = converted %T.ref, %T.as_type.loc11_23.1 [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple3.%T.as_type.loc11_23.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound3.decl: %Compound3.type = fn_decl @Compound3 [concrete = constants.%Compound3] {
// CHECK:STDOUT:     %V.patt.loc17_14.1: %K2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc17_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %y.patt: @Compound3.%V.as_type.loc17_25.2 (%V.as_type) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @Compound3.%V.as_type.loc17_25.2 (%V.as_type) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K2.ref.loc17: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     %V.loc17_14.1: %K2.type = bind_symbolic_name V, 0 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound3.%V.as_type.loc17_25.2 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_25.1: type = splice_block %.loc17_25.2 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %K2.type = name_ref V, %V.loc17_14.1 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc17_25.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc17_25.2: type = converted %V.ref, %V.as_type.loc17_25.1 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound3.%V.as_type.loc17_25.2 (%V.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K2 {
// CHECK:STDOUT:   %Self: %K2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Q2.decl: %Q2.type = fn_decl @Q2 [concrete = constants.%Q2] {} {}
// CHECK:STDOUT:   %assoc0: %K2.assoc_type = assoc_entity element0, %Q2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q2 = %assoc0
// CHECK:STDOUT:   witness = (%Q2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q2(@K2.%Self: %K2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple3(%T.loc11_12.1: %K2.type) {
// CHECK:STDOUT:   %T.loc11_12.2: %K2.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_12.2: %K2.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc11_23.2: type = facet_access_type %T.loc11_12.2 [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Simple3.%T.as_type.loc11_23.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.022e81.1)]
// CHECK:STDOUT:   %T.as_wit.loc12_4.2: <witness> = facet_access_witness %T.loc11_12.2 [symbolic = %T.as_wit.loc12_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %K2.facet: %K2.type = facet_value %T.as_type.loc11_23.2, %T.as_wit.loc12_4.2 [symbolic = %K2.facet (constants.%K2.facet)]
// CHECK:STDOUT:   %.loc12_4.2: type = fn_type_with_self_type constants.%Q2.type, %K2.facet [symbolic = %.loc12_4.2 (constants.%.813)]
// CHECK:STDOUT:   %impl.elem0.loc12_4.2: @Simple3.%.loc12_4.2 (%.813) = impl_witness_access %T.as_wit.loc12_4.2, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc12_4.2: <specific function> = specific_function %impl.elem0.loc12_4.2, @Q2(%K2.facet) [symbolic = %specific_fn.loc12_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_12.1: %K2.type, %x.param_patt: @Simple3.%T.as_type.loc11_23.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Simple3.%T.as_type.loc11_23.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc12: type = facet_access_type constants.%T [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_4.1: type = converted constants.%T, %T.as_type.loc12 [symbolic = %T.as_type.loc11_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc12_4.1: <witness> = facet_access_witness constants.%T [symbolic = %T.as_wit.loc12_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc12_4.1: @Simple3.%.loc12_4.2 (%.813) = impl_witness_access %T.as_wit.loc12_4.1, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc12_4.1: <specific function> = specific_function %impl.elem0.loc12_4.1, @Q2(constants.%K2.facet) [symbolic = %specific_fn.loc12_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %Q2.call: init %empty_tuple.type = call %specific_fn.loc12_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound3(%V.loc17_14.1: %K2.type) {
// CHECK:STDOUT:   %V.loc17_14.2: %K2.type = bind_symbolic_name V, 0 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc17_14.2: %K2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc17_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.as_type.loc17_25.2: type = facet_access_type %V.loc17_14.2 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Compound3.%V.as_type.loc17_25.2 (%V.as_type) [symbolic = %require_complete (constants.%require_complete.022e81.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc17_14.1: %K2.type, %y.param_patt: @Compound3.%V.as_type.loc17_25.2 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Compound3.%V.as_type.loc17_25.2 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     %K2.ref.loc22: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple3(constants.%T) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc11_23.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q2(constants.%K2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q2(@Simple3.%K2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound3(constants.%V) {
// CHECK:STDOUT:   %V.loc17_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc17_14.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc17_25.2 => constants.%V.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L1.type: type = facet_type <@L1> [concrete]
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %R1.type: type = fn_type @R1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %R1: %R1.type = struct_value () [concrete]
// CHECK:STDOUT:   %L1.assoc_type: type = assoc_entity_type %L1.type [concrete]
// CHECK:STDOUT:   %assoc0: %L1.assoc_type = assoc_entity element0, @L1.%R1.decl [concrete]
// CHECK:STDOUT:   %ptr.72f01d.1: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %S1.type: type = fn_type @S1 [concrete]
// CHECK:STDOUT:   %S1: %S1.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, @L1.%S1.decl [concrete]
// CHECK:STDOUT:   %T: %L1.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %L1.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Simple4.type: type = fn_type @Simple4 [concrete]
// CHECK:STDOUT:   %Simple4: %Simple4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.97cd40.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %L1.facet: %L1.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.1e8: type = fn_type_with_self_type %R1.type, %L1.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.1e8 = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %.d0e: type = fn_type_with_self_type %S1.type, %L1.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.d0e = impl_witness_access %T.as_wit, element1 [symbolic]
// CHECK:STDOUT:   %ptr.72f01d.2: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.932: <witness> = require_complete_type %ptr.72f01d.2 [symbolic]
// CHECK:STDOUT:   %V: %L1.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %L1.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %Compound4.type: type = fn_type @Compound4 [concrete]
// CHECK:STDOUT:   %Compound4: %Compound4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.97cd40.2: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L1 = %L1.decl
// CHECK:STDOUT:     .Simple4 = %Simple4.decl
// CHECK:STDOUT:     .Compound4 = %Compound4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L1.decl: type = interface_decl @L1 [concrete = constants.%L1.type] {} {}
// CHECK:STDOUT:   %Simple4.decl: %Simple4.type = fn_decl @Simple4 [concrete = constants.%Simple4] {
// CHECK:STDOUT:     %T.patt.loc10_12.1: %L1.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L1.ref: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %T.loc10_12.1: %L1.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_23.1: type = splice_block %.loc10_23.2 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc10: %L1.type = name_ref T, %T.loc10_12.1 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc10_23.1: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_23.2: type = converted %T.ref.loc10, %T.as_type.loc10_23.1 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound4.decl: %Compound4.type = fn_decl @Compound4 [concrete = constants.%Compound4] {
// CHECK:STDOUT:     %V.patt.loc17_14.1: %L1.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc17_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %y.patt: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L1.ref.loc17: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %V.loc17_14.1: %L1.type = bind_symbolic_name V, 0 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_25.1: type = splice_block %.loc17_25.2 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref.loc17: %L1.type = name_ref V, %V.loc17_14.1 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc17_25.1: type = facet_access_type %V.ref.loc17 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc17_25.2: type = converted %V.ref.loc17, %V.as_type.loc17_25.1 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L1 {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %R1.decl: %R1.type = fn_decl @R1 [concrete = constants.%R1] {
// CHECK:STDOUT:     %self.patt: @R1.%Self.as_type.loc4_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @R1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R1.%Self.as_type.loc4_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L1.assoc_type = assoc_entity element0, %R1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %S1.decl: %S1.type = fn_decl @S1 [concrete = constants.%S1] {
// CHECK:STDOUT:     %self.patt: @S1.%ptr.loc5_24.1 (%ptr.72f01d.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @S1.%ptr.loc5_24.1 (%ptr.72f01d.1) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc5_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @S1.%ptr.loc5_24.1 (%ptr.72f01d.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5_24.1: type = splice_block %ptr.loc5_24.2 [symbolic = %ptr.loc5_24.1 (constants.%ptr.72f01d.1)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_24.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_24.2: type = converted %Self.ref, %Self.as_type.loc5_24.2 [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc5_24.2: type = ptr_type %Self.as_type [symbolic = %ptr.loc5_24.1 (constants.%ptr.72f01d.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @S1.%ptr.loc5_24.1 (%ptr.72f01d.1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, %S1.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R1 = %assoc0
// CHECK:STDOUT:   .S1 = %assoc1
// CHECK:STDOUT:   witness = (%R1.decl, %S1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @R1.%Self.as_type.loc4_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @S1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_24.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc5_24.1: type = ptr_type @S1.%Self.as_type.loc5_24.1 (%Self.as_type) [symbolic = %ptr.loc5_24.1 (constants.%ptr.72f01d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @S1.%ptr.loc5_24.1 (%ptr.72f01d.1)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple4(%T.loc10_12.1: %L1.type) {
// CHECK:STDOUT:   %T.loc10_12.2: %L1.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_12.2: %L1.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc10_23.2: type = facet_access_type %T.loc10_12.2 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type @Simple4.%T.as_type.loc10_23.2 (%T.as_type) [symbolic = %require_complete.loc10 (constants.%require_complete.97cd40.1)]
// CHECK:STDOUT:   %T.as_wit.loc11_4.2: <witness> = facet_access_witness %T.loc10_12.2 [symbolic = %T.as_wit.loc11_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %L1.facet: %L1.type = facet_value %T.as_type.loc10_23.2, %T.as_wit.loc11_4.2 [symbolic = %L1.facet (constants.%L1.facet)]
// CHECK:STDOUT:   %.loc11_4.2: type = fn_type_with_self_type constants.%R1.type, %L1.facet [symbolic = %.loc11_4.2 (constants.%.1e8)]
// CHECK:STDOUT:   %impl.elem0.loc11_4.2: @Simple4.%.loc11_4.2 (%.1e8) = impl_witness_access %T.as_wit.loc11_4.2, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %.loc13_8.2: type = fn_type_with_self_type constants.%S1.type, %L1.facet [symbolic = %.loc13_8.2 (constants.%.d0e)]
// CHECK:STDOUT:   %impl.elem1.loc13_8.2: @Simple4.%.loc13_8.2 (%.d0e) = impl_witness_access %T.as_wit.loc11_4.2, element1 [symbolic = %impl.elem1.loc13_8.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %ptr: type = ptr_type @Simple4.%T.as_type.loc10_23.2 (%T.as_type) [symbolic = %ptr (constants.%ptr.72f01d.2)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type @Simple4.%ptr (%ptr.72f01d.2) [symbolic = %require_complete.loc13 (constants.%require_complete.932)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc10_12.1: %L1.type, %x.param_patt: @Simple4.%T.as_type.loc10_23.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc11: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc11: type = facet_access_type constants.%T [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_4.1: type = converted constants.%T, %T.as_type.loc11 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc11_4.1: <witness> = facet_access_witness constants.%T [symbolic = %T.as_wit.loc11_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc11_4.1: @Simple4.%.loc11_4.2 (%.1e8) = impl_witness_access %T.as_wit.loc11_4.1, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11: <bound method> = bound_method %x.ref.loc11, %impl.elem0.loc11_4.1
// CHECK:STDOUT:     %specific_fn.loc11: <specific function> = specific_function %bound_method.loc11, @R1(constants.%L1.facet)
// CHECK:STDOUT:     %R1.call: init %empty_tuple.type = call %specific_fn.loc11(%x.ref.loc11)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %xcopy.patt: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = binding_pattern xcopy
// CHECK:STDOUT:       %.loc12_3: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = var_pattern %xcopy.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xcopy.var: ref @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = var xcopy
// CHECK:STDOUT:     %x.ref.loc12: @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     assign %xcopy.var, %x.ref.loc12
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc12: %L1.type = name_ref T, %T.loc10_12.1 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc12: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %T.ref.loc12, %T.as_type.loc12 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xcopy: ref @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = bind_name xcopy, %xcopy.var
// CHECK:STDOUT:     %xcopy.ref: ref @Simple4.%T.as_type.loc10_23.2 (%T.as_type) = name_ref xcopy, %xcopy
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type constants.%T [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_8.1: type = converted constants.%T, %T.as_type.loc13 [symbolic = %T.as_type.loc10_23.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc13: <witness> = facet_access_witness constants.%T [symbolic = %T.as_wit.loc11_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem1.loc13_8.1: @Simple4.%.loc13_8.2 (%.d0e) = impl_witness_access %T.as_wit.loc13, element1 [symbolic = %impl.elem1.loc13_8.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc13: <bound method> = bound_method %xcopy.ref, %impl.elem1.loc13_8.1
// CHECK:STDOUT:     %specific_fn.loc13: <specific function> = specific_function %bound_method.loc13, @S1(constants.%L1.facet)
// CHECK:STDOUT:     %addr: @Simple4.%ptr (%ptr.72f01d.2) = addr_of %xcopy.ref
// CHECK:STDOUT:     %S1.call: init %empty_tuple.type = call %specific_fn.loc13(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound4(%V.loc17_14.1: %L1.type) {
// CHECK:STDOUT:   %V.loc17_14.2: %L1.type = bind_symbolic_name V, 0 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc17_14.2: %L1.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc17_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.as_type.loc17_25.2: type = facet_access_type %V.loc17_14.2 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Compound4.%V.as_type.loc17_25.2 (%V.as_type) [symbolic = %require_complete (constants.%require_complete.97cd40.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc17_14.1: %L1.type, %y.param_patt: @Compound4.%V.as_type.loc17_25.2 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref.loc22: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     %L1.ref.loc22: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %ycopy.patt: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = binding_pattern ycopy
// CHECK:STDOUT:       %.loc23_3: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = var_pattern %ycopy.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ycopy.var: ref @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = var ycopy
// CHECK:STDOUT:     %y.ref.loc23: @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     assign %ycopy.var, %y.ref.loc23
// CHECK:STDOUT:     %.loc23_14.1: type = splice_block %.loc23_14.2 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref.loc23: %L1.type = name_ref V, %V.loc17_14.1 [symbolic = %V.loc17_14.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc23: type = facet_access_type %V.ref.loc23 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc23_14.2: type = converted %V.ref.loc23, %V.as_type.loc23 [symbolic = %V.as_type.loc17_25.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ycopy: ref @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = bind_name ycopy, %ycopy.var
// CHECK:STDOUT:     %ycopy.ref: ref @Compound4.%V.as_type.loc17_25.2 (%V.as_type) = name_ref ycopy, %ycopy
// CHECK:STDOUT:     %L1.ref.loc28: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @S1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_24.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc5_24.1 => constants.%ptr.72f01d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple4(constants.%T) {
// CHECK:STDOUT:   %T.loc10_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_12.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc10_23.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R1(constants.%L1.facet) {
// CHECK:STDOUT:   %Self => constants.%L1.facet
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @S1(constants.%L1.facet) {
// CHECK:STDOUT:   %Self => constants.%L1.facet
// CHECK:STDOUT:   %Self.as_type.loc5_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc5_24.1 => constants.%ptr.72f01d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound4(constants.%V) {
// CHECK:STDOUT:   %V.loc17_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc17_14.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc17_25.2 => constants.%V.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_interface_instance_caller_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L2.type: type = facet_type <@L2> [concrete]
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %R2.type: type = fn_type @R2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %R2: %R2.type = struct_value () [concrete]
// CHECK:STDOUT:   %L2.assoc_type: type = assoc_entity_type %L2.type [concrete]
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%R2.decl [concrete]
// CHECK:STDOUT:   %ptr.f3e54b.1: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %S2.type: type = fn_type @S2 [concrete]
// CHECK:STDOUT:   %S2: %S2.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, @L2.%S2.decl [concrete]
// CHECK:STDOUT:   %T: %L2.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %L2.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Simple5.type: type = fn_type @Simple5 [concrete]
// CHECK:STDOUT:   %Simple5: %Simple5.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %L2.facet: %L2.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.973: type = fn_type_with_self_type %R2.type, %L2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.973 = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn.e11: <specific function> = specific_function %impl.elem0, @R2(%L2.facet) [symbolic]
// CHECK:STDOUT:   %.3ca: type = fn_type_with_self_type %S2.type, %L2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.3ca = impl_witness_access %T.as_wit, element1 [symbolic]
// CHECK:STDOUT:   %ptr.f3e54b.2: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %specific_fn.cf2: <specific function> = specific_function %impl.elem1, @S2(%L2.facet) [symbolic]
// CHECK:STDOUT:   %V: %L2.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %L2.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %Compound5.type: type = fn_type @Compound5 [concrete]
// CHECK:STDOUT:   %Compound5: %Compound5.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L2 = %L2.decl
// CHECK:STDOUT:     .Simple5 = %Simple5.decl
// CHECK:STDOUT:     .Compound5 = %Compound5.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L2.decl: type = interface_decl @L2 [concrete = constants.%L2.type] {} {}
// CHECK:STDOUT:   %Simple5.decl: %Simple5.type = fn_decl @Simple5 [concrete = constants.%Simple5] {
// CHECK:STDOUT:     %T.patt.loc10_12.1: %L2.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L2.ref: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %T.loc10_12.1: %L2.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound5.decl: %Compound5.type = fn_decl @Compound5 [concrete = constants.%Compound5] {
// CHECK:STDOUT:     %V.patt.loc30_14.1: %L2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc30_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L2.ref.loc30: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %V.loc30_14.1: %L2.type = bind_symbolic_name V, 0 [symbolic = %V.loc30_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L2 {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %R2.decl: %R2.type = fn_decl @R2 [concrete = constants.%R2] {
// CHECK:STDOUT:     %self.patt: @R2.%Self.as_type.loc5_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @R2.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R2.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_15.1: type = splice_block %.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_15.2: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R2.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, %R2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %S2.decl: %S2.type = fn_decl @S2 [concrete = constants.%S2] {
// CHECK:STDOUT:     %self.patt: @S2.%ptr.loc6_24.1 (%ptr.f3e54b.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @S2.%ptr.loc6_24.1 (%ptr.f3e54b.1) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc6_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @S2.%ptr.loc6_24.1 (%ptr.f3e54b.1) = value_param call_param0
// CHECK:STDOUT:     %.loc6_24.1: type = splice_block %ptr.loc6_24.2 [symbolic = %ptr.loc6_24.1 (constants.%ptr.f3e54b.1)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc6_24.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_24.2: type = converted %Self.ref, %Self.as_type.loc6_24.2 [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc6_24.2: type = ptr_type %Self.as_type [symbolic = %ptr.loc6_24.1 (constants.%ptr.f3e54b.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @S2.%ptr.loc6_24.1 (%ptr.f3e54b.1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, %S2.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R2 = %assoc0
// CHECK:STDOUT:   .S2 = %assoc1
// CHECK:STDOUT:   witness = (%R2.decl, %S2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @R2.%Self.as_type.loc5_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @S2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc6_24.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc6_24.1: type = ptr_type @S2.%Self.as_type.loc6_24.1 (%Self.as_type) [symbolic = %ptr.loc6_24.1 (constants.%ptr.f3e54b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @S2.%ptr.loc6_24.1 (%ptr.f3e54b.1)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple5(%T.loc10_12.1: %L2.type) {
// CHECK:STDOUT:   %T.loc10_12.2: %L2.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_12.2: %L2.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc18_4.2: type = facet_access_type %T.loc10_12.2 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc18_4.2: <witness> = facet_access_witness %T.loc10_12.2 [symbolic = %T.as_wit.loc18_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %L2.facet: %L2.type = facet_value %T.as_type.loc18_4.2, %T.as_wit.loc18_4.2 [symbolic = %L2.facet (constants.%L2.facet)]
// CHECK:STDOUT:   %.loc18_4.2: type = fn_type_with_self_type constants.%R2.type, %L2.facet [symbolic = %.loc18_4.2 (constants.%.973)]
// CHECK:STDOUT:   %impl.elem0.loc18_4.2: @Simple5.%.loc18_4.2 (%.973) = impl_witness_access %T.as_wit.loc18_4.2, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc18_4.2: <specific function> = specific_function %impl.elem0.loc18_4.2, @R2(%L2.facet) [symbolic = %specific_fn.loc18_4.2 (constants.%specific_fn.e11)]
// CHECK:STDOUT:   %.loc26_4.2: type = fn_type_with_self_type constants.%S2.type, %L2.facet [symbolic = %.loc26_4.2 (constants.%.3ca)]
// CHECK:STDOUT:   %impl.elem1.loc26_4.2: @Simple5.%.loc26_4.2 (%.3ca) = impl_witness_access %T.as_wit.loc18_4.2, element1 [symbolic = %impl.elem1.loc26_4.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_fn.loc26_4.2: <specific function> = specific_function %impl.elem1.loc26_4.2, @S2(%L2.facet) [symbolic = %specific_fn.loc26_4.2 (constants.%specific_fn.cf2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc10_12.1: %L2.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc18: %L2.type = name_ref T, %T.loc10_12.1 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc18_4.1: type = facet_access_type %T.ref.loc18 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_4.1: type = converted %T.ref.loc18, %T.as_type.loc18_4.1 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc18_4.1: <witness> = facet_access_witness %T.ref.loc18 [symbolic = %T.as_wit.loc18_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc18_4.1: @Simple5.%.loc18_4.2 (%.973) = impl_witness_access %T.as_wit.loc18_4.1, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc18_4.1: <specific function> = specific_function %impl.elem0.loc18_4.1, @R2(constants.%L2.facet) [symbolic = %specific_fn.loc18_4.2 (constants.%specific_fn.e11)]
// CHECK:STDOUT:     %R2.call: init %empty_tuple.type = call %specific_fn.loc18_4.1(<error>)
// CHECK:STDOUT:     %T.ref.loc26: %L2.type = name_ref T, %T.loc10_12.1 [symbolic = %T.loc10_12.2 (constants.%T)]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_4.1: type = converted %T.ref.loc26, %T.as_type.loc26 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc26: <witness> = facet_access_witness %T.ref.loc26 [symbolic = %T.as_wit.loc18_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem1.loc26_4.1: @Simple5.%.loc26_4.2 (%.3ca) = impl_witness_access %T.as_wit.loc26, element1 [symbolic = %impl.elem1.loc26_4.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %specific_fn.loc26_4.1: <specific function> = specific_function %impl.elem1.loc26_4.1, @S2(constants.%L2.facet) [symbolic = %specific_fn.loc26_4.2 (constants.%specific_fn.cf2)]
// CHECK:STDOUT:     %addr: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:     %S2.call: init %empty_tuple.type = call %specific_fn.loc26_4.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound5(%V.loc30_14.1: %L2.type) {
// CHECK:STDOUT:   %V.loc30_14.2: %L2.type = bind_symbolic_name V, 0 [symbolic = %V.loc30_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc30_14.2: %L2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc30_14.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc30_14.1: %L2.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref.loc35: %L2.type = name_ref V, %V.loc30_14.1 [symbolic = %V.loc30_14.2 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc35: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %V.ref.loc40: %L2.type = name_ref V, %V.loc30_14.1 [symbolic = %V.loc30_14.2 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc40: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @S2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc6_24.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc6_24.1 => constants.%ptr.f3e54b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple5(constants.%T) {
// CHECK:STDOUT:   %T.loc10_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R2(constants.%L2.facet) {
// CHECK:STDOUT:   %Self => constants.%L2.facet
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @S2(constants.%L2.facet) {
// CHECK:STDOUT:   %Self => constants.%L2.facet
// CHECK:STDOUT:   %Self.as_type.loc6_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_24.1 => constants.%ptr.f3e54b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R2(@Simple5.%L2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @S2(@Simple5.%L2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound5(constants.%V) {
// CHECK:STDOUT:   %V.loc30_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc30_14.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.c3a: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G.11a: %G.type.c3a = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0.773: %A.assoc_type = assoc_entity element0, @A.%G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.a7a: <witness> = impl_witness (@impl.77d.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.486: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.001: %G.type.486 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, %impl_witness.a7a [concrete]
// CHECK:STDOUT:   %Fails.type: type = fn_type @Fails [concrete]
// CHECK:STDOUT:   %Fails: %Fails.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.a63: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.1, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.1, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, %impl_witness.3ea [concrete]
// CHECK:STDOUT:   %.d4d: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %A.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.bound, @Op.1(type) [concrete]
// CHECK:STDOUT:   %.faa: type = fn_type_with_self_type %G.type.c3a, %A.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//default, inst100 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//default, loc12_41, loaded [concrete = constants.%assoc0.a63]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//default, loc15_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//default, loc15_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//default, loc15_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//default, loc16_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//default, inst100 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Fails = %Fails.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.77d [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.77d.%G.decl) [concrete = constants.%impl_witness.a7a]
// CHECK:STDOUT:   %Fails.decl: %Fails.type = fn_decl @Fails [concrete = constants.%Fails] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %G.decl: %G.type.c3a = fn_decl @G.1 [concrete = constants.%G.11a] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.773]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77d: %C.ref as %A.ref {
// CHECK:STDOUT:   %G.decl: %G.type.486 = fn_decl @G.2 [concrete = constants.%G.001] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Fails() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_5.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc17_5.3: init %C = class_init (), %.loc17_5.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc17_5.4: ref %C = temporary %.loc17_5.2, %.loc17_5.3
// CHECK:STDOUT:   %.loc17_7: ref %C = converted %.loc17_5.1, %.loc17_5.4
// CHECK:STDOUT:   %A.ref.loc17_15: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc17_19: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc17_17: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc17: <bound method> = bound_method %A.ref.loc17_15, %impl.elem0.loc17_17 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %bound_method.loc17, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc17: init type = call %specific_fn.loc17(%A.ref.loc17_15, %A.ref.loc17_19) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc17: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   %impl.elem0.loc17_12: %.faa = impl_witness_access constants.%impl_witness.a7a, element0 [concrete = constants.%G.001]
// CHECK:STDOUT:   %G.call.loc17: init %empty_tuple.type = call %impl.elem0.loc17_12()
// CHECK:STDOUT:   %.loc18_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc18_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc18_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_6.3: init %C = class_init (), %.loc18_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc18_6.4: ref %C = temporary %.loc18_6.2, %.loc18_6.3
// CHECK:STDOUT:   %.loc18_8: ref %C = converted %.loc18_6.1, %.loc18_6.4
// CHECK:STDOUT:   %C.ref.loc18_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc18_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc18_28: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc18_26: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc18_26: <bound method> = bound_method %A.ref.loc18_24, %impl.elem0.loc18_26 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc18_26: <specific function> = specific_function %bound_method.loc18_26, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc18_26: init type = call %specific_fn.loc18_26(%A.ref.loc18_24, %A.ref.loc18_28) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc18_29.1: type = value_of_initializer %type.and.loc18_26 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc18_29.2: type = converted %type.and.loc18_26, %.loc18_29.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc18: %A.type = facet_value constants.%C, constants.%impl_witness.a7a [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc18_20: %A.type = converted %C.ref.loc18_18, %A.facet.loc18 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc18: type = facet_access_type %.loc18_20 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc18_30: type = converted %.loc18_20, %as_type.loc18 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc18_35: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc18_39: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc18_37: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc18_37: <bound method> = bound_method %A.ref.loc18_35, %impl.elem0.loc18_37 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc18_37: <specific function> = specific_function %bound_method.loc18_37, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc18_37: init type = call %specific_fn.loc18_37(%A.ref.loc18_35, %A.ref.loc18_39) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc18: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   %impl.elem0.loc18_32: %.faa = impl_witness_access constants.%impl_witness.a7a, element0 [concrete = constants.%G.001]
// CHECK:STDOUT:   %G.call.loc18: init %empty_tuple.type = call %impl.elem0.loc18_32()
// CHECK:STDOUT:   %.loc19_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc19_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc19_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_6.3: init %C = class_init (), %.loc19_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_6.4: ref %C = temporary %.loc19_6.2, %.loc19_6.3
// CHECK:STDOUT:   %.loc19_8: ref %C = converted %.loc19_6.1, %.loc19_6.4
// CHECK:STDOUT:   %C.ref.loc19_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc19_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc19_28: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc19_26: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %A.ref.loc19_24, %impl.elem0.loc19_26 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %bound_method.loc19, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc19: init type = call %specific_fn.loc19(%A.ref.loc19_24, %A.ref.loc19_28) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc19_29.1: type = value_of_initializer %type.and.loc19 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc19_29.2: type = converted %type.and.loc19, %.loc19_29.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc19: %A.type = facet_value constants.%C, constants.%impl_witness.a7a [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc19_20: %A.type = converted %C.ref.loc19_18, %A.facet.loc19 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc19: type = facet_access_type %.loc19_20 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc19_30: type = converted %.loc19_20, %as_type.loc19 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc19_34: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc19: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   %impl.elem0.loc19_32: %.faa = impl_witness_access constants.%impl_witness.a7a, element0 [concrete = constants.%G.001]
// CHECK:STDOUT:   %G.call.loc19: init %empty_tuple.type = call %impl.elem0.loc19_32()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.5ab3ec.2: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%T (%T)](%other.param_patt: @Op.1.%T (%T)) -> @Op.1.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.040: %BitAnd.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.2.%Self.as_type (%Self.as_type)) -> @Op.2.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_allowed_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.c3a: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.11a: %G.type.c3a = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0.773: %A.assoc_type = assoc_entity element0, @A.%G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.a7a: <witness> = impl_witness (@impl.77d.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.486: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.001: %G.type.486 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, %impl_witness.a7a [concrete]
// CHECK:STDOUT:   %Works.type: type = fn_type @Works [concrete]
// CHECK:STDOUT:   %Works: %Works.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.a63: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.1, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.1, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, %impl_witness.3ea [concrete]
// CHECK:STDOUT:   %.d4d: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %A.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.bound, @Op.1(type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//default, inst100 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//default, loc12_41, loaded [concrete = constants.%assoc0.a63]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//default, loc15_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//default, loc15_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//default, loc15_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//default, loc16_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//default, inst100 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Works = %Works.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.77d [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.77d.%G.decl) [concrete = constants.%impl_witness.a7a]
// CHECK:STDOUT:   %Works.decl: %Works.type = fn_decl @Works [concrete = constants.%Works] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %G.decl: %G.type.c3a = fn_decl @G.1 [concrete = constants.%G.11a] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.773]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77d: %C.ref as %A.ref {
// CHECK:STDOUT:   %G.decl: %G.type.486 = fn_decl @G.2 [concrete = constants.%G.001] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Works() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc19_7: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc19_11: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc19: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %A.ref.loc19_7, %impl.elem0.loc19 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %bound_method.loc19, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc19: init type = call %specific_fn.loc19(%A.ref.loc19_7, %A.ref.loc19_11) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc19: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc24_10: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc24_14: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc24_12: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc24_12: <bound method> = bound_method %A.ref.loc24_10, %impl.elem0.loc24_12 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc24_12: <specific function> = specific_function %bound_method.loc24_12, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc24_12: init type = call %specific_fn.loc24_12(%A.ref.loc24_10, %A.ref.loc24_14) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc24_15.1: type = value_of_initializer %type.and.loc24_12 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc24_15.2: type = converted %type.and.loc24_12, %.loc24_15.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc24: %A.type = facet_value constants.%C, constants.%impl_witness.a7a [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc24_6: %A.type = converted %C.ref.loc24, %A.facet.loc24 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc24_20: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc24_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc24_22: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc24_22: <bound method> = bound_method %A.ref.loc24_20, %impl.elem0.loc24_22 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc24_22: <specific function> = specific_function %bound_method.loc24_22, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc24_22: init type = call %specific_fn.loc24_22(%A.ref.loc24_20, %A.ref.loc24_24) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc24: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   %C.ref.loc29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc29_10: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc29_14: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc29: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %A.ref.loc29_10, %impl.elem0.loc29 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %bound_method.loc29, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %type.and.loc29: init type = call %specific_fn.loc29(%A.ref.loc29_10, %A.ref.loc29_14) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc29_15.1: type = value_of_initializer %type.and.loc29 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc29_15.2: type = converted %type.and.loc29, %.loc29_15.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc29: %A.type = facet_value constants.%C, constants.%impl_witness.a7a [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc29_6: %A.type = converted %C.ref.loc29, %A.facet.loc29 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc29_19: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc29: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.773]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.5ab3ec.2: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%T (%T)](%other.param_patt: @Op.1.%T (%T)) -> @Op.1.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.040: %BitAnd.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.2.%Self.as_type (%Self.as_type)) -> @Op.2.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
