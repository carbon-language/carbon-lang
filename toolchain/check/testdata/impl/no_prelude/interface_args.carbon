// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/interface_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/interface_args.carbon

// --- core.carbon
package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- action.carbon

library "[[@TEST_NAME]]";

interface Action(T:! type) {
  fn Op[self: Self]();
}

class A {}
class B {}
class C {}

impl A as Action(B) {
  fn Op[self: Self]() {}
}

fn F(a: A) { a.(Action(B).Op)(); }

// --- action.impl.carbon

impl library "[[@TEST_NAME]]";

fn G(a: A) { a.(Action(B).Op)(); }

// --- fail_action.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_action.impl.carbon:[[@LINE+4]]:14: error: cannot access member of interface `Action(C)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: fn G(a: A) { a.(Action(C).Op)(); }
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(a: A) { a.(Action(C).Op)(); }

// --- factory.carbon

library "[[@TEST_NAME]]";

interface Factory(T:! type) {
  // Non-instance class function
  fn Make() -> T;
  // Instance method
  fn Method[self: Self]() -> T;
}

class A {}
class B {}

impl A as Factory(B) {
  fn Make() -> B;
  fn Method[self: Self]() -> B;
}

// --- factory.impl.carbon

impl library "[[@TEST_NAME]]";

fn MakeB() -> B {
  return A.(Factory(B).Make)();
}
fn InstanceB(a: A) -> B {
  return a.(Factory(B).Method)();
}

// --- fail_factory.impl.carbon

impl library "[[@TEST_NAME]]";

import Core;

class C {}

fn MakeC() -> C {
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+7]]:10: error: cannot implicitly convert value of type `type` to `Factory(C)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return A.(Factory(C).Make)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+4]]:10: note: type `type` does not implement interface `Core.ImplicitAs(Factory(C))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return A.(Factory(C).Make)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return A.(Factory(C).Make)();
}

fn InstanceC(a: A) -> C {
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Factory(C)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a.(Factory(C).Method)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a.(Factory(C).Method)();
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
// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.8f9: type = assoc_entity_type %Action.type.cca [symbolic]
// CHECK:STDOUT:   %assoc0.69b: %Action.assoc_type.8f9 = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op.1, @Action(%B) [concrete]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.827: type = assoc_entity_type %Action.type.cb0 [concrete]
// CHECK:STDOUT:   %assoc0.035: %Action.assoc_type.827 = assoc_entity element0, @Action.%Op.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [concrete]
// CHECK:STDOUT:   %Op.type.4b4: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.40d: %Op.type.4b4 = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.cb0 = facet_value %A, (%impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dde: type = fn_type_with_self_type %Op.type.54d, %Action.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = %Action.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Action.decl: %Action.type.29c = interface_decl @Action [concrete = constants.%Action.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self.2: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.8f9)]
// CHECK:STDOUT:   %assoc0.loc5_22.2: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.69b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.036) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.6ed)] {
// CHECK:STDOUT:       %self.patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @Op.1.%Action.type (%Action.type.cca) = specific_constant @Action.%Self.1, @Action(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Op.1.%Action.type (%Action.type.cca) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_22.1: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.69b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_22.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.4b4 = fn_decl @Op.2 [concrete = constants.%Op.40d] {
// CHECK:STDOUT:     %self.patt: %A = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %A = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.cca)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %A]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.827 = specific_constant @Action.%assoc0.loc5_22.1, @Action(constants.%B) [concrete = constants.%assoc0.035]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.827 = name_ref Op, %.loc16 [concrete = constants.%assoc0.035]
// CHECK:STDOUT:   %impl.elem0: %.dde = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.40d]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Op.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T.loc4_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.54d
// CHECK:STDOUT:   %Op => constants.%Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.827
// CHECK:STDOUT:   %assoc0.loc5_22.2 => constants.%assoc0.035
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.8f9: type = assoc_entity_type %Action.type.cca [symbolic]
// CHECK:STDOUT:   %assoc0.905ab9.1: %Action.assoc_type.8f9 = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op.1, @Action(%B) [concrete]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.827: type = assoc_entity_type %Action.type.cb0 [concrete]
// CHECK:STDOUT:   %assoc0.4cc: %Action.assoc_type.827 = assoc_entity element0, imports.%Main.import_ref.1f6 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.905ab9.2: %Action.assoc_type.8f9 = assoc_entity element0, imports.%Main.import_ref.0e3753.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.d0b) [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.cb0 = facet_value %A, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.584: type = fn_type_with_self_type %Op.type.54d, %Action.facet [concrete]
// CHECK:STDOUT:   %Op.type.4b4: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.40d: %Op.type.4b4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.29c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//action, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst58 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ddc = import_ref Main//action, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.318: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.905ab9.2)]
// CHECK:STDOUT:   %Main.Op: @Action.%Op.type (%Op.type.036) = import_ref Main//action, Op, loaded [symbolic = @Action.%Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Main.import_ref.71c: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst53 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %Main.import_ref.7b5 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.835: @Action.%Action.type (%Action.type.cca) = import_ref Main//action, inst24 [no loc], loaded [symbolic = @Action.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.0e3753.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.5ab3ec.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.8f9)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic = %assoc0 (constants.%assoc0.905ab9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.318
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7b5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.71c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.835: @Action.%Action.type (%Action.type.cca)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.827 = specific_constant imports.%Main.import_ref.318, @Action(constants.%B) [concrete = constants.%assoc0.4cc]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.827 = name_ref Op, %.loc4 [concrete = constants.%assoc0.4cc]
// CHECK:STDOUT:   %impl.elem0: %.584 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.40d]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %A]() [from "action.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %T.patt => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.54d
// CHECK:STDOUT:   %Op => constants.%Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.827
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Op.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.8f9: type = assoc_entity_type %Action.type.cca [symbolic]
// CHECK:STDOUT:   %assoc0.905ab9.1: %Action.assoc_type.8f9 = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.827: type = assoc_entity_type %Action.type.cb0 [concrete]
// CHECK:STDOUT:   %assoc0.8f8: %Action.assoc_type.827 = assoc_entity element0, imports.%Main.import_ref.0e3753.2 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.e2b: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.5ad: type = fn_type @Op, @Action(%C) [concrete]
// CHECK:STDOUT:   %Op.b10: %Op.type.5ad = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.0a7: type = assoc_entity_type %Action.type.e2b [concrete]
// CHECK:STDOUT:   %assoc0.85d: %Action.assoc_type.0a7 = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [concrete]
// CHECK:STDOUT:   %assoc0.905ab9.2: %Action.assoc_type.8f9 = assoc_entity element0, imports.%Main.import_ref.0e3753.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.29c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//action, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst58 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ddc = import_ref Main//action, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.318: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.905ab9.2)]
// CHECK:STDOUT:   %Main.Op: @Action.%Op.type (%Op.type.036) = import_ref Main//action, Op, loaded [symbolic = @Action.%Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Main.import_ref.7a1 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst53 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %Main.import_ref.7b5 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.835: @Action.%Action.type (%Action.type.cca) = import_ref Main//action, inst24 [no loc], loaded [symbolic = @Action.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.0e3753.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//action, inst61 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.5ab3ec.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.8f9)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.8f9) = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic = %assoc0 (constants.%assoc0.905ab9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.318
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7b5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.835: @Action.%Action.type (%Action.type.cca)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.e2b]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.0a7 = specific_constant imports.%Main.import_ref.318, @Action(constants.%C) [concrete = constants.%assoc0.85d]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.0a7 = name_ref Op, %.loc8 [concrete = constants.%assoc0.85d]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %T.patt => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.54d
// CHECK:STDOUT:   %Op => constants.%Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.827
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Op.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.e2b
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.5ad
// CHECK:STDOUT:   %Op => constants.%Op.b10
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.0a7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.85d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.ca7: type = assoc_entity_type %Factory.type.c96 [symbolic]
// CHECK:STDOUT:   %assoc0.d7a: %Factory.assoc_type.ca7 = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Method.type.7ee: type = fn_type @Method.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Method.a71: %Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.157: %Factory.assoc_type.ca7 = assoc_entity element1, @Factory.%Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make.1, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.668: type = assoc_entity_type %Factory.type.a5d [concrete]
// CHECK:STDOUT:   %assoc0.40c: %Factory.assoc_type.668 = assoc_entity element0, @Factory.%Make.decl [concrete]
// CHECK:STDOUT:   %Method.type.117: type = fn_type @Method.1, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Method.ea9: %Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.523: %Factory.assoc_type.668 = assoc_entity element1, @Factory.%Method.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl, @impl.%Method.decl) [concrete]
// CHECK:STDOUT:   %Make.type.ec4: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.377: %Make.type.ec4 = struct_value () [concrete]
// CHECK:STDOUT:   %Method.type.af5: type = fn_type @Method.2 [concrete]
// CHECK:STDOUT:   %Method.3d4: %Method.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value %A, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type.1a8 = interface_decl @Factory [concrete = constants.%Factory.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1a8 = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl, @impl.%Method.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.2)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self.2: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T.loc4_19.2) [symbolic = %Make.type (constants.%Make.type.598)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.598) = struct_value () [symbolic = %Make (constants.%Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.ca7)]
// CHECK:STDOUT:   %assoc0.loc6_17.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.d7a)]
// CHECK:STDOUT:   %Method.type: type = fn_type @Method.1, @Factory(%T.loc4_19.2) [symbolic = %Method.type (constants.%Method.type.7ee)]
// CHECK:STDOUT:   %Method: @Factory.%Method.type (%Method.type.7ee) = struct_value () [symbolic = %Method (constants.%Method.a71)]
// CHECK:STDOUT:   %assoc1.loc8_31.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element1, %Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.157)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.598) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.737)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.d7a)]
// CHECK:STDOUT:     %Method.decl: @Factory.%Method.type (%Method.type.7ee) = fn_decl @Method.1 [symbolic = @Factory.%Method (constants.%Method.a71)] {
// CHECK:STDOUT:       %self.patt: @Method.1.%Self.as_type.loc8_19.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Method.1.%Self.as_type.loc8_19.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Method.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Method.1.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Method.1.%Self.as_type.loc8_19.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_19.1: type = splice_block %.loc8_19.3 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc8_19.2: @Method.1.%Factory.type (%Factory.type.c96) = specific_constant @Factory.%Self.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Method.1.%Factory.type (%Factory.type.c96) = name_ref Self, %.loc8_19.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc8_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc8_19.3: type = converted %Self.ref, %Self.as_type.loc8_19.2 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Method.1.%Self.as_type.loc8_19.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Method.1.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Method.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_31.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element1, %Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.157)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %assoc0.loc6_17.1
// CHECK:STDOUT:     .Method = %assoc1.loc8_31.1
// CHECK:STDOUT:     witness = (%Make.decl, %Method.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.ec4 = fn_decl @Make.2 [concrete = constants.%Make.377] {
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Method.decl: %Method.type.af5 = fn_decl @Method.2 [concrete = constants.%Method.3d4] {
// CHECK:STDOUT:     %self.patt: %A = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %A = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .Method = %Method.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.c96)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Method.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.c96)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Method.1.%Self.as_type.loc8_19.1 (%Self.as_type)]() -> @Method.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Method.2[%self.param_patt: %A]() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Method.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.668
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.40c
// CHECK:STDOUT:   %Method.type => constants.%Method.type.117
// CHECK:STDOUT:   %Method => constants.%Method.ea9
// CHECK:STDOUT:   %assoc1.loc8_31.2 => constants.%assoc1.523
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.ca7: type = assoc_entity_type %Factory.type.c96 [symbolic]
// CHECK:STDOUT:   %assoc0.35472f.1: %Factory.assoc_type.ca7 = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic]
// CHECK:STDOUT:   %Method.type.7ee: type = fn_type @Method.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Method.a71: %Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %assoc1.1fdf32.1: %Factory.assoc_type.ca7 = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make.1, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.668: type = assoc_entity_type %Factory.type.a5d [concrete]
// CHECK:STDOUT:   %assoc0.228: %Factory.assoc_type.668 = assoc_entity element0, imports.%Main.import_ref.21018a.2 [concrete]
// CHECK:STDOUT:   %Method.type.117: type = fn_type @Method.1, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Method.ea9: %Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.51f: %Factory.assoc_type.668 = assoc_entity element1, imports.%Main.import_ref.5be [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.35472f.2: %Factory.assoc_type.ca7 = assoc_entity element0, imports.%Main.import_ref.21018a.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.9ec, imports.%Main.import_ref.7dd) [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value %A, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.b98: type = fn_type_with_self_type %Make.type.c59, %Factory.facet [concrete]
// CHECK:STDOUT:   %Make.type.ec4: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.377: %Make.type.ec4 = struct_value () [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.1fdf32.2: %Factory.assoc_type.ca7 = assoc_entity element1, imports.%Main.import_ref.46fc3c.2 [symbolic]
// CHECK:STDOUT:   %.ac0: type = fn_type_with_self_type %Method.type.117, %Factory.facet [concrete]
// CHECK:STDOUT:   %Method.type.af5: type = fn_type @Method.2 [concrete]
// CHECK:STDOUT:   %Method.3d4: %Method.type.af5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//factory, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst77 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//factory, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8d5: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.35472f.2)]
// CHECK:STDOUT:   %Main.import_ref.ff7: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.1fdf32.2)]
// CHECK:STDOUT:   %Main.Make: @Factory.%Make.type (%Make.type.598) = import_ref Main//factory, Make, loaded [symbolic = @Factory.%Make (constants.%Make.737)]
// CHECK:STDOUT:   %Main.Method: @Factory.%Method.type (%Method.type.7ee) = import_ref Main//factory, Method, loaded [symbolic = @Factory.%Method (constants.%Method.a71)]
// CHECK:STDOUT:   %Main.import_ref.f8d: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst72 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bd2: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %Main.import_ref.a27 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.163 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst24 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.21018a.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst24 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.46fc3c.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = imports.%Main.Factory
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .MakeB = %MakeB.decl
// CHECK:STDOUT:     .InstanceB = %InstanceB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %MakeB.decl: %MakeB.type = fn_decl @MakeB [concrete = constants.%MakeB] {
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceB.decl: %InstanceB.type = fn_decl @InstanceB [concrete = constants.%InstanceB] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc7: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.5ab3ec.1: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.598)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.598) = struct_value () [symbolic = %Make (constants.%Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.ca7)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic = %assoc0 (constants.%assoc0.35472f.1)]
// CHECK:STDOUT:   %Method.type: type = fn_type @Method.1, @Factory(%T) [symbolic = %Method.type (constants.%Method.type.7ee)]
// CHECK:STDOUT:   %Method: @Factory.%Method.type (%Method.type.7ee) = struct_value () [symbolic = %Method (constants.%Method.a71)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic = %assoc1 (constants.%assoc1.1fdf32.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.8d5
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.ff7
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.a27
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.163
// CHECK:STDOUT:   witness = imports.%Main.import_ref.f8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Method.1(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Method.1.%Self.as_type (%Self.as_type)]() -> @Method.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB() -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.668 = specific_constant imports.%Main.import_ref.8d5, @Factory(constants.%B) [concrete = constants.%assoc0.228]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.668 = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.228]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value constants.%A, (constants.%impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.a5d = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc5_11, element0 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.b98 = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%Make.377]
// CHECK:STDOUT:   %.loc4: ref %B = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %B = call %impl.elem0() to %.loc4
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceB(%a.param_patt: %A) -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.668 = specific_constant imports.%Main.import_ref.ff7, @Factory(constants.%B) [concrete = constants.%assoc1.51f]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.668 = name_ref Method, %.loc8 [concrete = constants.%assoc1.51f]
// CHECK:STDOUT:   %impl.elem1: %.ac0 = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%Method.3d4]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %.loc7: ref %B = splice_block %return {}
// CHECK:STDOUT:   %Method.call: init %B = call %bound_method(%a.ref) to %.loc7
// CHECK:STDOUT:   return %Method.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Method.2[%self.param_patt: %A]() -> %B [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %T.patt => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.668
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.228
// CHECK:STDOUT:   %Method.type => constants.%Method.type.117
// CHECK:STDOUT:   %Method => constants.%Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.51f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Method.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9ba: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.ca7: type = assoc_entity_type %Factory.type.c96 [symbolic]
// CHECK:STDOUT:   %assoc0.35472f.1: %Factory.assoc_type.ca7 = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic]
// CHECK:STDOUT:   %Method.type.7ee: type = fn_type @Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Method.a71: %Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.9ba [symbolic]
// CHECK:STDOUT:   %assoc1.1fdf32.1: %Factory.assoc_type.ca7 = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.668: type = assoc_entity_type %Factory.type.a5d [concrete]
// CHECK:STDOUT:   %assoc0.228: %Factory.assoc_type.668 = assoc_entity element0, imports.%Main.import_ref.21018a.2 [concrete]
// CHECK:STDOUT:   %Method.type.117: type = fn_type @Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Method.ea9: %Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.7a1: %Factory.assoc_type.668 = assoc_entity element1, imports.%Main.import_ref.46fc3c.2 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.5c5: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Make.type.0de: type = fn_type @Make, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Make.8ba: %Make.type.0de = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.709: type = assoc_entity_type %Factory.type.5c5 [concrete]
// CHECK:STDOUT:   %assoc0.ef9: %Factory.assoc_type.709 = assoc_entity element0, imports.%Main.import_ref.21018a.1 [concrete]
// CHECK:STDOUT:   %Method.type.d46: type = fn_type @Method, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Method.f9e: %Method.type.d46 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.ae5: %Factory.assoc_type.709 = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [concrete]
// CHECK:STDOUT:   %assoc0.35472f.2: %Factory.assoc_type.ca7 = assoc_entity element0, imports.%Main.import_ref.21018a.3 [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.40a: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d43: type = facet_type <@ImplicitAs, @ImplicitAs(%Factory.type.5c5)> [concrete]
// CHECK:STDOUT:   %Convert.type.9b5: type = fn_type @Convert, @ImplicitAs(%Factory.type.5c5) [concrete]
// CHECK:STDOUT:   %Convert.8cc: %Convert.type.9b5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.0f8: type = assoc_entity_type %ImplicitAs.type.d43 [concrete]
// CHECK:STDOUT:   %assoc0.ae3: %ImplicitAs.assoc_type.0f8 = assoc_entity element0, imports.%Core.import_ref.207961.1 [concrete]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.1fdf32.2: %Factory.assoc_type.ca7 = assoc_entity element1, imports.%Main.import_ref.46fc3c.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst77 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//factory, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8d5: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.35472f.2)]
// CHECK:STDOUT:   %Main.import_ref.ff7: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.1fdf32.2)]
// CHECK:STDOUT:   %Main.Make: @Factory.%Make.type (%Make.type.598) = import_ref Main//factory, Make, loaded [symbolic = @Factory.%Make (constants.%Make.737)]
// CHECK:STDOUT:   %Main.Method: @Factory.%Method.type (%Method.type.7ee) = import_ref Main//factory, Method, loaded [symbolic = @Factory.%Method (constants.%Method.a71)]
// CHECK:STDOUT:   %Main.import_ref.e41 = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst72 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bd2: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %Main.import_ref.a27 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.163 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst24 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.21018a.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst24 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.46fc3c.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc4_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc3_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc4_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = imports.%Main.Factory
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:     .InstanceC = %InstanceC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceC.decl: %InstanceC.type = fn_decl @InstanceC [concrete = constants.%InstanceC] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.5ab3ec.1: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.598)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.598) = struct_value () [symbolic = %Make (constants.%Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.ca7)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic = %assoc0 (constants.%assoc0.35472f.1)]
// CHECK:STDOUT:   %Method.type: type = fn_type @Method, @Factory(%T) [symbolic = %Method.type (constants.%Method.type.7ee)]
// CHECK:STDOUT:   %Method: @Factory.%Method.type (%Method.type.7ee) = struct_value () [symbolic = %Method (constants.%Method.a71)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.ca7) = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic = %assoc1 (constants.%assoc1.1fdf32.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.8d5
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.ff7
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.a27
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.163
// CHECK:STDOUT:   witness = imports.%Main.import_ref.e41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Method(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Method.%Self.as_type (%Self.as_type.56c)]() -> @Method.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.5c5]
// CHECK:STDOUT:   %.loc16_23: %Factory.assoc_type.709 = specific_constant imports.%Main.import_ref.8d5, @Factory(constants.%C) [concrete = constants.%assoc0.ef9]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.709 = name_ref Make, %.loc16_23 [concrete = constants.%assoc0.ef9]
// CHECK:STDOUT:   %.loc16_11: %Factory.type.5c5 = converted %A.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.40a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type.40a)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param_patt: %A) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.5c5]
// CHECK:STDOUT:   %.loc24: %Factory.assoc_type.709 = specific_constant imports.%Main.import_ref.ff7, @Factory(constants.%C) [concrete = constants.%assoc1.ae5]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.709 = name_ref Method, %.loc24 [concrete = constants.%assoc1.ae5]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %T.patt => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.668
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.228
// CHECK:STDOUT:   %Method.type => constants.%Method.type.117
// CHECK:STDOUT:   %Method => constants.%Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Method.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5c5
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Make.type => constants.%Make.type.0de
// CHECK:STDOUT:   %Make => constants.%Make.8ba
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.709
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ef9
// CHECK:STDOUT:   %Method.type => constants.%Method.type.d46
// CHECK:STDOUT:   %Method => constants.%Method.f9e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ae5
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.40a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Factory.type.5c5) {
// CHECK:STDOUT:   %Dest => constants.%Factory.type.5c5
// CHECK:STDOUT:   %Dest.patt => constants.%Factory.type.5c5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d43
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9b5
// CHECK:STDOUT:   %Convert => constants.%Convert.8cc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.0f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ae3
// CHECK:STDOUT: }
// CHECK:STDOUT:
