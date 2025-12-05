// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/interface_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/interface_args.carbon

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
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+4]]:10: error: cannot convert type `A` into type implementing `Factory(C)` [ConversionFailureTypeToFacet]
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.060: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.060 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.170: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Dest.loc3_22.2: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.2: type) {
// CHECK:STDOUT:   %Dest.loc3_22.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.060)]
// CHECK:STDOUT:   %Self.loc3_35.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.060) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.060) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.170) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.170) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.060) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.060) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_35.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%ImplicitAs.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.060)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.060)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.060) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.170)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.51d)]
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
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.060
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.170
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.7df: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c4c: %Action.type.7df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.c4c [symbolic]
// CHECK:STDOUT:   %pattern_type.bce: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Action.Op.type.0a6: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.cd2: %Action.Op.type.0a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.026: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.191: %Action.assoc_type.026 = assoc_entity element0, @Action.%Action.Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.74f: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness file.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.cdb: %Action.type.74f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.c5f: %Action.assoc_type.4ee = assoc_entity element0, @Action.%Action.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.74f = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.db8: type = fn_type_with_self_type %Action.Op.type.54d, %Action.facet [concrete]
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
// CHECK:STDOUT:     .Action = %Action.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Action.decl: %Action.type.29c = interface_decl @Action [concrete = constants.%Action.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Action.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.74f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (@A.as.Action.impl.%A.as.Action.impl.Op.decl), @A.as.Action.impl [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness %Action.impl_witness_table [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.1)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self.loc4_28.2: @Action.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.c4c)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T.loc4_18.1) [symbolic = %Action.Op.type (constants.%Action.Op.type.0a6)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.0a6) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.cd2)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T.loc4_18.1) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.026)]
// CHECK:STDOUT:   %assoc0.loc5_22.2: @Action.%Action.assoc_type (%Action.assoc_type.026) = assoc_entity element0, %Action.Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @Action.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.c4c)]
// CHECK:STDOUT:     %Action.Op.decl: @Action.%Action.Op.type (%Action.Op.type.0a6) = fn_decl @Action.Op [symbolic = @Action.%Action.Op (constants.%Action.Op.cd2)] {
// CHECK:STDOUT:       %self.patt: @Action.Op.%pattern_type (%pattern_type.bce) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Action.Op.%pattern_type (%pattern_type.bce) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Action.Op.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @Action.Op.%Action.type (%Action.type.7df) = specific_constant @Action.%Self.loc4_28.1, @Action(constants.%T) [symbolic = %Self (constants.%Self.c4c)]
// CHECK:STDOUT:         %Self.ref: @Action.Op.%Action.type (%Action.type.7df) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self.c4c)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Action.Op.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_22.1: @Action.%Action.assoc_type (%Action.assoc_type.026) = assoc_entity element0, %Action.Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_22.1
// CHECK:STDOUT:     witness = (%Action.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %A.as.Action.impl.Op.decl: %A.as.Action.impl.Op.type = fn_decl @A.as.Action.impl.Op [concrete = constants.%A.as.Action.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Action.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Action.impl.Op.decl
// CHECK:STDOUT:   witness = file.%Action.impl_witness
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.Op(@Action.%T.loc4_18.2: type, @Action.%Self.loc4_28.1: @Action.%Action.type (%Action.type.7df)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c4c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.bce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Action.Op.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Action.impl.Op(%self.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.74f]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.4ee = specific_constant @Action.%assoc0.loc5_22.1, @Action(constants.%B) [concrete = constants.%assoc0.c5f]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ee = name_ref Op, %.loc16 [concrete = constants.%assoc0.c5f]
// CHECK:STDOUT:   %impl.elem0: %.db8 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.c4c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.7df
// CHECK:STDOUT:   %Self => constants.%Self.c4c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.74f
// CHECK:STDOUT:   %Self.loc4_28.2 => constants.%Self.cdb
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0.loc5_22.2 => constants.%assoc0.c5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.74f
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c10
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.7df: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.eee: %Action.type.7df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.026: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.cea611.1: %Action.assoc_type.026 = assoc_entity element0, imports.%Main.import_ref.3f972e.1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.0a6: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.cd2: %Action.Op.type.0a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.eee [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Action.type.74f: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.faa: %Action.assoc_type.4ee = assoc_entity element0, imports.%Main.import_ref.a20 [concrete]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Self.cf6: %Action.type.74f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.cea611.2: %Action.assoc_type.026 = assoc_entity element0, imports.%Main.import_ref.3f972e.2 [symbolic]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness imports.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.74f = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %.452: type = fn_type_with_self_type %Action.Op.type.54d, %Action.facet [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.29c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//action, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.460 = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.0dc: @Action.%Action.assoc_type (%Action.assoc_type.026) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.cea611.2)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3f972e.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.451: @Action.%Action.type (%Action.type.7df) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Main.import_ref.c2f: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.a20: @Action.%Action.Op.type (%Action.Op.type.0a6) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%Action.Op (constants.%Action.Op.cd2)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.99f: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.74f]
// CHECK:STDOUT:   %Main.import_ref.8c4 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.3f972e.2 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.469: %A.as.Action.impl.Op.type = import_ref Main//action, loc13_23, loaded [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (%Main.import_ref.469), @A.as.Action.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T) [symbolic = %Action.Op.type (constants.%Action.Op.type.0a6)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.0a6) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.cd2)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.026)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.026) = assoc_entity element0, imports.%Main.import_ref.3f972e.1 [symbolic = %assoc0 (constants.%assoc0.cea611.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.460
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.0dc
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.99f [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.8c4
// CHECK:STDOUT:   witness = imports.%Main.import_ref.c2f
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
// CHECK:STDOUT: generic fn @Action.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.451: @Action.%Action.type (%Action.type.7df)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.74f]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.4ee = specific_constant imports.%Main.import_ref.0dc, @Action(constants.%B) [concrete = constants.%assoc0.faa]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ee = name_ref Op, %.loc4 [concrete = constants.%assoc0.faa]
// CHECK:STDOUT:   %impl.elem0: %.452 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Action.impl.Op [from "action.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.eee) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.7df
// CHECK:STDOUT:   %Self => constants.%Self.eee
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.74f
// CHECK:STDOUT:   %Self => constants.%Self.cf6
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.faa
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.7df: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.eee: %Action.type.7df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.026: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.4d0: %Action.assoc_type.026 = assoc_entity element0, imports.%Main.import_ref.a20 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.0a6: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.cd2: %Action.Op.type.0a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.eee [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Action.type.74f: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.54c: %Action.assoc_type.4ee = assoc_entity element0, imports.%Main.import_ref.3f972e.1 [concrete]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Self.cf6: %Action.type.74f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.eec: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Self.c5d: %Action.type.eec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.5ad: type = fn_type @Action.Op, @Action(%C) [concrete]
// CHECK:STDOUT:   %Action.Op.b10: %Action.Op.type.5ad = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.79f: type = assoc_entity_type @Action, @Action(%C) [concrete]
// CHECK:STDOUT:   %assoc0.4b0: %Action.assoc_type.79f = assoc_entity element0, imports.%Main.import_ref.a20 [concrete]
// CHECK:STDOUT:   %assoc0.cea: %Action.assoc_type.026 = assoc_entity element0, imports.%Main.import_ref.3f972e.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.29c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//action, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.460 = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.0dc: @Action.%Action.assoc_type (%Action.assoc_type.026) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.cea)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.a20: @Action.%Action.Op.type (%Action.Op.type.0a6) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%Action.Op (constants.%Action.Op.cd2)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.451: @Action.%Action.type (%Action.type.7df) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Main.import_ref.7bf = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.3f972e.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.99f: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.74f]
// CHECK:STDOUT:   %Main.import_ref.8c4 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.3f972e.2 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T) [symbolic = %Action.Op.type (constants.%Action.Op.type.0a6)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.0a6) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.cd2)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.026)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.026) = assoc_entity element0, imports.%Main.import_ref.a20 [symbolic = %assoc0 (constants.%assoc0.4d0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.460
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.0dc
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.99f [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.8c4
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7bf
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
// CHECK:STDOUT: generic fn @Action.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.451: @Action.%Action.type (%Action.type.7df)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.7df)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.7df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.eee)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.eec]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.79f = specific_constant imports.%Main.import_ref.0dc, @Action(constants.%C) [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.79f = name_ref Op, %.loc8 [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.eee) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.7df
// CHECK:STDOUT:   %Self => constants.%Self.eee
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.74f
// CHECK:STDOUT:   %Self => constants.%Self.cf6
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.eec
// CHECK:STDOUT:   %Self => constants.%Self.c5d
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.5ad
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.b10
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.79f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.ea0: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c88: %Factory.type.ea0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.fec: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.3d6: %Factory.Make.type.fec = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.80d: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.706: %Factory.assoc_type.80d = assoc_entity element0, @Factory.%Factory.Make.decl [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.c88 [symbolic]
// CHECK:STDOUT:   %pattern_type.cc1: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.d98: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.9a8: %Factory.Method.type.d98 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.7ef: %Factory.assoc_type.80d = assoc_entity element1, @Factory.%Factory.Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.3cb: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness file.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.bbe: %Factory.type.3cb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.e92: %Factory.assoc_type.579 = assoc_entity element0, @Factory.%Factory.Make.decl [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.f1d: %Factory.assoc_type.579 = assoc_entity element1, @Factory.%Factory.Method.decl [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.3cb = facet_value %A, (%Factory.impl_witness) [concrete]
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
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Factory.decl: %Factory.type.1a8 = interface_decl @Factory [concrete = constants.%Factory.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Factory.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1a8 = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.3cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (@A.as.Factory.impl.%A.as.Factory.impl.Make.decl, @A.as.Factory.impl.%A.as.Factory.impl.Method.decl), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness %Factory.impl_witness_table [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.1)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self.loc4_29.2: @Factory.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.c88)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T.loc4_19.1) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.fec)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.3d6)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T.loc4_19.1) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.80d)]
// CHECK:STDOUT:   %assoc0.loc6_17.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element0, %Factory.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.706)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T.loc4_19.1) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.d98)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.9a8)]
// CHECK:STDOUT:   %assoc1.loc8_31.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element1, %Factory.Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.7ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.c88)]
// CHECK:STDOUT:     %Factory.Make.decl: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = fn_decl @Factory.Make [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.3d6)] {
// CHECK:STDOUT:       %return.patt: @Factory.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Factory.Make.%pattern_type (%pattern_type.51d) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Factory.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Factory.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element0, %Factory.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.706)]
// CHECK:STDOUT:     %Factory.Method.decl: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = fn_decl @Factory.Method [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.9a8)] {
// CHECK:STDOUT:       %self.patt: @Factory.Method.%pattern_type.loc8_13 (%pattern_type.cc1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Factory.Method.%pattern_type.loc8_13 (%pattern_type.cc1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Factory.Method.%pattern_type.loc8_27 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Factory.Method.%pattern_type.loc8_27 (%pattern_type.51d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Factory.Method.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_19.1: type = splice_block %.loc8_19.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc8_19.2: @Factory.Method.%Factory.type (%Factory.type.ea0) = specific_constant @Factory.%Self.loc4_29.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self.c88)]
// CHECK:STDOUT:         %Self.ref: @Factory.Method.%Factory.type (%Factory.type.ea0) = name_ref Self, %.loc8_19.2 [symbolic = %Self (constants.%Self.c88)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc8_19.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Factory.Method.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Factory.Method.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Factory.Method.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_31.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element1, %Factory.Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.7ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_29.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %assoc0.loc6_17.1
// CHECK:STDOUT:     .Method = %assoc1.loc8_31.1
// CHECK:STDOUT:     witness = (%Factory.Make.decl, %Factory.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %A.as.Factory.impl.Make.decl: %A.as.Factory.impl.Make.type = fn_decl @A.as.Factory.impl.Make [concrete = constants.%A.as.Factory.impl.Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.as.Factory.impl.Method.decl: %A.as.Factory.impl.Method.type = fn_decl @A.as.Factory.impl.Method [concrete = constants.%A.as.Factory.impl.Method] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Factory.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .Make = %A.as.Factory.impl.Make.decl
// CHECK:STDOUT:   .Method = %A.as.Factory.impl.Method.decl
// CHECK:STDOUT:   witness = file.%Factory.impl_witness
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
// CHECK:STDOUT: generic fn @Factory.Make(@Factory.%T.loc4_19.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.ea0)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Factory.Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Method(@Factory.%T.loc4_19.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.ea0)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c88)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc8_13 (constants.%pattern_type.cc1)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %T [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Factory.Method.%Self.binding.as_type (%Self.binding.as_type)) -> @Factory.Method.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method(%self.param: %A) -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.c88) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.c88) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.ea0
// CHECK:STDOUT:   %Self => constants.%Self.c88
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.cc1
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3cb
// CHECK:STDOUT:   %Self.loc4_29.2 => constants.%Self.bbe
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.e92
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1.loc8_31.2 => constants.%assoc1.f1d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3cb
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%A
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.c10
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.049
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.ea0: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.024: %Factory.type.ea0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.80d: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc1.426954.1: %Factory.assoc_type.80d = assoc_entity element1, imports.%Main.import_ref.46619b.1 [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.d98: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.9a8: %Factory.Method.type.d98 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.024 [symbolic]
// CHECK:STDOUT:   %pattern_type.26a: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %assoc0.c89cd9.1: %Factory.assoc_type.80d = assoc_entity element0, imports.%Main.import_ref.d2478c.1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.fec: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.3d6: %Factory.Make.type.fec = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.3cb: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc1.208: %Factory.assoc_type.579 = assoc_entity element1, imports.%Main.import_ref.6a3 [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.cb4: %Factory.assoc_type.579 = assoc_entity element0, imports.%Main.import_ref.2b4 [concrete]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.ffb: %Factory.type.3cb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.c89cd9.2: %Factory.assoc_type.80d = assoc_entity element0, imports.%Main.import_ref.d2478c.2 [symbolic]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness imports.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.3cb = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %.549: type = fn_type_with_self_type %Factory.Make.type.c59, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.426954.2: %Factory.assoc_type.80d = assoc_entity element1, imports.%Main.import_ref.46619b.2 [symbolic]
// CHECK:STDOUT:   %.a61: type = fn_type_with_self_type %Factory.Method.type.117, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//factory, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b7e = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.11b: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.c89cd9.2)]
// CHECK:STDOUT:   %Main.import_ref.995: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.426954.2)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.46619b.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.704502.1: @Factory.%Factory.type (%Factory.type.ea0) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.024)]
// CHECK:STDOUT:   %Main.import_ref.d2478c.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.704502.2: @Factory.%Factory.type (%Factory.type.ea0) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.024)]
// CHECK:STDOUT:   %Main.import_ref.a6e: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.6a3: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.9a8)]
// CHECK:STDOUT:   %Main.import_ref.2b4: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.3d6)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.e91: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.3cb]
// CHECK:STDOUT:   %Main.import_ref.22f = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5a9 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.d2478c.2 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.4c9: %A.as.Factory.impl.Make.type = import_ref Main//factory, loc15_17, loaded [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %Main.import_ref.721: %A.as.Factory.impl.Method.type = import_ref Main//factory, loc16_31, loaded [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%Main.import_ref.4c9, %Main.import_ref.721), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.46619b.2 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = imports.%Main.Factory
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeB = %MakeB.decl
// CHECK:STDOUT:     .InstanceB = %InstanceB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_23.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_23.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeB.decl: %MakeB.type = fn_decl @MakeB [concrete = constants.%MakeB] {
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceB.decl: %InstanceB.type = fn_decl @InstanceB [concrete = constants.%InstanceB] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc7: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.1: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.024)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.fec)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.3d6)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.80d)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element0, imports.%Main.import_ref.d2478c.1 [symbolic = %assoc0 (constants.%assoc0.c89cd9.1)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.d98)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.9a8)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element1, imports.%Main.import_ref.46619b.1 [symbolic = %assoc1 (constants.%assoc1.426954.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b7e
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.11b
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.995
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.e91 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.22f
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.5a9
// CHECK:STDOUT:   witness = imports.%Main.import_ref.a6e
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
// CHECK:STDOUT: generic fn @Factory.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.704502.1: @Factory.%Factory.type (%Factory.type.ea0)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.024)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.26a)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.704502.2: @Factory.%Factory.type (%Factory.type.ea0)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB() -> %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.3cb]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.579 = specific_constant imports.%Main.import_ref.11b, @Factory(constants.%B) [concrete = constants.%assoc0.cb4]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.579 = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.cb4]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.3cb = facet_value %A.ref, (constants.%Factory.impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.3cb = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %impl.elem0: %.549 = impl_witness_access constants.%Factory.impl_witness, element0 [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %.loc4: ref %B = splice_block %return {}
// CHECK:STDOUT:   %A.as.Factory.impl.Make.call: init %B = call %impl.elem0() to %.loc4
// CHECK:STDOUT:   return %A.as.Factory.impl.Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceB(%a.param: %A) -> %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.3cb]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.579 = specific_constant imports.%Main.import_ref.995, @Factory(constants.%B) [concrete = constants.%assoc1.208]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.579 = name_ref Method, %.loc8 [concrete = constants.%assoc1.208]
// CHECK:STDOUT:   %impl.elem1: %.a61 = impl_witness_access constants.%Factory.impl_witness, element1 [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %.loc7: ref %B = splice_block %return {}
// CHECK:STDOUT:   %A.as.Factory.impl.Method.call: init %B = call %bound_method(%a.ref) to %.loc7
// CHECK:STDOUT:   return %A.as.Factory.impl.Method.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.024) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.ea0
// CHECK:STDOUT:   %Self => constants.%Self.024
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.26a
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.024) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3cb
// CHECK:STDOUT:   %Self => constants.%Self.ffb
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cb4
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.208
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.ea0: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.024: %Factory.type.ea0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.80d: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc1.832: %Factory.assoc_type.80d = assoc_entity element1, imports.%Main.import_ref.6a3 [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.d98: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.9a8: %Factory.Method.type.d98 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.024 [symbolic]
// CHECK:STDOUT:   %pattern_type.26a: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %assoc0.46c: %Factory.assoc_type.80d = assoc_entity element0, imports.%Main.import_ref.2b4 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.fec: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.3d6: %Factory.Make.type.fec = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.3cb: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc1.06b: %Factory.assoc_type.579 = assoc_entity element1, imports.%Main.import_ref.46619b.1 [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.65a: %Factory.assoc_type.579 = assoc_entity element0, imports.%Main.import_ref.d2478c.1 [concrete]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.ffb: %Factory.type.3cb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.a06: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Self.4f1: %Factory.type.a06 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.0de: type = fn_type @Factory.Make, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Factory.Make.8ba: %Factory.Make.type.0de = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.d01: type = assoc_entity_type @Factory, @Factory(%C) [concrete]
// CHECK:STDOUT:   %assoc0.0cb: %Factory.assoc_type.d01 = assoc_entity element0, imports.%Main.import_ref.2b4 [concrete]
// CHECK:STDOUT:   %Factory.Method.type.d46: type = fn_type @Factory.Method, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Factory.Method.f9e: %Factory.Method.type.d46 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.0b1: %Factory.assoc_type.d01 = assoc_entity element1, imports.%Main.import_ref.6a3 [concrete]
// CHECK:STDOUT:   %assoc0.c89: %Factory.assoc_type.80d = assoc_entity element0, imports.%Main.import_ref.d2478c.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.426: %Factory.assoc_type.80d = assoc_entity element1, imports.%Main.import_ref.46619b.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b7e = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.11b: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.c89)]
// CHECK:STDOUT:   %Main.import_ref.995: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.426)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.6a3: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.9a8)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.704502.1: @Factory.%Factory.type (%Factory.type.ea0) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.024)]
// CHECK:STDOUT:   %Main.import_ref.2b4: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.3d6)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.704502.2: @Factory.%Factory.type (%Factory.type.ea0) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.024)]
// CHECK:STDOUT:   %Main.import_ref.42c = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.46619b.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.d2478c.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.e91: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.3cb]
// CHECK:STDOUT:   %Main.import_ref.22f = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5a9 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.d2478c.2 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.46619b.2 = import_ref Main//factory, loc8_31, unloaded
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
// CHECK:STDOUT:   %default.import.loc2_23.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_23.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceC.decl: %InstanceC.type = fn_decl @InstanceC [concrete = constants.%InstanceC] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.1: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.024)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.fec)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.fec) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.3d6)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.80d)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element0, imports.%Main.import_ref.2b4 [symbolic = %assoc0 (constants.%assoc0.46c)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.d98)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.d98) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.9a8)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.80d) = assoc_entity element1, imports.%Main.import_ref.6a3 [symbolic = %assoc1 (constants.%assoc1.832)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b7e
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.11b
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.995
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.e91 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.22f
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.5a9
// CHECK:STDOUT:   witness = imports.%Main.import_ref.42c
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.704502.1: @Factory.%Factory.type (%Factory.type.ea0)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.ea0)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.ea0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.024)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.26a)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.704502.2: @Factory.%Factory.type (%Factory.type.ea0)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.a06]
// CHECK:STDOUT:   %.loc13: %Factory.assoc_type.d01 = specific_constant imports.%Main.import_ref.11b, @Factory(constants.%C) [concrete = constants.%assoc0.0cb]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.d01 = name_ref Make, %.loc13 [concrete = constants.%assoc0.0cb]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param: %A) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.a06]
// CHECK:STDOUT:   %.loc21: %Factory.assoc_type.d01 = specific_constant imports.%Main.import_ref.995, @Factory(constants.%C) [concrete = constants.%assoc1.0b1]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.d01 = name_ref Method, %.loc21 [concrete = constants.%assoc1.0b1]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.024) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.ea0
// CHECK:STDOUT:   %Self => constants.%Self.024
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.26a
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.024) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3cb
// CHECK:STDOUT:   %Self => constants.%Self.ffb
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.65a
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.06b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a06
// CHECK:STDOUT:   %Self => constants.%Self.4f1
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.0de
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.8ba
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.d01
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0cb
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.d46
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.f9e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
