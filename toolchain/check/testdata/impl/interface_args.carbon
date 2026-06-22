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
  fn Convert(self) -> Dest;
}

// --- action.carbon

library "[[@TEST_NAME]]";

interface Action(T:! type) {
  fn Op(self);
}

class A {}
class B {}
class C {}

impl A as Action(B) {
  fn Op(unused self) {}
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
  fn Method(self) -> T;
}

class A {}
class B {}

impl A as Factory(B) {
  fn Make() -> B;
  fn Method(self) -> B;
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.701: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.701 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.701 = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %Dest [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt: %pattern_type.51d = return_slot_pattern %return.param_patt, %Dest [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.0ff = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_26.1: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_26.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_29.1: type = splice_block %.loc3_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Dest.loc3_26.2: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_26.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_26.2: type) {
// CHECK:STDOUT:   %Dest.patt.loc3_26.2: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_26.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   %Dest.loc3_26.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_26.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_26.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self.loc3_35.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.param_patt.loc4_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.701) = value_param_pattern [symbolic = %self.param_patt.loc4_14.2 (constants.%self.param_patt)]
// CHECK:STDOUT:       %self.patt.loc4_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.701) = at_binding_pattern self, %self.param_patt.loc4_14.1 [symbolic = %self.patt.loc4_14.2 (constants.%self.patt)]
// CHECK:STDOUT:       %return.param_patt.loc4_23.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc4_23.2 (constants.%return.param_patt)]
// CHECK:STDOUT:       %return.patt.loc4_20.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc4_23.1, %Dest.ref [symbolic = %return.patt.loc4_20.2 (constants.%return.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_26.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %.loc4_23.2: Core.Form = init_form %Dest.ref [symbolic = %.loc4_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_14.1: type = splice_block %.loc4_14.3 [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_14.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = name_ref Self, %.loc4_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_14.3: type = converted %Self.ref, %Self.as_type.loc4_14.2 [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_27.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc4_27.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_35.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc4_27.1
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%Dest.loc3_26.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.as_type.loc4_14.1 [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.701)]
// CHECK:STDOUT:   %self.param_patt.loc4_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.701) = value_param_pattern [symbolic = %self.param_patt.loc4_14.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc4_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.701) = at_binding_pattern self, %self.param_patt.loc4_14.2 [symbolic = %self.patt.loc4_14.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc4_23.1: Core.Form = init_form %Dest [symbolic = %.loc4_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc4_23: type = pattern_type %Dest [symbolic = %pattern_type.loc4_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc4_23.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc4_23.2 (constants.%return.param_patt)]
// CHECK:STDOUT:   %return.patt.loc4_20.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc4_23.2, %Dest [symbolic = %return.patt.loc4_20.2 (constants.%return.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_14.1 (%Self.as_type)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.patt.loc3_26.2 => constants.%Dest.patt
// CHECK:STDOUT:   %Dest.loc3_26.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.701
// CHECK:STDOUT:   %self.param_patt.loc4_14.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc4_14.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc4_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc4_23 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc4_23.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc4_20.2 => constants.%return.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.cbf: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.67c: %Action.type.cbf = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.67c [symbolic]
// CHECK:STDOUT:   %pattern_type.085: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.256: %pattern_type.085 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.8a1: %pattern_type.085 = at_binding_pattern self, %self.param_patt.256 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.667: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.67c) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.217: %Action.WithSelf.Op.type.667 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1c3: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.26f: %Action.assoc_type.1c3 = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.5d2: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness @A.as.Action.impl.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.7ff: %Action.type.5d2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.cd3: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.67c) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.d98: %Action.WithSelf.Op.type.cd3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.4ce: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.005: %Action.assoc_type.4ce = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.5d2 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.e59: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.c3b: %Action.WithSelf.Op.type.e59 = struct_value () [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.898: type = fn_type_with_self_type %Action.WithSelf.Op.type.e59, %Action.facet [concrete]
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
// CHECK:STDOUT:   %Action.decl: %Action.type.538 = interface_decl @Action [concrete = constants.%Action.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_22.1: type = splice_block %.loc4_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Action.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type.538 = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.5d2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_19.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_19.1)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self.loc4_28.2: @Action.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.67c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @Action.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.67c)]
// CHECK:STDOUT:     %Action.WithSelf.decl = interface_with_self_decl @Action [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Action.WithSelf.Op.decl: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.667) = fn_decl @Action.WithSelf.Op [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.217)] {
// CHECK:STDOUT:       %self.param_patt.loc5_9.1: @Action.WithSelf.Op.%pattern_type (%pattern_type.085) = value_param_pattern [symbolic = %self.param_patt.loc5_9.2 (constants.%self.param_patt.256)]
// CHECK:STDOUT:       %self.patt.loc5_9.1: @Action.WithSelf.Op.%pattern_type (%pattern_type.085) = at_binding_pattern self, %self.param_patt.loc5_9.1 [symbolic = %self.patt.loc5_9.2 (constants.%self.patt.8a1)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Action.WithSelf.Op.%Self.as_type.loc5_9.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_9.1: type = splice_block %.loc5_9.3 [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_9.2: @Action.WithSelf.Op.%Action.type (%Action.type.cbf) = specific_constant @Action.%Self.loc4_28.1, @Action(constants.%T) [symbolic = %Self (constants.%Self.67c)]
// CHECK:STDOUT:         %Self.ref: @Action.WithSelf.Op.%Action.type (%Action.type.cbf) = name_ref Self, %.loc5_9.2 [symbolic = %Self (constants.%Self.67c)]
// CHECK:STDOUT:         %Self.as_type.loc5_9.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_9.3: type = converted %Self.ref, %Self.as_type.loc5_9.2 [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Action.WithSelf.Op.%Self.as_type.loc5_9.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1c3) = assoc_entity element0, %Action.WithSelf.Op.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.26f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
// CHECK:STDOUT:     .Op = @Action.WithSelf.%assoc0.loc5_14.1
// CHECK:STDOUT:     witness = (@Action.WithSelf.%Action.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %A.as.Action.impl.Op.decl: %A.as.Action.impl.Op.type = fn_decl @A.as.Action.impl.Op [concrete = constants.%A.as.Action.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Action.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (%A.as.Action.impl.Op.decl), @A.as.Action.impl [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness %Action.impl_witness_table [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Action.impl.Op.decl
// CHECK:STDOUT:   extend %Action.type
// CHECK:STDOUT:   witness = %Action.impl_witness
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
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(@Action.%T.loc4_19.2: type, @Action.%Self.loc4_28.1: @Action.%Action.type (%Action.type.cbf)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.67c)]
// CHECK:STDOUT:   %Self.as_type.loc5_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_9.1 [symbolic = %pattern_type (constants.%pattern_type.085)]
// CHECK:STDOUT:   %self.param_patt.loc5_9.2: @Action.WithSelf.Op.%pattern_type (%pattern_type.085) = value_param_pattern [symbolic = %self.param_patt.loc5_9.2 (constants.%self.param_patt.256)]
// CHECK:STDOUT:   %self.patt.loc5_9.2: @Action.WithSelf.Op.%pattern_type (%pattern_type.085) = at_binding_pattern self, %self.param_patt.loc5_9.2 [symbolic = %self.patt.loc5_9.2 (constants.%self.patt.8a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Action.WithSelf.Op.%Self.as_type.loc5_9.1 (%Self.as_type));
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
// CHECK:STDOUT:   %Action.ref: %Action.type.538 = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.5d2]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.4ce = specific_constant @Action.WithSelf.%assoc0.loc5_14.1, @Action.WithSelf(constants.%B, constants.%Self.67c) [concrete = constants.%assoc0.005]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ce = name_ref Op, %.loc16 [concrete = constants.%assoc0.005]
// CHECK:STDOUT:   %impl.elem0: %.898 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.67c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.67c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cbf
// CHECK:STDOUT:   %Self => constants.%Self.67c
// CHECK:STDOUT:   %Self.as_type.loc5_9.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.085
// CHECK:STDOUT:   %self.param_patt.loc5_9.2 => constants.%self.param_patt.256
// CHECK:STDOUT:   %self.patt.loc5_9.2 => constants.%self.patt.8a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self.loc4_28.2 => constants.%Self.7ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.67c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Self.67c
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.cd3
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.d98
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ce
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.005
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.e59
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.c3b
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ce
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.005
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.as_type.loc5_9.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc5_9.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc5_9.2 => constants.%self.patt.312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cbf: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.ea2: %Action.type.cbf = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1c3: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.a98: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.ea2) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.876: %Action.WithSelf.Op.type.a98 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ea2 [symbolic]
// CHECK:STDOUT:   %pattern_type.1f8: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.e11: %pattern_type.1f8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.f9b: %pattern_type.1f8 = at_binding_pattern self, %self.param_patt.e11 [symbolic]
// CHECK:STDOUT:   %Action.type.5d2: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.581: %Action.type.5d2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.070: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.ea2) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.84b: %Action.WithSelf.Op.type.070 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.4ce: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.2d9: %Action.assoc_type.4ce = assoc_entity element0, imports.%Main.import_ref.c56 [concrete]
// CHECK:STDOUT:   %assoc0.029: %Action.assoc_type.1c3 = assoc_entity element0, imports.%Main.import_ref.b3a [symbolic]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness imports.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.5d2 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.8c0: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.031: %Action.WithSelf.Op.type.8c0 = struct_value () [concrete]
// CHECK:STDOUT:   %.1f3: type = fn_type_with_self_type %Action.WithSelf.Op.type.8c0, %Action.facet [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.538 = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//action, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.7b8 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.00f: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1c3) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.029)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.c56: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.a98) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.876)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7dfc68.2: @Action.%Action.type (%Action.type.cbf) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.ea2)]
// CHECK:STDOUT:   %Main.import_ref.422 = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.371: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.4a4: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.5d2]
// CHECK:STDOUT:   %Main.import_ref.294 = import_ref Main//action, loc13_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.07b = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3a = import_ref Main//action, loc5_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.b97: %A.as.Action.impl.Op.type = import_ref Main//action, loc13_22, loaded [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (%Main.import_ref.b97), @A.as.Action.impl [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ea2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.422
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.00f
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.4a4 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.294
// CHECK:STDOUT:   extend imports.%Main.import_ref.07b
// CHECK:STDOUT:   witness = imports.%Main.import_ref.371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.7dfc68.2: @Action.%Action.type (%Action.type.cbf)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ea2)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.1f8)]
// CHECK:STDOUT:   %self.param_patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1f8) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e11)]
// CHECK:STDOUT:   %self.patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1f8) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.f9b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.538 = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.5d2]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.4ce = specific_constant imports.%Main.import_ref.00f, @Action.WithSelf(constants.%B, constants.%Self.ea2) [concrete = constants.%assoc0.2d9]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ce = name_ref Op, %.loc4 [concrete = constants.%assoc0.2d9]
// CHECK:STDOUT:   %impl.elem0: %.1f3 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Action.impl.Op [from "action.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.ea2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.ea2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cbf
// CHECK:STDOUT:   %Self => constants.%Self.ea2
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1f8
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e11
// CHECK:STDOUT:   %self.patt => constants.%self.patt.f9b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Self.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.ea2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Self.ea2
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.070
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.84b
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ce
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.8c0
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.031
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ce
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cbf: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.ea2: %Action.type.cbf = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1c3: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.a98: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.ea2) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.876: %Action.WithSelf.Op.type.a98 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ea2 [symbolic]
// CHECK:STDOUT:   %pattern_type.1f8: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.1f8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.1f8 = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %Action.type.5d2: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.581: %Action.type.5d2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.0f0: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Self.612: %Action.type.0f0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.529: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%C, %Self.ea2) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.ba9: %Action.WithSelf.Op.type.529 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.35c: type = assoc_entity_type @Action, @Action(%C) [concrete]
// CHECK:STDOUT:   %assoc0.940: %Action.assoc_type.35c = assoc_entity element0, imports.%Main.import_ref.c56 [concrete]
// CHECK:STDOUT:   %assoc0.029: %Action.assoc_type.1c3 = assoc_entity element0, imports.%Main.import_ref.b3a [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.538 = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//action, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.7b8 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.00f: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1c3) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.029)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.c56: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.a98) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.876)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7dfc68.2: @Action.%Action.type (%Action.type.cbf) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.ea2)]
// CHECK:STDOUT:   %Main.import_ref.422 = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.640 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.4a4: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.5d2]
// CHECK:STDOUT:   %Main.import_ref.294 = import_ref Main//action, loc13_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.07b = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3a = import_ref Main//action, loc5_14, unloaded
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ea2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.422
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.00f
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.4a4 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.294
// CHECK:STDOUT:   extend imports.%Main.import_ref.07b
// CHECK:STDOUT:   witness = imports.%Main.import_ref.640
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.7dfc68.2: @Action.%Action.type (%Action.type.cbf)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cbf)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.cbf) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ea2)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.1f8)]
// CHECK:STDOUT:   %self.param_patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1f8) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1f8) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.538 = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.0f0]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.35c = specific_constant imports.%Main.import_ref.00f, @Action.WithSelf(constants.%C, constants.%Self.ea2) [concrete = constants.%assoc0.940]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.35c = name_ref Op, %.loc8 [concrete = constants.%assoc0.940]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.ea2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.ea2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cbf
// CHECK:STDOUT:   %Self => constants.%Self.ea2
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1f8
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt => constants.%self.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.5d2
// CHECK:STDOUT:   %Self => constants.%Self.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.0f0
// CHECK:STDOUT:   %Self => constants.%Self.612
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%C, constants.%Self.ea2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Action.type => constants.%Action.type.0f0
// CHECK:STDOUT:   %Self => constants.%Self.ea2
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.529
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.ba9
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.35c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.940
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.711: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9c2: %Factory.type.711 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.515: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.9c2) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.bc6: %Factory.WithSelf.Make.type.515 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0ed: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.6f5: %Factory.assoc_type.0ed = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.9c2 [symbolic]
// CHECK:STDOUT:   %pattern_type.5ba: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.d68: %pattern_type.5ba = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.79a: %pattern_type.5ba = at_binding_pattern self, %self.param_patt.d68 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.63a: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.9c2) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.097: %Factory.WithSelf.Method.type.63a = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.b7b: %Factory.assoc_type.0ed = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.5e3: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness @A.as.Factory.impl.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.f44: %Factory.type.5e3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.264: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.9c2) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.26e: %Factory.WithSelf.Make.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.8bd: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.012: %Factory.assoc_type.8bd = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.dc9: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.9c2) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.5a4: %Factory.WithSelf.Method.type.dc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.682: %Factory.assoc_type.8bd = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.5e3 = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.0c4: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.f30: %Factory.WithSelf.Make.type.0c4 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.99a: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.637: %Factory.WithSelf.Method.type.99a = struct_value () [concrete]
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
// CHECK:STDOUT:   %Factory.decl: %Factory.type.9da = interface_decl @Factory [concrete = constants.%Factory.generic] {
// CHECK:STDOUT:     %T.patt.loc4_20.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Factory.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.9da = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.5e3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_20.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_20.1)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self.loc4_29.2: @Factory.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.9c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.9c2)]
// CHECK:STDOUT:     %Factory.WithSelf.decl = interface_with_self_decl @Factory [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Factory.WithSelf.Make.decl: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.515) = fn_decl @Factory.WithSelf.Make [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.bc6)] {
// CHECK:STDOUT:       %return.param_patt.loc6_16.1: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_13.1: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_16.1, %T.ref [symbolic = %return.patt.loc6_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_16.2: Core.Form = init_form %T.ref [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = assoc_entity element0, %Factory.WithSelf.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.6f5)]
// CHECK:STDOUT:     %Factory.WithSelf.Method.decl: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.63a) = fn_decl @Factory.WithSelf.Method [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.097)] {
// CHECK:STDOUT:       %self.param_patt.loc8_13.1: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.5ba) = value_param_pattern [symbolic = %self.param_patt.loc8_13.2 (constants.%self.param_patt.d68)]
// CHECK:STDOUT:       %self.patt.loc8_13.1: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.5ba) = at_binding_pattern self, %self.param_patt.loc8_13.1 [symbolic = %self.patt.loc8_13.2 (constants.%self.patt.79a)]
// CHECK:STDOUT:       %return.param_patt.loc8_22.1: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_22.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc8_19.1: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_22.1, %T.ref [symbolic = %return.patt.loc8_19.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc8_22.2: Core.Form = init_form %T.ref [symbolic = %.loc8_22.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Factory.WithSelf.Method.%Self.as_type.loc8_13.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_13.1: type = splice_block %.loc8_13.3 [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc8_13.2: @Factory.WithSelf.Method.%Factory.type (%Factory.type.711) = specific_constant @Factory.%Self.loc4_29.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self.9c2)]
// CHECK:STDOUT:         %Self.ref: @Factory.WithSelf.Method.%Factory.type (%Factory.type.711) = name_ref Self, %.loc8_13.2 [symbolic = %Self (constants.%Self.9c2)]
// CHECK:STDOUT:         %Self.as_type.loc8_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc8_13.3: type = converted %Self.ref, %Self.as_type.loc8_13.2 [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Factory.WithSelf.Method.%Self.as_type.loc8_13.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Method.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Method.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_23.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = assoc_entity element1, %Factory.WithSelf.Method.decl [symbolic = %assoc1.loc8_23.2 (constants.%assoc1.b7b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_29.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = @Factory.WithSelf.%assoc0.loc6_17.1
// CHECK:STDOUT:     .Method = @Factory.WithSelf.%assoc1.loc8_23.1
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     witness = (@Factory.WithSelf.%Factory.WithSelf.Make.decl, @Factory.WithSelf.%Factory.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %A.as.Factory.impl.Make.decl: %A.as.Factory.impl.Make.type = fn_decl @A.as.Factory.impl.Make [concrete = constants.%A.as.Factory.impl.Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %B.ref [concrete = constants.%.d83]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.as.Factory.impl.Method.decl: %A.as.Factory.impl.Method.type = fn_decl @A.as.Factory.impl.Method [concrete = constants.%A.as.Factory.impl.Method] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %B.ref [concrete = constants.%.d83]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Factory.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%A.as.Factory.impl.Make.decl, %A.as.Factory.impl.Method.decl), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness %Factory.impl_witness_table [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .Make = %A.as.Factory.impl.Make.decl
// CHECK:STDOUT:   .Method = %A.as.Factory.impl.Method.decl
// CHECK:STDOUT:   extend %Factory.type
// CHECK:STDOUT:   witness = %Factory.impl_witness
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.711)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc6_16.1: Core.Form = init_form %T [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc6_16.2: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_13.2: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_16.2, %T [symbolic = %return.patt.loc6_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Factory.WithSelf.Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.711)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9c2)]
// CHECK:STDOUT:   %Self.as_type.loc8_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_13: type = pattern_type %Self.as_type.loc8_13.1 [symbolic = %pattern_type.loc8_13 (constants.%pattern_type.5ba)]
// CHECK:STDOUT:   %self.param_patt.loc8_13.2: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.5ba) = value_param_pattern [symbolic = %self.param_patt.loc8_13.2 (constants.%self.param_patt.d68)]
// CHECK:STDOUT:   %self.patt.loc8_13.2: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.5ba) = at_binding_pattern self, %self.param_patt.loc8_13.2 [symbolic = %self.patt.loc8_13.2 (constants.%self.patt.79a)]
// CHECK:STDOUT:   %.loc8_22.1: Core.Form = init_form %T [symbolic = %.loc8_22.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc8_22: type = pattern_type %T [symbolic = %pattern_type.loc8_22 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc8_22.2: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_22.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc8_19.2: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_22.2, %T [symbolic = %return.patt.loc8_19.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Factory.WithSelf.Method.%Self.as_type.loc8_13.1 (%Self.as_type)) -> out %return.param: @Factory.WithSelf.Method.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method(%self.param: %A) -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.9c2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.9c2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc6_16.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_16.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_13.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.9c2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.711
// CHECK:STDOUT:   %Self => constants.%Self.9c2
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.5ba
// CHECK:STDOUT:   %self.param_patt.loc8_13.2 => constants.%self.param_patt.d68
// CHECK:STDOUT:   %self.patt.loc8_13.2 => constants.%self.patt.79a
// CHECK:STDOUT:   %.loc8_22.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc8_22.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc8_19.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc4_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_20.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self.loc4_29.2 => constants.%Self.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.9c2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Self.9c2
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.264
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.26e
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.8bd
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.012
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.dc9
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.5a4
// CHECK:STDOUT:   %assoc1.loc8_23.2 => constants.%assoc1.682
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.0c4
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.f30
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.8bd
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.012
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.99a
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.637
// CHECK:STDOUT:   %assoc1.loc8_23.2 => constants.%assoc1.682
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %.loc6_16.1 => constants.%.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %return.param_patt.loc6_16.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc6_13.2 => constants.%return.patt.911
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc8_13.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc8_13.2 => constants.%self.patt.312
// CHECK:STDOUT:   %.loc8_22.1 => constants.%.d83
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.e39
// CHECK:STDOUT:   %return.param_patt.loc8_22.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc8_19.2 => constants.%return.patt.911
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.711: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.3e3: %Factory.type.711 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0ed: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.d40: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.459: %Factory.WithSelf.Method.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.3e3 [symbolic]
// CHECK:STDOUT:   %pattern_type.b1a: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.f89: %pattern_type.b1a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.5e0: %pattern_type.b1a = at_binding_pattern self, %self.param_patt.f89 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.cf7: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.86c: %Factory.WithSelf.Make.type.cf7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.5e3: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.739: %Factory.type.5e3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.189: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.42f: %Factory.WithSelf.Make.type.189 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.8bd: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.e71: %Factory.assoc_type.8bd = assoc_entity element0, imports.%Main.import_ref.0a8 [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.85b: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.18d: %Factory.WithSelf.Method.type.85b = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.e2e: %Factory.assoc_type.8bd = assoc_entity element1, imports.%Main.import_ref.ad0 [concrete]
// CHECK:STDOUT:   %assoc0.c0a: %Factory.assoc_type.0ed = assoc_entity element0, imports.%Main.import_ref.587 [symbolic]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness imports.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.5e3 = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.f93: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.57a: %Factory.WithSelf.Make.type.f93 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.053: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.d76: %Factory.WithSelf.Method.type.053 = struct_value () [concrete]
// CHECK:STDOUT:   %.9f3: type = fn_type_with_self_type %Factory.WithSelf.Make.type.f93, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.c00: %Factory.assoc_type.0ed = assoc_entity element1, imports.%Main.import_ref.82f [symbolic]
// CHECK:STDOUT:   %.076: type = fn_type_with_self_type %Factory.WithSelf.Method.type.053, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.9da = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//factory, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.7b8 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a26: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.c0a)]
// CHECK:STDOUT:   %Main.import_ref.1ef: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.c00)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.ad0: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.d40) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.459)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.0f8e01.2: @Factory.%Factory.type (%Factory.type.711) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Main.import_ref.0a8: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.cf7) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.86c)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.0f8e01.3: @Factory.%Factory.type (%Factory.type.711) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Main.import_ref.a45 = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8c3: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.1d4: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.5e3]
// CHECK:STDOUT:   %Main.import_ref.213 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ee = import_ref Main//factory, loc16_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.ebb = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.587 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.127: %A.as.Factory.impl.Make.type = import_ref Main//factory, loc15_17, loaded [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %Main.import_ref.84c: %A.as.Factory.impl.Method.type = import_ref Main//factory, loc16_23, loaded [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%Main.import_ref.127, %Main.import_ref.84c), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.82f = import_ref Main//factory, loc8_23, unloaded
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref.loc4 [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc4_15.2: Core.Form = init_form %B.ref.loc4 [concrete = constants.%.d83]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceB.decl: %InstanceB.type = fn_decl @InstanceB [concrete = constants.%InstanceB] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref.loc7 [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc7: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7_23.2: Core.Form = init_form %B.ref.loc7 [concrete = constants.%.d83]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a45
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.a26
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.1ef
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.1d4 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.213
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.1ee
// CHECK:STDOUT:   extend imports.%Main.import_ref.ebb
// CHECK:STDOUT:   witness = imports.%Main.import_ref.8c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.0f8e01.2: @Factory.%Factory.type (%Factory.type.711)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.b1a)]
// CHECK:STDOUT:   %self.param_patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.b1a) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.f89)]
// CHECK:STDOUT:   %self.patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.b1a) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.5e0)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.0f8e01.3: @Factory.%Factory.type (%Factory.type.711)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB() -> out %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.9da = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.5e3]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.8bd = specific_constant imports.%Main.import_ref.a26, @Factory.WithSelf(constants.%B, constants.%Self.3e3) [concrete = constants.%assoc0.e71]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.8bd = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.e71]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.5e3 = facet_value %A.ref, (constants.%Factory.impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.5e3 = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %impl.elem0: %.9f3 = impl_witness_access constants.%Factory.impl_witness, element0 [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %.loc4_15.1: ref %B = splice_block %return.param {}
// CHECK:STDOUT:   %A.as.Factory.impl.Make.call: init %B to %.loc4_15.1 = call %impl.elem0()
// CHECK:STDOUT:   return %A.as.Factory.impl.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceB(%a.param: %A) -> out %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.9da = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.5e3]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.8bd = specific_constant imports.%Main.import_ref.1ef, @Factory.WithSelf(constants.%B, constants.%Self.3e3) [concrete = constants.%assoc1.e2e]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.8bd = name_ref Method, %.loc8 [concrete = constants.%assoc1.e2e]
// CHECK:STDOUT:   %impl.elem1: %.076 = impl_witness_access constants.%Factory.impl_witness, element1 [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %.loc7_23.1: ref %B = splice_block %return.param {}
// CHECK:STDOUT:   %A.as.Factory.impl.Method.call: init %B to %.loc7_23.1 = call %bound_method(%a.ref)
// CHECK:STDOUT:   return %A.as.Factory.impl.Method.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.3e3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.3e3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.711
// CHECK:STDOUT:   %Self => constants.%Self.3e3
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.b1a
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.f89
// CHECK:STDOUT:   %self.patt => constants.%self.patt.5e0
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.3e3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Self.739
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.3e3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Self.3e3
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.189
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.42f
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.8bd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e71
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.85b
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.18d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.f93
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.57a
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.8bd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e71
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.053
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.d76
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.711: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.3e3: %Factory.type.711 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0ed: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.d40: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.459: %Factory.WithSelf.Method.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.3e3 [symbolic]
// CHECK:STDOUT:   %pattern_type.b1a: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.b1a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.b1a = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.cf7: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.86c: %Factory.WithSelf.Make.type.cf7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.5e3: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.739: %Factory.type.5e3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.8fc: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Self.60c: %Factory.type.8fc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.f60: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%C, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.597: %Factory.WithSelf.Make.type.f60 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.fc0: type = assoc_entity_type @Factory, @Factory(%C) [concrete]
// CHECK:STDOUT:   %assoc0.c53: %Factory.assoc_type.fc0 = assoc_entity element0, imports.%Main.import_ref.0a8 [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.b2d: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%C, %Self.3e3) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.e0d: %Factory.WithSelf.Method.type.b2d = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.f6c: %Factory.assoc_type.fc0 = assoc_entity element1, imports.%Main.import_ref.ad0 [concrete]
// CHECK:STDOUT:   %assoc0.c0a: %Factory.assoc_type.0ed = assoc_entity element0, imports.%Main.import_ref.587 [symbolic]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.c00: %Factory.assoc_type.0ed = assoc_entity element1, imports.%Main.import_ref.82f [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.9da = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.7b8 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a26: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.c0a)]
// CHECK:STDOUT:   %Main.import_ref.1ef: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0ed) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.c00)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.ad0: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.d40) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.459)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.0f8e01.2: @Factory.%Factory.type (%Factory.type.711) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Main.import_ref.0a8: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.cf7) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.86c)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.0f8e01.3: @Factory.%Factory.type (%Factory.type.711) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Main.import_ref.a45 = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.97f = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.1d4: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.5e3]
// CHECK:STDOUT:   %Main.import_ref.213 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ee = import_ref Main//factory, loc16_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.ebb = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.587 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.82f = import_ref Main//factory, loc8_23, unloaded
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.a44]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceC.decl: %InstanceC.type = fn_decl @InstanceC [concrete = constants.%InstanceC] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc16 [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %C.ref.loc16 [concrete = constants.%.a44]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a45
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.a26
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.1ef
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.1d4 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.213
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.1ee
// CHECK:STDOUT:   extend imports.%Main.import_ref.ebb
// CHECK:STDOUT:   witness = imports.%Main.import_ref.97f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0ee
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.0f8e01.2: @Factory.%Factory.type (%Factory.type.711)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.711)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.711) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3e3)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.b1a)]
// CHECK:STDOUT:   %self.param_patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.b1a) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.b1a) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.0f8e01.3: @Factory.%Factory.type (%Factory.type.711)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.9da = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.8fc]
// CHECK:STDOUT:   %.loc13: %Factory.assoc_type.fc0 = specific_constant imports.%Main.import_ref.a26, @Factory.WithSelf(constants.%C, constants.%Self.3e3) [concrete = constants.%assoc0.c53]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.fc0 = name_ref Make, %.loc13 [concrete = constants.%assoc0.c53]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param: %A) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.9da = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.8fc]
// CHECK:STDOUT:   %.loc21: %Factory.assoc_type.fc0 = specific_constant imports.%Main.import_ref.1ef, @Factory.WithSelf(constants.%C, constants.%Self.3e3) [concrete = constants.%assoc1.f6c]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.fc0 = name_ref Method, %.loc21 [concrete = constants.%assoc1.f6c]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.3e3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.3e3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.711
// CHECK:STDOUT:   %Self => constants.%Self.3e3
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.b1a
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt => constants.%self.patt
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.3e3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5e3
// CHECK:STDOUT:   %Self => constants.%Self.739
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fc
// CHECK:STDOUT:   %Self => constants.%Self.60c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%C, constants.%Self.3e3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fc
// CHECK:STDOUT:   %Self => constants.%Self.3e3
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.f60
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.597
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.fc0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.b2d
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.e0d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
