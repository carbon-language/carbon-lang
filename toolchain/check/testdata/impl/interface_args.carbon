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

interface ImplicitAs(Dest: type) {
  fn Convert(self) -> Dest;
}

// --- action.carbon

library "[[@TEST_NAME]]";

interface Action(T: type) {
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

interface Factory(T: type) {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Dest.patt: %pattern_type.9a5 = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b02: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.b02 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.97b: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.97b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.97b = wrapper_binding_pattern self, %self.param_patt [symbolic]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.Destroy = import_ref Core//prelude/parts/destroy, Destroy, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Destroy = imports.%Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.0ff = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_26.1: %pattern_type.9a5 = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_26.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_28.1: type = splice_block %.loc3_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Dest.loc3_26.2: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_26.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_26.2: type) {
// CHECK:STDOUT:   %Dest.patt.loc3_26.2: %pattern_type.9a5 = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_26.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   %Dest.loc3_26.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_26.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_26.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self.loc3_34.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_34.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.param_patt.loc4_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.97b) = value_param_pattern [symbolic = %self.param_patt.loc4_14.2 (constants.%self.param_patt)]
// CHECK:STDOUT:       %self.patt.loc4_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.97b) = wrapper_binding_pattern self, %self.param_patt.loc4_14.1 [symbolic = %self.patt.loc4_14.2 (constants.%self.patt)]
// CHECK:STDOUT:       %return.param_patt.loc4_23.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc4_23.2 (constants.%return.param_patt)]
// CHECK:STDOUT:       %return.patt.loc4_20.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc4_23.1, %Dest.ref [symbolic = %return.patt.loc4_20.2 (constants.%return.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_26.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %.loc4_23.2: Core.Form = init_form %Dest.ref [symbolic = %.loc4_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_14.1: type = splice_block %.loc4_14.3 [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_14.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = specific_constant @ImplicitAs.%Self.loc3_34.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = name_ref Self, %.loc4_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_14.3: type = converted %Self.ref, %Self.as_type.loc4_14.2 [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_14.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_27.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc4_27.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_34.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc4_27.1
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%Dest.loc3_26.2: type, @ImplicitAs.%Self.loc3_34.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.as_type.loc4_14.1 [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.97b)]
// CHECK:STDOUT:   %self.param_patt.loc4_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.97b) = value_param_pattern [symbolic = %self.param_patt.loc4_14.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc4_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.97b) = wrapper_binding_pattern self, %self.param_patt.loc4_14.2 [symbolic = %self.patt.loc4_14.2 (constants.%self.patt)]
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
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.b02
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.97b
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.849: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fb7: %Action.type.849 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.fb7 [symbolic]
// CHECK:STDOUT:   %pattern_type.358: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.6b2: %pattern_type.358 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.816: %pattern_type.358 = wrapper_binding_pattern self, %self.param_patt.6b2 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.8a1: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.fb7) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.2f0: %Action.WithSelf.Op.type.8a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1e1: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.ef4: %Action.assoc_type.1e1 = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.d56: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %.cde: <witness> = impl_self_witness %A, @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness @A.as.Action.impl.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.d4a: %Action.type.d56 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.17f: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.fb7) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.fa7: %Action.WithSelf.Op.type.17f = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.2b4: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.6bf: %Action.assoc_type.2b4 = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7cd: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.d56 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.632: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.90d: %Action.WithSelf.Op.type.632 = struct_value () [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.603: type = fn_type_with_self_type %Action.WithSelf.Op.type.632, %Action.facet [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_19.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_21.2: type = type_literal type [concrete = type]
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
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.d56]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @A.as.Action.impl.%A.ref, @Action, @Action(constants.%B) [concrete = constants.%.cde]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_19.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_19.1)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self.loc4_27.2: @Action.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self.loc4_27.2 (constants.%Self.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_27.1: @Action.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self.loc4_27.2 (constants.%Self.fb7)]
// CHECK:STDOUT:     %Action.WithSelf.decl = interface_with_self_decl @Action [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Action.WithSelf.Op.decl: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.8a1) = fn_decl @Action.WithSelf.Op [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.2f0)] {
// CHECK:STDOUT:       %self.param_patt.loc5_9.1: @Action.WithSelf.Op.%pattern_type (%pattern_type.358) = value_param_pattern [symbolic = %self.param_patt.loc5_9.2 (constants.%self.param_patt.6b2)]
// CHECK:STDOUT:       %self.patt.loc5_9.1: @Action.WithSelf.Op.%pattern_type (%pattern_type.358) = wrapper_binding_pattern self, %self.param_patt.loc5_9.1 [symbolic = %self.patt.loc5_9.2 (constants.%self.patt.816)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Action.WithSelf.Op.%Self.as_type.loc5_9.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_9.1: type = splice_block %.loc5_9.3 [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_9.2: @Action.WithSelf.Op.%Action.type (%Action.type.849) = specific_constant @Action.%Self.loc4_27.1, @Action(constants.%T) [symbolic = %Self (constants.%Self.fb7)]
// CHECK:STDOUT:         %Self.ref: @Action.WithSelf.Op.%Action.type (%Action.type.849) = name_ref Self, %.loc5_9.2 [symbolic = %Self (constants.%Self.fb7)]
// CHECK:STDOUT:         %Self.as_type.loc5_9.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_9.3: type = converted %Self.ref, %Self.as_type.loc5_9.2 [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Action.WithSelf.Op.%Self.as_type.loc5_9.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1e1) = assoc_entity element0, %Action.WithSelf.Op.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.ef4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_27.1
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
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7cd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Action.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
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
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(@Action.%T.loc4_19.2: type, @Action.%Self.loc4_27.1: @Action.%Action.type (%Action.type.849)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.fb7)]
// CHECK:STDOUT:   %Self.as_type.loc5_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_9.1 [symbolic = %pattern_type (constants.%pattern_type.358)]
// CHECK:STDOUT:   %self.param_patt.loc5_9.2: @Action.WithSelf.Op.%pattern_type (%pattern_type.358) = value_param_pattern [symbolic = %self.param_patt.loc5_9.2 (constants.%self.param_patt.6b2)]
// CHECK:STDOUT:   %self.patt.loc5_9.2: @Action.WithSelf.Op.%pattern_type (%pattern_type.358) = wrapper_binding_pattern self, %self.param_patt.loc5_9.2 [symbolic = %self.patt.loc5_9.2 (constants.%self.patt.816)]
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.d56]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.2b4 = specific_constant @Action.WithSelf.%assoc0.loc5_14.1, @Action.WithSelf(constants.%B, constants.%Self.fb7) [concrete = constants.%assoc0.6bf]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.2b4 = name_ref Op, %.loc16 [concrete = constants.%assoc0.6bf]
// CHECK:STDOUT:   %impl.elem0: %.603 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
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
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.fb7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.fb7) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.849
// CHECK:STDOUT:   %Self => constants.%Self.fb7
// CHECK:STDOUT:   %Self.as_type.loc5_9.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.358
// CHECK:STDOUT:   %self.param_patt.loc5_9.2 => constants.%self.param_patt.6b2
// CHECK:STDOUT:   %self.patt.loc5_9.2 => constants.%self.patt.816
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self.loc4_27.2 => constants.%Self.d4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.fb7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Self.fb7
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.17f
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.fa7
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.2b4
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.6bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.632
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.90d
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.2b4
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.6bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.as_type.loc5_9.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc5_9.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc5_9.2 => constants.%self.patt.7cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.849: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.60a: %Action.type.849 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1e1: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.e35: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.60a) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.f36: %Action.WithSelf.Op.type.e35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.60a [symbolic]
// CHECK:STDOUT:   %pattern_type.1b3: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.e33: %pattern_type.1b3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.07e: %pattern_type.1b3 = wrapper_binding_pattern self, %self.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Action.type.d56: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.858: %Action.type.d56 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.5fa: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.60a) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.bfe: %Action.WithSelf.Op.type.5fa = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.2b4: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.f05: %Action.assoc_type.2b4 = assoc_entity element0, imports.%Main.import_ref.1bb [concrete]
// CHECK:STDOUT:   %assoc0.71a: %Action.assoc_type.1e1 = assoc_entity element0, imports.%Main.import_ref.c27 [symbolic]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness imports.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.d56 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.946: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.c67: %Action.WithSelf.Op.type.946 = struct_value () [concrete]
// CHECK:STDOUT:   %.d02: type = fn_type_with_self_type %Action.WithSelf.Op.type.946, %Action.facet [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.8fb: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1e1) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.71a)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.1bb: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.e35) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.f36)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3eaa6e.2: @Action.%Action.type (%Action.type.849) = import_ref Main//action, loc4_27, loaded [symbolic = @Action.%Self (constants.%Self.60a)]
// CHECK:STDOUT:   %Main.import_ref.ad4 = import_ref Main//action, loc4_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f0: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.a16: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.d56]
// CHECK:STDOUT:   %Main.import_ref.294 = import_ref Main//action, loc13_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.336 = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.c27 = import_ref Main//action, loc5_14, unloaded
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
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ad4
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.8fb
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.a16 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.294
// CHECK:STDOUT:   extend imports.%Main.import_ref.336
// CHECK:STDOUT:   witness = imports.%Main.import_ref.8f0
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
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.3eaa6e.2: @Action.%Action.type (%Action.type.849)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.60a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.1b3)]
// CHECK:STDOUT:   %self.param_patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1b3) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e33)]
// CHECK:STDOUT:   %self.patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1b3) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.07e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.538 = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.d56]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.2b4 = specific_constant imports.%Main.import_ref.8fb, @Action.WithSelf(constants.%B, constants.%Self.60a) [concrete = constants.%assoc0.f05]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.2b4 = name_ref Op, %.loc4 [concrete = constants.%assoc0.f05]
// CHECK:STDOUT:   %impl.elem0: %.d02 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
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
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.60a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.60a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.849
// CHECK:STDOUT:   %Self => constants.%Self.60a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1b3
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e33
// CHECK:STDOUT:   %self.patt => constants.%self.patt.07e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Self.858
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.60a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Self.60a
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.5fa
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.bfe
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.2b4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.946
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.c67
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.2b4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.538: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.849: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.60a: %Action.type.849 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1e1: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.e35: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.60a) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.f36: %Action.WithSelf.Op.type.e35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.60a [symbolic]
// CHECK:STDOUT:   %pattern_type.1b3: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.1b3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.1b3 = wrapper_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Action.type.d56: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.858: %Action.type.d56 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.d5e: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Self.a8f: %Action.type.d5e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.2e9: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%C, %Self.60a) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.727: %Action.WithSelf.Op.type.2e9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.1ac: type = assoc_entity_type @Action, @Action(%C) [concrete]
// CHECK:STDOUT:   %assoc0.933: %Action.assoc_type.1ac = assoc_entity element0, imports.%Main.import_ref.1bb [concrete]
// CHECK:STDOUT:   %assoc0.71a: %Action.assoc_type.1e1 = assoc_entity element0, imports.%Main.import_ref.c27 [symbolic]
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
// CHECK:STDOUT:   %Main.import_ref.8fb: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.1e1) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.71a)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.1bb: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.e35) = import_ref Main//action, loc5_14, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.f36)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3eaa6e.2: @Action.%Action.type (%Action.type.849) = import_ref Main//action, loc4_27, loaded [symbolic = @Action.%Self (constants.%Self.60a)]
// CHECK:STDOUT:   %Main.import_ref.ad4 = import_ref Main//action, loc4_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7ac = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.a16: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.d56]
// CHECK:STDOUT:   %Main.import_ref.294 = import_ref Main//action, loc13_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.336 = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c27 = import_ref Main//action, loc5_14, unloaded
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
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ad4
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.8fb
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.a16 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.294
// CHECK:STDOUT:   extend imports.%Main.import_ref.336
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7ac
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
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.3eaa6e.2: @Action.%Action.type (%Action.type.849)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.849)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.849) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.60a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.1b3)]
// CHECK:STDOUT:   %self.param_patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1b3) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.1b3) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.538 = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.d5e]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.1ac = specific_constant imports.%Main.import_ref.8fb, @Action.WithSelf(constants.%C, constants.%Self.60a) [concrete = constants.%assoc0.933]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.1ac = name_ref Op, %.loc8 [concrete = constants.%assoc0.933]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.60a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.60a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.849
// CHECK:STDOUT:   %Self => constants.%Self.60a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1b3
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt => constants.%self.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d56
// CHECK:STDOUT:   %Self => constants.%Self.858
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d5e
// CHECK:STDOUT:   %Self => constants.%Self.a8f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%C, constants.%Self.60a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Action.type => constants.%Action.type.d5e
// CHECK:STDOUT:   %Self => constants.%Self.60a
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.2e9
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.727
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.1ac
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.933
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.377: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c06: %Factory.type.377 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.250: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.c06) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.310: %Factory.WithSelf.Make.type.250 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0a9: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.a02: %Factory.assoc_type.0a9 = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c06 [symbolic]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.492: %pattern_type.8a5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.226: %pattern_type.8a5 = wrapper_binding_pattern self, %self.param_patt.492 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.2c7: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.c06) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.59a: %Factory.WithSelf.Method.type.2c7 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.8ee: %Factory.assoc_type.0a9 = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.25d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %.efe: <witness> = impl_self_witness %A, @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness @A.as.Factory.impl.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.89c: %Factory.type.25d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.b4d: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.c06) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.27b: %Factory.WithSelf.Make.type.b4d = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.ee0: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.e06: %Factory.assoc_type.ee0 = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.b0f: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.c06) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.455: %Factory.WithSelf.Method.type.b0f = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.b70: %Factory.assoc_type.ee0 = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7cd: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.25d = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.186: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.d69: %Factory.WithSelf.Make.type.186 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.1ca: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.731: %Factory.WithSelf.Method.type.1ca = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_22.1: type = splice_block %.loc4_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Factory.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.9da = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.25d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @A.as.Factory.impl.%A.ref, @Factory, @Factory(constants.%B) [concrete = constants.%.efe]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_20.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_20.1)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self.loc4_28.2: @Factory.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.c06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @Factory.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.c06)]
// CHECK:STDOUT:     %Factory.WithSelf.decl = interface_with_self_decl @Factory [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Factory.WithSelf.Make.decl: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.250) = fn_decl @Factory.WithSelf.Make [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.310)] {
// CHECK:STDOUT:       %return.param_patt.loc6_16.1: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_13.1: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_16.1, %T.ref [symbolic = %return.patt.loc6_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_16.2: Core.Form = init_form %T.ref [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = assoc_entity element0, %Factory.WithSelf.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.a02)]
// CHECK:STDOUT:     %Factory.WithSelf.Method.decl: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.2c7) = fn_decl @Factory.WithSelf.Method [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.59a)] {
// CHECK:STDOUT:       %self.param_patt.loc8_13.1: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.8a5) = value_param_pattern [symbolic = %self.param_patt.loc8_13.2 (constants.%self.param_patt.492)]
// CHECK:STDOUT:       %self.patt.loc8_13.1: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.8a5) = wrapper_binding_pattern self, %self.param_patt.loc8_13.1 [symbolic = %self.patt.loc8_13.2 (constants.%self.patt.226)]
// CHECK:STDOUT:       %return.param_patt.loc8_22.1: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc8_22.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc8_19.1: @Factory.WithSelf.Method.%pattern_type.loc8_22 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc8_22.1, %T.ref [symbolic = %return.patt.loc8_19.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc8_22.2: Core.Form = init_form %T.ref [symbolic = %.loc8_22.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Factory.WithSelf.Method.%Self.as_type.loc8_13.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_13.1: type = splice_block %.loc8_13.3 [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc8_13.2: @Factory.WithSelf.Method.%Factory.type (%Factory.type.377) = specific_constant @Factory.%Self.loc4_28.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self.c06)]
// CHECK:STDOUT:         %Self.ref: @Factory.WithSelf.Method.%Factory.type (%Factory.type.377) = name_ref Self, %.loc8_13.2 [symbolic = %Self (constants.%Self.c06)]
// CHECK:STDOUT:         %Self.as_type.loc8_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc8_13.3: type = converted %Self.ref, %Self.as_type.loc8_13.2 [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Factory.WithSelf.Method.%Self.as_type.loc8_13.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Method.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Method.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_23.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = assoc_entity element1, %Factory.WithSelf.Method.decl [symbolic = %assoc1.loc8_23.2 (constants.%assoc1.8ee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
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
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7cd]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %B.ref [concrete = constants.%.d83]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Factory.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_28.1: @Factory.%Factory.type (%Factory.type.377)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc6_16.1: Core.Form = init_form %T [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc6_16.2: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_13.2: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_16.2, %T [symbolic = %return.patt.loc6_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Factory.WithSelf.Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_28.1: @Factory.%Factory.type (%Factory.type.377)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c06)]
// CHECK:STDOUT:   %Self.as_type.loc8_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_13: type = pattern_type %Self.as_type.loc8_13.1 [symbolic = %pattern_type.loc8_13 (constants.%pattern_type.8a5)]
// CHECK:STDOUT:   %self.param_patt.loc8_13.2: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.8a5) = value_param_pattern [symbolic = %self.param_patt.loc8_13.2 (constants.%self.param_patt.492)]
// CHECK:STDOUT:   %self.patt.loc8_13.2: @Factory.WithSelf.Method.%pattern_type.loc8_13 (%pattern_type.8a5) = wrapper_binding_pattern self, %self.param_patt.loc8_13.2 [symbolic = %self.patt.loc8_13.2 (constants.%self.patt.226)]
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
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.c06) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.c06) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc6_16.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_16.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_13.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.c06) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.377
// CHECK:STDOUT:   %Self => constants.%Self.c06
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.8a5
// CHECK:STDOUT:   %self.param_patt.loc8_13.2 => constants.%self.param_patt.492
// CHECK:STDOUT:   %self.patt.loc8_13.2 => constants.%self.patt.226
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self.loc4_28.2 => constants.%Self.89c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.c06) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Self.c06
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.b4d
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.27b
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.ee0
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.e06
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.b0f
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.455
// CHECK:STDOUT:   %assoc1.loc8_23.2 => constants.%assoc1.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.186
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.d69
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.ee0
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.e06
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.1ca
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.731
// CHECK:STDOUT:   %assoc1.loc8_23.2 => constants.%assoc1.b70
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc8_13.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc8_13.2 => constants.%self.patt.7cd
// CHECK:STDOUT:   %.loc8_22.1 => constants.%.d83
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.e39
// CHECK:STDOUT:   %return.param_patt.loc8_22.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc8_19.2 => constants.%return.patt.911
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.377: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.0bb: %Factory.type.377 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0a9: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.cc6: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.38d: %Factory.WithSelf.Method.type.cc6 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.0bb [symbolic]
// CHECK:STDOUT:   %pattern_type.f7d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.71c: %pattern_type.f7d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.ae2: %pattern_type.f7d = wrapper_binding_pattern self, %self.param_patt.71c [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.ac6: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.de8: %Factory.WithSelf.Make.type.ac6 = struct_value () [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Factory.type.25d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.7e4: %Factory.type.25d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.be5: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.0af: %Factory.WithSelf.Make.type.be5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.ee0: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.cd0: %Factory.assoc_type.ee0 = assoc_entity element0, imports.%Main.import_ref.e1a [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.490: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.70b: %Factory.WithSelf.Method.type.490 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.120: %Factory.assoc_type.ee0 = assoc_entity element1, imports.%Main.import_ref.f7c [concrete]
// CHECK:STDOUT:   %assoc0.f92: %Factory.assoc_type.0a9 = assoc_entity element0, imports.%Main.import_ref.95c [symbolic]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness imports.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.25d = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.cdf: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.cca: %Factory.WithSelf.Make.type.cdf = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.009: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.632: %Factory.WithSelf.Method.type.009 = struct_value () [concrete]
// CHECK:STDOUT:   %.0f2: type = fn_type_with_self_type %Factory.WithSelf.Make.type.cdf, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.1c9: %Factory.assoc_type.0a9 = assoc_entity element1, imports.%Main.import_ref.240 [symbolic]
// CHECK:STDOUT:   %.dbb: type = fn_type_with_self_type %Factory.WithSelf.Method.type.009, %Factory.facet [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.0c4: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.f92)]
// CHECK:STDOUT:   %Main.import_ref.998: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.1c9)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.f7c: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.cc6) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.38d)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.23a1c3.2: @Factory.%Factory.type (%Factory.type.377) = import_ref Main//factory, loc4_28, loaded [symbolic = @Factory.%Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Main.import_ref.e1a: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.ac6) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.de8)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.23a1c3.3: @Factory.%Factory.type (%Factory.type.377) = import_ref Main//factory, loc4_28, loaded [symbolic = @Factory.%Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Main.import_ref.324 = import_ref Main//factory, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.00b: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.56d: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.25d]
// CHECK:STDOUT:   %Main.import_ref.213 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ee = import_ref Main//factory, loc16_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.633 = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.95c = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.127: %A.as.Factory.impl.Make.type = import_ref Main//factory, loc15_17, loaded [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %Main.import_ref.84c: %A.as.Factory.impl.Method.type = import_ref Main//factory, loc16_23, loaded [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%Main.import_ref.127, %Main.import_ref.84c), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.240 = import_ref Main//factory, loc8_23, unloaded
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
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref.loc7 [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc7: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7_23.2: Core.Form = init_form %B.ref.loc7 [concrete = constants.%.d83]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.324
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.0c4
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.998
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.56d [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.213
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.1ee
// CHECK:STDOUT:   extend imports.%Main.import_ref.633
// CHECK:STDOUT:   witness = imports.%Main.import_ref.00b
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.23a1c3.2: @Factory.%Factory.type (%Factory.type.377)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f7d)]
// CHECK:STDOUT:   %self.param_patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.f7d) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.71c)]
// CHECK:STDOUT:   %self.patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.f7d) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.ae2)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.23a1c3.3: @Factory.%Factory.type (%Factory.type.377)) [from "factory.carbon"] {
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.25d]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.ee0 = specific_constant imports.%Main.import_ref.0c4, @Factory.WithSelf(constants.%B, constants.%Self.0bb) [concrete = constants.%assoc0.cd0]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.ee0 = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.cd0]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.25d = facet_value %A.ref, (constants.%Factory.impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.25d = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %impl.elem0: %.0f2 = impl_witness_access constants.%Factory.impl_witness, element0 [concrete = constants.%A.as.Factory.impl.Make]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.25d]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.ee0 = specific_constant imports.%Main.import_ref.998, @Factory.WithSelf(constants.%B, constants.%Self.0bb) [concrete = constants.%assoc1.120]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.ee0 = name_ref Method, %.loc8 [concrete = constants.%assoc1.120]
// CHECK:STDOUT:   %impl.elem1: %.dbb = impl_witness_access constants.%Factory.impl_witness, element1 [concrete = constants.%A.as.Factory.impl.Method]
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
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.0bb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.0bb) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.377
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f7d
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.71c
// CHECK:STDOUT:   %self.patt => constants.%self.patt.ae2
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.0bb) {
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Self.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.0bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.be5
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.0af
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.ee0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cd0
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.490
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.70b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.cdf
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.cca
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.ee0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cd0
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.009
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.632
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.9da: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.9da = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.377: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.0bb: %Factory.type.377 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.0a9: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.cc6: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.38d: %Factory.WithSelf.Method.type.cc6 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.0bb [symbolic]
// CHECK:STDOUT:   %pattern_type.f7d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.f7d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.f7d = wrapper_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.ac6: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.de8: %Factory.WithSelf.Make.type.ac6 = struct_value () [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Factory.type.25d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.7e4: %Factory.type.25d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.120: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Self.d23: %Factory.type.120 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.b4f: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%C, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.429: %Factory.WithSelf.Make.type.b4f = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.b33: type = assoc_entity_type @Factory, @Factory(%C) [concrete]
// CHECK:STDOUT:   %assoc0.593: %Factory.assoc_type.b33 = assoc_entity element0, imports.%Main.import_ref.e1a [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.40d: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%C, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.88b: %Factory.WithSelf.Method.type.40d = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.4d7: %Factory.assoc_type.b33 = assoc_entity element1, imports.%Main.import_ref.f7c [concrete]
// CHECK:STDOUT:   %assoc0.f92: %Factory.assoc_type.0a9 = assoc_entity element0, imports.%Main.import_ref.95c [symbolic]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.1c9: %Factory.assoc_type.0a9 = assoc_entity element1, imports.%Main.import_ref.240 [symbolic]
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
// CHECK:STDOUT:   %Main.import_ref.0c4: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.f92)]
// CHECK:STDOUT:   %Main.import_ref.998: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.0a9) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.1c9)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.f7c: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.cc6) = import_ref Main//factory, loc8_23, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.38d)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.23a1c3.2: @Factory.%Factory.type (%Factory.type.377) = import_ref Main//factory, loc4_28, loaded [symbolic = @Factory.%Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Main.import_ref.e1a: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.ac6) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.de8)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.23a1c3.3: @Factory.%Factory.type (%Factory.type.377) = import_ref Main//factory, loc4_28, loaded [symbolic = @Factory.%Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Main.import_ref.324 = import_ref Main//factory, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e58 = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.0ee = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.239: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.56d: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.25d]
// CHECK:STDOUT:   %Main.import_ref.213 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ee = import_ref Main//factory, loc16_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.633 = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.95c = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.240 = import_ref Main//factory, loc8_23, unloaded
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
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc16 [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %C.ref.loc16 [concrete = constants.%.a44]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.324
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.0c4
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.998
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.239 as imports.%Main.import_ref.56d [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.213
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.1ee
// CHECK:STDOUT:   extend imports.%Main.import_ref.633
// CHECK:STDOUT:   witness = imports.%Main.import_ref.e58
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.23a1c3.2: @Factory.%Factory.type (%Factory.type.377)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.377)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.377) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f7d)]
// CHECK:STDOUT:   %self.param_patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.f7d) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt: @Factory.WithSelf.Method.%pattern_type.1 (%pattern_type.f7d) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @Factory.WithSelf.Method.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.23a1c3.3: @Factory.%Factory.type (%Factory.type.377)) [from "factory.carbon"] {
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.120]
// CHECK:STDOUT:   %.loc13: %Factory.assoc_type.b33 = specific_constant imports.%Main.import_ref.0c4, @Factory.WithSelf(constants.%C, constants.%Self.0bb) [concrete = constants.%assoc0.593]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.b33 = name_ref Make, %.loc13 [concrete = constants.%assoc0.593]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param: %A) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.9da = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.120]
// CHECK:STDOUT:   %.loc21: %Factory.assoc_type.b33 = specific_constant imports.%Main.import_ref.998, @Factory.WithSelf(constants.%C, constants.%Self.0bb) [concrete = constants.%assoc1.4d7]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.b33 = name_ref Method, %.loc21 [concrete = constants.%assoc1.4d7]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.0bb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.0bb) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.377
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f7d
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt => constants.%self.patt
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.0bb) {
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.25d
// CHECK:STDOUT:   %Self => constants.%Self.7e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.120
// CHECK:STDOUT:   %Self => constants.%Self.d23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%C, constants.%Self.0bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.120
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.b4f
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.429
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.b33
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.593
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.40d
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.88b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
