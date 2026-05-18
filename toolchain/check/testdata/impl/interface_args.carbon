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
  fn Op[unused self: Self]() {}
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.152: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.152 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.524: type = pattern_type %Self.as_type [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.b23 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
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
// CHECK:STDOUT:   %Dest.loc3_26.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_26.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_26.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self.loc3_35.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_18 (%pattern_type.524) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_18 (%pattern_type.524) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_31 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_31 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %Dest.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_26.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %.loc4_31.2: Core.Form = init_form %Dest.ref [symbolic = %.loc4_31.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_binding self, %self.param
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
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%Dest.loc3_26.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_18: type = pattern_type %Self.as_type.loc4_20.1 [symbolic = %pattern_type.loc4_18 (constants.%pattern_type.524)]
// CHECK:STDOUT:   %.loc4_31.1: Core.Form = init_form %Dest [symbolic = %.loc4_31.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc4_31: type = pattern_type %Dest [symbolic = %pattern_type.loc4_31 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc4_20.1 (%Self.as_type)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_26.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.152
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc4_18 => constants.%pattern_type.524
// CHECK:STDOUT:   %.loc4_31.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc4_31 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.62c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.62c = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.32b: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.760: %Action.type.32b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.760 [symbolic]
// CHECK:STDOUT:   %pattern_type.9df: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.aac: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.760) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.06c: %Action.WithSelf.Op.type.aac = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.e83: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.143: %Action.assoc_type.e83 = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.788: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness @A.as.Action.impl.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.5f5: %Action.type.788 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.22b: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.760) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.f6f: %Action.WithSelf.Op.type.22b = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.c15: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.eea: %Action.assoc_type.c15 = assoc_entity element0, @Action.WithSelf.%Action.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.788 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.715: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.f85: %Action.WithSelf.Op.type.715 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ab1: type = fn_type_with_self_type %Action.WithSelf.Op.type.715, %Action.facet [concrete]
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
// CHECK:STDOUT:   %Action.decl: %Action.type.62c = interface_decl @Action [concrete = constants.%Action.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %Action.ref: %Action.type.62c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.788]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bbf = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_19.1)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self.loc4_28.2: @Action.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.760)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @Action.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.760)]
// CHECK:STDOUT:     %Action.WithSelf.decl = interface_with_self_decl @Action [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Action.WithSelf.Op.decl: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.aac) = fn_decl @Action.WithSelf.Op [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.06c)] {
// CHECK:STDOUT:       %self.param_patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.9df) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Action.WithSelf.Op.%pattern_type (%pattern_type.9df) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Action.WithSelf.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @Action.WithSelf.Op.%Action.type (%Action.type.32b) = specific_constant @Action.%Self.loc4_28.1, @Action(constants.%T) [symbolic = %Self (constants.%Self.760)]
// CHECK:STDOUT:         %Self.ref: @Action.WithSelf.Op.%Action.type (%Action.type.32b) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self.760)]
// CHECK:STDOUT:         %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Action.WithSelf.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_22.1: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.e83) = assoc_entity element0, %Action.WithSelf.Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.143)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
// CHECK:STDOUT:     .Op = @Action.WithSelf.%assoc0.loc5_22.1
// CHECK:STDOUT:     witness = (@Action.WithSelf.%Action.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %A.as.Action.impl.Op.decl: %A.as.Action.impl.Op.type = fn_decl @A.as.Action.impl.Op [concrete = constants.%A.as.Action.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.bbf = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(@Action.%T.loc4_19.2: type, @Action.%Self.loc4_28.1: @Action.%Action.type (%Action.type.32b)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.760)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_15.1 [symbolic = %pattern_type (constants.%pattern_type.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Action.WithSelf.Op.%Self.as_type.loc5_15.1 (%Self.as_type));
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
// CHECK:STDOUT:   %Action.ref: %Action.type.62c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.788]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.c15 = specific_constant @Action.WithSelf.%assoc0.loc5_22.1, @Action.WithSelf(constants.%B, constants.%Self.760) [concrete = constants.%assoc0.eea]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.c15 = name_ref Op, %.loc16 [concrete = constants.%assoc0.eea]
// CHECK:STDOUT:   %impl.elem0: %.ab1 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.760) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.760) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.32b
// CHECK:STDOUT:   %Self => constants.%Self.760
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self.loc4_28.2 => constants.%Self.5f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.760) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Self.760
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.22b
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.f6f
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.c15
// CHECK:STDOUT:   %assoc0.loc5_22.2 => constants.%assoc0.eea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.715
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.f85
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.c15
// CHECK:STDOUT:   %assoc0.loc5_22.2 => constants.%assoc0.eea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bbf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.62c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.62c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.32b: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.45e: %Action.type.32b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.e83: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.e8f: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.45e) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.e11: %Action.WithSelf.Op.type.e8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.45e [symbolic]
// CHECK:STDOUT:   %pattern_type.c70: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Action.type.788: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.550: %Action.type.788 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.1a5: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Self.45e) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.3a1: %Action.WithSelf.Op.type.1a5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.c15: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.340: %Action.assoc_type.c15 = assoc_entity element0, imports.%Main.import_ref.1e6 [concrete]
// CHECK:STDOUT:   %assoc0.14b: %Action.assoc_type.e83 = assoc_entity element0, imports.%Main.import_ref.b3c [symbolic]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness imports.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.788 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.29a: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%B, %Action.facet) [concrete]
// CHECK:STDOUT:   %Action.WithSelf.Op.be6: %Action.WithSelf.Op.type.29a = struct_value () [concrete]
// CHECK:STDOUT:   %.cb6: type = fn_type_with_self_type %Action.WithSelf.Op.type.29a, %Action.facet [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.62c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//action, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.505 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d07: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.e83) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.14b)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.1e6: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.e8f) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.e11)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.6a3ea5.2: @Action.%Action.type (%Action.type.32b) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.45e)]
// CHECK:STDOUT:   %Main.import_ref.82f = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.4f3: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.02c = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.819: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.46e: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.788]
// CHECK:STDOUT:   %Main.import_ref.7df = import_ref Main//action, loc13_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.cf9 = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3c = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.7a0: %A.as.Action.impl.Op.type = import_ref Main//action, loc13_30, loaded [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (%Main.import_ref.7a0), @A.as.Action.impl [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bbf = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.82f
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.d07
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.819 as imports.%Main.import_ref.46e [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7df
// CHECK:STDOUT:   extend imports.%Main.import_ref.cf9
// CHECK:STDOUT:   witness = imports.%Main.import_ref.4f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.02c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.6a3ea5.2: @Action.%Action.type (%Action.type.32b)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.45e)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.c70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.62c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.788]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.c15 = specific_constant imports.%Main.import_ref.d07, @Action.WithSelf(constants.%B, constants.%Self.45e) [concrete = constants.%assoc0.340]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.c15 = name_ref Op, %.loc4 [concrete = constants.%assoc0.340]
// CHECK:STDOUT:   %impl.elem0: %.cb6 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
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
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.45e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.45e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.32b
// CHECK:STDOUT:   %Self => constants.%Self.45e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Self.45e
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.1a5
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.3a1
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.c15
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.29a
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.be6
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.c15
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Action.type.62c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.62c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.32b: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.45e: %Action.type.32b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.e83: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.e8f: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%T, %Self.45e) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.e11: %Action.WithSelf.Op.type.e8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.45e [symbolic]
// CHECK:STDOUT:   %pattern_type.c70: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Action.type.788: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.550: %Action.type.788 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.075: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Self.127: %Action.type.075 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.type.a75: type = fn_type @Action.WithSelf.Op, @Action.WithSelf(%C, %Self.45e) [symbolic]
// CHECK:STDOUT:   %Action.WithSelf.Op.ed0: %Action.WithSelf.Op.type.a75 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.880: type = assoc_entity_type @Action, @Action(%C) [concrete]
// CHECK:STDOUT:   %assoc0.3ab: %Action.assoc_type.880 = assoc_entity element0, imports.%Main.import_ref.1e6 [concrete]
// CHECK:STDOUT:   %assoc0.14b: %Action.assoc_type.e83 = assoc_entity element0, imports.%Main.import_ref.b3c [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.62c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//action, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.505 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d07: @Action.WithSelf.%Action.assoc_type (%Action.assoc_type.e83) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.WithSelf.%assoc0 (constants.%assoc0.14b)]
// CHECK:STDOUT:   %Main.Op = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.1e6: @Action.WithSelf.%Action.WithSelf.Op.type (%Action.WithSelf.Op.type.e8f) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.WithSelf.%Action.WithSelf.Op (constants.%Action.WithSelf.Op.e11)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.6a3ea5.2: @Action.%Action.type (%Action.type.32b) = import_ref Main//action, loc4_28, loaded [symbolic = @Action.%Self (constants.%Self.45e)]
// CHECK:STDOUT:   %Main.import_ref.82f = import_ref Main//action, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//action, loc4_19, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.966 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.02c = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.819: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.46e: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.788]
// CHECK:STDOUT:   %Main.import_ref.7df = import_ref Main//action, loc13_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.cf9 = import_ref Main//action, loc12_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.d96 = import_ref Main//action, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3c = import_ref Main//action, loc5_22, unloaded
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bbf = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.b3bc94.3: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.82f
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.d07
// CHECK:STDOUT:     witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.819 as imports.%Main.import_ref.46e [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7df
// CHECK:STDOUT:   extend imports.%Main.import_ref.cf9
// CHECK:STDOUT:   witness = imports.%Main.import_ref.966
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.02c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.WithSelf.Op(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.6a3ea5.2: @Action.%Action.type (%Action.type.32b)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.32b)]
// CHECK:STDOUT:   %Self: @Action.WithSelf.Op.%Action.type (%Action.type.32b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.45e)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.c70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.62c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.075]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.880 = specific_constant imports.%Main.import_ref.d07, @Action.WithSelf(constants.%C, constants.%Self.45e) [concrete = constants.%assoc0.3ab]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.880 = name_ref Op, %.loc8 [concrete = constants.%assoc0.3ab]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%T, constants.%Self.45e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf.Op(constants.%T, constants.%Self.45e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.32b
// CHECK:STDOUT:   %Self => constants.%Self.45e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.788
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.075
// CHECK:STDOUT:   %Self => constants.%Self.127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.WithSelf(constants.%C, constants.%Self.45e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Action.type => constants.%Action.type.075
// CHECK:STDOUT:   %Self => constants.%Self.45e
// CHECK:STDOUT:   %Action.WithSelf.Op.type => constants.%Action.WithSelf.Op.type.a75
// CHECK:STDOUT:   %Action.WithSelf.Op => constants.%Action.WithSelf.Op.ed0
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.880
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.319: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.319 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.d25: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b8f: %Factory.type.d25 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.586: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.b8f) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.e2c: %Factory.WithSelf.Make.type.586 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.a3c: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.c71: %Factory.assoc_type.a3c = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.b8f [symbolic]
// CHECK:STDOUT:   %pattern_type.e12: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.988: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.b8f) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.a9b: %Factory.WithSelf.Method.type.988 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.74c: %Factory.assoc_type.a3c = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.8fe: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness @A.as.Factory.impl.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.085: %Factory.type.8fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.221: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.b8f) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.2b5: %Factory.WithSelf.Make.type.221 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.436: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.44d: %Factory.assoc_type.436 = assoc_entity element0, @Factory.WithSelf.%Factory.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.96f: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.b8f) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.084: %Factory.WithSelf.Method.type.96f = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.1b9: %Factory.assoc_type.436 = assoc_entity element1, @Factory.WithSelf.%Factory.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %.c10: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.a58: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.8fe = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.57c: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.7ea: %Factory.WithSelf.Make.type.57c = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.389: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.bd7: %Factory.WithSelf.Method.type.389 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Factory.decl: %Factory.type.319 = interface_decl @Factory [concrete = constants.%Factory.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %Factory.ref: %Factory.type.319 = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.8fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_20.2: type) {
// CHECK:STDOUT:   %T.loc4_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_20.1)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self.loc4_29.2: @Factory.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.b8f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.b8f)]
// CHECK:STDOUT:     %Factory.WithSelf.decl = interface_with_self_decl @Factory [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Factory.WithSelf.Make.decl: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.586) = fn_decl @Factory.WithSelf.Make [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.e2c)] {
// CHECK:STDOUT:       %return.param_patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Factory.WithSelf.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_16.2: Core.Form = init_form %T.ref [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = assoc_entity element0, %Factory.WithSelf.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.c71)]
// CHECK:STDOUT:     %Factory.WithSelf.Method.decl: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.988) = fn_decl @Factory.WithSelf.Method [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.a9b)] {
// CHECK:STDOUT:       %self.param_patt: @Factory.WithSelf.Method.%pattern_type.loc8_17 (%pattern_type.e12) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Factory.WithSelf.Method.%pattern_type.loc8_17 (%pattern_type.e12) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Factory.WithSelf.Method.%pattern_type.loc8_30 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Factory.WithSelf.Method.%pattern_type.loc8_30 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc8_30.2: Core.Form = init_form %T.ref [symbolic = %.loc8_30.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Factory.WithSelf.Method.%Self.as_type.loc8_19.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_19.1: type = splice_block %.loc8_19.3 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc8_19.2: @Factory.WithSelf.Method.%Factory.type (%Factory.type.d25) = specific_constant @Factory.%Self.loc4_29.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self.b8f)]
// CHECK:STDOUT:         %Self.ref: @Factory.WithSelf.Method.%Factory.type (%Factory.type.d25) = name_ref Self, %.loc8_19.2 [symbolic = %Self (constants.%Self.b8f)]
// CHECK:STDOUT:         %Self.as_type.loc8_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc8_19.3: type = converted %Self.ref, %Self.as_type.loc8_19.2 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Factory.WithSelf.Method.%Self.as_type.loc8_19.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Factory.WithSelf.Method.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Factory.WithSelf.Method.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_31.1: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = assoc_entity element1, %Factory.WithSelf.Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.74c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_29.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = @Factory.WithSelf.%assoc0.loc6_17.1
// CHECK:STDOUT:     .Method = @Factory.WithSelf.%assoc1.loc8_31.1
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     witness = (@Factory.WithSelf.%Factory.WithSelf.Make.decl, @Factory.WithSelf.%Factory.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %A.as.Factory.impl.Make.decl: %A.as.Factory.impl.Make.type = fn_decl @A.as.Factory.impl.Make [concrete = constants.%A.as.Factory.impl.Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a58 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a58 = return_slot_pattern %return.param_patt, %B.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %B.ref [concrete = constants.%.c10]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.as.Factory.impl.Method.decl: %A.as.Factory.impl.Method.type = fn_decl @A.as.Factory.impl.Method [concrete = constants.%A.as.Factory.impl.Method] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.bbf = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a58 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a58 = return_slot_pattern %return.param_patt, %B.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %B.ref [concrete = constants.%.c10]
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.d25)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc6_16.1: Core.Form = init_form %T [symbolic = %.loc6_16.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Factory.WithSelf.Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(@Factory.%T.loc4_20.2: type, @Factory.%Self.loc4_29.1: @Factory.%Factory.type (%Factory.type.d25)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.b8f)]
// CHECK:STDOUT:   %Self.as_type.loc8_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_17: type = pattern_type %Self.as_type.loc8_19.1 [symbolic = %pattern_type.loc8_17 (constants.%pattern_type.e12)]
// CHECK:STDOUT:   %.loc8_30.1: Core.Form = init_form %T [symbolic = %.loc8_30.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc8_30: type = pattern_type %T [symbolic = %pattern_type.loc8_30 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Factory.WithSelf.Method.%Self.as_type.loc8_19.1 (%Self.as_type)) -> out %return.param: @Factory.WithSelf.Method.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method(%self.param: %A) -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.loc4_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.b8f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.b8f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc6_16.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.b8f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.d25
// CHECK:STDOUT:   %Self => constants.%Self.b8f
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc8_17 => constants.%pattern_type.e12
// CHECK:STDOUT:   %.loc8_30.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_20.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self.loc4_29.2 => constants.%Self.085
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.b8f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Self.b8f
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.221
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.2b5
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.436
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.44d
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.96f
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.084
// CHECK:STDOUT:   %assoc1.loc8_31.2 => constants.%assoc1.1b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.57c
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.7ea
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.436
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.44d
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.389
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.bd7
// CHECK:STDOUT:   %assoc1.loc8_31.2 => constants.%assoc1.1b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %.loc6_16.1 => constants.%.c10
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc8_17 => constants.%pattern_type.bbf
// CHECK:STDOUT:   %.loc8_30.1 => constants.%.c10
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.a58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.319: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.319 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.d25: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2e6: %Factory.type.d25 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.a3c: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.582: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.0d5: %Factory.WithSelf.Method.type.582 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.2e6 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f6: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.5fe: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.d16: %Factory.WithSelf.Make.type.5fe = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.8fe: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.078: %Factory.type.8fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %.c10: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.a58: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.7df: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.ce2: %Factory.WithSelf.Make.type.7df = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.436: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.bf6: %Factory.assoc_type.436 = assoc_entity element0, imports.%Main.import_ref.c77 [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.aea: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.c91: %Factory.WithSelf.Method.type.aea = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.d38: %Factory.assoc_type.436 = assoc_entity element1, imports.%Main.import_ref.d94 [concrete]
// CHECK:STDOUT:   %assoc0.2f9: %Factory.assoc_type.a3c = assoc_entity element0, imports.%Main.import_ref.db6 [symbolic]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness imports.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.8fe = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.b63: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Make.8f4: %Factory.WithSelf.Make.type.b63 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.af8: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%B, %Factory.facet) [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.893: %Factory.WithSelf.Method.type.af8 = struct_value () [concrete]
// CHECK:STDOUT:   %.b2a: type = fn_type_with_self_type %Factory.WithSelf.Make.type.b63, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.e42: %Factory.assoc_type.a3c = assoc_entity element1, imports.%Main.import_ref.b43 [symbolic]
// CHECK:STDOUT:   %.a27: type = fn_type_with_self_type %Factory.WithSelf.Method.type.af8, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.319 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//factory, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.505 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.ba3: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.2f9)]
// CHECK:STDOUT:   %Main.import_ref.6a7: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.e42)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.d94: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.582) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.0d5)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c8fa2a.2: @Factory.%Factory.type (%Factory.type.d25) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Main.import_ref.c77: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.5fe) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.d16)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c8fa2a.3: @Factory.%Factory.type (%Factory.type.d25) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Main.import_ref.83e = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.101: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.02c = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.819: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.481: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.8fe]
// CHECK:STDOUT:   %Main.import_ref.642 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.6c1 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.b60 = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.db6 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.eac: %A.as.Factory.impl.Make.type = import_ref Main//factory, loc15_17, loaded [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %Main.import_ref.b7d: %A.as.Factory.impl.Method.type = import_ref Main//factory, loc16_31, loaded [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%Main.import_ref.eac, %Main.import_ref.b7d), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b43 = import_ref Main//factory, loc8_31, unloaded
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a58 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a58 = return_slot_pattern %return.param_patt, %B.ref.loc4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc4_15.2: Core.Form = init_form %B.ref.loc4 [concrete = constants.%.c10]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceB.decl: %InstanceB.type = fn_decl @InstanceB [concrete = constants.%InstanceB] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bbf = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a58 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a58 = return_slot_pattern %return.param_patt, %B.ref.loc7 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc7: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7_23.2: Core.Form = init_form %B.ref.loc7 [concrete = constants.%.c10]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.2e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.83e
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.ba3
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.6a7
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.819 as imports.%Main.import_ref.481 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.642
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.6c1
// CHECK:STDOUT:   extend imports.%Main.import_ref.b60
// CHECK:STDOUT:   witness = imports.%Main.import_ref.101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.02c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.c8fa2a.2: @Factory.%Factory.type (%Factory.type.d25)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.7f6)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.c8fa2a.3: @Factory.%Factory.type (%Factory.type.d25)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB() -> out %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.319 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.8fe]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.436 = specific_constant imports.%Main.import_ref.ba3, @Factory.WithSelf(constants.%B, constants.%Self.2e6) [concrete = constants.%assoc0.bf6]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.436 = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.bf6]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.8fe = facet_value %A.ref, (constants.%Factory.impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.8fe = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %impl.elem0: %.b2a = impl_witness_access constants.%Factory.impl_witness, element0 [concrete = constants.%A.as.Factory.impl.Make]
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
// CHECK:STDOUT:   %Factory.ref: %Factory.type.319 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.8fe]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.436 = specific_constant imports.%Main.import_ref.6a7, @Factory.WithSelf(constants.%B, constants.%Self.2e6) [concrete = constants.%assoc1.d38]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.436 = name_ref Method, %.loc8 [concrete = constants.%assoc1.d38]
// CHECK:STDOUT:   %impl.elem1: %.a27 = impl_witness_access constants.%Factory.impl_witness, element1 [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %.loc7_23.1: ref %B = splice_block %return.param {}
// CHECK:STDOUT:   %A.as.Factory.impl.Method.call: init %B to %.loc7_23.1 = call %bound_method(%a.ref)
// CHECK:STDOUT:   return %A.as.Factory.impl.Method.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.2e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.2e6) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.d25
// CHECK:STDOUT:   %Self => constants.%Self.2e6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.7f6
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.2e6) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Self.078
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Self.2e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Self.2e6
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.7df
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.ce2
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.436
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf6
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.aea
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.c91
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.b63
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.8f4
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.436
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf6
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.af8
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.893
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Factory.type.319: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.319 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.d25: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2e6: %Factory.type.d25 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.a3c: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.582: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%T, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.0d5: %Factory.WithSelf.Method.type.582 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.2e6 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f6: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.5fe: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%T, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.d16: %Factory.WithSelf.Make.type.5fe = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.type.8fe: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.078: %Factory.type.8fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.b61: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.3bc: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Self.922: %Factory.type.3bc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.type.767: type = fn_type @Factory.WithSelf.Make, @Factory.WithSelf(%C, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Make.0fd: %Factory.WithSelf.Make.type.767 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.64a: type = assoc_entity_type @Factory, @Factory(%C) [concrete]
// CHECK:STDOUT:   %assoc0.366: %Factory.assoc_type.64a = assoc_entity element0, imports.%Main.import_ref.c77 [concrete]
// CHECK:STDOUT:   %Factory.WithSelf.Method.type.53b: type = fn_type @Factory.WithSelf.Method, @Factory.WithSelf(%C, %Self.2e6) [symbolic]
// CHECK:STDOUT:   %Factory.WithSelf.Method.4ba: %Factory.WithSelf.Method.type.53b = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.4db: %Factory.assoc_type.64a = assoc_entity element1, imports.%Main.import_ref.d94 [concrete]
// CHECK:STDOUT:   %assoc0.2f9: %Factory.assoc_type.a3c = assoc_entity element0, imports.%Main.import_ref.db6 [symbolic]
// CHECK:STDOUT:   %pattern_type.bbf: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.e42: %Factory.assoc_type.a3c = assoc_entity element1, imports.%Main.import_ref.b43 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.319 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.505 = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.ba3: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%assoc0 (constants.%assoc0.2f9)]
// CHECK:STDOUT:   %Main.import_ref.6a7: @Factory.WithSelf.%Factory.assoc_type (%Factory.assoc_type.a3c) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.WithSelf.%assoc1 (constants.%assoc1.e42)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.d94: @Factory.WithSelf.%Factory.WithSelf.Method.type (%Factory.WithSelf.Method.type.582) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Method (constants.%Factory.WithSelf.Method.0d5)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c8fa2a.2: @Factory.%Factory.type (%Factory.type.d25) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Main.import_ref.c77: @Factory.WithSelf.%Factory.WithSelf.Make.type (%Factory.WithSelf.Make.type.5fe) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.WithSelf.%Factory.WithSelf.Make (constants.%Factory.WithSelf.Make.d16)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c8fa2a.3: @Factory.%Factory.type (%Factory.type.d25) = import_ref Main//factory, loc4_29, loaded [symbolic = @Factory.%Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Main.import_ref.83e = import_ref Main//factory, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//factory, loc4_20, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.16e = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.02c = import_ref Main//factory, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.819: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.481: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.8fe]
// CHECK:STDOUT:   %Main.import_ref.642 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.6c1 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.b60 = import_ref Main//factory, loc14_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.db6 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.b43 = import_ref Main//factory, loc8_31, unloaded
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7e1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7e1 = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.b61]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceC.decl: %InstanceC.type = fn_decl @InstanceC [concrete = constants.%InstanceC] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bbf = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bbf = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7e1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7e1 = return_slot_pattern %return.param_patt, %C.ref.loc16 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %C.ref.loc16 [concrete = constants.%.b61]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(imports.%Main.import_ref.b3bc94.4: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.2e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.83e
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.ba3
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.6a7
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.819 as imports.%Main.import_ref.481 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.642
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.6c1
// CHECK:STDOUT:   extend imports.%Main.import_ref.b60
// CHECK:STDOUT:   witness = imports.%Main.import_ref.16e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.02c
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
// CHECK:STDOUT: generic fn @Factory.WithSelf.Method(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.c8fa2a.2: @Factory.%Factory.type (%Factory.type.d25)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d25)]
// CHECK:STDOUT:   %Self: @Factory.WithSelf.Method.%Factory.type (%Factory.type.d25) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.2e6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.7f6)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.WithSelf.Make(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.c8fa2a.3: @Factory.%Factory.type (%Factory.type.d25)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.319 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.3bc]
// CHECK:STDOUT:   %.loc13: %Factory.assoc_type.64a = specific_constant imports.%Main.import_ref.ba3, @Factory.WithSelf(constants.%C, constants.%Self.2e6) [concrete = constants.%assoc0.366]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.64a = name_ref Make, %.loc13 [concrete = constants.%assoc0.366]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param: %A) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.319 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.3bc]
// CHECK:STDOUT:   %.loc21: %Factory.assoc_type.64a = specific_constant imports.%Main.import_ref.6a7, @Factory.WithSelf(constants.%C, constants.%Self.2e6) [concrete = constants.%assoc1.4db]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.64a = name_ref Method, %.loc21 [concrete = constants.%assoc1.4db]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%T, constants.%Self.2e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Method(constants.%T, constants.%Self.2e6) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.d25
// CHECK:STDOUT:   %Self => constants.%Self.2e6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.7f6
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf.Make(constants.%T, constants.%Self.2e6) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8fe
// CHECK:STDOUT:   %Self => constants.%Self.078
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3bc
// CHECK:STDOUT:   %Self => constants.%Self.922
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.WithSelf(constants.%C, constants.%Self.2e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3bc
// CHECK:STDOUT:   %Self => constants.%Self.2e6
// CHECK:STDOUT:   %Factory.WithSelf.Make.type => constants.%Factory.WithSelf.Make.type.767
// CHECK:STDOUT:   %Factory.WithSelf.Make => constants.%Factory.WithSelf.Make.0fd
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.64a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.366
// CHECK:STDOUT:   %Factory.WithSelf.Method.type => constants.%Factory.WithSelf.Method.type.53b
// CHECK:STDOUT:   %Factory.WithSelf.Method => constants.%Factory.WithSelf.Method.4ba
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4db
// CHECK:STDOUT: }
// CHECK:STDOUT:
