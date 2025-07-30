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
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type [symbolic]
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
// CHECK:STDOUT:     %Dest.loc3_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.2: type) {
// CHECK:STDOUT:   %Dest.loc3_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = bind_name self, %self.param
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
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.as_type.loc4_20.1 [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.as_type.loc4_20.1 (%Self.as_type)) -> @ImplicitAs.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e98: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.566: type = facet_access_type %Self.e98 [symbolic]
// CHECK:STDOUT:   %pattern_type.8db: type = pattern_type %Self.as_type.566 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.036: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.6ed: %Action.Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.32e: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.268: %Action.assoc_type.32e = assoc_entity element0, @Action.%Action.Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b44: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Self.0d1: %Action.type.cb0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.f3b: %Action.assoc_type.4ee = assoc_entity element0, @Action.%Action.Op.decl [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness file.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op.type: type = fn_type @A.as.Action.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Action.impl.Op: %A.as.Action.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.cb0 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.584: type = fn_type_with_self_type %Action.Op.type.54d, %Action.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Action.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [concrete = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (@A.as.Action.impl.%A.as.Action.impl.Op.decl), @A.as.Action.impl [concrete]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness %Action.impl_witness_table [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.1)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self.2: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e98)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T.loc4_18.1) [symbolic = %Action.Op.type (constants.%Action.Op.type.036)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.036) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T.loc4_18.1) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.32e)]
// CHECK:STDOUT:   %assoc0.loc5_22.2: @Action.%Action.assoc_type (%Action.assoc_type.32e) = assoc_entity element0, %Action.Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.268)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e98)]
// CHECK:STDOUT:     %Action.Op.decl: @Action.%Action.Op.type (%Action.Op.type.036) = fn_decl @Action.Op [symbolic = @Action.%Action.Op (constants.%Action.Op.6ed)] {
// CHECK:STDOUT:       %self.patt: @Action.Op.%pattern_type (%pattern_type.8db) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Action.Op.%pattern_type (%pattern_type.8db) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Action.Op.%Self.as_type.loc5_15.1 (%Self.as_type.566) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type.566)] {
// CHECK:STDOUT:         %.loc5_15.2: @Action.Op.%Action.type (%Action.type.cca) = specific_constant @Action.%Self.1, @Action(constants.%T) [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:         %Self.ref: @Action.Op.%Action.type (%Action.type.cca) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:         %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type.566)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type.566)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Action.Op.%Self.as_type.loc5_15.1 (%Self.as_type.566) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_22.1: @Action.%Action.assoc_type (%Action.assoc_type.32e) = assoc_entity element0, %Action.Op.decl [symbolic = %assoc0.loc5_22.2 (constants.%assoc0.268)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_22.1
// CHECK:STDOUT:     witness = (%Action.Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: constants.%A as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: constants.%B as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc10: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %A.as.Action.impl.Op.decl: %A.as.Action.impl.Op.type = fn_decl @A.as.Action.impl.Op [concrete = constants.%A.as.Action.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Action.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Action.impl.Op.decl
// CHECK:STDOUT:   witness = file.%Action.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.b44]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.40d]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.Op(@Action.%T.loc4_18.2: type, @Action.%Self.1: @Action.%Action.type (%Action.type.cca)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type.566)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_15.1 [symbolic = %pattern_type (constants.%pattern_type.8db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Action.Op.%Self.as_type.loc5_15.1 (%Self.as_type.566));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc16: %Action.assoc_type.4ee = specific_constant @Action.%assoc0.loc5_22.1, @Action(constants.%B) [concrete = constants.%assoc0.f3b]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ee = name_ref Op, %.loc16 [concrete = constants.%assoc0.f3b]
// CHECK:STDOUT:   %impl.elem0: %.584 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %A.as.Action.impl.Op.call: init %empty_tuple.type = call %bound_method(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.e98) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self.e98
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type.566
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self.2 => constants.%Self.0d1
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0.loc5_22.2 => constants.%assoc0.f3b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%B, constants.%Action.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Action.facet
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e98: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.Op.type.036: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.6ed: %Action.Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e98 [symbolic]
// CHECK:STDOUT:   %pattern_type.8db: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.32e: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.f18741.1: %Action.assoc_type.32e = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic]
// CHECK:STDOUT:   %Self.0d1: %Action.type.cb0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.8a8: %Action.assoc_type.4ee = assoc_entity element0, imports.%Main.import_ref.1f6 [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.f18741.2: %Action.assoc_type.32e = assoc_entity element0, imports.%Main.import_ref.0e3753.2 [symbolic]
// CHECK:STDOUT:   %Action.impl_witness: <witness> = impl_witness imports.%Action.impl_witness_table [concrete]
// CHECK:STDOUT:   %Action.facet: %Action.type.cb0 = facet_value %A, (%Action.impl_witness) [concrete]
// CHECK:STDOUT:   %.6c7: type = fn_type_with_self_type %Action.Op.type.54d, %Action.facet [concrete]
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
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7b0 = import_ref Main//action, loc8_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst58 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93c: type = import_ref Main//action, inst58 [no loc], loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.2d5 = import_ref Main//action, loc8_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.ae7 = import_ref Main//action, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst102 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bfe: type = import_ref Main//action, inst102 [no loc], loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.fc3 = import_ref Main//action, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.34c = import_ref Main//action, loc10_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//action, inst122 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c0c: type = import_ref Main//action, inst122 [no loc], loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.cb9298.3: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.b3d = import_ref Main//action, loc10_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ddc = import_ref Main//action, inst27 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c55: @Action.%Action.assoc_type (%Action.assoc_type.32e) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.f18741.2)]
// CHECK:STDOUT:   %Main.Op.ae2 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.995: <witness> = import_ref Main//action, loc12_21, loaded [concrete = constants.%Action.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %Main.import_ref.7b5 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.835: @Action.%Action.type (%Action.type.cca) = import_ref Main//action, inst27 [no loc], loaded [symbolic = @Action.%Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Main.import_ref.0e3753.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1f6: @Action.%Action.Op.type (%Action.Op.type.036) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%Action.Op (constants.%Action.Op.6ed)]
// CHECK:STDOUT:   %Main.import_ref.0e3753.2 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.d0b: %A.as.Action.impl.Op.type = import_ref Main//action, loc13_23, loaded [concrete = constants.%A.as.Action.impl.Op]
// CHECK:STDOUT:   %Action.impl_witness_table = impl_witness_table (%Main.import_ref.d0b), @A.as.Action.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.5ab3ec.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T) [symbolic = %Action.Op.type (constants.%Action.Op.type.036)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.036) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.32e)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.32e) = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [symbolic = %assoc0 (constants.%assoc0.f18741.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.c55
// CHECK:STDOUT:     witness = (imports.%Main.Op.ae2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: imports.%Main.import_ref.93c as imports.%Main.import_ref.cb9298.1 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.2d5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: imports.%Main.import_ref.bfe as imports.%Main.import_ref.cb9298.2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.fc3
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ae7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: imports.%Main.import_ref.c0c as imports.%Main.import_ref.cb9298.3 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.b3d
// CHECK:STDOUT:   witness = imports.%Main.import_ref.34c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7b5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.995
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.Op(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.835: @Action.%Action.type (%Action.type.cca)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.8db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc4: %Action.assoc_type.4ee = specific_constant imports.%Main.import_ref.c55, @Action(constants.%B) [concrete = constants.%assoc0.8a8]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.4ee = name_ref Op, %.loc4 [concrete = constants.%assoc0.8a8]
// CHECK:STDOUT:   %impl.elem0: %.6c7 = impl_witness_access constants.%Action.impl_witness, element0 [concrete = constants.%A.as.Action.impl.Op]
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
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Self.0d1
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.e98) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self.e98
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [concrete]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e98: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [concrete]
// CHECK:STDOUT:   %Action.Op.type.036: type = fn_type @Action.Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Action.Op.6ed: %Action.Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e98 [symbolic]
// CHECK:STDOUT:   %pattern_type.8db: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Action.assoc_type.32e: type = assoc_entity_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.326: %Action.assoc_type.32e = assoc_entity element0, imports.%Main.import_ref.1f6 [symbolic]
// CHECK:STDOUT:   %Self.0d1: %Action.type.cb0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.54d: type = fn_type @Action.Op, @Action(%B) [concrete]
// CHECK:STDOUT:   %Action.Op.dba: %Action.Op.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.4ee: type = assoc_entity_type @Action, @Action(%B) [concrete]
// CHECK:STDOUT:   %assoc0.338: %Action.assoc_type.4ee = assoc_entity element0, imports.%Main.import_ref.0e3753.1 [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Action.type.e2b: type = facet_type <@Action, @Action(%C)> [concrete]
// CHECK:STDOUT:   %Self.4bb: %Action.type.e2b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Action.Op.type.5ad: type = fn_type @Action.Op, @Action(%C) [concrete]
// CHECK:STDOUT:   %Action.Op.b10: %Action.Op.type.5ad = struct_value () [concrete]
// CHECK:STDOUT:   %Action.assoc_type.79f: type = assoc_entity_type @Action, @Action(%C) [concrete]
// CHECK:STDOUT:   %assoc0.37d: %Action.assoc_type.79f = assoc_entity element0, imports.%Main.import_ref.1f6 [concrete]
// CHECK:STDOUT:   %assoc0.f18: %Action.assoc_type.32e = assoc_entity element0, imports.%Main.import_ref.0e3753.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Action: %Action.type.29c = import_ref Main//action, Action, loaded [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//action, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//action, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.F = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7b0 = import_ref Main//action, loc8_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//action, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//action, inst58 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93c: type = import_ref Main//action, inst58 [no loc], loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.2d5 = import_ref Main//action, loc8_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.ae7 = import_ref Main//action, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//action, loc9_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//action, inst102 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bfe: type = import_ref Main//action, inst102 [no loc], loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.fc3 = import_ref Main//action, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.34c = import_ref Main//action, loc10_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//action, inst122 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c0c: type = import_ref Main//action, inst122 [no loc], loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.cb9298.3: type = import_ref Main//action, inst61 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.b3d = import_ref Main//action, loc10_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ddc = import_ref Main//action, inst27 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c55: @Action.%Action.assoc_type (%Action.assoc_type.32e) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.f18)]
// CHECK:STDOUT:   %Main.Op.ae2 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.19c = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//action, loc12_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [concrete = constants.%Action.type.cb0]
// CHECK:STDOUT:   %Main.import_ref.7b5 = import_ref Main//action, loc13_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//action, loc4_18, loaded [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.835: @Action.%Action.type (%Action.type.cca) = import_ref Main//action, inst27 [no loc], loaded [symbolic = @Action.%Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Main.import_ref.1f6: @Action.%Action.Op.type (%Action.Op.type.036) = import_ref Main//action, loc5_22, loaded [symbolic = @Action.%Action.Op (constants.%Action.Op.6ed)]
// CHECK:STDOUT:   %Main.import_ref.0e3753.1 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.0e3753.2 = import_ref Main//action, loc5_22, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Action = imports.%Main.Action
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(imports.%Main.import_ref.5ab3ec.1: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Action.Op.type: type = fn_type @Action.Op, @Action(%T) [symbolic = %Action.Op.type (constants.%Action.Op.type.036)]
// CHECK:STDOUT:   %Action.Op: @Action.%Action.Op.type (%Action.Op.type.036) = struct_value () [symbolic = %Action.Op (constants.%Action.Op.6ed)]
// CHECK:STDOUT:   %Action.assoc_type: type = assoc_entity_type @Action, @Action(%T) [symbolic = %Action.assoc_type (constants.%Action.assoc_type.32e)]
// CHECK:STDOUT:   %assoc0: @Action.%Action.assoc_type (%Action.assoc_type.32e) = assoc_entity element0, imports.%Main.import_ref.1f6 [symbolic = %assoc0 (constants.%assoc0.326)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.c55
// CHECK:STDOUT:     witness = (imports.%Main.Op.ae2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: imports.%Main.import_ref.93c as imports.%Main.import_ref.cb9298.1 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.2d5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: imports.%Main.import_ref.bfe as imports.%Main.import_ref.cb9298.2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.fc3
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ae7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: imports.%Main.import_ref.c0c as imports.%Main.import_ref.cb9298.3 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.b3d
// CHECK:STDOUT:   witness = imports.%Main.import_ref.34c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Action.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.7b5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.19c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Action.Op(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.835: @Action.%Action.type (%Action.type.cca)) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: @Action.Op.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.e98)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.8db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%Main.Action [concrete = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [concrete = constants.%Action.type.e2b]
// CHECK:STDOUT:   %.loc8: %Action.assoc_type.79f = specific_constant imports.%Main.import_ref.c55, @Action(constants.%C) [concrete = constants.%assoc0.37d]
// CHECK:STDOUT:   %Op.ref: %Action.assoc_type.79f = name_ref Op, %.loc8 [concrete = constants.%assoc0.37d]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Self.0d1
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.54d
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.dba
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.4ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action.Op(constants.%T, constants.%Self.e98) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cca
// CHECK:STDOUT:   %Self => constants.%Self.e98
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.e2b
// CHECK:STDOUT:   %Self => constants.%Self.4bb
// CHECK:STDOUT:   %Action.Op.type => constants.%Action.Op.type.5ad
// CHECK:STDOUT:   %Action.Op => constants.%Action.Op.b10
// CHECK:STDOUT:   %Action.assoc_type => constants.%Action.assoc_type.79f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.37d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9ba: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.598: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.737: %Factory.Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.207: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.b3c: %Factory.assoc_type.207 = assoc_entity element0, @Factory.%Factory.Make.decl [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.9ba [symbolic]
// CHECK:STDOUT:   %pattern_type.5ea: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.7ee: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.a71: %Factory.Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.e30: %Factory.assoc_type.207 = assoc_entity element1, @Factory.%Factory.Method.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b44: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Self.187: %Factory.type.a5d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.fe7: %Factory.assoc_type.579 = assoc_entity element0, @Factory.%Factory.Make.decl [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.02a: %Factory.assoc_type.579 = assoc_entity element1, @Factory.%Factory.Method.decl [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness file.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @A.as.Factory.impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1a8 = name_ref Factory, file.%Factory.decl [concrete = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (@A.as.Factory.impl.%A.as.Factory.impl.Make.decl, @A.as.Factory.impl.%A.as.Factory.impl.Method.decl), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness %Factory.impl_witness_table [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.1)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self.2: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.9ba)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T.loc4_19.1) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.598)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.598) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T.loc4_19.1) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.207)]
// CHECK:STDOUT:   %assoc0.loc6_17.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element0, %Factory.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.b3c)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T.loc4_19.1) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.7ee)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.a71)]
// CHECK:STDOUT:   %assoc1.loc8_31.2: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element1, %Factory.Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.e30)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.9ba)]
// CHECK:STDOUT:     %Factory.Make.decl: @Factory.%Factory.Make.type (%Factory.Make.type.598) = fn_decl @Factory.Make [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.737)] {
// CHECK:STDOUT:       %return.patt: @Factory.Make.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Factory.Make.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Factory.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Factory.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_17.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element0, %Factory.Make.decl [symbolic = %assoc0.loc6_17.2 (constants.%assoc0.b3c)]
// CHECK:STDOUT:     %Factory.Method.decl: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = fn_decl @Factory.Method [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.a71)] {
// CHECK:STDOUT:       %self.patt: @Factory.Method.%pattern_type.loc8_13 (%pattern_type.5ea) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Factory.Method.%pattern_type.loc8_13 (%pattern_type.5ea) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Factory.Method.%pattern_type.loc8_27 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Factory.Method.%pattern_type.loc8_27 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Factory.Method.%Self.as_type.loc8_19.1 (%Self.as_type.56c) = value_param call_param0
// CHECK:STDOUT:       %.loc8_19.1: type = splice_block %.loc8_19.3 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type.56c)] {
// CHECK:STDOUT:         %.loc8_19.2: @Factory.Method.%Factory.type (%Factory.type.c96) = specific_constant @Factory.%Self.1, @Factory(constants.%T) [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:         %Self.ref: @Factory.Method.%Factory.type (%Factory.type.c96) = name_ref Self, %.loc8_19.2 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:         %Self.as_type.loc8_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type.56c)]
// CHECK:STDOUT:         %.loc8_19.3: type = converted %Self.ref, %Self.as_type.loc8_19.2 [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type.56c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Factory.Method.%Self.as_type.loc8_19.1 (%Self.as_type.56c) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Factory.Method.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Factory.Method.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc8_31.1: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element1, %Factory.Method.decl [symbolic = %assoc1.loc8_31.2 (constants.%assoc1.e30)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %assoc0.loc6_17.1
// CHECK:STDOUT:     .Method = %assoc1.loc8_31.1
// CHECK:STDOUT:     witness = (%Factory.Make.decl, %Factory.Method.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: constants.%A as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: constants.%B as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc12: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
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
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @A.as.Factory.impl.%A.ref [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
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
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.b44]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.40d]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Make(@Factory.%T.loc4_19.2: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.c96)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Factory.Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Method(@Factory.%T.loc4_19.2: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.c96)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Self.as_type.loc8_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_19.1 (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.loc8_13: type = pattern_type %Self.as_type.loc8_19.1 [symbolic = %pattern_type.loc8_13 (constants.%pattern_type.5ea)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %T [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Factory.Method.%Self.as_type.loc8_19.1 (%Self.as_type.56c)) -> @Factory.Method.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method(%self.param: %A) -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.5ea
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self.2 => constants.%Self.187
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0.loc6_17.2 => constants.%assoc0.fe7
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1.loc8_31.2 => constants.%assoc1.02a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Factory.facet
// CHECK:STDOUT:   %Self.as_type.loc8_19.1 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc8_13 => constants.%pattern_type.c10
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.049
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9ba: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.Make.type.598: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.737: %Factory.Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.207: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.46d25f.1: %Factory.assoc_type.207 = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.7ee: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.a71: %Factory.Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.9ba [symbolic]
// CHECK:STDOUT:   %pattern_type.5ea: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %assoc1.16541d.1: %Factory.assoc_type.207 = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic]
// CHECK:STDOUT:   %Self.187: %Factory.type.a5d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.153: %Factory.assoc_type.579 = assoc_entity element0, imports.%Main.import_ref.1aa [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.a48: %Factory.assoc_type.579 = assoc_entity element1, imports.%Main.import_ref.5be [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.46d25f.2: %Factory.assoc_type.207 = assoc_entity element0, imports.%Main.import_ref.21018a.2 [symbolic]
// CHECK:STDOUT:   %Factory.impl_witness: <witness> = impl_witness imports.%Factory.impl_witness_table [concrete]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value %A, (%Factory.impl_witness) [concrete]
// CHECK:STDOUT:   %.711: type = fn_type_with_self_type %Factory.Make.type.c59, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make.type: type = fn_type @A.as.Factory.impl.Make [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Make: %A.as.Factory.impl.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.5cf: <witness> = impl_witness imports.%Destroy.impl_witness_table.f6b [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceB.type: type = fn_type @InstanceB [concrete]
// CHECK:STDOUT:   %InstanceB: %InstanceB.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.16541d.2: %Factory.assoc_type.207 = assoc_entity element1, imports.%Main.import_ref.46fc3c.2 [symbolic]
// CHECK:STDOUT:   %.fed: type = fn_type_with_self_type %Factory.Method.type.117, %Factory.facet [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method.type: type = fn_type @A.as.Factory.impl.Method [concrete]
// CHECK:STDOUT:   %A.as.Factory.impl.Method: %A.as.Factory.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: type = import_ref Main//factory, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7b0 = import_ref Main//factory, loc11_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst80 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93c: type = import_ref Main//factory, inst80 [no loc], loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//factory, inst83 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.2d5 = import_ref Main//factory, loc11_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.6c7: <witness> = import_ref Main//factory, loc12_9, loaded [concrete = constants.%Destroy.impl_witness.5cf]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst124 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bfe: type = import_ref Main//factory, inst124 [no loc], loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//factory, inst83 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.fc3 = import_ref Main//factory, loc12_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//factory, inst27 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.46c: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.46d25f.2)]
// CHECK:STDOUT:   %Main.import_ref.2e4: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.16541d.2)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.6fd: <witness> = import_ref Main//factory, loc14_22, loaded [concrete = constants.%Factory.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bd2: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %Main.import_ref.a27 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.163 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst27 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.21018a.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst27 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.46fc3c.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.1aa: @Factory.%Factory.Make.type (%Factory.Make.type.598) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.737)]
// CHECK:STDOUT:   %Main.import_ref.5be: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.a71)]
// CHECK:STDOUT:   %Main.import_ref.21018a.2 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.9ec: %A.as.Factory.impl.Make.type = import_ref Main//factory, loc15_17, loaded [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %Main.import_ref.7dd: %A.as.Factory.impl.Method.type = import_ref Main//factory, loc16_31, loaded [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %Factory.impl_witness_table = impl_witness_table (%Main.import_ref.9ec, %Main.import_ref.7dd), @A.as.Factory.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.82a: %B.as.Destroy.impl.Op.type = import_ref Main//factory, loc12_9, loaded [concrete = constants.%B.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.f6b = impl_witness_table (%Main.import_ref.82a), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.46fc3c.2 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = imports.%Main.Factory
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .Core = imports.%Core.ece
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.598)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.598) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.207)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element0, imports.%Main.import_ref.21018a.1 [symbolic = %assoc0 (constants.%assoc0.46d25f.1)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.7ee)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.a71)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [symbolic = %assoc1 (constants.%assoc1.16541d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.46c
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.2e4
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: imports.%Main.import_ref.93c as imports.%Main.import_ref.cb9298.1 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.2d5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: imports.%Main.import_ref.bfe as imports.%Main.import_ref.cb9298.2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.fc3
// CHECK:STDOUT:   witness = imports.%Main.import_ref.6c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.a27
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.163
// CHECK:STDOUT:   witness = imports.%Main.import_ref.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Make(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Method(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.5ea)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB() -> %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %.loc5_23: %Factory.assoc_type.579 = specific_constant imports.%Main.import_ref.46c, @Factory(constants.%B) [concrete = constants.%assoc0.153]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.579 = name_ref Make, %.loc5_23 [concrete = constants.%assoc0.153]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.a5d = facet_value constants.%A, (constants.%Factory.impl_witness) [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %.loc5_11: %Factory.type.a5d = converted %A.ref, %Factory.facet [concrete = constants.%Factory.facet]
// CHECK:STDOUT:   %impl.elem0: %.711 = impl_witness_access constants.%Factory.impl_witness, element0 [concrete = constants.%A.as.Factory.impl.Make]
// CHECK:STDOUT:   %.loc4: ref %B = splice_block %return {}
// CHECK:STDOUT:   %A.as.Factory.impl.Make.call: init %B = call %impl.elem0() to %.loc4
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc4, constants.%B.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc4
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.call: init %empty_tuple.type = call %B.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %A.as.Factory.impl.Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Make [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op = "no_op" [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceB(%a.param: %A) -> %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %.loc8: %Factory.assoc_type.579 = specific_constant imports.%Main.import_ref.2e4, @Factory(constants.%B) [concrete = constants.%assoc1.a48]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.579 = name_ref Method, %.loc8 [concrete = constants.%assoc1.a48]
// CHECK:STDOUT:   %impl.elem1: %.fed = impl_witness_access constants.%Factory.impl_witness, element1 [concrete = constants.%A.as.Factory.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   %.loc7: ref %B = splice_block %return {}
// CHECK:STDOUT:   %A.as.Factory.impl.Method.call: init %B = call %bound_method(%a.ref) to %.loc7
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc7, constants.%B.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc7
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.call: init %empty_tuple.type = call %B.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %A.as.Factory.impl.Method.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Factory.impl.Method [from "factory.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Self.187
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.153
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.5ea
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [concrete]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9ba: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [concrete]
// CHECK:STDOUT:   %Factory.Make.type.598: type = fn_type @Factory.Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Make.737: %Factory.Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Factory.assoc_type.207: type = assoc_entity_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.fe4: %Factory.assoc_type.207 = assoc_entity element0, imports.%Main.import_ref.1aa [symbolic]
// CHECK:STDOUT:   %Factory.Method.type.7ee: type = fn_type @Factory.Method, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Factory.Method.a71: %Factory.Method.type.7ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.9ba [symbolic]
// CHECK:STDOUT:   %pattern_type.5ea: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %assoc1.ea6: %Factory.assoc_type.207 = assoc_entity element1, imports.%Main.import_ref.5be [symbolic]
// CHECK:STDOUT:   %Self.187: %Factory.type.a5d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.c59: type = fn_type @Factory.Make, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Make.efe: %Factory.Make.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.579: type = assoc_entity_type @Factory, @Factory(%B) [concrete]
// CHECK:STDOUT:   %assoc0.cc7: %Factory.assoc_type.579 = assoc_entity element0, imports.%Main.import_ref.21018a.1 [concrete]
// CHECK:STDOUT:   %Factory.Method.type.117: type = fn_type @Factory.Method, @Factory(%B) [concrete]
// CHECK:STDOUT:   %Factory.Method.ea9: %Factory.Method.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.535: %Factory.assoc_type.579 = assoc_entity element1, imports.%Main.import_ref.46fc3c.1 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.type.5c5: type = facet_type <@Factory, @Factory(%C)> [concrete]
// CHECK:STDOUT:   %Self.424: %Factory.type.5c5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Factory.Make.type.0de: type = fn_type @Factory.Make, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Factory.Make.8ba: %Factory.Make.type.0de = struct_value () [concrete]
// CHECK:STDOUT:   %Factory.assoc_type.d01: type = assoc_entity_type @Factory, @Factory(%C) [concrete]
// CHECK:STDOUT:   %assoc0.38e: %Factory.assoc_type.d01 = assoc_entity element0, imports.%Main.import_ref.1aa [concrete]
// CHECK:STDOUT:   %Factory.Method.type.d46: type = fn_type @Factory.Method, @Factory(%C) [concrete]
// CHECK:STDOUT:   %Factory.Method.f9e: %Factory.Method.type.d46 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.263: %Factory.assoc_type.d01 = assoc_entity element1, imports.%Main.import_ref.5be [concrete]
// CHECK:STDOUT:   %assoc0.46d: %Factory.assoc_type.207 = assoc_entity element0, imports.%Main.import_ref.21018a.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %InstanceC.type: type = fn_type @InstanceC [concrete]
// CHECK:STDOUT:   %InstanceC: %InstanceC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.165: %Factory.assoc_type.207 = assoc_entity element1, imports.%Main.import_ref.46fc3c.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Factory: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %Main.A: type = import_ref Main//factory, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7b0 = import_ref Main//factory, loc11_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc11_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.da3 = import_ref Main//factory, inst80 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.93c: type = import_ref Main//factory, inst80 [no loc], loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//factory, inst83 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.2d5 = import_ref Main//factory, loc11_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.ae7 = import_ref Main//factory, loc12_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc12_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//factory, inst124 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bfe: type = import_ref Main//factory, inst124 [no loc], loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//factory, inst83 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.fc3 = import_ref Main//factory, loc12_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//factory, inst27 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.46c: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.46d)]
// CHECK:STDOUT:   %Main.import_ref.2e4: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%assoc1 (constants.%assoc1.165)]
// CHECK:STDOUT:   %Main.Make = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %Main.Method = import_ref Main//factory, Method, unloaded
// CHECK:STDOUT:   %Main.import_ref.4c3 = import_ref Main//factory, loc14_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.984: type = import_ref Main//factory, loc14_6, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.import_ref.bd2: type = import_ref Main//factory, loc14_20, loaded [concrete = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %Main.import_ref.a27 = import_ref Main//factory, loc15_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.163 = import_ref Main//factory, loc16_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst27 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.1aa: @Factory.%Factory.Make.type (%Factory.Make.type.598) = import_ref Main//factory, loc6_17, loaded [symbolic = @Factory.%Factory.Make (constants.%Factory.Make.737)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//factory, loc4_19, loaded [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96) = import_ref Main//factory, inst27 [no loc], loaded [symbolic = @Factory.%Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Main.import_ref.5be: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = import_ref Main//factory, loc8_31, loaded [symbolic = @Factory.%Factory.Method (constants.%Factory.Method.a71)]
// CHECK:STDOUT:   %Main.import_ref.21018a.1 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.46fc3c.1 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.21018a.2 = import_ref Main//factory, loc6_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.46fc3c.2 = import_ref Main//factory, loc8_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Factory = imports.%Main.Factory
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .Core = imports.%Core.ece
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Factory.Make.type: type = fn_type @Factory.Make, @Factory(%T) [symbolic = %Factory.Make.type (constants.%Factory.Make.type.598)]
// CHECK:STDOUT:   %Factory.Make: @Factory.%Factory.Make.type (%Factory.Make.type.598) = struct_value () [symbolic = %Factory.Make (constants.%Factory.Make.737)]
// CHECK:STDOUT:   %Factory.assoc_type: type = assoc_entity_type @Factory, @Factory(%T) [symbolic = %Factory.assoc_type (constants.%Factory.assoc_type.207)]
// CHECK:STDOUT:   %assoc0: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element0, imports.%Main.import_ref.1aa [symbolic = %assoc0 (constants.%assoc0.fe4)]
// CHECK:STDOUT:   %Factory.Method.type: type = fn_type @Factory.Method, @Factory(%T) [symbolic = %Factory.Method.type (constants.%Factory.Method.type.7ee)]
// CHECK:STDOUT:   %Factory.Method: @Factory.%Factory.Method.type (%Factory.Method.type.7ee) = struct_value () [symbolic = %Factory.Method (constants.%Factory.Method.a71)]
// CHECK:STDOUT:   %assoc1: @Factory.%Factory.assoc_type (%Factory.assoc_type.207) = assoc_entity element1, imports.%Main.import_ref.5be [symbolic = %assoc1 (constants.%assoc1.ea6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.46c
// CHECK:STDOUT:     .Method = imports.%Main.import_ref.2e4
// CHECK:STDOUT:     witness = (imports.%Main.Make, imports.%Main.Method)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: imports.%Main.import_ref.93c as imports.%Main.import_ref.cb9298.1 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.2d5
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: imports.%Main.import_ref.bfe as imports.%Main.import_ref.cb9298.2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.fc3
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ae7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Factory.impl: imports.%Main.import_ref.984 as imports.%Main.import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.a27
// CHECK:STDOUT:   .Method = imports.%Main.import_ref.163
// CHECK:STDOUT:   witness = imports.%Main.import_ref.4c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Make(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.91b53a.1: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Factory.Method(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.91b53a.2: @Factory.%Factory.type (%Factory.type.c96)) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: @Factory.Method.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.9ba)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.5ea)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC() -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.5c5]
// CHECK:STDOUT:   %.loc13: %Factory.assoc_type.d01 = specific_constant imports.%Main.import_ref.46c, @Factory(constants.%C) [concrete = constants.%assoc0.38e]
// CHECK:STDOUT:   %Make.ref: %Factory.assoc_type.d01 = name_ref Make, %.loc13 [concrete = constants.%assoc0.38e]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceC(%a.param: %A) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%Main.Factory [concrete = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [concrete = constants.%Factory.type.5c5]
// CHECK:STDOUT:   %.loc21: %Factory.assoc_type.d01 = specific_constant imports.%Main.import_ref.2e4, @Factory(constants.%C) [concrete = constants.%assoc1.263]
// CHECK:STDOUT:   %Method.ref: %Factory.assoc_type.d01 = name_ref Method, %.loc21 [concrete = constants.%assoc1.263]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Self.187
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.c59
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.efe
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.579
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cc7
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.117
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.ea9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.535
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Make(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory.Method(constants.%T, constants.%Self.9ba) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.c96
// CHECK:STDOUT:   %Self => constants.%Self.9ba
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.5ea
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5c5
// CHECK:STDOUT:   %Self => constants.%Self.424
// CHECK:STDOUT:   %Factory.Make.type => constants.%Factory.Make.type.0de
// CHECK:STDOUT:   %Factory.Make => constants.%Factory.Make.8ba
// CHECK:STDOUT:   %Factory.assoc_type => constants.%Factory.assoc_type.d01
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.38e
// CHECK:STDOUT:   %Factory.Method.type => constants.%Factory.Method.type.d46
// CHECK:STDOUT:   %Factory.Method => constants.%Factory.Method.f9e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.263
// CHECK:STDOUT: }
// CHECK:STDOUT:
