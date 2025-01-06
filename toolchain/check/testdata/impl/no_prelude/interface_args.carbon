// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/interface_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/interface_args.carbon

// --- action.carbon

library "[[@TEST_NAME]]";

interface Action(T:! type) {
  fn Op();
}

class A {}
class B {}
class C {}

impl A as Action(B) {
  fn Op() {}
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
  fn Make() -> T;
}

class A {}
class B {}

impl A as Factory(B) {
  fn Make() -> B;
}

// --- factory.impl.carbon

impl library "[[@TEST_NAME]]";

fn MakeB(a: A) -> B {
  return a.(Factory(B).Make)();
}

// --- fail_factory.impl.carbon

impl library "[[@TEST_NAME]]";

class C {}

fn MakeC(a: A) -> C {
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+3]]:10: error: cannot access member of interface `Factory(C)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a.(Factory(C).Make)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  return a.(Factory(C).Make)();
}

// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.4: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Action.type.31: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.31 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.d: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.8: %Op.type.d = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.f: type = assoc_entity_type %Action.type.31, %Op.type.d [symbolic]
// CHECK:STDOUT:   %assoc0.2: %Op.assoc_type.f = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.38: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %Op.type.7: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.0b: %Op.type.7 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.1: type = assoc_entity_type %Action.type.38, %Op.type.7 [template]
// CHECK:STDOUT:   %assoc0.a: %Op.assoc_type.1 = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.9: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.01: %Op.type.9 = struct_value () [template]
// CHECK:STDOUT:   %Action.facet: %Action.type.31 = facet_value %A, %impl_witness [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = %Action.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Action.decl: %Action.type.4 = interface_decl @Action [template = constants.%Action.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_18.1, runtime_param<invalid> [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type.4 = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.38]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.31)]
// CHECK:STDOUT:   %Self.2: %Action.type.31 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.d)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.d) = struct_value () [symbolic = %Op (constants.%Op.8)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.31), @Action.%Op.type (%Op.type.d) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.f)]
// CHECK:STDOUT:   %assoc0.loc5_10.2: @Action.%Op.assoc_type (%Op.assoc_type.f) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.31) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.d) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.8)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_10.1: @Action.%Op.assoc_type (%Op.assoc_type.f) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_10.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.9 = fn_decl @Op.2 [template = constants.%Op.01] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.31)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.4 = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.38]
// CHECK:STDOUT:   %.loc16: %Op.assoc_type.1 = specific_constant @Action.%assoc0.loc5_10.1, @Action(constants.%B) [template = constants.%assoc0.a]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.1 = name_ref Op, %.loc16 [template = constants.%assoc0.a]
// CHECK:STDOUT:   %impl.elem0: %Op.type.7 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.01]
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T.loc4_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.38
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.7
// CHECK:STDOUT:   %Op => constants.%Op.0b
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.1
// CHECK:STDOUT:   %assoc0.loc5_10.2 => constants.%assoc0.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%B, constants.%Action.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Action.type.4: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.4 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.5: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.b: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.8: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.b: %Op.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.0: type = assoc_entity_type %Action.type.5, %Op.type.8 [symbolic]
// CHECK:STDOUT:   %assoc0.3fd18f.1: %Op.assoc_type.0 = assoc_entity element0, imports.%import_ref.78f365.1 [symbolic]
// CHECK:STDOUT:   %Op.type.6: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.0d: %Op.type.6 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.f: type = assoc_entity_type %Action.type.b, %Op.type.6 [template]
// CHECK:STDOUT:   %assoc0.a: %Op.assoc_type.f = assoc_entity element0, imports.%import_ref.78f365.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.3fd18f.2: %Op.assoc_type.0 = assoc_entity element0, imports.%import_ref.78f365.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.79) [template]
// CHECK:STDOUT:   %Op.type.9: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.01: %Op.type.9 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.0: %Action.type.4 = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.dc: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.a57: type = import_ref Main//action, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.82 = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %import_ref.7f = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.db0: <witness> = import_ref Main//action, loc12_21, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.54c = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.a51: @Action.%Op.assoc_type (%Op.assoc_type.0) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.3fd18f.2)]
// CHECK:STDOUT:   %import_ref.c = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.db3: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.543: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.b]
// CHECK:STDOUT:   %import_ref.f = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.78f365.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.0
// CHECK:STDOUT:     .A = imports.%import_ref.dc
// CHECK:STDOUT:     .B = imports.%import_ref.a57
// CHECK:STDOUT:     .C = imports.%import_ref.82
// CHECK:STDOUT:     .F = imports.%import_ref.7f
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.dc [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.5)]
// CHECK:STDOUT:   %Self: %Action.type.5 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.8)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.8) = struct_value () [symbolic = %Op (constants.%Op.b)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.5), @Action.%Op.type (%Op.type.8) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.0)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.0) = assoc_entity element0, imports.%import_ref.78f365.1 [symbolic = %assoc0 (constants.%assoc0.3fd18f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.54c
// CHECK:STDOUT:     .Op = imports.%import_ref.a51
// CHECK:STDOUT:     witness = (imports.%import_ref.c)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.db3 as imports.%import_ref.543 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.f
// CHECK:STDOUT:   witness = imports.%import_ref.db0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.1
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self: %Action.type.5) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.4 = name_ref Action, imports.%import_ref.0 [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.a57 [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.b]
// CHECK:STDOUT:   %.loc4: %Op.assoc_type.f = specific_constant imports.%import_ref.a51, @Action(constants.%B) [template = constants.%assoc0.a]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.f = name_ref Op, %.loc4 [template = constants.%assoc0.a]
// CHECK:STDOUT:   %impl.elem0: %Op.type.6 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.01]
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2() [from "action.carbon"];
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.b
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.6
// CHECK:STDOUT:   %Op => constants.%Op.0d
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Action.type.4: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.4 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.5: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.b: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.8: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.b: %Op.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.0: type = assoc_entity_type %Action.type.5, %Op.type.8 [symbolic]
// CHECK:STDOUT:   %assoc0.3fd18f.1: %Op.assoc_type.0 = assoc_entity element0, imports.%import_ref.78f365.1 [symbolic]
// CHECK:STDOUT:   %Op.type.6: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.0: %Op.type.6 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.f: type = assoc_entity_type %Action.type.b, %Op.type.6 [template]
// CHECK:STDOUT:   %assoc0.a: %Op.assoc_type.f = assoc_entity element0, imports.%import_ref.78f365.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.3: type = facet_type <@Action, @Action(%C)> [template]
// CHECK:STDOUT:   %Op.type.b: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.d: %Op.type.b = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.3: type = assoc_entity_type %Action.type.3, %Op.type.b [template]
// CHECK:STDOUT:   %assoc0.e: %Op.assoc_type.3 = assoc_entity element0, imports.%import_ref.78f365.1 [template]
// CHECK:STDOUT:   %assoc0.3fd18f.2: %Op.assoc_type.0 = assoc_entity element0, imports.%import_ref.78f365.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.0: %Action.type.4 = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.dc: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.9f5 = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %import_ref.76: type = import_ref Main//action, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.7f = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.93 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.9fb = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.54c = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.a: @Action.%Op.assoc_type (%Op.assoc_type.0) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.3fd18f.2)]
// CHECK:STDOUT:   %import_ref.c = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.db: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.543: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.b]
// CHECK:STDOUT:   %import_ref.f = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.78f365.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.b = import_ref Main//action, inst49 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.0
// CHECK:STDOUT:     .A = imports.%import_ref.dc
// CHECK:STDOUT:     .B = imports.%import_ref.9f5
// CHECK:STDOUT:     .C = imports.%import_ref.76
// CHECK:STDOUT:     .F = imports.%import_ref.7f
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.dc [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.5)]
// CHECK:STDOUT:   %Self: %Action.type.5 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.8)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.8) = struct_value () [symbolic = %Op (constants.%Op.b)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.5), @Action.%Op.type (%Op.type.8) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.0)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.0) = assoc_entity element0, imports.%import_ref.78f365.1 [symbolic = %assoc0 (constants.%assoc0.3fd18f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.54c
// CHECK:STDOUT:     .Op = imports.%import_ref.a
// CHECK:STDOUT:     witness = (imports.%import_ref.c)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.db as imports.%import_ref.543 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.f
// CHECK:STDOUT:   witness = imports.%import_ref.93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9fb
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.1
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.b
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self: %Action.type.5) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.4 = name_ref Action, imports.%import_ref.0 [template = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.76 [template = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [template = constants.%Action.type.3]
// CHECK:STDOUT:   %.loc8: %Op.assoc_type.3 = specific_constant imports.%import_ref.a, @Action(constants.%C) [template = constants.%assoc0.e]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.3 = name_ref Op, %.loc8 [template = constants.%assoc0.e]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.b
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.6
// CHECK:STDOUT:   %Op => constants.%Op.0
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.b
// CHECK:STDOUT:   %Op => constants.%Op.d
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.c = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.5: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.9: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.3: %Make.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.3: type = assoc_entity_type %Factory.type.5, %Make.type.9 [symbolic]
// CHECK:STDOUT:   %assoc0.e: %Make.assoc_type.3 = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.3: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.7: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.a: type = assoc_entity_type %Factory.type.3, %Make.type.3 [template]
// CHECK:STDOUT:   %assoc0.c: %Make.assoc_type.a = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template]
// CHECK:STDOUT:   %Make.type.b: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.8: %Make.type.b = struct_value () [template]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.5 = facet_value %A, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type.c = interface_decl @Factory [template = constants.%Factory.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<invalid> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type.c = name_ref Factory, file.%Factory.decl [template = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.2)> [symbolic = %Factory.type (constants.%Factory.type.5)]
// CHECK:STDOUT:   %Self.2: %Factory.type.5 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T.loc4_19.2) [symbolic = %Make.type (constants.%Make.type.9)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.9) = struct_value () [symbolic = %Make (constants.%Make.3)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.5), @Factory.%Make.type (%Make.type.9) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.3)]
// CHECK:STDOUT:   %assoc0.loc5_17.2: @Factory.%Make.assoc_type (%Make.assoc_type.3) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.5) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.9) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.3)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_17.1: @Factory.%Make.assoc_type (%Make.assoc_type.3) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %assoc0.loc5_17.1
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.b = fn_decl @Make.2 [template = constants.%Make.8] {
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.5)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B;
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
// CHECK:STDOUT: specific @Factory(%T.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.7
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.a
// CHECK:STDOUT:   %assoc0.loc5_17.2 => constants.%assoc0.c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%B, constants.%Factory.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Factory.type.c: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.c = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.8: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.8: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.2: %Make.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.5: type = assoc_entity_type %Factory.type.1, %Make.type.8 [symbolic]
// CHECK:STDOUT:   %assoc0.92248f.1: %Make.assoc_type.5 = assoc_entity element0, imports.%import_ref.2f9a36.1 [symbolic]
// CHECK:STDOUT:   %Make.type.6: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.88: %Make.type.6 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.8: type = assoc_entity_type %Factory.type.8, %Make.type.6 [template]
// CHECK:STDOUT:   %assoc0.0: %Make.assoc_type.8 = assoc_entity element0, imports.%import_ref.2f9a36.2 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.92248f.2: %Make.assoc_type.5 = assoc_entity element0, imports.%import_ref.2f9a36.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.6) [template]
// CHECK:STDOUT:   %Make.type.b: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.8f: %Make.type.b = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.7: %Factory.type.c = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.dc: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.a: type = import_ref Main//factory, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.1b: <witness> = import_ref Main//factory, loc11_22, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.20: @Factory.%Make.assoc_type (%Make.assoc_type.5) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.92248f.2)]
// CHECK:STDOUT:   %import_ref.22 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.db: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.c: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.8]
// CHECK:STDOUT:   %import_ref.8a = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.2f9a36.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.7
// CHECK:STDOUT:     .A = imports.%import_ref.dc
// CHECK:STDOUT:     .B = imports.%import_ref.a
// CHECK:STDOUT:     .MakeB = %MakeB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %MakeB.decl: %MakeB.type = fn_decl @MakeB [template = constants.%MakeB] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%import_ref.a [template = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.dc [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.1)]
// CHECK:STDOUT:   %Self: %Factory.type.1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.8)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.8) = struct_value () [symbolic = %Make (constants.%Make.2)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.1), @Factory.%Make.type (%Make.type.8) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.5)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.5) = assoc_entity element0, imports.%import_ref.2f9a36.1 [symbolic = %assoc0 (constants.%assoc0.92248f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.16
// CHECK:STDOUT:     .Make = imports.%import_ref.20
// CHECK:STDOUT:     witness = (imports.%import_ref.22)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.db as imports.%import_ref.c [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.8a
// CHECK:STDOUT:   witness = imports.%import_ref.1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.17
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self: %Factory.type.1) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a.param_patt: %A) -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.c = name_ref Factory, imports.%import_ref.7 [template = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%import_ref.a [template = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.8]
// CHECK:STDOUT:   %.loc5: %Make.assoc_type.8 = specific_constant imports.%import_ref.20, @Factory(constants.%B) [template = constants.%assoc0.0]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.8 = name_ref Make, %.loc5 [template = constants.%assoc0.0]
// CHECK:STDOUT:   %impl.elem0: %Make.type.6 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Make.8f]
// CHECK:STDOUT:   %.loc4: ref %B = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %B = call %impl.elem0() to %.loc4
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B [from "factory.carbon"];
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.6
// CHECK:STDOUT:   %Make => constants.%Make.88
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Factory.type.c: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.c = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.8: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.89: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.2: %Make.type.89 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.5: type = assoc_entity_type %Factory.type.1, %Make.type.89 [symbolic]
// CHECK:STDOUT:   %assoc0.92248f.1: %Make.assoc_type.5 = assoc_entity element0, imports.%import_ref.2f9a36.1 [symbolic]
// CHECK:STDOUT:   %Make.type.6: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.8: %Make.type.6 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.8: type = assoc_entity_type %Factory.type.8, %Make.type.6 [template]
// CHECK:STDOUT:   %assoc0.0: %Make.assoc_type.8 = assoc_entity element0, imports.%import_ref.2f9a36.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.7: type = facet_type <@Factory, @Factory(%C)> [template]
// CHECK:STDOUT:   %Make.type.84: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.7: %Make.type.84 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.d: type = assoc_entity_type %Factory.type.7, %Make.type.84 [template]
// CHECK:STDOUT:   %assoc0.d: %Make.assoc_type.d = assoc_entity element0, imports.%import_ref.2f9a36.1 [template]
// CHECK:STDOUT:   %assoc0.92248f.2: %Make.assoc_type.5 = assoc_entity element0, imports.%import_ref.2f9a36.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.7: %Factory.type.c = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.dc: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.9f5 = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %import_ref.e = import_ref Main//factory, loc11_22, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.9fb = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.20: @Factory.%Make.assoc_type (%Make.assoc_type.5) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.92248f.2)]
// CHECK:STDOUT:   %import_ref.22 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.db: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.c: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.8]
// CHECK:STDOUT:   %import_ref.8a = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.2f9a36.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.7
// CHECK:STDOUT:     .A = imports.%import_ref.dc
// CHECK:STDOUT:     .B = imports.%import_ref.9f5
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .MakeC = %MakeC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.dc [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.1)]
// CHECK:STDOUT:   %Self: %Factory.type.1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.89)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.89) = struct_value () [symbolic = %Make (constants.%Make.2)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.1), @Factory.%Make.type (%Make.type.89) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.5)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.5) = assoc_entity element0, imports.%import_ref.2f9a36.1 [symbolic = %assoc0 (constants.%assoc0.92248f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.16
// CHECK:STDOUT:     .Make = imports.%import_ref.20
// CHECK:STDOUT:     witness = (imports.%import_ref.22)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.db as imports.%import_ref.c [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.8a
// CHECK:STDOUT:   witness = imports.%import_ref.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9fb
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.17
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self: %Factory.type.1) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a.param_patt: %A) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.c = name_ref Factory, imports.%import_ref.7 [template = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [template = constants.%Factory.type.7]
// CHECK:STDOUT:   %.loc10: %Make.assoc_type.d = specific_constant imports.%import_ref.20, @Factory(constants.%C) [template = constants.%assoc0.d]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.d = name_ref Make, %.loc10 [template = constants.%assoc0.d]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.8
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.6
// CHECK:STDOUT:   %Make => constants.%Make.8
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.7
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.84
// CHECK:STDOUT:   %Make => constants.%Make.7
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d
// CHECK:STDOUT: }
// CHECK:STDOUT:
