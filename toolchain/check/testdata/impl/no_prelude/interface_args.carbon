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
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Factory(C)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a.(Factory(C).Make)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a.(Factory(C).Make)();
}

// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [template]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2ba: type = assoc_entity_type %Action.type.cca, %Op.type.036 [symbolic]
// CHECK:STDOUT:   %assoc0.0aa: %Op.assoc_type.2ba = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.e7a: type = assoc_entity_type %Action.type.cb0, %Op.type.54d [template]
// CHECK:STDOUT:   %assoc0.64a: %Op.assoc_type.e7a = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.4b4: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.40d: %Op.type.4b4 = struct_value () [template]
// CHECK:STDOUT:   %Action.facet: %Action.type.cca = facet_value %A, %impl_witness [symbolic]
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
// CHECK:STDOUT:   %Action.decl: %Action.type.29c = interface_decl @Action [template = constants.%Action.generic] {
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
// CHECK:STDOUT:     %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.cb0]
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self.2: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca), @Action.%Op.type (%Op.type.036) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2ba)]
// CHECK:STDOUT:   %assoc0.loc5_10.2: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.0aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.cca) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.036) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.6ed)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_10.1: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.0aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_10.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.4b4 = fn_decl @Op.2 [template = constants.%Op.40d] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.cca)) {
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
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc16: %Op.assoc_type.e7a = specific_constant @Action.%assoc0.loc5_10.1, @Action(constants.%B) [template = constants.%assoc0.64a]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.e7a = name_ref Op, %.loc16 [template = constants.%assoc0.64a]
// CHECK:STDOUT:   %impl.elem0: %Op.type.54d = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.40d]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.54d
// CHECK:STDOUT:   %Op => constants.%Op.dba
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.e7a
// CHECK:STDOUT:   %assoc0.loc5_10.2 => constants.%assoc0.64a
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
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2ba: type = assoc_entity_type %Action.type.cca, %Op.type.036 [symbolic]
// CHECK:STDOUT:   %assoc0.9b9b19.1: %Op.assoc_type.2ba = assoc_entity element0, imports.%import_ref.0e3753.1 [symbolic]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.e7a: type = assoc_entity_type %Action.type.cb0, %Op.type.54d [template]
// CHECK:STDOUT:   %assoc0.b39: %Op.assoc_type.e7a = assoc_entity element0, imports.%import_ref.0e3753.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.9b9b19.2: %Op.assoc_type.2ba = assoc_entity element0, imports.%import_ref.0e3753.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.d0b) [template]
// CHECK:STDOUT:   %Op.type.4b4: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.40d: %Op.type.4b4 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10a: %Action.type.29c = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.e62: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.dd8: type = import_ref Main//action, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.f42 = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %import_ref.bde = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.71c: <witness> = import_ref Main//action, loc12_21, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.ddc = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.f86: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.9b9b19.2)]
// CHECK:STDOUT:   %import_ref.ae2 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.984: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.cb0]
// CHECK:STDOUT:   %import_ref.7b5 = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.0e3753.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.10a
// CHECK:STDOUT:     .A = imports.%import_ref.e62
// CHECK:STDOUT:     .B = imports.%import_ref.dd8
// CHECK:STDOUT:     .C = imports.%import_ref.f42
// CHECK:STDOUT:     .F = imports.%import_ref.bde
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.e62 [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca), @Action.%Op.type (%Op.type.036) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2ba)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = assoc_entity element0, imports.%import_ref.0e3753.1 [symbolic = %assoc0 (constants.%assoc0.9b9b19.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%import_ref.f86
// CHECK:STDOUT:     witness = (imports.%import_ref.ae2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.984 as imports.%import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.7b5
// CHECK:STDOUT:   witness = imports.%import_ref.71c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self: %Action.type.cca) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%import_ref.10a [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.dd8 [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.cb0]
// CHECK:STDOUT:   %.loc4: %Op.assoc_type.e7a = specific_constant imports.%import_ref.f86, @Action(constants.%B) [template = constants.%assoc0.b39]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.e7a = name_ref Op, %.loc4 [template = constants.%assoc0.b39]
// CHECK:STDOUT:   %impl.elem0: %Op.type.54d = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.40d]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.cb0
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.54d
// CHECK:STDOUT:   %Op => constants.%Op.dba
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.e7a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b39
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
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cca: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.cb0: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.036: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.6ed: %Op.type.036 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2ba: type = assoc_entity_type %Action.type.cca, %Op.type.036 [symbolic]
// CHECK:STDOUT:   %assoc0.9b9b19.1: %Op.assoc_type.2ba = assoc_entity element0, imports.%import_ref.0e3753.1 [symbolic]
// CHECK:STDOUT:   %Op.type.54d: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.dba: %Op.type.54d = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.e7a: type = assoc_entity_type %Action.type.cb0, %Op.type.54d [template]
// CHECK:STDOUT:   %assoc0.b39: %Op.assoc_type.e7a = assoc_entity element0, imports.%import_ref.0e3753.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.e2b: type = facet_type <@Action, @Action(%C)> [template]
// CHECK:STDOUT:   %Op.type.5ad: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.b10: %Op.type.5ad = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.17b: type = assoc_entity_type %Action.type.e2b, %Op.type.5ad [template]
// CHECK:STDOUT:   %assoc0.ccd: %Op.assoc_type.17b = assoc_entity element0, imports.%import_ref.0e3753.1 [template]
// CHECK:STDOUT:   %assoc0.9b9b19.2: %Op.assoc_type.2ba = assoc_entity element0, imports.%import_ref.0e3753.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10a: %Action.type.29c = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.e62: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.81b = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//action, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.bde = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.7a1 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.ddc = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.f86: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.9b9b19.2)]
// CHECK:STDOUT:   %import_ref.ae2 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.984: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.cb0]
// CHECK:STDOUT:   %import_ref.7b5 = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.0e3753.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.10a
// CHECK:STDOUT:     .A = imports.%import_ref.e62
// CHECK:STDOUT:     .B = imports.%import_ref.81b
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .F = imports.%import_ref.bde
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.e62 [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.cca)]
// CHECK:STDOUT:   %Self: %Action.type.cca = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.036)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.036) = struct_value () [symbolic = %Op (constants.%Op.6ed)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.cca), @Action.%Op.type (%Op.type.036) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2ba)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2ba) = assoc_entity element0, imports.%import_ref.0e3753.1 [symbolic = %assoc0 (constants.%assoc0.9b9b19.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ddc
// CHECK:STDOUT:     .Op = imports.%import_ref.f86
// CHECK:STDOUT:     witness = (imports.%import_ref.ae2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.984 as imports.%import_ref.bb2 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.7b5
// CHECK:STDOUT:   witness = imports.%import_ref.7a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self: %Action.type.cca) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%import_ref.10a [template = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [template = constants.%Action.type.e2b]
// CHECK:STDOUT:   %.loc8: %Op.assoc_type.17b = specific_constant imports.%import_ref.f86, @Action(constants.%C) [template = constants.%assoc0.ccd]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.17b = name_ref Op, %.loc8 [template = constants.%assoc0.ccd]
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
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.e7a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b39
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.e2b
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.5ad
// CHECK:STDOUT:   %Op => constants.%Op.b10
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.17b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ccd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.4d9: type = assoc_entity_type %Factory.type.c96, %Make.type.598 [symbolic]
// CHECK:STDOUT:   %assoc0.a4a: %Make.assoc_type.4d9 = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.6c6: type = assoc_entity_type %Factory.type.a5d, %Make.type.c59 [template]
// CHECK:STDOUT:   %assoc0.925: %Make.assoc_type.6c6 = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template]
// CHECK:STDOUT:   %Make.type.ec4: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.377: %Make.type.ec4 = struct_value () [template]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.c96 = facet_value %A, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type.1a8 = interface_decl @Factory [template = constants.%Factory.generic] {
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
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1a8 = name_ref Factory, file.%Factory.decl [template = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.a5d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template = constants.%impl_witness]
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
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96), @Factory.%Make.type (%Make.type.598) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.4d9)]
// CHECK:STDOUT:   %assoc0.loc5_17.2: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.a4a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.c96) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.598) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.737)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_17.1: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.a4a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %assoc0.loc5_17.1
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.ec4 = fn_decl @Make.2 [template = constants.%Make.377] {
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
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.c96)) {
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.6c6
// CHECK:STDOUT:   %assoc0.loc5_17.2 => constants.%assoc0.925
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
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.4d9: type = assoc_entity_type %Factory.type.c96, %Make.type.598 [symbolic]
// CHECK:STDOUT:   %assoc0.9270f3.1: %Make.assoc_type.4d9 = assoc_entity element0, imports.%import_ref.21018a.1 [symbolic]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.6c6: type = assoc_entity_type %Factory.type.a5d, %Make.type.c59 [template]
// CHECK:STDOUT:   %assoc0.c38: %Make.assoc_type.6c6 = assoc_entity element0, imports.%import_ref.21018a.2 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.9270f3.2: %Make.assoc_type.4d9 = assoc_entity element0, imports.%import_ref.21018a.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.9ec) [template]
// CHECK:STDOUT:   %Make.type.ec4: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.377: %Make.type.ec4 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3d: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.e62: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.dd8: type = import_ref Main//factory, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.72b: <witness> = import_ref Main//factory, loc11_22, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.fbb = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b0a: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.9270f3.2)]
// CHECK:STDOUT:   %import_ref.c20 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.984: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.bd2: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %import_ref.a27 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.21018a.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.e3d
// CHECK:STDOUT:     .A = imports.%import_ref.e62
// CHECK:STDOUT:     .B = imports.%import_ref.dd8
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
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%import_ref.dd8 [template = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.e62 [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.598)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.598) = struct_value () [symbolic = %Make (constants.%Make.737)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96), @Factory.%Make.type (%Make.type.598) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.4d9)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = assoc_entity element0, imports.%import_ref.21018a.1 [symbolic = %assoc0 (constants.%assoc0.9270f3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%import_ref.b0a
// CHECK:STDOUT:     witness = (imports.%import_ref.c20)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.984 as imports.%import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.a27
// CHECK:STDOUT:   witness = imports.%import_ref.72b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self: %Factory.type.c96) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a.param_patt: %A) -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%import_ref.e3d [template = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%import_ref.dd8 [template = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %.loc5: %Make.assoc_type.6c6 = specific_constant imports.%import_ref.b0a, @Factory(constants.%B) [template = constants.%assoc0.c38]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.6c6 = name_ref Make, %.loc5 [template = constants.%assoc0.c38]
// CHECK:STDOUT:   %impl.elem0: %Make.type.c59 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Make.377]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.a5d
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.6c6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c38
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
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.c96: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.a5d: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.598: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.737: %Make.type.598 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.4d9: type = assoc_entity_type %Factory.type.c96, %Make.type.598 [symbolic]
// CHECK:STDOUT:   %assoc0.9270f3.1: %Make.assoc_type.4d9 = assoc_entity element0, imports.%import_ref.21018a.1 [symbolic]
// CHECK:STDOUT:   %Make.type.c59: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.efe: %Make.type.c59 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.6c6: type = assoc_entity_type %Factory.type.a5d, %Make.type.c59 [template]
// CHECK:STDOUT:   %assoc0.c38: %Make.assoc_type.6c6 = assoc_entity element0, imports.%import_ref.21018a.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.5c5: type = facet_type <@Factory, @Factory(%C)> [template]
// CHECK:STDOUT:   %Make.type.0de: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.8ba: %Make.type.0de = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.2ea: type = assoc_entity_type %Factory.type.5c5, %Make.type.0de [template]
// CHECK:STDOUT:   %assoc0.9a3: %Make.assoc_type.2ea = assoc_entity element0, imports.%import_ref.21018a.1 [template]
// CHECK:STDOUT:   %assoc0.9270f3.2: %Make.assoc_type.4d9 = assoc_entity element0, imports.%import_ref.21018a.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3d: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.e62: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.81b = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %import_ref.3e9 = import_ref Main//factory, loc11_22, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.fbb = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b0a: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.9270f3.2)]
// CHECK:STDOUT:   %import_ref.c20 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.984: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.bd2: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.a5d]
// CHECK:STDOUT:   %import_ref.a27 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.21018a.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.e3d
// CHECK:STDOUT:     .A = imports.%import_ref.e62
// CHECK:STDOUT:     .B = imports.%import_ref.81b
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.e62 [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.c96)]
// CHECK:STDOUT:   %Self: %Factory.type.c96 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.598)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.598) = struct_value () [symbolic = %Make (constants.%Make.737)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.c96), @Factory.%Make.type (%Make.type.598) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.4d9)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.4d9) = assoc_entity element0, imports.%import_ref.21018a.1 [symbolic = %assoc0 (constants.%assoc0.9270f3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.fbb
// CHECK:STDOUT:     .Make = imports.%import_ref.b0a
// CHECK:STDOUT:     witness = (imports.%import_ref.c20)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.984 as imports.%import_ref.bd2 [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.a27
// CHECK:STDOUT:   witness = imports.%import_ref.3e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self: %Factory.type.c96) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a.param_patt: %A) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%import_ref.e3d [template = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc11: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [template = constants.%Factory.type.5c5]
// CHECK:STDOUT:   %.loc11: %Make.assoc_type.2ea = specific_constant imports.%import_ref.b0a, @Factory(constants.%C) [template = constants.%assoc0.9a3]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.2ea = name_ref Make, %.loc11 [template = constants.%assoc0.9a3]
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
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.c59
// CHECK:STDOUT:   %Make => constants.%Make.efe
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.6c6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c38
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.5c5
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.0de
// CHECK:STDOUT:   %Make => constants.%Make.8ba
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.2ea
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
