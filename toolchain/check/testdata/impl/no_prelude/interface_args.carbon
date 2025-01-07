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
// CHECK:STDOUT:   %Action.type.29c7: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c7 = struct_value () [template]
// CHECK:STDOUT:   %Action.type.55b2: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.55b2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.7d81: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.edae: %Op.type.7d81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.e54f: type = assoc_entity_type %Action.type.55b2, %Op.type.7d81 [symbolic]
// CHECK:STDOUT:   %assoc0.814b: %Op.assoc_type.e54f = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.2f81: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %Op.type.db27: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.a26a: %Op.type.db27 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.390e: type = assoc_entity_type %Action.type.2f81, %Op.type.db27 [template]
// CHECK:STDOUT:   %assoc0.2572: %Op.assoc_type.390e = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.447a: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.aa40: %Op.type.447a = struct_value () [template]
// CHECK:STDOUT:   %Action.facet: %Action.type.55b2 = facet_value %A, %impl_witness [symbolic]
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
// CHECK:STDOUT:   %Action.decl: %Action.type.29c7 = interface_decl @Action [template = constants.%Action.generic] {
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
// CHECK:STDOUT:     %Action.ref: %Action.type.29c7 = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.2f81]
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.55b2)]
// CHECK:STDOUT:   %Self.2: %Action.type.55b2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.7d81)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.7d81) = struct_value () [symbolic = %Op (constants.%Op.edae)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.55b2), @Action.%Op.type (%Op.type.7d81) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.e54f)]
// CHECK:STDOUT:   %assoc0.loc5_10.2: @Action.%Op.assoc_type (%Op.assoc_type.e54f) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.814b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.55b2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.7d81) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.edae)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_10.1: @Action.%Op.assoc_type (%Op.assoc_type.e54f) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.814b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_10.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.447a = fn_decl @Op.2 [template = constants.%Op.aa40] {} {}
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
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.55b2)) {
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
// CHECK:STDOUT:   %Action.ref: %Action.type.29c7 = name_ref Action, file.%Action.decl [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.2f81]
// CHECK:STDOUT:   %.loc16: %Op.assoc_type.390e = specific_constant @Action.%assoc0.loc5_10.1, @Action(constants.%B) [template = constants.%assoc0.2572]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.390e = name_ref Op, %.loc16 [template = constants.%assoc0.2572]
// CHECK:STDOUT:   %impl.elem0: %Op.type.db27 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.aa40]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.2f81
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.db27
// CHECK:STDOUT:   %Op => constants.%Op.a26a
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.390e
// CHECK:STDOUT:   %assoc0.loc5_10.2 => constants.%assoc0.2572
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
// CHECK:STDOUT:   %Action.type.29c7: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c7 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.a6c6: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.a6c6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.6ead: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.5efe: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1d02: %Op.type.5efe = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2aef: type = assoc_entity_type %Action.type.a6c6, %Op.type.5efe [symbolic]
// CHECK:STDOUT:   %assoc0.fd2454.1: %Op.assoc_type.2aef = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic]
// CHECK:STDOUT:   %Op.type.2178: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.5558: %Op.type.2178 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.ef29: type = assoc_entity_type %Action.type.6ead, %Op.type.2178 [template]
// CHECK:STDOUT:   %assoc0.042a: %Op.assoc_type.ef29 = assoc_entity element0, imports.%import_ref.341b2b.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.fd2454.2: %Op.assoc_type.2aef = assoc_entity element0, imports.%import_ref.341b2b.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.e613) [template]
// CHECK:STDOUT:   %Op.type.9b5c: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.796e: %Op.type.9b5c = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3e4: %Action.type.29c7 = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.8a9a: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.ba08: type = import_ref Main//action, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.85dd = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %import_ref.46e6 = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.a8a8: <witness> = import_ref Main//action, loc12_21, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a9 = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ac44 = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.543a: @Action.%Op.assoc_type (%Op.assoc_type.2aef) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.fd2454.2)]
// CHECK:STDOUT:   %import_ref.4d92 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da33 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2f: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2b5c: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.6ead]
// CHECK:STDOUT:   %import_ref.be6f = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.341b2b.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.e3e4
// CHECK:STDOUT:     .A = imports.%import_ref.8a9a
// CHECK:STDOUT:     .B = imports.%import_ref.ba08
// CHECK:STDOUT:     .C = imports.%import_ref.85dd
// CHECK:STDOUT:     .F = imports.%import_ref.46e6
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9a [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.a6c6)]
// CHECK:STDOUT:   %Self: %Action.type.a6c6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.5efe)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.5efe) = struct_value () [symbolic = %Op (constants.%Op.1d02)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.a6c6), @Action.%Op.type (%Op.type.5efe) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2aef)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2aef) = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic = %assoc0 (constants.%assoc0.fd2454.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ac44
// CHECK:STDOUT:     .Op = imports.%import_ref.543a
// CHECK:STDOUT:     witness = (imports.%import_ref.4d92)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2f as imports.%import_ref.2b5c [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.be6f
// CHECK:STDOUT:   witness = imports.%import_ref.a8a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da33
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self: %Action.type.a6c6) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c7 = name_ref Action, imports.%import_ref.e3e4 [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.ba08 [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.6ead]
// CHECK:STDOUT:   %.loc4: %Op.assoc_type.ef29 = specific_constant imports.%import_ref.543a, @Action(constants.%B) [template = constants.%assoc0.042a]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.ef29 = name_ref Op, %.loc4 [template = constants.%assoc0.042a]
// CHECK:STDOUT:   %impl.elem0: %Op.type.2178 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.796e]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.6ead
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2178
// CHECK:STDOUT:   %Op => constants.%Op.5558
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.ef29
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.042a
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
// CHECK:STDOUT:   %Action.type.29c7: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c7 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.a6c6: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.a6c6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.6ead: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.5efe: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1d02: %Op.type.5efe = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2aef: type = assoc_entity_type %Action.type.a6c6, %Op.type.5efe [symbolic]
// CHECK:STDOUT:   %assoc0.fd2454.1: %Op.assoc_type.2aef = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic]
// CHECK:STDOUT:   %Op.type.2178: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.5558: %Op.type.2178 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.ef29: type = assoc_entity_type %Action.type.6ead, %Op.type.2178 [template]
// CHECK:STDOUT:   %assoc0.042a: %Op.assoc_type.ef29 = assoc_entity element0, imports.%import_ref.341b2b.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.2169: type = facet_type <@Action, @Action(%C)> [template]
// CHECK:STDOUT:   %Op.type.ffbe: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.4666: %Op.type.ffbe = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.fa38: type = assoc_entity_type %Action.type.2169, %Op.type.ffbe [template]
// CHECK:STDOUT:   %assoc0.5996: %Op.assoc_type.fa38 = assoc_entity element0, imports.%import_ref.341b2b.1 [template]
// CHECK:STDOUT:   %assoc0.fd2454.2: %Op.assoc_type.2aef = assoc_entity element0, imports.%import_ref.341b2b.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3e4: %Action.type.29c7 = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.8a9a: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.874b = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %import_ref.0acc: type = import_ref Main//action, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.46e6 = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.4405 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a9 = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ac44 = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.543a: @Action.%Op.assoc_type (%Op.assoc_type.2aef) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.fd2454.2)]
// CHECK:STDOUT:   %import_ref.4d92 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da33 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2f: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2b5c: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.6ead]
// CHECK:STDOUT:   %import_ref.be6f = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.341b2b.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c47 = import_ref Main//action, inst49 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.e3e4
// CHECK:STDOUT:     .A = imports.%import_ref.8a9a
// CHECK:STDOUT:     .B = imports.%import_ref.874b
// CHECK:STDOUT:     .C = imports.%import_ref.0acc
// CHECK:STDOUT:     .F = imports.%import_ref.46e6
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9a [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.a6c6)]
// CHECK:STDOUT:   %Self: %Action.type.a6c6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.5efe)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.5efe) = struct_value () [symbolic = %Op (constants.%Op.1d02)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.a6c6), @Action.%Op.type (%Op.type.5efe) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2aef)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2aef) = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic = %assoc0 (constants.%assoc0.fd2454.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ac44
// CHECK:STDOUT:     .Op = imports.%import_ref.543a
// CHECK:STDOUT:     witness = (imports.%import_ref.4d92)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2f as imports.%import_ref.2b5c [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.be6f
// CHECK:STDOUT:   witness = imports.%import_ref.4405
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da33
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c47
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self: %Action.type.a6c6) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c7 = name_ref Action, imports.%import_ref.e3e4 [template = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.0acc [template = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [template = constants.%Action.type.2169]
// CHECK:STDOUT:   %.loc8: %Op.assoc_type.fa38 = specific_constant imports.%import_ref.543a, @Action(constants.%C) [template = constants.%assoc0.5996]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.fa38 = name_ref Op, %.loc8 [template = constants.%assoc0.5996]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.6ead
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2178
// CHECK:STDOUT:   %Op => constants.%Op.5558
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.ef29
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.042a
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.2169
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.ffbe
// CHECK:STDOUT:   %Op => constants.%Op.4666
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.fa38
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5996
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1a83: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a83 = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.eb36: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.eb36 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.d005: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.daaf: %Make.type.d005 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.45dc: type = assoc_entity_type %Factory.type.eb36, %Make.type.d005 [symbolic]
// CHECK:STDOUT:   %assoc0.4623: %Make.assoc_type.45dc = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.4e9e: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %Make.type.a70f: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.643c: %Make.type.a70f = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f8d3: type = assoc_entity_type %Factory.type.4e9e, %Make.type.a70f [template]
// CHECK:STDOUT:   %assoc0.b991: %Make.assoc_type.f8d3 = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template]
// CHECK:STDOUT:   %Make.type.8829: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.e083: %Make.type.8829 = struct_value () [template]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.eb36 = facet_value %A, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type.1a83 = interface_decl @Factory [template = constants.%Factory.generic] {
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
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1a83 = name_ref Factory, file.%Factory.decl [template = constants.%Factory.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.4e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.2)> [symbolic = %Factory.type (constants.%Factory.type.eb36)]
// CHECK:STDOUT:   %Self.2: %Factory.type.eb36 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T.loc4_19.2) [symbolic = %Make.type (constants.%Make.type.d005)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.d005) = struct_value () [symbolic = %Make (constants.%Make.daaf)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.eb36), @Factory.%Make.type (%Make.type.d005) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.45dc)]
// CHECK:STDOUT:   %assoc0.loc5_17.2: @Factory.%Make.assoc_type (%Make.assoc_type.45dc) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.4623)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.eb36) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.d005) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.daaf)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_17.1: @Factory.%Make.assoc_type (%Make.assoc_type.45dc) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.4623)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %assoc0.loc5_17.1
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.8829 = fn_decl @Make.2 [template = constants.%Make.e083] {
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
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.eb36)) {
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.4e9e
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.a70f
// CHECK:STDOUT:   %Make => constants.%Make.643c
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f8d3
// CHECK:STDOUT:   %assoc0.loc5_17.2 => constants.%assoc0.b991
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
// CHECK:STDOUT:   %Factory.type.1a83: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a83 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.d2d3: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.d2d3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.91a6: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.a5c3: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.0f02: %Make.type.a5c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.56fd: type = assoc_entity_type %Factory.type.d2d3, %Make.type.a5c3 [symbolic]
// CHECK:STDOUT:   %assoc0.08b852.1: %Make.assoc_type.56fd = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic]
// CHECK:STDOUT:   %Make.type.e84c: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.ea4b: %Make.type.e84c = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f242: type = assoc_entity_type %Factory.type.91a6, %Make.type.e84c [template]
// CHECK:STDOUT:   %assoc0.fe9c: %Make.assoc_type.f242 = assoc_entity element0, imports.%import_ref.930de4.2 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.08b852.2: %Make.assoc_type.56fd = assoc_entity element0, imports.%import_ref.930de4.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.d29f) [template]
// CHECK:STDOUT:   %Make.type.4c5a: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.2877: %Make.type.4c5a = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10c1: %Factory.type.1a83 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.8a9a: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.ba08: type = import_ref Main//factory, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.8b96: <witness> = import_ref Main//factory, loc11_22, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a9 = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c9b3 = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2a25: @Factory.%Make.assoc_type (%Make.assoc_type.56fd) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.08b852.2)]
// CHECK:STDOUT:   %import_ref.1335 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da33 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2f: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.5efc: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.91a6]
// CHECK:STDOUT:   %import_ref.0440 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.930de4.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.10c1
// CHECK:STDOUT:     .A = imports.%import_ref.8a9a
// CHECK:STDOUT:     .B = imports.%import_ref.ba08
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
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%import_ref.ba08 [template = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9a [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d2d3)]
// CHECK:STDOUT:   %Self: %Factory.type.d2d3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.a5c3)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.a5c3) = struct_value () [symbolic = %Make (constants.%Make.0f02)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.d2d3), @Factory.%Make.type (%Make.type.a5c3) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.56fd)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.56fd) = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic = %assoc0 (constants.%assoc0.08b852.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.c9b3
// CHECK:STDOUT:     .Make = imports.%import_ref.2a25
// CHECK:STDOUT:     witness = (imports.%import_ref.1335)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2f as imports.%import_ref.5efc [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.0440
// CHECK:STDOUT:   witness = imports.%import_ref.8b96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da33
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self: %Factory.type.d2d3) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a.param_patt: %A) -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a83 = name_ref Factory, imports.%import_ref.10c1 [template = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%import_ref.ba08 [template = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.91a6]
// CHECK:STDOUT:   %.loc5: %Make.assoc_type.f242 = specific_constant imports.%import_ref.2a25, @Factory(constants.%B) [template = constants.%assoc0.fe9c]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.f242 = name_ref Make, %.loc5 [template = constants.%assoc0.fe9c]
// CHECK:STDOUT:   %impl.elem0: %Make.type.e84c = impl_witness_access constants.%impl_witness, element0 [template = constants.%Make.2877]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.91a6
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.e84c
// CHECK:STDOUT:   %Make => constants.%Make.ea4b
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f242
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fe9c
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
// CHECK:STDOUT:   %Factory.type.1a83: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a83 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.d2d3: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.d2d3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.91a6: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.a5c3: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.0f02: %Make.type.a5c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.56fd: type = assoc_entity_type %Factory.type.d2d3, %Make.type.a5c3 [symbolic]
// CHECK:STDOUT:   %assoc0.08b852.1: %Make.assoc_type.56fd = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic]
// CHECK:STDOUT:   %Make.type.e84c: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.ea4b: %Make.type.e84c = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f242: type = assoc_entity_type %Factory.type.91a6, %Make.type.e84c [template]
// CHECK:STDOUT:   %assoc0.fe9c: %Make.assoc_type.f242 = assoc_entity element0, imports.%import_ref.930de4.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.f0b8: type = facet_type <@Factory, @Factory(%C)> [template]
// CHECK:STDOUT:   %Make.type.42c1: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.29c5: %Make.type.42c1 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.392d: type = assoc_entity_type %Factory.type.f0b8, %Make.type.42c1 [template]
// CHECK:STDOUT:   %assoc0.3a7a: %Make.assoc_type.392d = assoc_entity element0, imports.%import_ref.930de4.1 [template]
// CHECK:STDOUT:   %assoc0.08b852.2: %Make.assoc_type.56fd = assoc_entity element0, imports.%import_ref.930de4.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10c1: %Factory.type.1a83 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.8a9a: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.874b = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %import_ref.826f = import_ref Main//factory, loc11_22, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a9 = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c9b3 = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2a25: @Factory.%Make.assoc_type (%Make.assoc_type.56fd) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.08b852.2)]
// CHECK:STDOUT:   %import_ref.1335 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da33 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2f: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.5efc: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.91a6]
// CHECK:STDOUT:   %import_ref.0440 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.930de4.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.10c1
// CHECK:STDOUT:     .A = imports.%import_ref.8a9a
// CHECK:STDOUT:     .B = imports.%import_ref.874b
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9a [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d2d3)]
// CHECK:STDOUT:   %Self: %Factory.type.d2d3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.a5c3)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.a5c3) = struct_value () [symbolic = %Make (constants.%Make.0f02)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.d2d3), @Factory.%Make.type (%Make.type.a5c3) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.56fd)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.56fd) = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic = %assoc0 (constants.%assoc0.08b852.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.c9b3
// CHECK:STDOUT:     .Make = imports.%import_ref.2a25
// CHECK:STDOUT:     witness = (imports.%import_ref.1335)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2f as imports.%import_ref.5efc [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.0440
// CHECK:STDOUT:   witness = imports.%import_ref.826f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da33
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
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self: %Factory.type.d2d3) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a.param_patt: %A) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a83 = name_ref Factory, imports.%import_ref.10c1 [template = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [template = constants.%Factory.type.f0b8]
// CHECK:STDOUT:   %.loc10: %Make.assoc_type.392d = specific_constant imports.%import_ref.2a25, @Factory(constants.%C) [template = constants.%assoc0.3a7a]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.392d = name_ref Make, %.loc10 [template = constants.%assoc0.3a7a]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.91a6
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.e84c
// CHECK:STDOUT:   %Make => constants.%Make.ea4b
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f242
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fe9c
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.f0b8
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.42c1
// CHECK:STDOUT:   %Make => constants.%Make.29c5
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.392d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3a7a
// CHECK:STDOUT: }
// CHECK:STDOUT:
