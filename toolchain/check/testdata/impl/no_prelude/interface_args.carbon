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
// CHECK:STDOUT:   %Action.type.29c: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action.generic: %Action.type.29c = struct_value () [template]
// CHECK:STDOUT:   %Action.type.55b: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.55b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.7d8: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.eda: %Op.type.7d8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.e54: type = assoc_entity_type %Action.type.55b, %Op.type.7d8 [symbolic]
// CHECK:STDOUT:   %assoc0.814: %Op.assoc_type.e54 = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.2f8: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %Op.type.db2: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.a26: %Op.type.db2 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.390: type = assoc_entity_type %Action.type.2f8, %Op.type.db2 [template]
// CHECK:STDOUT:   %assoc0.257: %Op.assoc_type.390 = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.447: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.aa4: %Op.type.447 = struct_value () [template]
// CHECK:STDOUT:   %Action.facet: %Action.type.55b = facet_value %A, %impl_witness [symbolic]
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
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.2f8]
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.55b)]
// CHECK:STDOUT:   %Self.2: %Action.type.55b = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.7d8)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.7d8) = struct_value () [symbolic = %Op (constants.%Op.eda)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.55b), @Action.%Op.type (%Op.type.7d8) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.e54)]
// CHECK:STDOUT:   %assoc0.loc5_10.2: @Action.%Op.assoc_type (%Op.assoc_type.e54) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.814)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.55b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.7d8) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.eda)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_10.1: @Action.%Op.assoc_type (%Op.assoc_type.e54) = assoc_entity element0, %Op.decl [symbolic = %assoc0.loc5_10.2 (constants.%assoc0.814)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc5_10.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.447 = fn_decl @Op.2 [template = constants.%Op.aa4] {} {}
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
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.55b)) {
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
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.2f8]
// CHECK:STDOUT:   %.loc16: %Op.assoc_type.390 = specific_constant @Action.%assoc0.loc5_10.1, @Action(constants.%B) [template = constants.%assoc0.257]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.390 = name_ref Op, %.loc16 [template = constants.%assoc0.257]
// CHECK:STDOUT:   %impl.elem0: %Op.type.db2 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.aa4]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.2f8
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.db2
// CHECK:STDOUT:   %Op => constants.%Op.a26
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.390
// CHECK:STDOUT:   %assoc0.loc5_10.2 => constants.%assoc0.257
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
// CHECK:STDOUT:   %Action.type.a6c: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.a6c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.6ea: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.5ef: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1d0: %Op.type.5ef = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2ae: type = assoc_entity_type %Action.type.a6c, %Op.type.5ef [symbolic]
// CHECK:STDOUT:   %assoc0.fd2454.1: %Op.assoc_type.2ae = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic]
// CHECK:STDOUT:   %Op.type.217: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.555: %Op.type.217 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.ef2: type = assoc_entity_type %Action.type.6ea, %Op.type.217 [template]
// CHECK:STDOUT:   %assoc0.042: %Op.assoc_type.ef2 = assoc_entity element0, imports.%import_ref.341b2b.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.fd2454.2: %Op.assoc_type.2ae = assoc_entity element0, imports.%import_ref.341b2b.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.e61) [template]
// CHECK:STDOUT:   %Op.type.9b5: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.796: %Op.type.9b5 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3e: %Action.type.29c = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.8a9: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.ba0: type = import_ref Main//action, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.85d = import_ref Main//action, C, unloaded
// CHECK:STDOUT:   %import_ref.46e = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.a8a: <witness> = import_ref Main//action, loc12_21, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ac4 = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.543: @Action.%Op.assoc_type (%Op.assoc_type.2ae) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.fd2454.2)]
// CHECK:STDOUT:   %import_ref.4d9 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da3 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2b5: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.6ea]
// CHECK:STDOUT:   %import_ref.be6 = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.341b2b.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.e3e
// CHECK:STDOUT:     .A = imports.%import_ref.8a9
// CHECK:STDOUT:     .B = imports.%import_ref.ba0
// CHECK:STDOUT:     .C = imports.%import_ref.85d
// CHECK:STDOUT:     .F = imports.%import_ref.46e
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9 [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.a6c)]
// CHECK:STDOUT:   %Self: %Action.type.a6c = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.5ef)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.5ef) = struct_value () [symbolic = %Op (constants.%Op.1d0)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.a6c), @Action.%Op.type (%Op.type.5ef) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2ae)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2ae) = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic = %assoc0 (constants.%assoc0.fd2454.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ac4
// CHECK:STDOUT:     .Op = imports.%import_ref.543
// CHECK:STDOUT:     witness = (imports.%import_ref.4d9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2 as imports.%import_ref.2b5 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.be6
// CHECK:STDOUT:   witness = imports.%import_ref.a8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da3
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self: %Action.type.a6c) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%import_ref.e3e [template = constants.%Action.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.ba0 [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.6ea]
// CHECK:STDOUT:   %.loc4: %Op.assoc_type.ef2 = specific_constant imports.%import_ref.543, @Action(constants.%B) [template = constants.%assoc0.042]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.ef2 = name_ref Op, %.loc4 [template = constants.%assoc0.042]
// CHECK:STDOUT:   %impl.elem0: %Op.type.217 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Op.796]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.6ea
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.217
// CHECK:STDOUT:   %Op => constants.%Op.555
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.ef2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.042
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
// CHECK:STDOUT:   %Action.type.a6c: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.a6c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.6ea: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Op.type.5ef: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1d0: %Op.type.5ef = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.assoc_type.2ae: type = assoc_entity_type %Action.type.a6c, %Op.type.5ef [symbolic]
// CHECK:STDOUT:   %assoc0.fd2454.1: %Op.assoc_type.2ae = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic]
// CHECK:STDOUT:   %Op.type.217: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.555: %Op.type.217 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.ef2: type = assoc_entity_type %Action.type.6ea, %Op.type.217 [template]
// CHECK:STDOUT:   %assoc0.042: %Op.assoc_type.ef2 = assoc_entity element0, imports.%import_ref.341b2b.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.216: type = facet_type <@Action, @Action(%C)> [template]
// CHECK:STDOUT:   %Op.type.ffb: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.466: %Op.type.ffb = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type.fa3: type = assoc_entity_type %Action.type.216, %Op.type.ffb [template]
// CHECK:STDOUT:   %assoc0.599: %Op.assoc_type.fa3 = assoc_entity element0, imports.%import_ref.341b2b.1 [template]
// CHECK:STDOUT:   %assoc0.fd2454.2: %Op.assoc_type.2ae = assoc_entity element0, imports.%import_ref.341b2b.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.e3e: %Action.type.29c = import_ref Main//action, Action, loaded [template = constants.%Action.generic]
// CHECK:STDOUT:   %import_ref.8a9: type = import_ref Main//action, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.874 = import_ref Main//action, B, unloaded
// CHECK:STDOUT:   %import_ref.0ac: type = import_ref Main//action, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.46e = import_ref Main//action, F, unloaded
// CHECK:STDOUT:   %import_ref.440 = import_ref Main//action, loc12_21, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//action, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a = import_ref Main//action, inst46 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ac4 = import_ref Main//action, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.543: @Action.%Op.assoc_type (%Op.assoc_type.2ae) = import_ref Main//action, loc5_10, loaded [symbolic = @Action.%assoc0 (constants.%assoc0.fd2454.2)]
// CHECK:STDOUT:   %import_ref.4d9 = import_ref Main//action, Op, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//action, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da3 = import_ref Main//action, inst41 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//action, loc12_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2b5: type = import_ref Main//action, loc12_19, loaded [template = constants.%Action.type.6ea]
// CHECK:STDOUT:   %import_ref.be6 = import_ref Main//action, loc13_11, unloaded
// CHECK:STDOUT:   %import_ref.341b2b.1 = import_ref Main//action, loc5_10, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//action, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref Main//action, inst49 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.e3e
// CHECK:STDOUT:     .A = imports.%import_ref.8a9
// CHECK:STDOUT:     .B = imports.%import_ref.874
// CHECK:STDOUT:     .C = imports.%import_ref.0ac
// CHECK:STDOUT:     .F = imports.%import_ref.46e
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9 [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) [from "action.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.a6c)]
// CHECK:STDOUT:   %Self: %Action.type.a6c = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.5ef)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.5ef) = struct_value () [symbolic = %Op (constants.%Op.1d0)]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type @Action.%Action.type (%Action.type.a6c), @Action.%Op.type (%Op.type.5ef) [symbolic = %Op.assoc_type (constants.%Op.assoc_type.2ae)]
// CHECK:STDOUT:   %assoc0: @Action.%Op.assoc_type (%Op.assoc_type.2ae) = assoc_entity element0, imports.%import_ref.341b2b.1 [symbolic = %assoc0 (constants.%assoc0.fd2454.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ac4
// CHECK:STDOUT:     .Op = imports.%import_ref.543
// CHECK:STDOUT:     witness = (imports.%import_ref.4d9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2 as imports.%import_ref.2b5 [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.be6
// CHECK:STDOUT:   witness = imports.%import_ref.440
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da3
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "action.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self: %Action.type.a6c) [from "action.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.29c = name_ref Action, imports.%import_ref.e3e [template = constants.%Action.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [template = constants.%Action.type.216]
// CHECK:STDOUT:   %.loc8: %Op.assoc_type.fa3 = specific_constant imports.%import_ref.543, @Action(constants.%C) [template = constants.%assoc0.599]
// CHECK:STDOUT:   %Op.ref: %Op.assoc_type.fa3 = name_ref Op, %.loc8 [template = constants.%assoc0.599]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.6ea
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.217
// CHECK:STDOUT:   %Op => constants.%Op.555
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.ef2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.042
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.216
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.ffb
// CHECK:STDOUT:   %Op => constants.%Op.466
// CHECK:STDOUT:   %Op.assoc_type => constants.%Op.assoc_type.fa3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.599
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1a8: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory.generic: %Factory.type.1a8 = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.eb3: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.eb3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.d00: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.daa: %Make.type.d00 = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.45d: type = assoc_entity_type %Factory.type.eb3, %Make.type.d00 [symbolic]
// CHECK:STDOUT:   %assoc0.462: %Make.assoc_type.45d = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.4e9: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %Make.type.a70: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.643: %Make.type.a70 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f8d: type = assoc_entity_type %Factory.type.4e9, %Make.type.a70 [template]
// CHECK:STDOUT:   %assoc0.b99: %Make.assoc_type.f8d = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template]
// CHECK:STDOUT:   %Make.type.882: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.e08: %Make.type.882 = struct_value () [template]
// CHECK:STDOUT:   %Factory.facet: %Factory.type.eb3 = facet_value %A, %impl_witness [symbolic]
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
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.4e9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Make.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.2)> [symbolic = %Factory.type (constants.%Factory.type.eb3)]
// CHECK:STDOUT:   %Self.2: %Factory.type.eb3 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T.loc4_19.2) [symbolic = %Make.type (constants.%Make.type.d00)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.d00) = struct_value () [symbolic = %Make (constants.%Make.daa)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.eb3), @Factory.%Make.type (%Make.type.d00) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.45d)]
// CHECK:STDOUT:   %assoc0.loc5_17.2: @Factory.%Make.assoc_type (%Make.assoc_type.45d) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.462)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.eb3) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.d00) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.daa)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_17.1: @Factory.%Make.assoc_type (%Make.assoc_type.45d) = assoc_entity element0, %Make.decl [symbolic = %assoc0.loc5_17.2 (constants.%assoc0.462)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %assoc0.loc5_17.1
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.882 = fn_decl @Make.2 [template = constants.%Make.e08] {
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
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.eb3)) {
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.4e9
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.a70
// CHECK:STDOUT:   %Make => constants.%Make.643
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f8d
// CHECK:STDOUT:   %assoc0.loc5_17.2 => constants.%assoc0.b99
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
// CHECK:STDOUT:   %Factory.type.d2d: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.d2d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.91a: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.a5c: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.0f0: %Make.type.a5c = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.56f: type = assoc_entity_type %Factory.type.d2d, %Make.type.a5c [symbolic]
// CHECK:STDOUT:   %assoc0.08b852.1: %Make.assoc_type.56f = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic]
// CHECK:STDOUT:   %Make.type.e84: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.ea4: %Make.type.e84 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f24: type = assoc_entity_type %Factory.type.91a, %Make.type.e84 [template]
// CHECK:STDOUT:   %assoc0.fe9: %Make.assoc_type.f24 = assoc_entity element0, imports.%import_ref.930de4.2 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %assoc0.08b852.2: %Make.assoc_type.56f = assoc_entity element0, imports.%import_ref.930de4.3 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.d29) [template]
// CHECK:STDOUT:   %Make.type.4c5: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.287: %Make.type.4c5 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10c: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.8a9: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.ba0: type = import_ref Main//factory, B, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.8b9: <witness> = import_ref Main//factory, loc11_22, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c9b = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2a2: @Factory.%Make.assoc_type (%Make.assoc_type.56f) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.08b852.2)]
// CHECK:STDOUT:   %import_ref.133 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da3 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.5ef: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.91a]
// CHECK:STDOUT:   %import_ref.044 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.930de4.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.10c
// CHECK:STDOUT:     .A = imports.%import_ref.8a9
// CHECK:STDOUT:     .B = imports.%import_ref.ba0
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
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%import_ref.ba0 [template = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9 [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d2d)]
// CHECK:STDOUT:   %Self: %Factory.type.d2d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.a5c)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.a5c) = struct_value () [symbolic = %Make (constants.%Make.0f0)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.d2d), @Factory.%Make.type (%Make.type.a5c) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.56f)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.56f) = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic = %assoc0 (constants.%assoc0.08b852.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.c9b
// CHECK:STDOUT:     .Make = imports.%import_ref.2a2
// CHECK:STDOUT:     witness = (imports.%import_ref.133)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2 as imports.%import_ref.5ef [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.044
// CHECK:STDOUT:   witness = imports.%import_ref.8b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da3
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self: %Factory.type.d2d) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a.param_patt: %A) -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%import_ref.10c [template = constants.%Factory.generic]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%import_ref.ba0 [template = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.91a]
// CHECK:STDOUT:   %.loc5: %Make.assoc_type.f24 = specific_constant imports.%import_ref.2a2, @Factory(constants.%B) [template = constants.%assoc0.fe9]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.f24 = name_ref Make, %.loc5 [template = constants.%assoc0.fe9]
// CHECK:STDOUT:   %impl.elem0: %Make.type.e84 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Make.287]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.91a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.e84
// CHECK:STDOUT:   %Make => constants.%Make.ea4
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f24
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fe9
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
// CHECK:STDOUT:   %Factory.type.d2d: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.d2d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.91a: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.a5c: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.0f0: %Make.type.a5c = struct_value () [symbolic]
// CHECK:STDOUT:   %Make.assoc_type.56f: type = assoc_entity_type %Factory.type.d2d, %Make.type.a5c [symbolic]
// CHECK:STDOUT:   %assoc0.08b852.1: %Make.assoc_type.56f = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic]
// CHECK:STDOUT:   %Make.type.e84: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.ea4: %Make.type.e84 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.f24: type = assoc_entity_type %Factory.type.91a, %Make.type.e84 [template]
// CHECK:STDOUT:   %assoc0.fe9: %Make.assoc_type.f24 = assoc_entity element0, imports.%import_ref.930de4.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.f0b: type = facet_type <@Factory, @Factory(%C)> [template]
// CHECK:STDOUT:   %Make.type.42c: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.29c: %Make.type.42c = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type.392: type = assoc_entity_type %Factory.type.f0b, %Make.type.42c [template]
// CHECK:STDOUT:   %assoc0.3a7: %Make.assoc_type.392 = assoc_entity element0, imports.%import_ref.930de4.1 [template]
// CHECK:STDOUT:   %assoc0.08b852.2: %Make.assoc_type.56f = assoc_entity element0, imports.%import_ref.930de4.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.10c: %Factory.type.1a8 = import_ref Main//factory, Factory, loaded [template = constants.%Factory.generic]
// CHECK:STDOUT:   %import_ref.8a9: type = import_ref Main//factory, A, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.874 = import_ref Main//factory, B, unloaded
// CHECK:STDOUT:   %import_ref.826 = import_ref Main//factory, loc11_22, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//factory, loc9_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.54a = import_ref Main//factory, inst52 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c9b = import_ref Main//factory, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2a2: @Factory.%Make.assoc_type (%Make.assoc_type.56f) = import_ref Main//factory, loc5_17, loaded [symbolic = @Factory.%assoc0 (constants.%assoc0.08b852.2)]
// CHECK:STDOUT:   %import_ref.133 = import_ref Main//factory, Make, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//factory, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.da3 = import_ref Main//factory, inst47 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.bb2: type = import_ref Main//factory, loc11_6, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.5ef: type = import_ref Main//factory, loc11_20, loaded [template = constants.%Factory.type.91a]
// CHECK:STDOUT:   %import_ref.044 = import_ref Main//factory, loc12_17, unloaded
// CHECK:STDOUT:   %import_ref.930de4.1 = import_ref Main//factory, loc5_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.10c
// CHECK:STDOUT:     .A = imports.%import_ref.8a9
// CHECK:STDOUT:     .B = imports.%import_ref.874
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.8a9 [template = constants.%A]
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
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.d2d)]
// CHECK:STDOUT:   %Self: %Factory.type.d2d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.a5c)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.a5c) = struct_value () [symbolic = %Make (constants.%Make.0f0)]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.d2d), @Factory.%Make.type (%Make.type.a5c) [symbolic = %Make.assoc_type (constants.%Make.assoc_type.56f)]
// CHECK:STDOUT:   %assoc0: @Factory.%Make.assoc_type (%Make.assoc_type.56f) = assoc_entity element0, imports.%import_ref.930de4.1 [symbolic = %assoc0 (constants.%assoc0.08b852.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.c9b
// CHECK:STDOUT:     .Make = imports.%import_ref.2a2
// CHECK:STDOUT:     witness = (imports.%import_ref.133)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.bb2 as imports.%import_ref.5ef [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.044
// CHECK:STDOUT:   witness = imports.%import_ref.826
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "factory.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.da3
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
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self: %Factory.type.d2d) [from "factory.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a.param_patt: %A) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1a8 = name_ref Factory, imports.%import_ref.10c [template = constants.%Factory.generic]
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [template = constants.%Factory.type.f0b]
// CHECK:STDOUT:   %.loc10: %Make.assoc_type.392 = specific_constant imports.%import_ref.2a2, @Factory(constants.%C) [template = constants.%assoc0.3a7]
// CHECK:STDOUT:   %Make.ref: %Make.assoc_type.392 = name_ref Make, %.loc10 [template = constants.%assoc0.3a7]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.91a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.e84
// CHECK:STDOUT:   %Make => constants.%Make.ea4
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.f24
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fe9
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.f0b
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.42c
// CHECK:STDOUT:   %Make => constants.%Make.29c
// CHECK:STDOUT:   %Make.assoc_type => constants.%Make.assoc_type.392
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
