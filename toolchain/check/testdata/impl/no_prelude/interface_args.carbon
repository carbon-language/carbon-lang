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
// CHECK:STDOUT:   %Action.type.1: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action: %Action.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Action.type.2: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Action.type.2, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.3: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Action.type.3, %Op.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %.7: %Action.type.2 = facet_value %A, %A [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Op.2) [template]
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
// CHECK:STDOUT:   %Action.decl: %Action.type.1 = interface_decl @Action [template = constants.%Action] {
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
// CHECK:STDOUT:     %Action.ref: %Action.type.1 = name_ref Action, file.%Action.decl [template = constants.%Action]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T.loc4_18.2)> [symbolic = %Action.type (constants.%Action.type.2)]
// CHECK:STDOUT:   %Self.2: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T.loc4_18.2) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.loc5_10.2: type = assoc_entity_type @Action.%Action.type (%Action.type.2), @Action.%Op.type (%Op.type.1) [symbolic = %.loc5_10.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_10.3: @Action.%.loc5_10.2 (%.1) = assoc_entity element0, %Op.decl [symbolic = %.loc5_10.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%Action.type (%Action.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.1) = fn_decl @Op.1 [symbolic = @Action.%Op (constants.%Op.1)] {} {}
// CHECK:STDOUT:     %.loc5_10.1: @Action.%.loc5_10.2 (%.1) = assoc_entity element0, %Op.decl [symbolic = %.loc5_10.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %.loc5_10.1
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Action.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.2 = fn_decl @Op.2 [template = constants.%Op.2] {} {}
// CHECK:STDOUT:   %.loc12: <witness> = interface_witness (%Op.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Action.%T.loc4_18.1: type, @Action.%Self.1: @Action.%Action.type (%Action.type.2)) {
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
// CHECK:STDOUT:   %Action.ref: %Action.type.1 = name_ref Action, file.%Action.decl [template = constants.%Action]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.3]
// CHECK:STDOUT:   %.loc16_26: %.5 = specific_constant @Action.%.loc5_10.1, @Action(constants.%B) [template = constants.%.6]
// CHECK:STDOUT:   %Op.ref: %.5 = name_ref Op, %.loc16_26 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_15: %Op.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %.loc16_15()
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
// CHECK:STDOUT: specific @Action(%T.loc4_18.2) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.3
// CHECK:STDOUT:   %Op => constants.%Op.3
// CHECK:STDOUT:   %.loc5_10.2 => constants.%.5
// CHECK:STDOUT:   %.loc5_10.3 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%B, constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Action.type.1: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Action: %Action.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.2: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @Action.%Action.type (%Action.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.3: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Self.2: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Action.type.2, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Action.type.3, %Op.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.16 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.8: %.3 = assoc_entity element0, imports.%import_ref.17 [symbolic]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Action.type.1 = import_ref Main//action, inst+7, loaded [template = constants.%Action]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//action, inst+28, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//action, inst+33, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//action, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//action, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//action, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//action, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.8: @Action.%.1 (%.3) = import_ref Main//action, inst+20, loaded [symbolic = @Action.%.2 (constants.%.8)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//action, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//action, inst+29, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Main//action, inst+39, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Main//action, inst+42, loaded [template = constants.%Action.type.3]
// CHECK:STDOUT:   %import_ref.13: <witness> = import_ref Main//action, inst+53, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//action, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//action, inst+16, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.1
// CHECK:STDOUT:     .A = imports.%import_ref.2
// CHECK:STDOUT:     .B = imports.%import_ref.3
// CHECK:STDOUT:     .C = imports.%import_ref.4
// CHECK:STDOUT:     .F = imports.%import_ref.5
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.2)]
// CHECK:STDOUT:   %Self: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @Action.%Action.type (%Action.type.2), @Action.%Op.type (%Op.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Action.%.1 (%.3) = assoc_entity element0, imports.%import_ref.15 [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Op = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.14
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self.1: @Action.%Action.type (%Action.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.1 = name_ref Action, imports.%import_ref.1 [template = constants.%Action]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%B)> [template = constants.%Action.type.3]
// CHECK:STDOUT:   %.loc4_26: %.5 = specific_constant imports.%import_ref.8, @Action(constants.%B) [template = constants.%.6]
// CHECK:STDOUT:   %Op.ref: %.5 = name_ref Op, %.loc4_26 [template = constants.%.6]
// CHECK:STDOUT:   %.loc4_15: %Op.type.2 = interface_witness_access constants.%.9, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %.loc4_15()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2();
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2
// CHECK:STDOUT:   %Op => constants.%Op.2
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Action.type.1: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %Action: %Action.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.2: type = facet_type <@Action, @Action(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @Action.%Action.type (%Action.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Action.type.3: type = facet_type <@Action, @Action(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Self.2: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Action.type.2, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Action.type.3, %Op.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.16 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Action.type.4: type = facet_type <@Action, @Action(%C)> [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %Action.type.4, %Op.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %.10: %.3 = assoc_entity element0, imports.%import_ref.18 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Action.type.1 = import_ref Main//action, inst+7, loaded [template = constants.%Action]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//action, inst+28, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//action, inst+33, unloaded
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//action, inst+36, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//action, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//action, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//action, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.8: @Action.%.1 (%.3) = import_ref Main//action, inst+20, loaded [symbolic = @Action.%.2 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//action, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//action, inst+29, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Main//action, inst+39, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Main//action, inst+42, loaded [template = constants.%Action.type.3]
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//action, inst+53, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//action, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//action, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//action, inst+37, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Action = imports.%import_ref.1
// CHECK:STDOUT:     .A = imports.%import_ref.2
// CHECK:STDOUT:     .B = imports.%import_ref.3
// CHECK:STDOUT:     .C = imports.%import_ref.4
// CHECK:STDOUT:     .F = imports.%import_ref.5
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(%T)> [symbolic = %Action.type (constants.%Action.type.2)]
// CHECK:STDOUT:   %Self: %Action.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @Action.%Action.type (%Action.type.2), @Action.%Op.type (%Op.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Action.%.1 (%.3) = assoc_entity element0, imports.%import_ref.15 [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Op = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = imports.%import_ref.14
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self.1: @Action.%Action.type (%Action.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type.1 = name_ref Action, imports.%import_ref.1 [template = constants.%Action]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %Action.type: type = facet_type <@Action, @Action(constants.%C)> [template = constants.%Action.type.4]
// CHECK:STDOUT:   %.loc8: %.8 = specific_constant imports.%import_ref.8, @Action(constants.%C) [template = constants.%.9]
// CHECK:STDOUT:   %Op.ref: %.8 = name_ref Op, %.loc8 [template = constants.%.9]
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
// CHECK:STDOUT:   %Action.type => constants.%Action.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2
// CHECK:STDOUT:   %Op => constants.%Op.2
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T, constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Action.type => constants.%Action.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.3
// CHECK:STDOUT:   %Op => constants.%Op.3
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.1: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory: %Factory.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.2: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Factory.type.2, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.3: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Factory.type.3, %Make.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %.7: %Factory.type.2 = facet_value %A, %A [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Make.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type.1 = interface_decl @Factory [template = constants.%Factory] {
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
// CHECK:STDOUT:     %Factory.ref: %Factory.type.1 = name_ref Factory, file.%Factory.decl [template = constants.%Factory]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T.loc4_19.2)> [symbolic = %Factory.type (constants.%Factory.type.2)]
// CHECK:STDOUT:   %Self.2: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T.loc4_19.2) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.loc5_17.2: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.2), @Factory.%Make.type (%Make.type.1) [symbolic = %.loc5_17.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_17.3: @Factory.%.loc5_17.2 (%.1) = assoc_entity element0, %Make.decl [symbolic = %.loc5_17.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%Factory.type (%Factory.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.1) = fn_decl @Make.1 [symbolic = @Factory.%Make (constants.%Make.1)] {
// CHECK:STDOUT:       %return.patt: @Make.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Factory.%T.loc4_19.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5_17.1: @Factory.%.loc5_17.2 (%.1) = assoc_entity element0, %Make.decl [symbolic = %.loc5_17.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %.loc5_17.1
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %Factory.type {
// CHECK:STDOUT:   %Make.decl: %Make.type.2 = fn_decl @Make.2 [template = constants.%Make.2] {
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = interface_witness (%Make.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   witness = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@Factory.%T.loc4_19.1: type, @Factory.%Self.1: @Factory.%Factory.type (%Factory.type.2)) {
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
// CHECK:STDOUT: specific @Factory(%T.loc4_19.2) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT:   %.loc5_17.2 => constants.%.5
// CHECK:STDOUT:   %.loc5_17.3 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%B, constants.%.7) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type.1: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory: %Factory.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.2: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @Factory.%Factory.type (%Factory.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.3: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Self.2: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Factory.type.2, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Factory.type.3, %Make.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %.8: %.3 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Make.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Factory.type.1 = import_ref Main//factory, inst+7, loaded [template = constants.%Factory]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//factory, inst+34, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//factory, inst+39, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//factory, inst+40, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.6: @Factory.%.1 (%.3) = import_ref Main//factory, inst+26, loaded [symbolic = @Factory.%.2 (constants.%.8)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//factory, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//factory, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Main//factory, inst+42, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//factory, inst+45, loaded [template = constants.%Factory.type.3]
// CHECK:STDOUT:   %import_ref.11: <witness> = import_ref Main//factory, inst+61, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//factory, inst+53, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//factory, inst+21, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.1
// CHECK:STDOUT:     .A = imports.%import_ref.2
// CHECK:STDOUT:     .B = imports.%import_ref.3
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %B.ref.loc4: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %B = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.2)]
// CHECK:STDOUT:   %Self: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.2), @Factory.%Make.type (%Make.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Factory.%.1 (%.3) = assoc_entity element0, imports.%import_ref.13 [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .Make = imports.%import_ref.6
// CHECK:STDOUT:     witness = (imports.%import_ref.7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.12
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self.1: @Factory.%Factory.type (%Factory.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a.param_patt: %A) -> %return: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1 = name_ref Factory, imports.%import_ref.1 [template = constants.%Factory]
// CHECK:STDOUT:   %B.ref.loc5: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%B)> [template = constants.%Factory.type.3]
// CHECK:STDOUT:   %.loc5_23: %.5 = specific_constant imports.%import_ref.6, @Factory(constants.%B) [template = constants.%.6]
// CHECK:STDOUT:   %Make.ref: %.5 = name_ref Make, %.loc5_23 [template = constants.%.6]
// CHECK:STDOUT:   %.loc5_11: %Make.type.2 = interface_witness_access constants.%.9, element0 [template = constants.%Make.3]
// CHECK:STDOUT:   %.loc4_16.2: ref %B = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %B = call %.loc5_11() to %.loc4_16.2
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B;
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Factory.type.1: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %Factory: %Factory.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.2: type = facet_type <@Factory, @Factory(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @Factory.%Factory.type (%Factory.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Factory.type.3: type = facet_type <@Factory, @Factory(%B)> [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Self.2: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Factory.type.2, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Factory.type.3, %Make.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %Factory.type.4: type = facet_type <@Factory, @Factory(%C)> [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %Factory.type.4, %Make.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.13 [template]
// CHECK:STDOUT:   %.10: %.3 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Factory.type.1 = import_ref Main//factory, inst+7, loaded [template = constants.%Factory]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//factory, inst+34, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//factory, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//factory, inst+40, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.6: @Factory.%.1 (%.3) = import_ref Main//factory, inst+26, loaded [symbolic = @Factory.%.2 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//factory, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//factory, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Main//factory, inst+42, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//factory, inst+45, loaded [template = constants.%Factory.type.3]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//factory, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//factory, inst+53, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//factory, inst+21, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = imports.%import_ref.1
// CHECK:STDOUT:     .A = imports.%import_ref.2
// CHECK:STDOUT:     .B = imports.%import_ref.3
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(%T)> [symbolic = %Factory.type (constants.%Factory.type.2)]
// CHECK:STDOUT:   %Self: %Factory.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @Factory.%Factory.type (%Factory.type.2), @Factory.%Make.type (%Make.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Factory.%.1 (%.3) = assoc_entity element0, imports.%import_ref.13 [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .Make = imports.%import_ref.6
// CHECK:STDOUT:     witness = (imports.%import_ref.7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = imports.%import_ref.12
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self.1: @Factory.%Factory.type (%Factory.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a.param_patt: %A) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type.1 = name_ref Factory, imports.%import_ref.1 [template = constants.%Factory]
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Factory.type: type = facet_type <@Factory, @Factory(constants.%C)> [template = constants.%Factory.type.4]
// CHECK:STDOUT:   %.loc10: %.8 = specific_constant imports.%import_ref.6, @Factory(constants.%C) [template = constants.%.9]
// CHECK:STDOUT:   %Make.ref: %.8 = name_ref Make, %.loc10 [template = constants.%.9]
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
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Factory.type => constants.%Factory.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
