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

library "action";

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

impl library "action";

fn G(a: A) { a.(Action(B).Op)(); }

// --- fail_action.impl.carbon

impl library "action";

// CHECK:STDERR: fail_action.impl.carbon:[[@LINE+4]]:14: ERROR: Cannot access member of interface Action in type A that does not implement that interface.
// CHECK:STDERR: fn G(a: A) { a.(Action(C).Op)(); }
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(a: A) { a.(Action(C).Op)(); }

// --- factory.carbon

library "factory";

interface Factory(T:! type) {
  fn Make() -> T;
}

class A {}
class B {}

impl A as Factory(B) {
  fn Make() -> B;
}

// --- factory.impl.carbon

impl library "factory";

fn MakeB(a: A) -> B {
  return a.(Factory(B).Make)();
}

// --- fail_factory.impl.carbon

impl library "factory";

class C {}

fn MakeC(a: A) -> C {
  // CHECK:STDERR: fail_factory.impl.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Factory in type A that does not implement that interface.
  // CHECK:STDERR:   return a.(Factory(C).Make)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  return a.(Factory(C).Make)();
}

// CHECK:STDOUT: --- action.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Action.type: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Action: %Action.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Action.%Op.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.6: type = interface_type @Action, @Action(%B) [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Op.type.3 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, @Action.%Op.decl [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.5 [template]
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
// CHECK:STDOUT:   %Action.decl: %Action.type = interface_decl @Action [template = constants.%Action] {
// CHECK:STDOUT:     %T.loc4_18.1: type = param T
// CHECK:STDOUT:     %T.loc4_18.2: type = bind_symbolic_name T 0, %T.loc4_18.1 [symbolic = @Action.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %.loc12_19.1: type = value_of_initializer %.loc12_17 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_19.2: type = converted %.loc12_17, %.loc12_19.1 [template = constants.%.6]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %A.ref.loc12: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %Action.ref: %Action.type = name_ref Action, %Action.decl [template = constants.%Action]
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc12_17: init type = call %Action.ref(%B.ref) [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %A.ref.loc16: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %a.loc16_6.1: %A = param a
// CHECK:STDOUT:     @F.%a: %A = bind_name a, %a.loc16_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(file.%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Action, @Action(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Action.%.1 (%.2), @Action.%Op.type (%Op.type.1) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @Action.%.2 (%.3) = assoc_entity element0, %Op.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Action.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Op.decl: @Action.%Op.type (%Op.type.1) = fn_decl @Op.1 [symbolic = %Op (constants.%Op.1)] {}
// CHECK:STDOUT:     %.loc5: @Action.%.2 (%.3) = assoc_entity element0, %Op.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %.loc5
// CHECK:STDOUT:     witness = (%Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.6 {
// CHECK:STDOUT:   %Op.decl: %Op.type.2 = fn_decl @Op.2 [template = constants.%Op.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(file.%T.loc4_18.2: type, @Action.%Self.1: @Action.%.1 (%.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type = name_ref Action, file.%Action.decl [template = constants.%Action]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc16_23: init type = call %Action.ref(%B.ref) [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_26: %.7 = specific_constant @Action.%.loc5, @Action(constants.%B) [template = constants.%.8]
// CHECK:STDOUT:   %Op.ref: %.7 = name_ref Op, %.loc16_26 [template = constants.%.8]
// CHECK:STDOUT:   %.1: %Op.type.3 = interface_witness_access @impl.%.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.call: init %.1 = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Action.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Op.type => constants.%Op.type.3
// CHECK:STDOUT:   %Op => constants.%Op.3
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%B, constants.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Action.type: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Action: %Action.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @Action.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @Action, @Action(%B) [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.3, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.1, @Action(%B) [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %Op.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.10: %.5 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Action.type = import_ref Main//action, inst+4, loaded [template = constants.%Action]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//action, inst+24, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//action, inst+27, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//action, inst+29, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//action, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//action, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//action, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//action, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.9: @Action.%.2 (%.5) = import_ref Main//action, inst+16, loaded [symbolic = @Action.%.3 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.13: <witness> = import_ref Main//action, inst+46, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//action, inst+12, unloaded
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.loc4_6.1: %A = param a
// CHECK:STDOUT:     @G.%a: %A = bind_name a, %a.loc4_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Action, @Action(%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Action(%T) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Action.%.1 (%.3), @Action.%Op.type (%Op.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @Action.%.2 (%.5) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .Op = imports.%import_ref.9
// CHECK:STDOUT:     witness = (imports.%import_ref.10)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%T: type, constants.%Self.1: @Action.%.1 (%.3)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type = name_ref Action, imports.%import_ref.1 [template = constants.%Action]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:   %.loc4_23: init type = call %Action.ref(%B.ref) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_26: %.7 = specific_constant imports.%import_ref.9, @Action(constants.%B) [template = constants.%.8]
// CHECK:STDOUT:   %Op.ref: %.7 = name_ref Op, %.loc4_26 [template = constants.%.8]
// CHECK:STDOUT:   %.1: %Op.type.2 = interface_witness_access imports.%import_ref.13, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %Op.call: init %.2 = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2
// CHECK:STDOUT:   %Op => constants.%Op.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Action.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T, constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_action.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Action.type: type = generic_interface_type @Action [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Action: %Action.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @Action, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @Action.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @Action, @Action(%B) [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op, @Action(%T) [symbolic]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.3, %Op.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op, @Action(%B) [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %Op.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.10: type = interface_type @Action, @Action(%C) [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op, @Action(%C) [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.10, %Op.type.3 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.13: %.5 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Action.type = import_ref Main//action, inst+4, loaded [template = constants.%Action]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//action, inst+24, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//action, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//action, inst+29, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//action, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//action, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//action, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//action, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.9: @Action.%.2 (%.5) = import_ref Main//action, inst+16, loaded [symbolic = @Action.%.3 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//action, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//action, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//action, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//action, inst+12, unloaded
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.loc8_6.1: %A = param a
// CHECK:STDOUT:     @G.%a: %A = bind_name a, %a.loc8_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Action(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Action, @Action(%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @Action(%T) [symbolic = %Op.type (constants.%Op.type.1)]
// CHECK:STDOUT:   %Op: @Action.%Op.type (%Op.type.1) = struct_value () [symbolic = %Op (constants.%Op.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Action.%.1 (%.3), @Action.%Op.type (%Op.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @Action.%.2 (%.5) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .Op = imports.%import_ref.9
// CHECK:STDOUT:     witness = (imports.%import_ref.10)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%T: type, constants.%Self.1: @Action.%.1 (%.3)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Action.ref: %Action.type = name_ref Action, imports.%import_ref.1 [template = constants.%Action]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %.loc8_23: init type = call %Action.ref(%C.ref) [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_26: %.11 = specific_constant imports.%import_ref.9, @Action(constants.%C) [template = constants.%.12]
// CHECK:STDOUT:   %Op.ref: %.11 = name_ref Op, %.loc8_26 [template = constants.%.12]
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.2
// CHECK:STDOUT:   %Op => constants.%Op.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(@Action.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T, constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Action(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Op.type => constants.%Op.type.3
// CHECK:STDOUT:   %Op => constants.%Op.3
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Factory.type: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Factory: %Factory.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Factory.%Make.decl [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.6: type = interface_type @Factory, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Make.type.3 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, @Factory.%Make.decl [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Make.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Factory = %Factory.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Factory.decl: %Factory.type = interface_decl @Factory [template = constants.%Factory] {
// CHECK:STDOUT:     %T.loc4_19.1: type = param T
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T 0, %T.loc4_19.1 [symbolic = @Factory.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %.loc11_20.1: type = value_of_initializer %.loc11_18 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_20.2: type = converted %.loc11_18, %.loc11_20.1 [template = constants.%.6]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %Factory.ref: %Factory.type = name_ref Factory, %Factory.decl [template = constants.%Factory]
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc11_18: init type = call %Factory.ref(%B.ref) [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(file.%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Factory, @Factory(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Factory.%.1 (%.2), @Factory.%Make.type (%Make.type.1) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @Factory.%.2 (%.3) = assoc_entity element0, %Make.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Factory.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Make.decl: @Factory.%Make.type (%Make.type.1) = fn_decl @Make.1 [symbolic = %Make (constants.%Make.1)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc4_19.2 [symbolic = @Make.1.%T (constants.%T)]
// CHECK:STDOUT:       %return.var: ref @Make.1.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5: @Factory.%.2 (%.3) = assoc_entity element0, %Make.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Make = %.loc5
// CHECK:STDOUT:     witness = (%Make.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.6 {
// CHECK:STDOUT:   %Make.decl: %Make.type.2 = fn_decl @Make.2 [template = constants.%Make.2] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %return.var: ref %B = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Make.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(file.%T.loc4_19.2: type, @Factory.%Self.1: @Factory.%.1 (%.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Factory.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%B, constants.%A) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Factory: %Factory.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @Factory.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @Factory, @Factory(%B) [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.3, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.9 [symbolic]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.1, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %Make.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [template]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.10: %.5 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%Make.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Factory.type = import_ref Main//factory, inst+4, loaded [template = constants.%Factory]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//factory, inst+27, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//factory, inst+30, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//factory, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//factory, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//factory, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.7: @Factory.%.2 (%.5) = import_ref Main//factory, inst+19, loaded [symbolic = @Factory.%.3 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.11: <witness> = import_ref Main//factory, inst+49, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//factory, inst+14, unloaded
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
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.loc4_10.1: %A = param a
// CHECK:STDOUT:     @MakeB.%a: %A = bind_name a, %a.loc4_10.1
// CHECK:STDOUT:     %B.ref: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:     @MakeB.%return: ref %B = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Factory, @Factory(%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make.1, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Factory.%.1 (%.3), @Factory.%Make.type (%Make.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @Factory.%.2 (%.5) = assoc_entity element0, imports.%import_ref.9 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:     .Make = imports.%import_ref.7
// CHECK:STDOUT:     witness = (imports.%import_ref.8)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(constants.%T: type, constants.%Self.1: @Factory.%.1 (%.3)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeB(%a: %A) -> %return: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type = name_ref Factory, imports.%import_ref.1 [template = constants.%Factory]
// CHECK:STDOUT:   %B.ref: type = name_ref B, imports.%import_ref.3 [template = constants.%B]
// CHECK:STDOUT:   %.loc5_20: init type = call %Factory.ref(%B.ref) [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_23: %.7 = specific_constant imports.%import_ref.7, @Factory(constants.%B) [template = constants.%.8]
// CHECK:STDOUT:   %Make.ref: %.7 = name_ref Make, %.loc5_23 [template = constants.%.8]
// CHECK:STDOUT:   %.1: %Make.type.2 = interface_witness_access imports.%import_ref.11, element0 [template = constants.%Make.3]
// CHECK:STDOUT:   %.loc4: ref %B = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %B = call %.1() to %.loc4
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %B;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Factory.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_factory.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Factory.type: type = generic_interface_type @Factory [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Factory: %Factory.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @Factory, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @Factory.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @Factory, @Factory(%B) [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make, @Factory(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.3, %Make.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.9 [symbolic]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make, @Factory(%B) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.4, %Make.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [template]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.10: type = interface_type @Factory, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make, @Factory(%C) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.10, %Make.type.3 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.9 [template]
// CHECK:STDOUT:   %.13: %.5 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Factory.type = import_ref Main//factory, inst+4, loaded [template = constants.%Factory]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//factory, inst+27, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//factory, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//factory, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//factory, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//factory, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.7: @Factory.%.2 (%.5) = import_ref Main//factory, inst+19, loaded [symbolic = @Factory.%.3 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//factory, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//factory, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//factory, inst+14, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [template = constants.%MakeC] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %a.loc6_10.1: %A = param a
// CHECK:STDOUT:     @MakeC.%a: %A = bind_name a, %a.loc6_10.1
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @MakeC.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Factory(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Factory, @Factory(%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Factory(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Factory.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Factory.%.1 (%.3), @Factory.%Make.type (%Make.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @Factory.%.2 (%.5) = assoc_entity element0, imports.%import_ref.9 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:     .Make = imports.%import_ref.7
// CHECK:STDOUT:     witness = (imports.%import_ref.8)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A as %.4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(constants.%T: type, constants.%Self.1: @Factory.%.1 (%.3)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC(%a: %A) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %Factory.ref: %Factory.type = name_ref Factory, imports.%import_ref.1 [template = constants.%Factory]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc10_20: init type = call %Factory.ref(%C.ref) [template = constants.%.10]
// CHECK:STDOUT:   %.loc10_23: %.11 = specific_constant imports.%import_ref.7, @Factory(constants.%C) [template = constants.%.12]
// CHECK:STDOUT:   %Make.ref: %.11 = name_ref Make, %.loc10_23 [template = constants.%.12]
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
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(@Factory.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Factory(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
