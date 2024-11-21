// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic.carbon

// --- generic.carbon

library "[[@TEST_NAME]]";

interface Simple(T:! type) {}

class X {}

interface WithAssocFn(T:! type) {
  // TODO: Take `Self`, return `T`, once that works.
  fn F() -> X;
}

class C {
  impl as Simple(C) {}
  impl as WithAssocFn(C) {
    fn F() -> X {
      return {};
    }
  }
}

interface WithImplicitArgs[T:! type](N:! T);

fn Receive(T:! Simple(C)) {}
fn Pass(T:! Simple(C)) {
  Receive(T);
}

// --- fail_mismatched_args.carbon

library "[[@TEST_NAME]]";

interface Generic(T:! type) {}

class A {}
class B {}

fn F(T:! Generic(A));
fn G(T:! Generic(B)) {
  // TODO: Include generic arguments in the type name.
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+6]]:3: error: package `Core` implicitly referenced here, but not found [CoreNotFound]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(T:! Generic(A));
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Simple.type.1: type = generic_interface_type @Simple [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Simple: %Simple.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Simple.type.2: type = facet_type <@Simple, @Simple(%T.1)> [symbolic]
// CHECK:STDOUT:   %Self.1: %Simple.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %WithAssocFn.type.1: type = generic_interface_type @WithAssocFn [template]
// CHECK:STDOUT:   %WithAssocFn: %WithAssocFn.type.1 = struct_value () [template]
// CHECK:STDOUT:   %WithAssocFn.type.2: type = facet_type <@WithAssocFn, @WithAssocFn(%T.1)> [symbolic]
// CHECK:STDOUT:   %Self.2: %WithAssocFn.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @WithAssocFn(%T.1) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %WithAssocFn.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @WithAssocFn.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Simple.type.3: type = facet_type <@Simple, @Simple(%C)> [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %WithAssocFn.type.3: type = facet_type <@WithAssocFn, @WithAssocFn(%C)> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @WithAssocFn(%C) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %WithAssocFn.type.3, %F.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @WithAssocFn.%F.decl [template]
// CHECK:STDOUT:   %.8: %WithAssocFn.type.2 = facet_value %C, %C [symbolic]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %struct: %X = struct_value () [template]
// CHECK:STDOUT:   %N: %T.1 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T.1 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [template]
// CHECK:STDOUT:   %WithImplicitArgs: %WithImplicitArgs.type = struct_value () [template]
// CHECK:STDOUT:   %T.2: %Simple.type.3 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %Simple.type.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [template]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [template]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [template]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [template]
// CHECK:STDOUT:   %.11: <specific function> = specific_function %Receive, @Receive(%T.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .WithAssocFn = %WithAssocFn.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .WithImplicitArgs = %WithImplicitArgs.decl
// CHECK:STDOUT:     .Receive = %Receive.decl
// CHECK:STDOUT:     .Pass = %Pass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.decl: %Simple.type.1 = interface_decl @Simple [template = constants.%Simple] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_18.1, runtime_param<invalid> [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_18.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type.1 = interface_decl @WithAssocFn [template = constants.%WithAssocFn] {
// CHECK:STDOUT:     %T.patt.loc8_23.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_23.1, runtime_param<invalid> [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_23.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_23.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [template = constants.%WithImplicitArgs] {
// CHECK:STDOUT:     %T.patt.loc22_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc22_28.1, runtime_param<invalid> [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %N.patt.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.1) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @WithImplicitArgs.%T.loc22_28.2 (%T.1) = value_param_pattern %N.patt.loc22_38.1, runtime_param<invalid> [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_28.1 [symbolic = %T.loc22_28.2 (constants.%T.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22_28.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_28.2 (constants.%T.1)]
// CHECK:STDOUT:     %N.param: @WithImplicitArgs.%T.loc22_28.2 (%T.1) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.1) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [template = constants.%Receive] {
// CHECK:STDOUT:     %T.patt.loc24_12.1: %Simple.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.3 = value_param_pattern %T.patt.loc24_12.1, runtime_param<invalid> [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Simple.ref: %Simple.type.1 = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.3]
// CHECK:STDOUT:     %T.param: %Simple.type.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc24_12.1: %Simple.type.3 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc24_12.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [template = constants.%Pass] {
// CHECK:STDOUT:     %T.patt.loc25_9.1: %Simple.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.3 = value_param_pattern %T.patt.loc25_9.1, runtime_param<invalid> [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Simple.ref: %Simple.type.1 = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.3]
// CHECK:STDOUT:     %T.param: %Simple.type.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc25_9.1: %Simple.type.3 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_9.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple, @Simple(%T.loc4_18.2)> [symbolic = %Simple.type (constants.%Simple.type.2)]
// CHECK:STDOUT:   %Self.2: %Simple.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Simple.%Simple.type (%Simple.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8_23.1: type) {
// CHECK:STDOUT:   %T.loc8_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_23.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc8_23.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(%T.loc8_23.2)> [symbolic = %WithAssocFn.type (constants.%WithAssocFn.type.2)]
// CHECK:STDOUT:   %Self.2: %WithAssocFn.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @WithAssocFn(%T.loc8_23.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @WithAssocFn.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc10_14.2: type = assoc_entity_type @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.2), @WithAssocFn.%F.type (%F.type.1) [symbolic = %.loc10_14.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc10_14.3: @WithAssocFn.%.loc10_14.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.loc10_14.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:     %F.decl: @WithAssocFn.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @WithAssocFn.%F (constants.%F.1)] {
// CHECK:STDOUT:       %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:       %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %X = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_14.1: @WithAssocFn.%.loc10_14.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.loc10_14.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc10_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22_28.1: type, %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.1)) {
// CHECK:STDOUT:   %T.loc22_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_28.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc22_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   %N.loc22_38.2: %T.1 = bind_symbolic_name N, 1 [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc22_38.2: %T.1 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %Simple.type {
// CHECK:STDOUT:   %.loc14: <witness> = interface_witness () [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %WithAssocFn.type {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = interface_witness (%F.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %Simple.ref: %Simple.type.1 = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type.1 = name_ref WithAssocFn, file.%WithAssocFn.decl [template = constants.%WithAssocFn]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(constants.%C)> [template = constants.%WithAssocFn.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@WithAssocFn.%T.loc8_23.1: type, @WithAssocFn.%Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc17_16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24_12.1: %Simple.type.3) {
// CHECK:STDOUT:   %T.loc24_12.2: %Simple.type.3 = bind_symbolic_name T, 0 [symbolic = %T.loc24_12.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc24_12.2: %Simple.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25_9.1: %Simple.type.3) {
// CHECK:STDOUT:   %T.loc25_9.2: %Simple.type.3 = bind_symbolic_name T, 0 [symbolic = %T.loc25_9.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc25_9.2: %Simple.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc26_3.2: <specific function> = specific_function constants.%Receive, @Receive(%T.loc25_9.2) [symbolic = %.loc26_3.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [template = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %Simple.type.3 = name_ref T, %T.loc25_9.1 [symbolic = %T.loc25_9.2 (constants.%T.2)]
// CHECK:STDOUT:     %.loc26_3.1: <specific function> = specific_function %Receive.ref, @Receive(constants.%T.2) [symbolic = %.loc26_3.2 (constants.%.11)]
// CHECK:STDOUT:     %Receive.call: init %empty_tuple.type = call %.loc26_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.1) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(%T.loc4_18.2) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.1) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.1, constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(%T.loc8_23.2) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type => constants.%Simple.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type => constants.%WithAssocFn.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc10_14.2 => constants.%.6
// CHECK:STDOUT:   %.loc10_14.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%.8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.1, constants.%N) {
// CHECK:STDOUT:   %T.loc22_28.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc22_28.2 => constants.%T.1
// CHECK:STDOUT:   %N.loc22_38.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc22_38.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.2) {
// CHECK:STDOUT:   %T.loc24_12.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc24_12.2 => constants.%T.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.2) {
// CHECK:STDOUT:   %T.loc25_9.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc25_9.2 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(@Pass.%T.loc25_9.2) {
// CHECK:STDOUT:   %T.loc24_12.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc24_12.2 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.1: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %Generic: %Generic.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Generic.type.2: type = facet_type <@Generic, @Generic(%T.1)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Generic.type.3: type = facet_type <@Generic, @Generic(%A)> [template]
// CHECK:STDOUT:   %T.2: %Generic.type.3 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %Generic.type.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.type.4: type = facet_type <@Generic, @Generic(%B)> [template]
// CHECK:STDOUT:   %T.3: %Generic.type.4 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3: %Generic.type.4 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type.1 = interface_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<invalid> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %Generic.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.3 = value_param_pattern %T.patt.loc9_6.1, runtime_param<invalid> [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Generic.ref: %Generic.type.1 = name_ref Generic, file.%Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%A)> [template = constants.%Generic.type.3]
// CHECK:STDOUT:     %T.param: %Generic.type.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc9_6.1: %Generic.type.3 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc9_6.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc10_6.1: %Generic.type.4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.4 = value_param_pattern %T.patt.loc10_6.1, runtime_param<invalid> [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Generic.ref: %Generic.type.1 = name_ref Generic, file.%Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%B)> [template = constants.%Generic.type.4]
// CHECK:STDOUT:     %T.param: %Generic.type.4 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_6.1: %Generic.type.4 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_6.2 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc4_19.2)> [symbolic = %Generic.type (constants.%Generic.type.2)]
// CHECK:STDOUT:   %Self.2: %Generic.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Generic.type.3) {
// CHECK:STDOUT:   %T.loc9_6.2: %Generic.type.3 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %Generic.type.3 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_6.1: %Generic.type.4) {
// CHECK:STDOUT:   %T.loc10_6.2: %Generic.type.4 = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.2 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.loc10_6.2: %Generic.type.4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: %Generic.type.4 = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T.3)]
// CHECK:STDOUT:     %.loc18: %Generic.type.3 = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.1) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_19.2) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.2) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.3) {
// CHECK:STDOUT:   %T.loc10_6.2 => constants.%T.3
// CHECK:STDOUT:   %T.patt.loc10_6.2 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
