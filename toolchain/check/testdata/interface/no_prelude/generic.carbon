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

fn Receive(T:! Simple(C));
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
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+3]]:3: error: package `Core` implicitly referenced here, but not found
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %Simple.type: type = generic_interface_type @Simple [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Simple: %Simple.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Simple, @Simple(%T.1) [symbolic]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %WithAssocFn.type: type = generic_interface_type @WithAssocFn [template]
// CHECK:STDOUT:   %WithAssocFn: %WithAssocFn.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = interface_type @WithAssocFn, @WithAssocFn(%T.1) [symbolic]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @WithAssocFn(%T.1) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.5, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @WithAssocFn.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.8: type = interface_type @Simple, @Simple(%C) [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.10: type = interface_type @WithAssocFn, @WithAssocFn(%C) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @WithAssocFn(%C) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.10, %F.type.3 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, @WithAssocFn.%F.decl [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %struct: %X = struct_value () [template]
// CHECK:STDOUT:   %T.patt.3: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %N: %T.1 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T.1 = symbolic_binding_pattern N 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [template]
// CHECK:STDOUT:   %WithImplicitArgs: %WithImplicitArgs.type = struct_value () [template]
// CHECK:STDOUT:   %T.2: %.8 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.4: %.8 = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [template]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [template]
// CHECK:STDOUT:   %T.patt.5: %.8 = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [template]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [template]
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
// CHECK:STDOUT:   %Simple.decl: %Simple.type = interface_decl @Simple [template = constants.%Simple] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type = interface_decl @WithAssocFn [template = constants.%WithAssocFn] {
// CHECK:STDOUT:     %T.patt.loc8: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc8, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8: type = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [template = constants.%WithImplicitArgs] {
// CHECK:STDOUT:     %T.patt.loc22: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc22, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:     %N.patt.loc22: @WithImplicitArgs.%T.1 (%T.1) = symbolic_binding_pattern N 1 [symbolic = %N.patt.1 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @WithImplicitArgs.%T.1 (%T.1) = param_pattern %N.patt.loc22, runtime_param<invalid> [symbolic = %N.patt.1 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %param.loc22_28: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22: type = bind_symbolic_name T 0, %param.loc22_28 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %param.loc22_38: @WithImplicitArgs.%T.1 (%T.1) = param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc22: @WithImplicitArgs.%T.1 (%T.1) = bind_symbolic_name N 1, %param.loc22_38 [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [template = constants.%Receive] {
// CHECK:STDOUT:     %T.patt.loc24: %.8 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.4)]
// CHECK:STDOUT:     %T.param_patt: %.8 = param_pattern %T.patt.loc24, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.4)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Simple.ref: %Simple.type = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc24: type = interface_type @Simple, @Simple(constants.%C) [template = constants.%.8]
// CHECK:STDOUT:     %param: %.8 = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc24: %.8 = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [template = constants.%Pass] {
// CHECK:STDOUT:     %T.patt.loc25: %.8 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.5)]
// CHECK:STDOUT:     %T.param_patt: %.8 = param_pattern %T.patt.loc25, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.5)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Simple.ref: %Simple.type = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc25: type = interface_type @Simple, @Simple(constants.%C) [template = constants.%.8]
// CHECK:STDOUT:     %param: %.8 = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc25: %.8 = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Simple, @Simple(%T.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Simple.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @WithAssocFn, @WithAssocFn(%T.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @WithAssocFn(%T.1) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @WithAssocFn.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @WithAssocFn.%.1 (%.5), @WithAssocFn.%F.type (%F.type.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:   %.3: @WithAssocFn.%.2 (%.6) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:     %F.decl: @WithAssocFn.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @WithAssocFn.%F (constants.%F.1)] {} {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:       %return: ref %X = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: @WithAssocFn.%.2 (%.6) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc10
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22: type, %N.loc22: @WithImplicitArgs.%T.1 (%T.1)) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:   %N.1: %T.1 = bind_symbolic_name N 1 [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   %N.patt.1: %T.1 = symbolic_binding_pattern N 1 [symbolic = %N.patt.1 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.8 {
// CHECK:STDOUT:   %.loc14_21: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc14_21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.10 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return: ref %X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_26: <witness> = interface_witness (%F.decl) [template = constants.%.13]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc15_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 {} {
// CHECK:STDOUT:     %Simple.ref: %Simple.type = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14_17: type = interface_type @Simple, @Simple(constants.%C) [template = constants.%.8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {} {
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type = name_ref WithAssocFn, file.%WithAssocFn.decl [template = constants.%WithAssocFn]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc15_22: type = interface_type @WithAssocFn, @WithAssocFn(constants.%C) [template = constants.%.10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@WithAssocFn.%T.loc8: type, @WithAssocFn.%Self.1: @WithAssocFn.%.1 (%.5)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %.3 = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc17_16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24: %.8) {
// CHECK:STDOUT:   %T.1: %.8 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.1: %.8 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %.8);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25: %.8) {
// CHECK:STDOUT:   %T.1: %.8 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.1: %.8 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %.8) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [template = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %.8 = name_ref T, %T.loc25 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %Receive.call: init %.1 = call %Receive.ref()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(@Simple.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.1, constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(@WithAssocFn.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.1 => constants.%C
// CHECK:STDOUT:   %T.patt.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.1 => constants.%C
// CHECK:STDOUT:   %T.patt.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%C) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.1, constants.%N) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT:   %N.1 => constants.%N
// CHECK:STDOUT:   %N.patt.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT:   %T.patt.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT:   %T.patt.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Generic, @Generic(%T.1) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = interface_type @Generic, @Generic(%A) [template]
// CHECK:STDOUT:   %T.2: %.5 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %.5 = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = interface_type @Generic, @Generic(%B) [template]
// CHECK:STDOUT:   %T.3: %.6 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3: %.6 = symbolic_binding_pattern T 0 [symbolic]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type = interface_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9: %.5 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %.5 = param_pattern %T.patt.loc9, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Generic.ref: %Generic.type = name_ref Generic, file.%Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc9: type = interface_type @Generic, @Generic(constants.%A) [template = constants.%.5]
// CHECK:STDOUT:     %param: %.5 = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc9: %.5 = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc10: %.6 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: %.6 = param_pattern %T.patt.loc10, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Generic.ref: %Generic.type = name_ref Generic, file.%Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc10: type = interface_type @Generic, @Generic(constants.%B) [template = constants.%.6]
// CHECK:STDOUT:     %param: %.6 = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10: %.6 = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Generic, @Generic(%T.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9: %.5) {
// CHECK:STDOUT:   %T.1: %.5 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.1: %.5 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %.5);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10: %.6) {
// CHECK:STDOUT:   %T.1: %.6 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.1: %.6 = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %.6) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: %.6 = name_ref T, %T.loc10 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %.loc15: %.5 = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@Generic.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %T.patt.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.1 => constants.%A
// CHECK:STDOUT:   %T.patt.1 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT:   %T.patt.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.1 => constants.%B
// CHECK:STDOUT:   %T.patt.1 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.3) {
// CHECK:STDOUT:   %T.1 => constants.%T.3
// CHECK:STDOUT:   %T.patt.1 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
