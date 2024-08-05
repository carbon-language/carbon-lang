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

library "generic";

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

library "fail_mismatched_args";

interface Generic(T:! type) {}

class A {}
class B {}

fn F(T:! Generic(A));
fn G(T:! Generic(B)) {
  // TODO: Include generic arguments in the type name.
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `Generic` to `Generic`.
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE-6]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn F(T:! Generic(A));
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Simple.type: type = generic_interface_type @Simple [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Simple: %Simple.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Simple, @Simple(%T.1) [symbolic]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %WithAssocFn.type: type = generic_interface_type @WithAssocFn [template]
// CHECK:STDOUT:   %WithAssocFn: %WithAssocFn.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @WithAssocFn, @WithAssocFn(%T.1) [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @WithAssocFn(%T.1) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @WithAssocFn.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.7: type = interface_type @Simple, @Simple(%C) [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.9: type = interface_type @WithAssocFn, @WithAssocFn(%C) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %struct: %X = struct_value () [template]
// CHECK:STDOUT:   %N: %T.1 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [template]
// CHECK:STDOUT:   %WithImplicitArgs: %WithImplicitArgs.type = struct_value () [template]
// CHECK:STDOUT:   %T.2: %.7 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [template]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [template]
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
// CHECK:STDOUT:     %T.loc4_18.1: type = param T
// CHECK:STDOUT:     %T.loc4_18.2: type = bind_symbolic_name T 0, %T.loc4_18.1 [symbolic = @Simple.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type = interface_decl @WithAssocFn [template = constants.%WithAssocFn] {
// CHECK:STDOUT:     %T.loc8_23.1: type = param T
// CHECK:STDOUT:     %T.loc8_23.2: type = bind_symbolic_name T 0, %T.loc8_23.1 [symbolic = @WithAssocFn.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [template = constants.%WithImplicitArgs] {
// CHECK:STDOUT:     %T.loc22_28.1: type = param T
// CHECK:STDOUT:     %T.loc22_28.2: type = bind_symbolic_name T 0, %T.loc22_28.1 [symbolic = @WithImplicitArgs.%T (constants.%T.1)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_28.2 [symbolic = @WithImplicitArgs.%T (constants.%T.1)]
// CHECK:STDOUT:     %N.loc22_38.1: @WithImplicitArgs.%T (%T.1) = param N
// CHECK:STDOUT:     %N.loc22_38.2: @WithImplicitArgs.%T (%T.1) = bind_symbolic_name N 1, %N.loc22_38.1 [symbolic = @WithImplicitArgs.%N (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [template = constants.%Receive] {
// CHECK:STDOUT:     %Simple.ref.loc24: %Simple.type = name_ref Simple, %Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc24_22: init type = call %Simple.ref.loc24(%C.ref.loc24) [template = constants.%.7]
// CHECK:STDOUT:     %.loc24_24.1: type = value_of_initializer %.loc24_22 [template = constants.%.7]
// CHECK:STDOUT:     %.loc24_24.2: type = converted %.loc24_22, %.loc24_24.1 [template = constants.%.7]
// CHECK:STDOUT:     %T.loc24_12.1: %.7 = param T
// CHECK:STDOUT:     @Receive.%T.loc24: %.7 = bind_symbolic_name T 0, %T.loc24_12.1 [symbolic = @Receive.%T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [template = constants.%Pass] {
// CHECK:STDOUT:     %Simple.ref.loc25: %Simple.type = name_ref Simple, %Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc25_19: init type = call %Simple.ref.loc25(%C.ref.loc25) [template = constants.%.7]
// CHECK:STDOUT:     %.loc25_21.1: type = value_of_initializer %.loc25_19 [template = constants.%.7]
// CHECK:STDOUT:     %.loc25_21.2: type = converted %.loc25_19, %.loc25_21.1 [template = constants.%.7]
// CHECK:STDOUT:     %T.loc25_9.1: %.7 = param T
// CHECK:STDOUT:     @Pass.%T.loc25: %.7 = bind_symbolic_name T 0, %T.loc25_9.1 [symbolic = @Pass.%T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(file.%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Simple, @Simple(%T) [symbolic = %.1 (constants.%.2)]
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
// CHECK:STDOUT: generic interface @WithAssocFn(file.%T.loc8_23.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @WithAssocFn, @WithAssocFn(%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @WithAssocFn(%T) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @WithAssocFn.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @WithAssocFn.%.1 (%.4), @WithAssocFn.%F.type (%F.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @WithAssocFn.%.2 (%.5) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:     %F.decl: @WithAssocFn.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = %F (constants.%F.1)] {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:       %return.var: ref %X = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: @WithAssocFn.%.2 (%.5) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc10
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(file.%T.loc22_28.2: type, file.%N.loc22_38.2: @WithImplicitArgs.%T (%T.1)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:   %N: %T.1 = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.7 {
// CHECK:STDOUT:   %.1: <witness> = interface_witness () [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.9 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.var: ref %X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc14_19.1: type = value_of_initializer %.loc14_17 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_19.2: type = converted %.loc14_17, %.loc14_19.1 [template = constants.%.7]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %Simple.ref: %Simple.type = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14_17: init type = call %Simple.ref(%C.ref.loc14) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_24.1: type = value_of_initializer %.loc15_22 [template = constants.%.9]
// CHECK:STDOUT:   %.loc15_24.2: type = converted %.loc15_22, %.loc15_24.1 [template = constants.%.9]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type = name_ref WithAssocFn, file.%WithAssocFn.decl [template = constants.%WithAssocFn]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc15_22: init type = call %WithAssocFn.ref(%C.ref.loc15) [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(file.%T.loc8_23.2: type, @WithAssocFn.%Self.1: @WithAssocFn.%.1 (%.4)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> @impl.2.%return.var: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %.3 = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), @impl.2.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc17_16 to @impl.2.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24: %.7) {
// CHECK:STDOUT:   %T.1: %.7 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc24: %.7);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25: %.7) {
// CHECK:STDOUT:   %T.1: %.7 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc25: %.7) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [template = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %.7 = name_ref T, %T.loc25 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %Receive.call: init %.1 = call %Receive.ref(%T.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(@Simple.%T) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.1, constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(@WithAssocFn.%T) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.1, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Generic, @Generic(%T.1) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = interface_type @Generic, @Generic(%A) [template]
// CHECK:STDOUT:   %T.2: %.4 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = interface_type @Generic, @Generic(%B) [template]
// CHECK:STDOUT:   %T.3: %.5 = bind_symbolic_name T 0 [symbolic]
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
// CHECK:STDOUT:     %T.loc4_19.1: type = param T
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T 0, %T.loc4_19.1 [symbolic = @Generic.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Generic.ref.loc9: %Generic.type = name_ref Generic, %Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc9_17: init type = call %Generic.ref.loc9(%A.ref) [template = constants.%.4]
// CHECK:STDOUT:     %.loc9_19.1: type = value_of_initializer %.loc9_17 [template = constants.%.4]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %.loc9_17, %.loc9_19.1 [template = constants.%.4]
// CHECK:STDOUT:     %T.loc9_6.1: %.4 = param T
// CHECK:STDOUT:     @F.%T.loc9: %.4 = bind_symbolic_name T 0, %T.loc9_6.1 [symbolic = @F.%T.1 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Generic.ref.loc10: %Generic.type = name_ref Generic, %Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc10_17: init type = call %Generic.ref.loc10(%B.ref) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_19.1: type = value_of_initializer %.loc10_17 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_17, %.loc10_19.1 [template = constants.%.5]
// CHECK:STDOUT:     %T.loc10_6.1: %.5 = param T
// CHECK:STDOUT:     @G.%T.loc10: %.5 = bind_symbolic_name T 0, %T.loc10_6.1 [symbolic = @G.%T.1 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(file.%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Generic, @Generic(%T) [symbolic = %.1 (constants.%.2)]
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
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9: %.4) {
// CHECK:STDOUT:   %T.1: %.4 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc9: %.4);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10: %.5) {
// CHECK:STDOUT:   %T.1: %.5 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc10: %.5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: %.5 = name_ref T, %T.loc10 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %F.call: init %.1 = call %F.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@Generic.%T) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.3) {
// CHECK:STDOUT:   %T.1 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.3) {
// CHECK:STDOUT:   %T.1 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
