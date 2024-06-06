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
// CHECK:STDOUT:   %Simple: <type of Simple> = struct_value () [template]
// CHECK:STDOUT:   %Self.1: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %WithAssocFn.type: type = generic_interface_type @WithAssocFn [template]
// CHECK:STDOUT:   %WithAssocFn: <type of WithAssocFn> = struct_value () [template]
// CHECK:STDOUT:   %Self.2: <cannot stringify inst+17> = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @WithAssocFn, <type of F> [template]
// CHECK:STDOUT:   %.4: <associated <type of F> in WithAssocFn> = assoc_entity element0, @WithAssocFn.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: type = interface_type @Simple, (%C) [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.7: type = interface_type @WithAssocFn, (%C) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: X = struct_value () [template]
// CHECK:STDOUT:   %N: T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [template]
// CHECK:STDOUT:   %WithImplicitArgs: <type of WithImplicitArgs> = struct_value () [template]
// CHECK:STDOUT:   %T.2: Simple = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [template]
// CHECK:STDOUT:   %Receive: <type of Receive> = struct_value () [template]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [template]
// CHECK:STDOUT:   %Pass: <type of Pass> = struct_value () [template]
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
// CHECK:STDOUT:   %Simple.decl: <type of Simple> = interface_decl @Simple [template = constants.%Simple] {
// CHECK:STDOUT:     %T.loc4_18.1: type = param T
// CHECK:STDOUT:     %T.loc4_18.2: type = bind_symbolic_name T 0, %T.loc4_18.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %WithAssocFn.decl: <type of WithAssocFn> = interface_decl @WithAssocFn [template = constants.%WithAssocFn] {
// CHECK:STDOUT:     %T.loc8_23.1: type = param T
// CHECK:STDOUT:     %T.loc8_23.2: type = bind_symbolic_name T 0, %T.loc8_23.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: <type of WithImplicitArgs> = interface_decl @WithImplicitArgs [template = constants.%WithImplicitArgs] {
// CHECK:STDOUT:     %T.loc26_28.1: type = param T
// CHECK:STDOUT:     %T.loc26_28.2: type = bind_symbolic_name T 0, %T.loc26_28.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_28.2 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %N.loc26_38.1: T = param N
// CHECK:STDOUT:     %N.loc26_38.2: T = bind_symbolic_name N 1, %N.loc26_38.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: <type of Receive> = fn_decl @Receive [template = constants.%Receive] {
// CHECK:STDOUT:     %Simple.ref.loc28: <type of Simple> = name_ref Simple, %Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc28: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc28: type = interface_type @Simple, (%C.ref.loc28) [template = constants.%.5]
// CHECK:STDOUT:     %T.loc28_12.1: Simple = param T
// CHECK:STDOUT:     @Receive.%T: Simple = bind_symbolic_name T 0, %T.loc28_12.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: <type of Pass> = fn_decl @Pass [template = constants.%Pass] {
// CHECK:STDOUT:     %Simple.ref.loc29: <type of Simple> = name_ref Simple, %Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc29: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc29: type = interface_type @Simple, (%C.ref.loc29) [template = constants.%.5]
// CHECK:STDOUT:     %T.loc29_9.1: Simple = param T
// CHECK:STDOUT:     @Pass.%T: Simple = bind_symbolic_name T 0, %T.loc29_9.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @WithAssocFn {
// CHECK:STDOUT:   %Self: <cannot stringify inst+17> = bind_symbolic_name Self 1 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.var: ref X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <associated <type of F> in WithAssocFn> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc10
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @WithImplicitArgs;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as Simple {
// CHECK:STDOUT:   %.1: <witness> = interface_witness () [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: C as WithAssocFn {
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.var: ref X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.8]
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
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %Simple.ref: <type of Simple> = name_ref Simple, file.%Simple.decl [template = constants.%Simple]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14: type = interface_type @Simple, (%C.ref.loc14) [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %WithAssocFn.ref: <type of WithAssocFn> = name_ref WithAssocFn, file.%WithAssocFn.decl [template = constants.%WithAssocFn]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc19: type = interface_type @WithAssocFn, (%C.ref.loc19) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> @impl.2.%return.var: X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc21_15.2: init X = class_init (), @impl.2.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_16: init X = converted %.loc21_15.1, %.loc21_15.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc21_16 to @impl.2.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Receive(%T: Simple);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Pass(%T: Simple) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Receive.ref: <type of Receive> = name_ref Receive, file.%Receive.decl [template = constants.%Receive]
// CHECK:STDOUT:   %T.ref: Simple = name_ref T, %T [symbolic = constants.%T.2]
// CHECK:STDOUT:   %Receive.call: init () = call %Receive.ref(%T.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic: <type of Generic> = struct_value () [template]
// CHECK:STDOUT:   %Self: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = interface_type @Generic, (%A) [template]
// CHECK:STDOUT:   %T.2: Generic = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @Generic, (%B) [template]
// CHECK:STDOUT:   %T.3: Generic = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: <type of G> = struct_value () [template]
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
// CHECK:STDOUT:   %Generic.decl: <type of Generic> = interface_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.loc4_19.1: type = param T
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T 0, %T.loc4_19.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Generic.ref.loc9: <type of Generic> = name_ref Generic, %Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc9: type = interface_type @Generic, (%A.ref) [template = constants.%.3]
// CHECK:STDOUT:     %T.loc9_6.1: Generic = param T
// CHECK:STDOUT:     @F.%T: Generic = bind_symbolic_name T 0, %T.loc9_6.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: <type of G> = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Generic.ref.loc10: <type of Generic> = name_ref Generic, %Generic.decl [template = constants.%Generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc10: type = interface_type @Generic, (%B.ref) [template = constants.%.4]
// CHECK:STDOUT:     %T.loc10_6.1: Generic = param T
// CHECK:STDOUT:     @G.%T: Generic = bind_symbolic_name T 0, %T.loc10_6.1 [symbolic = constants.%T.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Generic {
// CHECK:STDOUT:   %Self: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
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
// CHECK:STDOUT: fn @F(%T: Generic);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%T: Generic) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <type of F> = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %T.ref: Generic = name_ref T, %T [symbolic = constants.%T.3]
// CHECK:STDOUT:   %F.call: init () = call %F.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
