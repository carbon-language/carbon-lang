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
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+7]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(T:! Generic(A));
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Simple.type.3b3: type = generic_interface_type @Simple [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Simple.generic: %Simple.type.3b3 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.type.d08: type = facet_type <@Simple, @Simple(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.faf: %Simple.type.d08 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.509: type = generic_interface_type @WithAssocFn [concrete]
// CHECK:STDOUT:   %WithAssocFn.generic: %WithAssocFn.type.509 = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.ce6: type = facet_type <@WithAssocFn, @WithAssocFn(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.088: %WithAssocFn.type.ce6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1af: type = fn_type @F.1, @WithAssocFn(%T.8b3) [symbolic]
// CHECK:STDOUT:   %F.b7d: %F.type.1af = struct_value () [symbolic]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.02b: type = assoc_entity_type %WithAssocFn.type.ce6 [symbolic]
// CHECK:STDOUT:   %assoc0.470: %WithAssocFn.assoc_type.02b = assoc_entity element0, @WithAssocFn.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Simple.type.51f: type = facet_type <@Simple, @Simple(%C)> [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.683: type = facet_type <@WithAssocFn, @WithAssocFn(%C)> [concrete]
// CHECK:STDOUT:   %F.type.18c: type = fn_type @F.1, @WithAssocFn(%C) [concrete]
// CHECK:STDOUT:   %F.e46: %F.type.18c = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.480: type = assoc_entity_type %WithAssocFn.type.683 [concrete]
// CHECK:STDOUT:   %assoc0.136: %WithAssocFn.assoc_type.480 = assoc_entity element0, @WithAssocFn.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness.49b: <witness> = impl_witness (@impl.2.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.005: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.317: %F.type.005 = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.facet: %WithAssocFn.type.683 = facet_value %C, %impl_witness.49b [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %N: %T.8b3 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T.8b3 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [concrete]
// CHECK:STDOUT:   %WithImplicitArgs.generic: %WithImplicitArgs.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.692: %Simple.type.51f = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6ae: %Simple.type.51f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [concrete]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [concrete]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [concrete]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [concrete]
// CHECK:STDOUT:   %Receive.specific_fn: <specific function> = specific_function %Receive, @Receive(%T.692) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .WithAssocFn = %WithAssocFn.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .WithImplicitArgs = %WithImplicitArgs.decl
// CHECK:STDOUT:     .Receive = %Receive.decl
// CHECK:STDOUT:     .Pass = %Pass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.decl: %Simple.type.3b3 = interface_decl @Simple [concrete = constants.%Simple.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_18.1, runtime_param<none> [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_18.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type.509 = interface_decl @WithAssocFn [concrete = constants.%WithAssocFn.generic] {
// CHECK:STDOUT:     %T.patt.loc8_23.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_23.1, runtime_param<none> [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc8_23.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_23.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [concrete = constants.%WithImplicitArgs.generic] {
// CHECK:STDOUT:     %T.patt.loc22_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc22_28.1, runtime_param<none> [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %N.patt.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.8b3) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @WithImplicitArgs.%T.loc22_28.2 (%T.8b3) = value_param_pattern %N.patt.loc22_38.1, runtime_param<none> [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc22_28.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_28.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %N.param: @WithImplicitArgs.%T.loc22_28.2 (%T.8b3) = value_param runtime_param<none>
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_28.1 [symbolic = %T.loc22_28.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.8b3) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [concrete = constants.%Receive] {
// CHECK:STDOUT:     %T.patt.loc24_12.1: %Simple.type.51f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.51f = value_param_pattern %T.patt.loc24_12.1, runtime_param<none> [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Simple.type.51f = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc24: type = splice_block %Simple.type [concrete = constants.%Simple.type.51f] {
// CHECK:STDOUT:       %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.51f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_12.1: %Simple.type.51f = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc24_12.2 (constants.%T.692)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [concrete = constants.%Pass] {
// CHECK:STDOUT:     %T.patt.loc25_9.1: %Simple.type.51f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.51f = value_param_pattern %T.patt.loc25_9.1, runtime_param<none> [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Simple.type.51f = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc25: type = splice_block %Simple.type [concrete = constants.%Simple.type.51f] {
// CHECK:STDOUT:       %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.51f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_9.1: %Simple.type.51f = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_9.2 (constants.%T.692)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple, @Simple(%T.loc4_18.2)> [symbolic = %Simple.type (constants.%Simple.type.d08)]
// CHECK:STDOUT:   %Self.2: %Simple.type.d08 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.faf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Simple.%Simple.type (%Simple.type.d08) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.faf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8_23.1: type) {
// CHECK:STDOUT:   %T.loc8_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_23.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc8_23.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(%T.loc8_23.2)> [symbolic = %WithAssocFn.type (constants.%WithAssocFn.type.ce6)]
// CHECK:STDOUT:   %Self.2: %WithAssocFn.type.ce6 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.088)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @WithAssocFn(%T.loc8_23.2) [symbolic = %F.type (constants.%F.type.1af)]
// CHECK:STDOUT:   %F: @WithAssocFn.%F.type (%F.type.1af) = struct_value () [symbolic = %F (constants.%F.b7d)]
// CHECK:STDOUT:   %WithAssocFn.assoc_type: type = assoc_entity_type @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.ce6) [symbolic = %WithAssocFn.assoc_type (constants.%WithAssocFn.assoc_type.02b)]
// CHECK:STDOUT:   %assoc0.loc10_14.2: @WithAssocFn.%WithAssocFn.assoc_type (%WithAssocFn.assoc_type.02b) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.470)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.ce6) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.088)]
// CHECK:STDOUT:     %F.decl: @WithAssocFn.%F.type (%F.type.1af) = fn_decl @F.1 [symbolic = @WithAssocFn.%F (constants.%F.b7d)] {
// CHECK:STDOUT:       %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %X = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_14.1: @WithAssocFn.%WithAssocFn.assoc_type (%WithAssocFn.assoc_type.02b) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.470)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc10_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22_28.1: type, %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.8b3)) {
// CHECK:STDOUT:   %T.loc22_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_28.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc22_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %N.loc22_38.2: %T.8b3 = bind_symbolic_name N, 1 [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc22_38.2: %T.8b3 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %Simple.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %WithAssocFn.type {
// CHECK:STDOUT:   %F.decl: %F.type.005 = fn_decl @F.2 [concrete = constants.%F.317] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.51f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.2 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type.509 = name_ref WithAssocFn, file.%WithAssocFn.decl [concrete = constants.%WithAssocFn.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(constants.%C)> [concrete = constants.%WithAssocFn.type.683]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.2.%F.decl) [concrete = constants.%impl_witness.49b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Simple = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .WithAssocFn = <poisoned>
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@WithAssocFn.%T.loc8_23.1: type, @WithAssocFn.%Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.ce6)) {
// CHECK:STDOUT:   fn() -> %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), %return [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   return %.loc17_16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24_12.1: %Simple.type.51f) {
// CHECK:STDOUT:   %T.loc24_12.2: %Simple.type.51f = bind_symbolic_name T, 0 [symbolic = %T.loc24_12.2 (constants.%T.692)]
// CHECK:STDOUT:   %T.patt.loc24_12.2: %Simple.type.51f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.51f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25_9.1: %Simple.type.51f) {
// CHECK:STDOUT:   %T.loc25_9.2: %Simple.type.51f = bind_symbolic_name T, 0 [symbolic = %T.loc25_9.2 (constants.%T.692)]
// CHECK:STDOUT:   %T.patt.loc25_9.2: %Simple.type.51f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.6ae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Receive.specific_fn.loc26_3.2: <specific function> = specific_function constants.%Receive, @Receive(%T.loc25_9.2) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.51f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [concrete = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %Simple.type.51f = name_ref T, %T.loc25_9.1 [symbolic = %T.loc25_9.2 (constants.%T.692)]
// CHECK:STDOUT:     %Receive.specific_fn.loc26_3.1: <specific function> = specific_function %Receive.ref, @Receive(constants.%T.692) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:     %Receive.call: init %empty_tuple.type = call %Receive.specific_fn.loc26_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(%T.loc4_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.8b3, constants.%Self.088) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(%T.loc8_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type => constants.%Simple.type.51f
// CHECK:STDOUT:   %Self.2 => constants.%Self.faf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type => constants.%WithAssocFn.type.683
// CHECK:STDOUT:   %Self.2 => constants.%Self.088
// CHECK:STDOUT:   %F.type => constants.%F.type.18c
// CHECK:STDOUT:   %F => constants.%F.e46
// CHECK:STDOUT:   %WithAssocFn.assoc_type => constants.%WithAssocFn.assoc_type.480
// CHECK:STDOUT:   %assoc0.loc10_14.2 => constants.%assoc0.136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%WithAssocFn.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.8b3, constants.%N) {
// CHECK:STDOUT:   %T.loc22_28.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc22_28.2 => constants.%T.8b3
// CHECK:STDOUT:   %N.loc22_38.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc22_38.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.692) {
// CHECK:STDOUT:   %T.loc24_12.2 => constants.%T.692
// CHECK:STDOUT:   %T.patt.loc24_12.2 => constants.%T.692
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.692) {
// CHECK:STDOUT:   %T.loc25_9.2 => constants.%T.692
// CHECK:STDOUT:   %T.patt.loc25_9.2 => constants.%T.692
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(@Pass.%T.loc25_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91c: type = facet_type <@Generic, @Generic(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.91c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Generic.type.c7c: type = facet_type <@Generic, @Generic(%A)> [concrete]
// CHECK:STDOUT:   %T.a53: %Generic.type.c7c = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3e0: %Generic.type.c7c = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.4ce: type = facet_type <@Generic, @Generic(%B)> [concrete]
// CHECK:STDOUT:   %T.bae: %Generic.type.4ce = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.53a: %Generic.type.4ce = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<none> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %Generic.type.c7c = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.3e0)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.c7c = value_param_pattern %T.patt.loc9_6.1, runtime_param<none> [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.3e0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Generic.type.c7c = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc9: type = splice_block %Generic.type [concrete = constants.%Generic.type.c7c] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%A)> [concrete = constants.%Generic.type.c7c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.1: %Generic.type.c7c = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc9_6.2 (constants.%T.a53)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc10_6.1: %Generic.type.4ce = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.53a)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.4ce = value_param_pattern %T.patt.loc10_6.1, runtime_param<none> [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.53a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Generic.type.4ce = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc10: type = splice_block %Generic.type [concrete = constants.%Generic.type.4ce] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%B)> [concrete = constants.%Generic.type.4ce]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_6.1: %Generic.type.4ce = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_6.2 (constants.%T.bae)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc4_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91c)]
// CHECK:STDOUT:   %Self.2: %Generic.type.91c = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91c) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Generic.type.c7c) {
// CHECK:STDOUT:   %T.loc9_6.2: %Generic.type.c7c = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.a53)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %Generic.type.c7c = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.3e0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.c7c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_6.1: %Generic.type.4ce) {
// CHECK:STDOUT:   %T.loc10_6.2: %Generic.type.4ce = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.2 (constants.%T.bae)]
// CHECK:STDOUT:   %T.patt.loc10_6.2: %Generic.type.4ce = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.53a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.4ce) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Generic.type.4ce = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T.bae)]
// CHECK:STDOUT:     %.loc19: %Generic.type.c7c = converted %T.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.c7c
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.a53) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.a53
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.a53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.bae) {
// CHECK:STDOUT:   %T.loc10_6.2 => constants.%T.bae
// CHECK:STDOUT:   %T.patt.loc10_6.2 => constants.%T.bae
// CHECK:STDOUT: }
// CHECK:STDOUT:
