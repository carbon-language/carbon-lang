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
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+6]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(T:! Generic(A));
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.6: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.d: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Simple.type.c: type = generic_interface_type @Simple [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Simple.generic: %Simple.type.c = struct_value () [template]
// CHECK:STDOUT:   %Simple.type.94: type = facet_type <@Simple, @Simple(%T.6)> [symbolic]
// CHECK:STDOUT:   %Self.e: %Simple.type.94 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %WithAssocFn.type.1: type = generic_interface_type @WithAssocFn [template]
// CHECK:STDOUT:   %WithAssocFn.generic: %WithAssocFn.type.1 = struct_value () [template]
// CHECK:STDOUT:   %WithAssocFn.type.7: type = facet_type <@WithAssocFn, @WithAssocFn(%T.6)> [symbolic]
// CHECK:STDOUT:   %Self.6: %WithAssocFn.type.7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.8: type = fn_type @F.1, @WithAssocFn(%T.6) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3: type = assoc_entity_type %WithAssocFn.type.7, %F.type.8 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.3 = assoc_entity element0, @WithAssocFn.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Simple.type.96: type = facet_type <@Simple, @Simple(%C)> [template]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %WithAssocFn.type.e: type = facet_type <@WithAssocFn, @WithAssocFn(%C)> [template]
// CHECK:STDOUT:   %F.type.0: type = fn_type @F.1, @WithAssocFn(%C) [template]
// CHECK:STDOUT:   %F.1: %F.type.0 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.4: type = assoc_entity_type %WithAssocFn.type.e, %F.type.0 [template]
// CHECK:STDOUT:   %assoc0.9: %F.assoc_type.4 = assoc_entity element0, @WithAssocFn.%F.decl [template]
// CHECK:STDOUT:   %impl_witness.8: <witness> = impl_witness (@impl.2.%F.decl) [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %WithAssocFn.facet: %WithAssocFn.type.7 = facet_value %C, %impl_witness.8 [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [template]
// CHECK:STDOUT:   %N: %T.6 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T.6 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [template]
// CHECK:STDOUT:   %WithImplicitArgs.generic: %WithImplicitArgs.type = struct_value () [template]
// CHECK:STDOUT:   %T.5: %Simple.type.96 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.f: %Simple.type.96 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [template]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [template]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [template]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [template]
// CHECK:STDOUT:   %Receive.specific_fn: <specific function> = specific_function %Receive, @Receive(%T.5) [symbolic]
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
// CHECK:STDOUT:   %Simple.decl: %Simple.type.c = interface_decl @Simple [template = constants.%Simple.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.d)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_18.1, runtime_param<invalid> [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_18.2 (constants.%T.6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type.1 = interface_decl @WithAssocFn [template = constants.%WithAssocFn.generic] {
// CHECK:STDOUT:     %T.patt.loc8_23.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.d)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_23.1, runtime_param<invalid> [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_23.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_23.2 (constants.%T.6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [template = constants.%WithImplicitArgs.generic] {
// CHECK:STDOUT:     %T.patt.loc22_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.d)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc22_28.1, runtime_param<invalid> [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.d)]
// CHECK:STDOUT:     %N.patt.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.6) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @WithImplicitArgs.%T.loc22_28.2 (%T.6) = value_param_pattern %N.patt.loc22_38.1, runtime_param<invalid> [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22_28.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_28.2 (constants.%T.6)]
// CHECK:STDOUT:     %N.param: @WithImplicitArgs.%T.loc22_28.2 (%T.6) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_28.1 [symbolic = %T.loc22_28.2 (constants.%T.6)]
// CHECK:STDOUT:     %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.6) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [template = constants.%Receive] {
// CHECK:STDOUT:     %T.patt.loc24_12.1: %Simple.type.96 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.f)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.96 = value_param_pattern %T.patt.loc24_12.1, runtime_param<invalid> [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Simple.type.96 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc24: type = splice_block %Simple.type [template = constants.%Simple.type.96] {
// CHECK:STDOUT:       %Simple.ref: %Simple.type.c = name_ref Simple, file.%Simple.decl [template = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.96]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_12.1: %Simple.type.96 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc24_12.2 (constants.%T.5)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [template = constants.%Pass] {
// CHECK:STDOUT:     %T.patt.loc25_9.1: %Simple.type.96 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.f)]
// CHECK:STDOUT:     %T.param_patt: %Simple.type.96 = value_param_pattern %T.patt.loc25_9.1, runtime_param<invalid> [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Simple.type.96 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc25: type = splice_block %Simple.type [template = constants.%Simple.type.96] {
// CHECK:STDOUT:       %Simple.ref: %Simple.type.c = name_ref Simple, file.%Simple.decl [template = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.96]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_9.1: %Simple.type.96 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_9.2 (constants.%T.5)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.2 (constants.%T.6)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple, @Simple(%T.loc4_18.2)> [symbolic = %Simple.type (constants.%Simple.type.94)]
// CHECK:STDOUT:   %Self.2: %Simple.type.94 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Simple.%Simple.type (%Simple.type.94) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8_23.1: type) {
// CHECK:STDOUT:   %T.loc8_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_23.2 (constants.%T.6)]
// CHECK:STDOUT:   %T.patt.loc8_23.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_23.2 (constants.%T.patt.d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(%T.loc8_23.2)> [symbolic = %WithAssocFn.type (constants.%WithAssocFn.type.7)]
// CHECK:STDOUT:   %Self.2: %WithAssocFn.type.7 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.6)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @WithAssocFn(%T.loc8_23.2) [symbolic = %F.type (constants.%F.type.8)]
// CHECK:STDOUT:   %F: @WithAssocFn.%F.type (%F.type.8) = struct_value () [symbolic = %F (constants.%F.3)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.7), @WithAssocFn.%F.type (%F.type.8) [symbolic = %F.assoc_type (constants.%F.assoc_type.3)]
// CHECK:STDOUT:   %assoc0.loc10_14.2: @WithAssocFn.%F.assoc_type (%F.assoc_type.3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.6)]
// CHECK:STDOUT:     %F.decl: @WithAssocFn.%F.type (%F.type.8) = fn_decl @F.1 [symbolic = @WithAssocFn.%F (constants.%F.3)] {
// CHECK:STDOUT:       %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:       %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %X = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_14.1: @WithAssocFn.%F.assoc_type (%F.assoc_type.3) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc10_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22_28.1: type, %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.2 (%T.6)) {
// CHECK:STDOUT:   %T.loc22_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_28.2 (constants.%T.6)]
// CHECK:STDOUT:   %T.patt.loc22_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_28.2 (constants.%T.patt.d)]
// CHECK:STDOUT:   %N.loc22_38.2: %T.6 = bind_symbolic_name N, 1 [symbolic = %N.loc22_38.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc22_38.2: %T.6 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc22_38.2 (constants.%N.patt)]
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
// CHECK:STDOUT:   %F.decl: %F.type.7 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %Simple.ref: %Simple.type.c = name_ref Simple, file.%Simple.decl [template = constants.%Simple.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [template = constants.%Simple.type.96]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type.1 = name_ref WithAssocFn, file.%WithAssocFn.decl [template = constants.%WithAssocFn.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(constants.%C)> [template = constants.%WithAssocFn.type.e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.2.%F.decl) [template = constants.%impl_witness.8]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@WithAssocFn.%T.loc8_23.1: type, @WithAssocFn.%Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.7)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), %return [template = constants.%X.val]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [template = constants.%X.val]
// CHECK:STDOUT:   return %.loc17_16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24_12.1: %Simple.type.96) {
// CHECK:STDOUT:   %T.loc24_12.2: %Simple.type.96 = bind_symbolic_name T, 0 [symbolic = %T.loc24_12.2 (constants.%T.5)]
// CHECK:STDOUT:   %T.patt.loc24_12.2: %Simple.type.96 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_12.2 (constants.%T.patt.f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.96) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25_9.1: %Simple.type.96) {
// CHECK:STDOUT:   %T.loc25_9.2: %Simple.type.96 = bind_symbolic_name T, 0 [symbolic = %T.loc25_9.2 (constants.%T.5)]
// CHECK:STDOUT:   %T.patt.loc25_9.2: %Simple.type.96 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_9.2 (constants.%T.patt.f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Receive.specific_fn.loc26_3.2: <specific function> = specific_function constants.%Receive, @Receive(%T.loc25_9.2) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Simple.type.96) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [template = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %Simple.type.96 = name_ref T, %T.loc25_9.1 [symbolic = %T.loc25_9.2 (constants.%T.5)]
// CHECK:STDOUT:     %Receive.specific_fn.loc26_3.1: <specific function> = specific_function %Receive.ref, @Receive(constants.%T.5) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:     %Receive.call: init %empty_tuple.type = call %Receive.specific_fn.loc26_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.6) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T.6
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(%T.loc4_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.6) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%T.6
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%T.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.6, constants.%Self.6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(%T.loc8_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type => constants.%Simple.type.96
// CHECK:STDOUT:   %Self.2 => constants.%Self.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.loc8_23.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc8_23.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type => constants.%WithAssocFn.type.e
// CHECK:STDOUT:   %Self.2 => constants.%Self.6
// CHECK:STDOUT:   %F.type => constants.%F.type.0
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.4
// CHECK:STDOUT:   %assoc0.loc10_14.2 => constants.%assoc0.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%WithAssocFn.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.6, constants.%N) {
// CHECK:STDOUT:   %T.loc22_28.2 => constants.%T.6
// CHECK:STDOUT:   %T.patt.loc22_28.2 => constants.%T.6
// CHECK:STDOUT:   %N.loc22_38.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc22_38.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.5) {
// CHECK:STDOUT:   %T.loc24_12.2 => constants.%T.5
// CHECK:STDOUT:   %T.patt.loc24_12.2 => constants.%T.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.5) {
// CHECK:STDOUT:   %T.loc25_9.2 => constants.%T.5
// CHECK:STDOUT:   %T.patt.loc25_9.2 => constants.%T.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(@Pass.%T.loc25_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.6: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.d: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.0: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.0 = struct_value () [template]
// CHECK:STDOUT:   %Generic.type.38: type = facet_type <@Generic, @Generic(%T.6)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.38 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Generic.type.3a: type = facet_type <@Generic, @Generic(%A)> [template]
// CHECK:STDOUT:   %T.b: %Generic.type.3a = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.9: %Generic.type.3a = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.type.d: type = facet_type <@Generic, @Generic(%B)> [template]
// CHECK:STDOUT:   %T.d: %Generic.type.d = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.a: %Generic.type.d = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.0 = interface_decl @Generic [template = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.d)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<invalid> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T.6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %Generic.type.3a = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.9)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.3a = value_param_pattern %T.patt.loc9_6.1, runtime_param<invalid> [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.9)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Generic.type.3a = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc9: type = splice_block %Generic.type [template = constants.%Generic.type.3a] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.0 = name_ref Generic, file.%Generic.decl [template = constants.%Generic.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%A)> [template = constants.%Generic.type.3a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.1: %Generic.type.3a = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc9_6.2 (constants.%T.b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc10_6.1: %Generic.type.d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.a)]
// CHECK:STDOUT:     %T.param_patt: %Generic.type.d = value_param_pattern %T.patt.loc10_6.1, runtime_param<invalid> [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %Generic.type.d = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc10: type = splice_block %Generic.type [template = constants.%Generic.type.d] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.0 = name_ref Generic, file.%Generic.decl [template = constants.%Generic.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%B)> [template = constants.%Generic.type.d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_6.1: %Generic.type.d = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_6.2 (constants.%T.d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T.6)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt.d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc4_19.2)> [symbolic = %Generic.type (constants.%Generic.type.38)]
// CHECK:STDOUT:   %Self.2: %Generic.type.38 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.38) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Generic.type.3a) {
// CHECK:STDOUT:   %T.loc9_6.2: %Generic.type.3a = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.b)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %Generic.type.3a = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.3a);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_6.1: %Generic.type.d) {
// CHECK:STDOUT:   %T.loc10_6.2: %Generic.type.d = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.2 (constants.%T.d)]
// CHECK:STDOUT:   %T.patt.loc10_6.2: %Generic.type.d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt.a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Generic.type.d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: %Generic.type.d = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T.d)]
// CHECK:STDOUT:     %.loc18: %Generic.type.3a = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.6) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T.6
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.b) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.b
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.d) {
// CHECK:STDOUT:   %T.loc10_6.2 => constants.%T.d
// CHECK:STDOUT:   %T.patt.loc10_6.2 => constants.%T.d
// CHECK:STDOUT: }
// CHECK:STDOUT:
