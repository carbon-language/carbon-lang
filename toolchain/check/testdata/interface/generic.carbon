// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic.carbon

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
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `Generic(B)` into type implementing `Generic(A)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_mismatched_args.carbon:[[@LINE-5]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! Generic(A));
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Simple.type.3b3: type = generic_interface_type @Simple [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Simple.generic: %Simple.type.3b3 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.type.a1f: type = facet_type <@Simple, @Simple(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.4c2: %Simple.type.a1f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.509: type = generic_interface_type @WithAssocFn [concrete]
// CHECK:STDOUT:   %WithAssocFn.generic: %WithAssocFn.type.509 = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.61b: type = facet_type <@WithAssocFn, @WithAssocFn(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.c02: %WithAssocFn.type.61b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %WithAssocFn.F.type.1af: type = fn_type @WithAssocFn.F, @WithAssocFn(%T.8b3) [symbolic]
// CHECK:STDOUT:   %WithAssocFn.F.b7d: %WithAssocFn.F.type.1af = struct_value () [symbolic]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.b2f: type = assoc_entity_type @WithAssocFn, @WithAssocFn(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fc5: %WithAssocFn.assoc_type.b2f = assoc_entity element0, @WithAssocFn.%WithAssocFn.F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Simple.type.ebf: type = facet_type <@Simple, @Simple(%C)> [concrete]
// CHECK:STDOUT:   %Self.ae2: %Simple.type.ebf = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @C.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.fc9: type = facet_type <@WithAssocFn, @WithAssocFn(%C)> [concrete]
// CHECK:STDOUT:   %Self.8cc: %WithAssocFn.type.fc9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %WithAssocFn.F.type.18c: type = fn_type @WithAssocFn.F, @WithAssocFn(%C) [concrete]
// CHECK:STDOUT:   %WithAssocFn.F.e46: %WithAssocFn.F.type.18c = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.9ea: type = assoc_entity_type @WithAssocFn, @WithAssocFn(%C) [concrete]
// CHECK:STDOUT:   %assoc0.cad: %WithAssocFn.assoc_type.9ea = assoc_entity element0, @WithAssocFn.%WithAssocFn.F.decl [concrete]
// CHECK:STDOUT:   %WithAssocFn.impl_witness: <witness> = impl_witness @C.%WithAssocFn.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F.type: type = fn_type @C.as.WithAssocFn.impl.F [concrete]
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F: %C.as.WithAssocFn.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.facet: %WithAssocFn.type.fc9 = facet_value %C, (%WithAssocFn.impl_witness) [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %N: %T.8b3 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [concrete]
// CHECK:STDOUT:   %WithImplicitArgs.generic: %WithImplicitArgs.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.0a9: %Simple.type.ebf = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %Simple.type.ebf [concrete]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [concrete]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [concrete]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [concrete]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [concrete]
// CHECK:STDOUT:   %Receive.specific_fn: <specific function> = specific_function %Receive, @Receive(%T.0a9) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type.509 = interface_decl @WithAssocFn [concrete = constants.%WithAssocFn.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_23.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [concrete = constants.%WithImplicitArgs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @WithImplicitArgs.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc22_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_28.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc22: type = splice_block %T.ref [symbolic = %T.loc22_28.1 (constants.%T.8b3)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc22_28.2 [symbolic = %T.loc22_28.1 (constants.%T.8b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc22_38.2: @WithImplicitArgs.%T.loc22_28.1 (%T.8b3) = bind_symbolic_name N, 1 [symbolic = %N.loc22_38.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [concrete = constants.%Receive] {
// CHECK:STDOUT:     %T.patt: %pattern_type.eee = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24: type = splice_block %Simple.type [concrete = constants.%Simple.type.ebf] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.ebf]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_12.2: %Simple.type.ebf = bind_symbolic_name T, 0 [symbolic = %T.loc24_12.1 (constants.%T.0a9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [concrete = constants.%Pass] {
// CHECK:STDOUT:     %T.patt: %pattern_type.eee = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %Simple.type [concrete = constants.%Simple.type.ebf] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.ebf]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_9.2: %Simple.type.ebf = bind_symbolic_name T, 0 [symbolic = %T.loc25_9.1 (constants.%T.0a9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_18.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple, @Simple(%T.loc4_18.1)> [symbolic = %Simple.type (constants.%Simple.type.a1f)]
// CHECK:STDOUT:   %Self.2: @Simple.%Simple.type (%Simple.type.a1f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Simple.%Simple.type (%Simple.type.a1f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8_23.2: type) {
// CHECK:STDOUT:   %T.loc8_23.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_23.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(%T.loc8_23.1)> [symbolic = %WithAssocFn.type (constants.%WithAssocFn.type.61b)]
// CHECK:STDOUT:   %Self.2: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.61b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c02)]
// CHECK:STDOUT:   %WithAssocFn.F.type: type = fn_type @WithAssocFn.F, @WithAssocFn(%T.loc8_23.1) [symbolic = %WithAssocFn.F.type (constants.%WithAssocFn.F.type.1af)]
// CHECK:STDOUT:   %WithAssocFn.F: @WithAssocFn.%WithAssocFn.F.type (%WithAssocFn.F.type.1af) = struct_value () [symbolic = %WithAssocFn.F (constants.%WithAssocFn.F.b7d)]
// CHECK:STDOUT:   %WithAssocFn.assoc_type: type = assoc_entity_type @WithAssocFn, @WithAssocFn(%T.loc8_23.1) [symbolic = %WithAssocFn.assoc_type (constants.%WithAssocFn.assoc_type.b2f)]
// CHECK:STDOUT:   %assoc0.loc10_14.2: @WithAssocFn.%WithAssocFn.assoc_type (%WithAssocFn.assoc_type.b2f) = assoc_entity element0, %WithAssocFn.F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.fc5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.61b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c02)]
// CHECK:STDOUT:     %WithAssocFn.F.decl: @WithAssocFn.%WithAssocFn.F.type (%WithAssocFn.F.type.1af) = fn_decl @WithAssocFn.F [symbolic = @WithAssocFn.%WithAssocFn.F (constants.%WithAssocFn.F.b7d)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.019 = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.019 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:       %return: ref %X = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_14.1: @WithAssocFn.%WithAssocFn.assoc_type (%WithAssocFn.assoc_type.b2f) = assoc_entity element0, %WithAssocFn.F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.fc5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc10_14.1
// CHECK:STDOUT:     witness = (%WithAssocFn.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22_28.2: type, %N.loc22_38.2: @WithImplicitArgs.%T.loc22_28.1 (%T.8b3)) {
// CHECK:STDOUT:   %T.loc22_28.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_28.1 (constants.%T.8b3)]
// CHECK:STDOUT:   %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.1 (%T.8b3) = bind_symbolic_name N, 1 [symbolic = %N.loc22_38.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc22_28.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Simple.impl: %Self.ref as %Simple.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C.%Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.WithAssocFn.impl: %Self.ref as %WithAssocFn.type {
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F.decl: %C.as.WithAssocFn.impl.F.type = fn_decl @C.as.WithAssocFn.impl.F [concrete = constants.%C.as.WithAssocFn.impl.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.019 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.019 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .F = %C.as.WithAssocFn.impl.F.decl
// CHECK:STDOUT:   witness = @C.%WithAssocFn.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Simple.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.ref: %Simple.type.3b3 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.ebf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (), @C.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.WithAssocFn.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type.509 = name_ref WithAssocFn, file.%WithAssocFn.decl [concrete = constants.%WithAssocFn.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(constants.%C)> [concrete = constants.%WithAssocFn.type.fc9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithAssocFn.impl_witness_table = impl_witness_table (@C.as.WithAssocFn.impl.%C.as.WithAssocFn.impl.F.decl), @C.as.WithAssocFn.impl [concrete]
// CHECK:STDOUT:   %WithAssocFn.impl_witness: <witness> = impl_witness %WithAssocFn.impl_witness_table [concrete = constants.%WithAssocFn.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @WithAssocFn.F(@WithAssocFn.%T.loc8_23.2: type, @WithAssocFn.%Self.1: @WithAssocFn.%WithAssocFn.type (%WithAssocFn.type.61b)) {
// CHECK:STDOUT:   fn() -> %return.param: %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.WithAssocFn.impl.F() -> %return.param: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_15.2: init %X = class_init (), %return [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   return %.loc17_16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24_12.2: %Simple.type.ebf) {
// CHECK:STDOUT:   %T.loc24_12.1: %Simple.type.ebf = bind_symbolic_name T, 0 [symbolic = %T.loc24_12.1 (constants.%T.0a9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25_9.2: %Simple.type.ebf) {
// CHECK:STDOUT:   %T.loc25_9.1: %Simple.type.ebf = bind_symbolic_name T, 0 [symbolic = %T.loc25_9.1 (constants.%T.0a9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Receive.specific_fn.loc26_3.2: <specific function> = specific_function constants.%Receive, @Receive(%T.loc25_9.1) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [concrete = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %Simple.type.ebf = name_ref T, %T.loc25_9.2 [symbolic = %T.loc25_9.1 (constants.%T.0a9)]
// CHECK:STDOUT:     %Receive.specific_fn.loc26_3.1: <specific function> = specific_function %Receive.ref, @Receive(constants.%T.0a9) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:     %Receive.call: init %empty_tuple.type = call %Receive.specific_fn.loc26_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc8_23.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn.F(constants.%T.8b3, constants.%Self.c02) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type => constants.%Simple.type.ebf
// CHECK:STDOUT:   %Self.2 => constants.%Self.ae2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.loc8_23.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type => constants.%WithAssocFn.type.fc9
// CHECK:STDOUT:   %Self.2 => constants.%Self.8cc
// CHECK:STDOUT:   %WithAssocFn.F.type => constants.%WithAssocFn.F.type.18c
// CHECK:STDOUT:   %WithAssocFn.F => constants.%WithAssocFn.F.e46
// CHECK:STDOUT:   %WithAssocFn.assoc_type => constants.%WithAssocFn.assoc_type.9ea
// CHECK:STDOUT:   %assoc0.loc10_14.2 => constants.%assoc0.cad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn.F(constants.%C, constants.%WithAssocFn.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.8b3, constants.%N) {
// CHECK:STDOUT:   %T.loc22_28.1 => constants.%T.8b3
// CHECK:STDOUT:   %N.loc22_38.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.0a9) {
// CHECK:STDOUT:   %T.loc24_12.1 => constants.%T.0a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.0a9) {
// CHECK:STDOUT:   %T.loc25_9.1 => constants.%T.0a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.2db: type = facet_type <@Generic, @Generic(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.899: %Generic.type.2db = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Generic.type.185: type = facet_type <@Generic, @Generic(%A)> [concrete]
// CHECK:STDOUT:   %T.000: %Generic.type.185 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %Generic.type.185 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.227: type = facet_type <@Generic, @Generic(%B)> [concrete]
// CHECK:STDOUT:   %T.03b: %Generic.type.227 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b43: type = pattern_type %Generic.type.227 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.a3e: %Generic.type.227 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.99a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %Generic.type [concrete = constants.%Generic.type.185] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%A)> [concrete = constants.%Generic.type.185]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Generic.type.185 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.000)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.b43 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %Generic.type [concrete = constants.%Generic.type.227] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%B)> [concrete = constants.%Generic.type.227]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_6.2: %Generic.type.227 = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.1 (constants.%T.03b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.2db)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.2db) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.899)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.2db) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.899)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Generic.type.185) {
// CHECK:STDOUT:   %T.loc9_6.1: %Generic.type.185 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.000)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_6.2: %Generic.type.227) {
// CHECK:STDOUT:   %T.loc10_6.1: %Generic.type.227 = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.1 (constants.%T.03b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Generic.type.227 = name_ref T, %T.loc10_6.2 [symbolic = %T.loc10_6.1 (constants.%T.03b)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.000) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.000
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.227
// CHECK:STDOUT:   %Self.2 => constants.%Self.a3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.03b) {
// CHECK:STDOUT:   %T.loc10_6.1 => constants.%T.03b
// CHECK:STDOUT: }
// CHECK:STDOUT:
