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

fn Receive(unused T:! Simple(C)) {}
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

// --- fail_args_count_mismatch.carbon
library "[[@TEST_NAME]]";

interface Generic(T:! type) {}

// CHECK:STDERR: fail_args_count_mismatch.carbon:[[@LINE+7]]:17: error: 2 arguments passed to generic interface expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: fn F(unused T:! Generic((), ())) {}
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_args_count_mismatch.carbon:[[@LINE-5]]:1: note: calling generic interface declared here [InCallToEntity]
// CHECK:STDERR: interface Generic(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused T:! Generic((), ())) {}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Simple.type.673: type = generic_interface_type @Simple [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Simple.generic: %Simple.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.type.e29: type = facet_type <@Simple, @Simple(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.862: %Simple.type.e29 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.daa: type = generic_interface_type @WithAssocFn [concrete]
// CHECK:STDOUT:   %WithAssocFn.generic: %WithAssocFn.type.daa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.d36: type = facet_type <@WithAssocFn, @WithAssocFn(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.629: %WithAssocFn.type.d36 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.22a: Core.Form = init_form %X, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.type.6c9: type = fn_type @WithAssocFn.WithSelf.F, @WithAssocFn(%T.67d, %Self.629) [symbolic]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.6d9: %WithAssocFn.WithSelf.F.type.6c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.b93: type = assoc_entity_type @WithAssocFn, @WithAssocFn(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.395: %WithAssocFn.assoc_type.b93 = assoc_entity element0, @WithAssocFn.%WithAssocFn.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Simple.type.12a: type = facet_type <@Simple, @Simple(%C)> [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @C.as.Simple.impl.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.0b4: %Simple.type.12a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Simple.facet: %Simple.type.12a = facet_value %C, (%Simple.impl_witness) [concrete]
// CHECK:STDOUT:   %WithAssocFn.type.19d: type = facet_type <@WithAssocFn, @WithAssocFn(%C)> [concrete]
// CHECK:STDOUT:   %WithAssocFn.impl_witness: <witness> = impl_witness @C.as.WithAssocFn.impl.%WithAssocFn.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.63e: %WithAssocFn.type.19d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.type.836: type = fn_type @WithAssocFn.WithSelf.F, @WithAssocFn(%C, %Self.629) [symbolic]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.a5a: %WithAssocFn.WithSelf.F.type.836 = struct_value () [symbolic]
// CHECK:STDOUT:   %WithAssocFn.assoc_type.cc0: type = assoc_entity_type @WithAssocFn, @WithAssocFn(%C) [concrete]
// CHECK:STDOUT:   %assoc0.50d: %WithAssocFn.assoc_type.cc0 = assoc_entity element0, @WithAssocFn.%WithAssocFn.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F.type: type = fn_type @C.as.WithAssocFn.impl.F [concrete]
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F: %C.as.WithAssocFn.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAssocFn.facet: %WithAssocFn.type.19d = facet_value %C, (%WithAssocFn.impl_witness) [concrete]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.type.a47: type = fn_type @WithAssocFn.WithSelf.F, @WithAssocFn(%C, %WithAssocFn.facet) [concrete]
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.975: %WithAssocFn.WithSelf.F.type.a47 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %N: %T.67d = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %WithImplicitArgs.type: type = generic_interface_type @WithImplicitArgs [concrete]
// CHECK:STDOUT:   %WithImplicitArgs.generic: %WithImplicitArgs.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.7608f6.1: %Simple.type.12a = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.08a: type = pattern_type %Simple.type.12a [concrete]
// CHECK:STDOUT:   %T.7608f6.2: %Simple.type.12a = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Receive.type: type = fn_type @Receive [concrete]
// CHECK:STDOUT:   %Receive: %Receive.type = struct_value () [concrete]
// CHECK:STDOUT:   %Pass.type: type = fn_type @Pass [concrete]
// CHECK:STDOUT:   %Pass: %Pass.type = struct_value () [concrete]
// CHECK:STDOUT:   %Receive.specific_fn: <specific function> = specific_function %Receive, @Receive(%T.7608f6.1) [symbolic]
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
// CHECK:STDOUT:   %Simple.decl: %Simple.type.673 = interface_decl @Simple [concrete = constants.%Simple.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_22.1: type = splice_block %.loc4_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %WithAssocFn.decl: %WithAssocFn.type.daa = interface_decl @WithAssocFn [concrete = constants.%WithAssocFn.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_27.1: type = splice_block %.loc8_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_23.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %WithImplicitArgs.decl: %WithImplicitArgs.type = interface_decl @WithImplicitArgs [concrete = constants.%WithImplicitArgs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @WithImplicitArgs.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_32.1: type = splice_block %.loc22_32.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc22_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc22_28.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc22_42: type = splice_block %T.ref [symbolic = %T.loc22_28.1 (constants.%T.67d)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc22_28.2 [symbolic = %T.loc22_28.1 (constants.%T.67d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc22_38.2: @WithImplicitArgs.%T.loc22_28.1 (%T.67d) = symbolic_binding N, 1 [symbolic = %N.loc22_38.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Receive.decl: %Receive.type = fn_decl @Receive [concrete = constants.%Receive] {
// CHECK:STDOUT:     %T.patt: %pattern_type.08a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24: type = splice_block %Simple.type [concrete = constants.%Simple.type.12a] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Simple.ref: %Simple.type.673 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.12a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_19.3: %Simple.type.12a = symbolic_binding T, 0 [symbolic = %T.loc24_19.2 (constants.%T.7608f6.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pass.decl: %Pass.type = fn_decl @Pass [concrete = constants.%Pass] {
// CHECK:STDOUT:     %T.patt: %pattern_type.08a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %Simple.type [concrete = constants.%Simple.type.12a] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Simple.ref: %Simple.type.673 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.12a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_9.2: %Simple.type.12a = symbolic_binding T, 0 [symbolic = %T.loc25_9.1 (constants.%T.7608f6.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Simple(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple, @Simple(%T.loc4_18.1)> [symbolic = %Simple.type (constants.%Simple.type.e29)]
// CHECK:STDOUT:   %Self.loc4_28.2: @Simple.%Simple.type (%Simple.type.e29) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.862)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @Simple.%Simple.type (%Simple.type.e29) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self.862)]
// CHECK:STDOUT:     interface_with_self_decl @Simple [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithAssocFn(%T.loc8_23.2: type) {
// CHECK:STDOUT:   %T.loc8_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_23.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type.loc8: type = facet_type <@WithAssocFn, @WithAssocFn(%T.loc8_23.1)> [symbolic = %WithAssocFn.type.loc8 (constants.%WithAssocFn.type.d36)]
// CHECK:STDOUT:   %Self.loc8_33.2: @WithAssocFn.%WithAssocFn.type.loc8 (%WithAssocFn.type.d36) = symbolic_binding Self, 1 [symbolic = %Self.loc8_33.2 (constants.%Self.629)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_33.1: @WithAssocFn.%WithAssocFn.type.loc8 (%WithAssocFn.type.d36) = symbolic_binding Self, 1 [symbolic = %Self.loc8_33.2 (constants.%Self.629)]
// CHECK:STDOUT:     interface_with_self_decl @WithAssocFn [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %WithAssocFn.WithSelf.F.decl: @WithAssocFn.%WithAssocFn.WithSelf.F.type (%WithAssocFn.WithSelf.F.type.6c9) = fn_decl @WithAssocFn.WithSelf.F [symbolic = @WithAssocFn.%WithAssocFn.WithSelf.F (constants.%WithAssocFn.WithSelf.F.6d9)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.05f = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.05f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %.loc10: Core.Form = init_form %X.ref, call_param0 [concrete = constants.%.22a]
// CHECK:STDOUT:       %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:       %return: ref %X = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_14.1: @WithAssocFn.%WithAssocFn.assoc_type (%WithAssocFn.assoc_type.b93) = assoc_entity element0, %WithAssocFn.WithSelf.F.decl [symbolic = %assoc0.loc10_14.2 (constants.%assoc0.395)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_33.1
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc10_14.1
// CHECK:STDOUT:     witness = (%WithAssocFn.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @WithImplicitArgs(%T.loc22_28.2: type, %N.loc22_38.2: @WithImplicitArgs.%T.loc22_28.1 (%T.67d)) {
// CHECK:STDOUT:   %T.loc22_28.1: type = symbolic_binding T, 0 [symbolic = %T.loc22_28.1 (constants.%T.67d)]
// CHECK:STDOUT:   %N.loc22_38.1: @WithImplicitArgs.%T.loc22_28.1 (%T.67d) = symbolic_binding N, 1 [symbolic = %N.loc22_38.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc22_28.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Simple.impl: %Self.ref as %Simple.type {
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (), @C.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.WithAssocFn.impl: %Self.ref as %WithAssocFn.type {
// CHECK:STDOUT:   %C.as.WithAssocFn.impl.F.decl: %C.as.WithAssocFn.impl.F.type = fn_decl @C.as.WithAssocFn.impl.F [concrete = constants.%C.as.WithAssocFn.impl.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.05f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.05f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %X.ref, call_param0 [concrete = constants.%.22a]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithAssocFn.impl_witness_table = impl_witness_table (%C.as.WithAssocFn.impl.F.decl), @C.as.WithAssocFn.impl [concrete]
// CHECK:STDOUT:   %WithAssocFn.impl_witness: <witness> = impl_witness %WithAssocFn.impl_witness_table [concrete = constants.%WithAssocFn.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .F = %C.as.WithAssocFn.impl.F.decl
// CHECK:STDOUT:   witness = %WithAssocFn.impl_witness
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
// CHECK:STDOUT:     %Simple.ref: %Simple.type.673 = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.type: type = facet_type <@Simple, @Simple(constants.%C)> [concrete = constants.%Simple.type.12a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.WithAssocFn.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.ref: %WithAssocFn.type.daa = name_ref WithAssocFn, file.%WithAssocFn.decl [concrete = constants.%WithAssocFn.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %WithAssocFn.type: type = facet_type <@WithAssocFn, @WithAssocFn(constants.%C)> [concrete = constants.%WithAssocFn.type.19d]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @WithAssocFn.WithSelf.F(@WithAssocFn.%T.loc8_23.2: type, @WithAssocFn.%Self.loc8_33.1: @WithAssocFn.%WithAssocFn.type.loc8 (%WithAssocFn.type.d36)) {
// CHECK:STDOUT:   fn() -> out %return.param: %X;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.WithAssocFn.impl.F() -> out %return.param: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_15.2: init %X to %return.param = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc17_16: init %X = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   return %.loc17_16 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Receive(%T.loc24_19.3: %Simple.type.12a) {
// CHECK:STDOUT:   %T.loc24_19.1: %Simple.type.12a = symbolic_binding T, 0 [symbolic = %T.loc24_19.1 (constants.%T.7608f6.1)]
// CHECK:STDOUT:   %T.loc24_19.2: %Simple.type.12a = symbolic_binding T, 0 [symbolic = %T.loc24_19.2 (constants.%T.7608f6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Pass(%T.loc25_9.2: %Simple.type.12a) {
// CHECK:STDOUT:   %T.loc25_9.1: %Simple.type.12a = symbolic_binding T, 0 [symbolic = %T.loc25_9.1 (constants.%T.7608f6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Receive.specific_fn.loc26_3.2: <specific function> = specific_function constants.%Receive, @Receive(%T.loc25_9.1) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Receive.ref: %Receive.type = name_ref Receive, file.%Receive.decl [concrete = constants.%Receive]
// CHECK:STDOUT:     %T.ref: %Simple.type.12a = name_ref T, %T.loc25_9.2 [symbolic = %T.loc25_9.1 (constants.%T.7608f6.1)]
// CHECK:STDOUT:     %Receive.specific_fn.loc26_3.1: <specific function> = specific_function %Receive.ref, @Receive(constants.%T.7608f6.1) [symbolic = %Receive.specific_fn.loc26_3.2 (constants.%Receive.specific_fn)]
// CHECK:STDOUT:     %Receive.call: init %empty_tuple.type = call %Receive.specific_fn.loc26_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%T.67d, constants.%Self.862) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc8_23.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%T.67d, constants.%Self.629) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn.WithSelf.F(constants.%T.67d, constants.%Self.629) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Simple.type => constants.%Simple.type.12a
// CHECK:STDOUT:   %Self.loc4_28.2 => constants.%Self.0b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C, constants.%Self.862) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple(constants.%C, constants.%Simple.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C) {
// CHECK:STDOUT:   %T.loc8_23.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WithAssocFn.type.loc8 => constants.%WithAssocFn.type.19d
// CHECK:STDOUT:   %Self.loc8_33.2 => constants.%Self.63e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C, constants.%Self.629) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.loc10 => constants.%C
// CHECK:STDOUT:   %WithAssocFn.type.loc10 => constants.%WithAssocFn.type.19d
// CHECK:STDOUT:   %Self.loc10 => constants.%Self.629
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.type => constants.%WithAssocFn.WithSelf.F.type.836
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F => constants.%WithAssocFn.WithSelf.F.a5a
// CHECK:STDOUT:   %WithAssocFn.assoc_type => constants.%WithAssocFn.assoc_type.cc0
// CHECK:STDOUT:   %assoc0.loc10_14.2 => constants.%assoc0.50d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn(constants.%C, constants.%WithAssocFn.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.loc10 => constants.%C
// CHECK:STDOUT:   %WithAssocFn.type.loc10 => constants.%WithAssocFn.type.19d
// CHECK:STDOUT:   %Self.loc10 => constants.%WithAssocFn.facet
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F.type => constants.%WithAssocFn.WithSelf.F.type.a47
// CHECK:STDOUT:   %WithAssocFn.WithSelf.F => constants.%WithAssocFn.WithSelf.F.975
// CHECK:STDOUT:   %WithAssocFn.assoc_type => constants.%WithAssocFn.assoc_type.cc0
// CHECK:STDOUT:   %assoc0.loc10_14.2 => constants.%assoc0.50d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithAssocFn.WithSelf.F(constants.%C, constants.%WithAssocFn.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithImplicitArgs(constants.%T.67d, constants.%N) {
// CHECK:STDOUT:   %T.loc22_28.1 => constants.%T.67d
// CHECK:STDOUT:   %N.loc22_38.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.7608f6.2) {
// CHECK:STDOUT:   %T.loc24_19.1 => constants.%T.7608f6.2
// CHECK:STDOUT:   %T.loc24_19.2 => constants.%T.7608f6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Pass(constants.%T.7608f6.1) {
// CHECK:STDOUT:   %T.loc25_9.1 => constants.%T.7608f6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Receive(constants.%T.7608f6.1) {
// CHECK:STDOUT:   %T.loc24_19.1 => constants.%T.7608f6.1
// CHECK:STDOUT:   %T.loc24_19.2 => constants.%T.7608f6.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.835: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.835 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.03d: type = facet_type <@Generic, @Generic(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.15d: %Generic.type.03d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Generic.type.478: type = facet_type <@Generic, @Generic(%A)> [concrete]
// CHECK:STDOUT:   %T.c1d: %Generic.type.478 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.708: type = pattern_type %Generic.type.478 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.cb7: type = facet_type <@Generic, @Generic(%B)> [concrete]
// CHECK:STDOUT:   %T.3ee: %Generic.type.cb7 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.0c9: type = pattern_type %Generic.type.cb7 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.b24: %Generic.type.cb7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.3ee [symbolic]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.835 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.708 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %Generic.type [concrete = constants.%Generic.type.478] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%A)> [concrete = constants.%Generic.type.478]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Generic.type.478 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.c1d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.0c9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %Generic.type [concrete = constants.%Generic.type.cb7] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %Generic.type: type = facet_type <@Generic, @Generic(constants.%B)> [concrete = constants.%Generic.type.cb7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_6.2: %Generic.type.cb7 = symbolic_binding T, 0 [symbolic = %T.loc10_6.1 (constants.%T.3ee)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.03d)]
// CHECK:STDOUT:   %Self.loc4_29.2: @Generic.%Generic.type (%Generic.type.03d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.15d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_29.1: @Generic.%Generic.type (%Generic.type.03d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self.15d)]
// CHECK:STDOUT:     interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_29.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
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
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Generic.type.478) {
// CHECK:STDOUT:   %T.loc9_6.1: %Generic.type.478 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.c1d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc10_6.2: %Generic.type.cb7) {
// CHECK:STDOUT:   %T.loc10_6.1: %Generic.type.cb7 = symbolic_binding T, 0 [symbolic = %T.loc10_6.1 (constants.%T.3ee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc10_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Generic.type.cb7 = name_ref T, %T.loc10_6.2 [symbolic = %T.loc10_6.1 (constants.%T.3ee)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.67d, constants.%Self.15d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%A) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.c1d) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.c1d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.cb7
// CHECK:STDOUT:   %Self.loc4_29.2 => constants.%Self.b24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.3ee) {
// CHECK:STDOUT:   %T.loc10_6.1 => constants.%T.3ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%B, constants.%Self.15d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_args_count_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.835: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.835 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.03d: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.03d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type.835 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_23.1: type = splice_block %.loc3_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %.loc12_26: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc12_30: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = symbolic_binding T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc3_19.2: type) {
// CHECK:STDOUT:   %T.loc3_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%T.loc3_19.1)> [symbolic = %Generic.type (constants.%Generic.type.03d)]
// CHECK:STDOUT:   %Self.loc3_29.2: @Generic.%Generic.type (%Generic.type.03d) = symbolic_binding Self, 1 [symbolic = %Self.loc3_29.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_29.1: @Generic.%Generic.type (%Generic.type.03d) = symbolic_binding Self, 1 [symbolic = %Self.loc3_29.2 (constants.%Self)]
// CHECK:STDOUT:     interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_29.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc3_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {}
// CHECK:STDOUT:
