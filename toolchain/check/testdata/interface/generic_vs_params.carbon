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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic_vs_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic_vs_params.carbon

// --- params.carbon

library "[[@TEST_NAME]]";

interface NotGenericNoParams {}
interface NotGenericButParams() {}
interface GenericAndParams(T:! type) {}

class C(T:! type) {
  interface GenericNoParams {}
  interface GenericAndParams(U:! type) {}
}

class X {}
impl X as NotGenericNoParams {}
impl X as NotGenericButParams() {}
impl X as GenericAndParams(X) {}
impl X as C(X).GenericNoParams {}
impl X as C(X).GenericAndParams(X) {}

// --- fail_non_generic_implicit_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_implicit_params.carbon:[[@LINE+4]]:13: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: interface A[T: type]() {}
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
interface A[T: type]() {}

// --- fail_non_generic_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_params.carbon:[[@LINE+4]]:13: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: interface A(T: type) {}
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
interface A(T: type) {}

// --- fail_implicit_params_only_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit_params_only_empty.carbon:[[@LINE+4]]:14: error: expected explicit parameters after implicit parameters [GenericMissingExplicitParameters]
// CHECK:STDERR: interface Bar[] {}
// CHECK:STDERR:              ^~
// CHECK:STDERR:
interface Bar[] {}

// --- fail_implicit_params_only.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit_params_only.carbon:[[@LINE+4]]:14: error: expected explicit parameters after implicit parameters [GenericMissingExplicitParameters]
// CHECK:STDERR: interface Bar[T:! type] {}
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR:
interface Bar[T:! type] {}

// CHECK:STDOUT: --- params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGenericNoParams.type: type = facet_type <@NotGenericNoParams> [concrete]
// CHECK:STDOUT:   %Self.9fc: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type.647: type = generic_interface_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.generic: %NotGenericButParams.type.647 = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericButParams.type.a08: type = facet_type <@NotGenericButParams> [concrete]
// CHECK:STDOUT:   %Self.290: %NotGenericButParams.type.a08 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.17f: type = generic_interface_type @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.253: %GenericAndParams.type.17f = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.a01: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T)> [symbolic]
// CHECK:STDOUT:   %Self.268: %GenericAndParams.type.a01 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.type.7cc: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic]
// CHECK:STDOUT:   %Self.d3a: %GenericNoParams.type.7cc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.a5d: type = generic_interface_type @GenericAndParams.loc10, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.fc5: %GenericAndParams.type.a5d = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.542: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U)> [symbolic]
// CHECK:STDOUT:   %Self.c75: %GenericAndParams.type.542 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness: <witness> = impl_witness @X.as.NotGenericNoParams.impl.%NotGenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.facet: %NotGenericNoParams.type = facet_value %X, (%NotGenericNoParams.impl_witness) [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness @X.as.NotGenericButParams.impl.%NotGenericButParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %NotGenericButParams.facet: %NotGenericButParams.type.a08 = facet_value %X, (%NotGenericButParams.impl_witness) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.2bb: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.12e: <witness> = impl_witness @X.as.GenericAndParams.impl.c55.%GenericAndParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.fec: %GenericAndParams.type.2bb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.facet.109: %GenericAndParams.type.2bb = facet_value %X, (%GenericAndParams.impl_witness.12e) [concrete]
// CHECK:STDOUT:   %C.513: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericNoParams.type.771: type = facet_type <@GenericNoParams, @GenericNoParams(%X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.8ab: type = generic_interface_type @GenericAndParams.loc10, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.22e: %GenericAndParams.type.8ab = struct_value () [concrete]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness @X.as.GenericNoParams.impl.%GenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.b86: %GenericNoParams.type.771 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericNoParams.facet: %GenericNoParams.type.771 = facet_value %X, (%GenericNoParams.impl_witness) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.2f8: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%X, %X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.e7f: <witness> = impl_witness @X.as.GenericAndParams.impl.c97.%GenericAndParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.292: %GenericAndParams.type.2f8 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.facet.ac4: %GenericAndParams.type.2f8 = facet_value %X, (%GenericAndParams.impl_witness.e7f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGenericNoParams = %NotGenericNoParams.decl
// CHECK:STDOUT:     .NotGenericButParams = %NotGenericButParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericNoParams.decl: type = interface_decl @NotGenericNoParams [concrete = constants.%NotGenericNoParams.type] {} {}
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type.647 = interface_decl @NotGenericButParams [concrete = constants.%NotGenericButParams.generic] {} {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.17f = interface_decl @GenericAndParams.loc6 [concrete = constants.%GenericAndParams.generic.253] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_32.1: type = splice_block %.loc6_32.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_28.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @X.as.NotGenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, file.%NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.NotGenericButParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type.647 = name_ref NotGenericButParams, file.%NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams.type: type = facet_type <@NotGenericButParams> [concrete = constants.%NotGenericButParams.type.a08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.c55 [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.17f = name_ref GenericAndParams, file.%GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.253]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(constants.%X)> [concrete = constants.%GenericAndParams.type.2bb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.513]
// CHECK:STDOUT:     %.loc17: type = specific_constant @C.%GenericNoParams.decl, @C(constants.%X) [concrete = constants.%GenericNoParams.type.771]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, %.loc17 [concrete = constants.%GenericNoParams.type.771]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.c97 [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.513]
// CHECK:STDOUT:     %.loc18: %GenericAndParams.type.8ab = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.22e]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.8ab = name_ref GenericAndParams, %.loc18 [concrete = constants.%GenericAndParams.generic.22e]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(constants.%X, constants.%X)> [concrete = constants.%GenericAndParams.type.2f8]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9fc]
// CHECK:STDOUT:   interface_with_self_decl @NotGenericNoParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %NotGenericButParams.type.a08 = symbolic_binding Self, 0 [symbolic = constants.%Self.290]
// CHECK:STDOUT:   interface_with_self_decl @NotGenericButParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.loc6(%T.loc6_28.2: type) {
// CHECK:STDOUT:   %T.loc6_28.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_28.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T.loc6_28.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.a01)]
// CHECK:STDOUT:   %Self.loc6_38.2: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.a01) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.268)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_38.1: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.a01) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.268)]
// CHECK:STDOUT:     interface_with_self_decl @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc6_38.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericNoParams(@C.%T.loc8_9.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.7cc)]
// CHECK:STDOUT:   %Self.loc9_29.2: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.7cc) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.d3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc9_29.1: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.7cc) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.d3a)]
// CHECK:STDOUT:     interface_with_self_decl @GenericNoParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_29.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.loc10(@C.%T.loc8_9.2: type, %U.loc10_30.2: type) {
// CHECK:STDOUT:   %U.loc10_30.1: type = symbolic_binding U, 1 [symbolic = %U.loc10_30.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U.loc10_30.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.542)]
// CHECK:STDOUT:   %Self.loc10_40.2: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.542) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.c75)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc10_40.1: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.542) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.c75)]
// CHECK:STDOUT:     interface_with_self_decl @GenericAndParams.loc10 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc10_40.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.NotGenericNoParams.impl: %X.ref as %NotGenericNoParams.ref {
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness_table = impl_witness_table (), @X.as.NotGenericNoParams.impl [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness: <witness> = impl_witness %NotGenericNoParams.impl_witness_table [concrete = constants.%NotGenericNoParams.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %NotGenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.NotGenericButParams.impl: %X.ref as %NotGenericButParams.type {
// CHECK:STDOUT:   %NotGenericButParams.impl_witness_table = impl_witness_table (), @X.as.NotGenericButParams.impl [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness %NotGenericButParams.impl_witness_table [concrete = constants.%NotGenericButParams.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %NotGenericButParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.c55: %X.ref.loc16_6 as %GenericAndParams.type {
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table = impl_witness_table (), @X.as.GenericAndParams.impl.c55 [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness: <witness> = impl_witness %GenericAndParams.impl_witness_table [concrete = constants.%GenericAndParams.impl_witness.12e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %GenericAndParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericNoParams.impl: %X.ref.loc17_6 as %GenericNoParams.ref {
// CHECK:STDOUT:   %GenericNoParams.impl_witness_table = impl_witness_table (), @X.as.GenericNoParams.impl [concrete]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness %GenericNoParams.impl_witness_table [concrete = constants.%GenericNoParams.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %GenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.c97: %X.ref.loc18_6 as %GenericAndParams.type {
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table = impl_witness_table (), @X.as.GenericAndParams.impl.c97 [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness: <witness> = impl_witness %GenericAndParams.impl_witness_table [concrete = constants.%GenericAndParams.impl_witness.e7f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %GenericAndParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T.loc8_9.1)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.7cc)]
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_interface_type @GenericAndParams.loc10, @C(%T.loc8_9.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.a5d)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.a5d) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.fc5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [symbolic = @C.%GenericNoParams.type (constants.%GenericNoParams.type.7cc)] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.a5d) = interface_decl @GenericAndParams.loc10 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.fc5)] {
// CHECK:STDOUT:       %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_34.1: type = splice_block %.loc10_34.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc10_34.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc10_30.2: type = symbolic_binding U, 1 [symbolic = %U.loc10_30.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: specific @NotGenericNoParams(constants.%Self.9fc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericButParams(constants.%Self.290) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%T) {
// CHECK:STDOUT:   %T.loc6_28.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%T, constants.%Self.268) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T, constants.%Self.d3a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc10_30.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%T, constants.%U, constants.%Self.c75) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericNoParams(constants.%NotGenericNoParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericButParams(constants.%NotGenericButParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X) {
// CHECK:STDOUT:   %T.loc6_28.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.2bb
// CHECK:STDOUT:   %Self.loc6_38.2 => constants.%Self.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X, constants.%Self.268) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X, constants.%GenericAndParams.facet.109) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.771
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.8ab
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.22e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.771
// CHECK:STDOUT:   %Self.loc9_29.2 => constants.%Self.b86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X, constants.%Self.d3a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X, constants.%GenericNoParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc10_30.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.2f8
// CHECK:STDOUT:   %Self.loc10_40.2 => constants.%Self.292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X, constants.%Self.c75) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X, constants.%GenericAndParams.facet.ac4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.464: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.464 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.4c4: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.4c4 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.464 = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.4c4 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.464: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.464 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.4c4: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.4c4 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.464 = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.4c4 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar.type.d8f: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.d8f = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.c56: type = facet_type <@Bar> [concrete]
// CHECK:STDOUT:   %Self: %Bar.type.c56 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.d8f = interface_decl @Bar [concrete = constants.%Bar.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Bar {
// CHECK:STDOUT:   %Self: %Bar.type.c56 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Bar.type.d8f: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.d8f = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.24d: type = facet_type <@Bar, @Bar(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Bar.type.24d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.d8f = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%T.loc8_15.1)> [symbolic = %Bar.type (constants.%Bar.type.24d)]
// CHECK:STDOUT:   %Self.loc8_25.2: @Bar.%Bar.type (%Bar.type.24d) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_25.1: @Bar.%Bar.type (%Bar.type.24d) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
// CHECK:STDOUT:     interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_25.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
