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
// CHECK:STDOUT:   %Self.52c: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type.aee: type = generic_interface_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.generic: %NotGenericButParams.type.aee = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericButParams.type.25f: type = facet_type <@NotGenericButParams> [concrete]
// CHECK:STDOUT:   %Self.e41: %NotGenericButParams.type.25f = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.b2d: type = generic_interface_type @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.cbb: %GenericAndParams.type.b2d = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.de5: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T)> [symbolic]
// CHECK:STDOUT:   %Self.651: %GenericAndParams.type.de5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1f9: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.type.41a: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic]
// CHECK:STDOUT:   %Self.be7: %GenericNoParams.type.41a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.3a6: type = generic_interface_type @GenericAndParams.loc10, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.83d: %GenericAndParams.type.3a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.af8: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U)> [symbolic]
// CHECK:STDOUT:   %Self.528: %GenericAndParams.type.af8 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness: <witness> = impl_witness @X.as.NotGenericNoParams.impl.%NotGenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.facet: %NotGenericNoParams.type = facet_value %X, (%NotGenericNoParams.impl_witness) [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness @X.as.NotGenericButParams.impl.%NotGenericButParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %NotGenericButParams.facet: %NotGenericButParams.type.25f = facet_value %X, (%NotGenericButParams.impl_witness) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.0d2: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.194: <witness> = impl_witness @X.as.GenericAndParams.impl.c9e.%GenericAndParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.15c: %GenericAndParams.type.0d2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.facet.845: %GenericAndParams.type.0d2 = facet_value %X, (%GenericAndParams.impl_witness.194) [concrete]
// CHECK:STDOUT:   %C.7b5: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericNoParams.type.ff6: type = facet_type <@GenericNoParams, @GenericNoParams(%X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.7d5: type = generic_interface_type @GenericAndParams.loc10, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.13c: %GenericAndParams.type.7d5 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness @X.as.GenericNoParams.impl.%GenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.018: %GenericNoParams.type.ff6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericNoParams.facet: %GenericNoParams.type.ff6 = facet_value %X, (%GenericNoParams.impl_witness) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.504: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%X, %X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.d85: <witness> = impl_witness @X.as.GenericAndParams.impl.918.%GenericAndParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.500: %GenericAndParams.type.504 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.facet.f27: %GenericAndParams.type.504 = facet_value %X, (%GenericAndParams.impl_witness.d85) [concrete]
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
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type.aee = interface_decl @NotGenericButParams [concrete = constants.%NotGenericButParams.generic] {} {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.b2d = interface_decl @GenericAndParams.loc6 [concrete = constants.%GenericAndParams.generic.cbb] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_32.1: type = splice_block %.loc6_32.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_29.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_29.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @X.as.NotGenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, file.%NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.NotGenericButParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type.aee = name_ref NotGenericButParams, file.%NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams.type: type = facet_type <@NotGenericButParams> [concrete = constants.%NotGenericButParams.type.25f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.c9e [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.b2d = name_ref GenericAndParams, file.%GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.cbb]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(constants.%X)> [concrete = constants.%GenericAndParams.type.0d2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.7b5]
// CHECK:STDOUT:     %.loc17: type = specific_constant @C.%GenericNoParams.decl, @C(constants.%X) [concrete = constants.%GenericNoParams.type.ff6]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, %.loc17 [concrete = constants.%GenericNoParams.type.ff6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.918 [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.7b5]
// CHECK:STDOUT:     %.loc18: %GenericAndParams.type.7d5 = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.13c]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.7d5 = name_ref GenericAndParams, %.loc18 [concrete = constants.%GenericAndParams.generic.13c]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(constants.%X, constants.%X)> [concrete = constants.%GenericAndParams.type.504]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic = constants.%Self.52c]
// CHECK:STDOUT:   %NotGenericNoParams.WithSelf.decl = interface_with_self_decl @NotGenericNoParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %NotGenericButParams.type.25f = symbolic_binding Self, 0 [symbolic = constants.%Self.e41]
// CHECK:STDOUT:   %NotGenericButParams.WithSelf.decl = interface_with_self_decl @NotGenericButParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.loc6(%T.loc6_29.2: type) {
// CHECK:STDOUT:   %T.loc6_29.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_29.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T.loc6_29.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.de5)]
// CHECK:STDOUT:   %Self.loc6_38.2: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.de5) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.651)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_38.1: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.de5) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.651)]
// CHECK:STDOUT:     %GenericAndParams.WithSelf.decl = interface_with_self_decl @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc6_38.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericNoParams(@C.%T.loc8_10.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.41a)]
// CHECK:STDOUT:   %Self.loc9_29.2: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.41a) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.be7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc9_29.1: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.41a) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.be7)]
// CHECK:STDOUT:     %GenericNoParams.WithSelf.decl = interface_with_self_decl @GenericNoParams [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_29.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.loc10(@C.%T.loc8_10.2: type, %U.loc10_31.2: type) {
// CHECK:STDOUT:   %U.loc10_31.1: type = symbolic_binding U, 1 [symbolic = %U.loc10_31.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U.loc10_31.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.af8)]
// CHECK:STDOUT:   %Self.loc10_40.2: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.af8) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.528)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc10_40.1: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.af8) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.528)]
// CHECK:STDOUT:     %GenericAndParams.WithSelf.decl = interface_with_self_decl @GenericAndParams.loc10 [concrete]
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
// CHECK:STDOUT:   extend %NotGenericNoParams.ref
// CHECK:STDOUT:   witness = %NotGenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.NotGenericButParams.impl: %X.ref as %NotGenericButParams.type {
// CHECK:STDOUT:   %NotGenericButParams.impl_witness_table = impl_witness_table (), @X.as.NotGenericButParams.impl [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness %NotGenericButParams.impl_witness_table [concrete = constants.%NotGenericButParams.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %NotGenericButParams.type
// CHECK:STDOUT:   witness = %NotGenericButParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.c9e: %X.ref.loc16_6 as %GenericAndParams.type {
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table = impl_witness_table (), @X.as.GenericAndParams.impl.c9e [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness: <witness> = impl_witness %GenericAndParams.impl_witness_table [concrete = constants.%GenericAndParams.impl_witness.194]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %GenericAndParams.type
// CHECK:STDOUT:   witness = %GenericAndParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericNoParams.impl: %X.ref.loc17_6 as %GenericNoParams.ref {
// CHECK:STDOUT:   %GenericNoParams.impl_witness_table = impl_witness_table (), @X.as.GenericNoParams.impl [concrete]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness %GenericNoParams.impl_witness_table [concrete = constants.%GenericNoParams.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %GenericNoParams.ref
// CHECK:STDOUT:   witness = %GenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.918: %X.ref.loc18_6 as %GenericAndParams.type {
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table = impl_witness_table (), @X.as.GenericAndParams.impl.918 [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness: <witness> = impl_witness %GenericAndParams.impl_witness_table [concrete = constants.%GenericAndParams.impl_witness.d85]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %GenericAndParams.type
// CHECK:STDOUT:   witness = %GenericAndParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_10.2: type) {
// CHECK:STDOUT:   %T.loc8_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T.loc8_10.1)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.41a)]
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_interface_type @GenericAndParams.loc10, @C(%T.loc8_10.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.3a6)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.3a6) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.83d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [symbolic = @C.%GenericNoParams.type (constants.%GenericNoParams.type.41a)] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.3a6) = interface_decl @GenericAndParams.loc10 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.83d)] {
// CHECK:STDOUT:       %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_34.1: type = splice_block %.loc10_34.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc10_34.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc10_31.2: type = symbolic_binding U, 1 [symbolic = %U.loc10_31.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1f9
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
// CHECK:STDOUT: specific @NotGenericNoParams.WithSelf(constants.%Self.52c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericButParams.WithSelf(constants.%Self.e41) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%T) {
// CHECK:STDOUT:   %T.loc6_29.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc6(constants.%T, constants.%Self.651) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams.WithSelf(constants.%T, constants.%Self.be7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc10_31.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc10(constants.%T, constants.%U, constants.%Self.528) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericNoParams.WithSelf(constants.%NotGenericNoParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGenericButParams.WithSelf(constants.%NotGenericButParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X) {
// CHECK:STDOUT:   %T.loc6_29.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.0d2
// CHECK:STDOUT:   %Self.loc6_38.2 => constants.%Self.15c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc6(constants.%X, constants.%Self.651) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc6(constants.%X, constants.%GenericAndParams.facet.845) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.ff6
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.7d5
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.13c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.ff6
// CHECK:STDOUT:   %Self.loc9_29.2 => constants.%Self.018
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams.WithSelf(constants.%X, constants.%Self.be7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams.WithSelf(constants.%X, constants.%GenericNoParams.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc10_31.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.504
// CHECK:STDOUT:   %Self.loc10_40.2 => constants.%Self.500
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc10(constants.%X, constants.%X, constants.%Self.528) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.WithSelf.loc10(constants.%X, constants.%X, constants.%GenericAndParams.facet.f27) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.026: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.026 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.0af = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.026 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.026: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.026 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.0af = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.026 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar.type.d3b: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.d3b = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.1c8: type = facet_type <@Bar> [concrete]
// CHECK:STDOUT:   %Self: %Bar.type.1c8 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.d3b = interface_decl @Bar [concrete = constants.%Bar.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Bar {
// CHECK:STDOUT:   %Self: %Bar.type.1c8 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Bar.WithSelf.decl = interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Bar.type.d3b: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.d3b = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.a09: type = facet_type <@Bar, @Bar(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Bar.type.a09 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.d3b = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%T.loc8_16.1)> [symbolic = %Bar.type (constants.%Bar.type.a09)]
// CHECK:STDOUT:   %Self.loc8_25.2: @Bar.%Bar.type (%Bar.type.a09) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_25.1: @Bar.%Bar.type (%Bar.type.a09) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar [concrete]
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
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
