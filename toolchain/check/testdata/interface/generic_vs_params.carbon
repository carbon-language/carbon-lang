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
// CHECK:STDOUT:   %Self.f47: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type.f26: type = generic_interface_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.generic: %NotGenericButParams.type.f26 = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericButParams.type.5a1: type = facet_type <@NotGenericButParams> [concrete]
// CHECK:STDOUT:   %Self.109: %NotGenericButParams.type.5a1 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.cde: type = generic_interface_type @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.827: %GenericAndParams.type.cde = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.741: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a56: %GenericAndParams.type.741 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.type.672: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic]
// CHECK:STDOUT:   %Self.ed2: %GenericNoParams.type.672 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.677: type = generic_interface_type @GenericAndParams.loc10, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.571: %GenericAndParams.type.677 = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.506: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U)> [symbolic]
// CHECK:STDOUT:   %Self.202: %GenericAndParams.type.506 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness: <witness> = impl_witness file.%NotGenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness file.%NotGenericButParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.674: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%X)> [concrete]
// CHECK:STDOUT:   %Self.1d4: %GenericAndParams.type.674 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.4a7: <witness> = impl_witness file.%GenericAndParams.impl_witness_table.loc16 [concrete]
// CHECK:STDOUT:   %C.fac: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericNoParams.type.7ce: type = facet_type <@GenericNoParams, @GenericNoParams(%X)> [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.425: type = generic_interface_type @GenericAndParams.loc10, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.b50: %GenericAndParams.type.425 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.96e: %GenericNoParams.type.7ce = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness file.%GenericNoParams.impl_witness_table [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.e77: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%X, %X)> [concrete]
// CHECK:STDOUT:   %Self.b70: %GenericAndParams.type.e77 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.87c: <witness> = impl_witness file.%GenericAndParams.impl_witness_table.loc18 [concrete]
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
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type.f26 = interface_decl @NotGenericButParams [concrete = constants.%NotGenericButParams.generic] {} {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.cde = interface_decl @GenericAndParams.loc6 [concrete = constants.%GenericAndParams.generic.827] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_28.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @X.as.NotGenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, file.%NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness_table = impl_witness_table (), @X.as.NotGenericNoParams.impl [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.impl_witness: <witness> = impl_witness %NotGenericNoParams.impl_witness_table [concrete = constants.%NotGenericNoParams.impl_witness]
// CHECK:STDOUT:   impl_decl @X.as.NotGenericButParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type.f26 = name_ref NotGenericButParams, file.%NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams.type: type = facet_type <@NotGenericButParams> [concrete = constants.%NotGenericButParams.type.5a1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericButParams.impl_witness_table = impl_witness_table (), @X.as.NotGenericButParams.impl [concrete]
// CHECK:STDOUT:   %NotGenericButParams.impl_witness: <witness> = impl_witness %NotGenericButParams.impl_witness_table [concrete = constants.%NotGenericButParams.impl_witness]
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.19c [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.cde = name_ref GenericAndParams, file.%GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.827]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(constants.%X)> [concrete = constants.%GenericAndParams.type.674]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table.loc16 = impl_witness_table (), @X.as.GenericAndParams.impl.19c [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.loc16: <witness> = impl_witness %GenericAndParams.impl_witness_table.loc16 [concrete = constants.%GenericAndParams.impl_witness.4a7]
// CHECK:STDOUT:   impl_decl @X.as.GenericNoParams.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %.loc17: type = specific_constant @C.%GenericNoParams.decl, @C(constants.%X) [concrete = constants.%GenericNoParams.type.7ce]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, %.loc17 [concrete = constants.%GenericNoParams.type.7ce]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericNoParams.impl_witness_table = impl_witness_table (), @X.as.GenericNoParams.impl [concrete]
// CHECK:STDOUT:   %GenericNoParams.impl_witness: <witness> = impl_witness %GenericNoParams.impl_witness_table [concrete = constants.%GenericNoParams.impl_witness]
// CHECK:STDOUT:   impl_decl @X.as.GenericAndParams.impl.6a2 [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %.loc18: %GenericAndParams.type.425 = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.b50]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.425 = name_ref GenericAndParams, %.loc18 [concrete = constants.%GenericAndParams.generic.b50]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(constants.%X, constants.%X)> [concrete = constants.%GenericAndParams.type.e77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericAndParams.impl_witness_table.loc18 = impl_witness_table (), @X.as.GenericAndParams.impl.6a2 [concrete]
// CHECK:STDOUT:   %GenericAndParams.impl_witness.loc18: <witness> = impl_witness %GenericAndParams.impl_witness_table.loc18 [concrete = constants.%GenericAndParams.impl_witness.87c]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %NotGenericNoParams.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f47]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %NotGenericButParams.type.5a1 = symbolic_binding Self, 0 [symbolic = constants.%Self.109]
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
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc6, @GenericAndParams.loc6(%T.loc6_28.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.741)]
// CHECK:STDOUT:   %Self.loc6_38.2: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.741) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.a56)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_38.1: @GenericAndParams.loc6.%GenericAndParams.type (%GenericAndParams.type.741) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.a56)]
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
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.672)]
// CHECK:STDOUT:   %Self.loc9_29.2: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.672) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.ed2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc9_29.1: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.672) = symbolic_binding Self, 1 [symbolic = %Self.loc9_29.2 (constants.%Self.ed2)]
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
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U.loc10_30.1)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.506)]
// CHECK:STDOUT:   %Self.loc10_40.2: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.506) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.202)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc10_40.1: @GenericAndParams.loc10.%GenericAndParams.type (%GenericAndParams.type.506) = symbolic_binding Self, 2 [symbolic = %Self.loc10_40.2 (constants.%Self.202)]
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
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%NotGenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.NotGenericButParams.impl: %X.ref as %NotGenericButParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%NotGenericButParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.19c: %X.ref.loc16_6 as %GenericAndParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%GenericAndParams.impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericNoParams.impl: %X.ref.loc17_6 as %GenericNoParams.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%GenericNoParams.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.GenericAndParams.impl.6a2: %X.ref.loc18_6 as %GenericAndParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%GenericAndParams.impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T.loc8_9.1)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.672)]
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_interface_type @GenericAndParams.loc10, @C(%T.loc8_9.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.677)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.677) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.571)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [symbolic = @C.%GenericNoParams.type (constants.%GenericNoParams.type.672)] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.677) = interface_decl @GenericAndParams.loc10 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.571)] {
// CHECK:STDOUT:       %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc10_30.2: type = symbolic_binding U, 1 [symbolic = %U.loc10_30.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
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
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%T) {
// CHECK:STDOUT:   %T.loc6_28.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc10_30.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X) {
// CHECK:STDOUT:   %T.loc6_28.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.674
// CHECK:STDOUT:   %Self.loc6_38.2 => constants.%Self.1d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.7ce
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.425
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.b50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericNoParams.type => constants.%GenericNoParams.type.7ce
// CHECK:STDOUT:   %Self.loc9_29.2 => constants.%Self.96e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc10_30.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.e77
// CHECK:STDOUT:   %Self.loc10_40.2 => constants.%Self.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.233: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.233 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.233 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.233: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.233 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type.233 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar.type.982: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.982 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.6b3: type = facet_type <@Bar> [concrete]
// CHECK:STDOUT:   %Self: %Bar.type.6b3 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.982 = interface_decl @Bar [concrete = constants.%Bar.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Bar {
// CHECK:STDOUT:   %Self: %Bar.type.6b3 = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Bar.type.982: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.982 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.70d: type = facet_type <@Bar, @Bar(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Bar.type.70d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.982 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%T.loc8_15.1)> [symbolic = %Bar.type (constants.%Bar.type.70d)]
// CHECK:STDOUT:   %Self.loc8_25.2: @Bar.%Bar.type (%Bar.type.70d) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_25.1: @Bar.%Bar.type (%Bar.type.70d) = symbolic_binding Self, 1 [symbolic = %Self.loc8_25.2 (constants.%Self)]
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
