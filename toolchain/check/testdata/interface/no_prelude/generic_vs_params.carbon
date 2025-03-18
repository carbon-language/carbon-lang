// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic_vs_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic_vs_params.carbon

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
// CHECK:STDOUT:   %Self.238: %NotGenericNoParams.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type.f26: type = generic_interface_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.generic: %NotGenericButParams.type.f26 = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericButParams.type.014: type = facet_type <@NotGenericButParams> [concrete]
// CHECK:STDOUT:   %Self.43b: %NotGenericButParams.type.014 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.cde: type = generic_interface_type @GenericAndParams.1 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.827: %GenericAndParams.type.cde = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.73e: type = facet_type <@GenericAndParams.1, @GenericAndParams.1(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2f4: %GenericAndParams.type.73e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.type.f90: type = facet_type <@GenericNoParams> [concrete]
// CHECK:STDOUT:   %GenericNoParams.type.b79: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0f6: %GenericNoParams.type.b79 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.597: type = generic_interface_type @GenericAndParams.2, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.2ec: %GenericAndParams.type.597 = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.d96: type = facet_type <@GenericAndParams.2, @GenericAndParams.2(%T, %U)> [symbolic]
// CHECK:STDOUT:   %Self.37a: %GenericAndParams.type.d96 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.4b6: type = facet_type <@GenericAndParams.1, @GenericAndParams.1(%X)> [concrete]
// CHECK:STDOUT:   %C.fac: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.425: type = generic_interface_type @GenericAndParams.2, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.b50: %GenericAndParams.type.425 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.79c: type = facet_type <@GenericAndParams.2, @GenericAndParams.2(%X, %X)> [concrete]
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
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.cde = interface_decl @GenericAndParams.1 [concrete = constants.%GenericAndParams.generic.827] {
// CHECK:STDOUT:     %T.patt.loc6_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_28.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_28.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc8_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.c96 [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, file.%NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.f59 [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type.f26 = name_ref NotGenericButParams, file.%NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams.type: type = facet_type <@NotGenericButParams> [concrete = constants.%NotGenericButParams.type.014]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.87f [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.cde = name_ref GenericAndParams, file.%GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.827]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.1, @GenericAndParams.1(constants.%X)> [concrete = constants.%GenericAndParams.type.4b6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.3f8 [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, @C.%GenericNoParams.decl [concrete = constants.%GenericNoParams.type.f90]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.15f [concrete] {} {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %.loc18: %GenericAndParams.type.425 = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.b50]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.425 = name_ref GenericAndParams, %.loc18 [concrete = constants.%GenericAndParams.generic.b50]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.type: type = facet_type <@GenericAndParams.2, @GenericAndParams.2(constants.%X, constants.%X)> [concrete = constants.%GenericAndParams.type.79c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %NotGenericNoParams.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.238]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %NotGenericButParams.type.014 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.43b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.1(%T.loc6_28.1: type) {
// CHECK:STDOUT:   %T.loc6_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_28.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_28.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.1, @GenericAndParams.1(%T.loc6_28.2)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.73e)]
// CHECK:STDOUT:   %Self.2: %GenericAndParams.type.73e = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.1.%GenericAndParams.type (%GenericAndParams.type.73e) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericNoParams(@C.%T.loc8_9.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericNoParams.type: type = facet_type <@GenericNoParams, @GenericNoParams(%T)> [symbolic = %GenericNoParams.type (constants.%GenericNoParams.type.b79)]
// CHECK:STDOUT:   %Self.2: %GenericNoParams.type.b79 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericNoParams.%GenericNoParams.type (%GenericNoParams.type.b79) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.2(@C.%T.loc8_9.1: type, %U.loc10_30.1: type) {
// CHECK:STDOUT:   %U.loc10_30.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc10_30.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc10_30.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_30.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericAndParams.type: type = facet_type <@GenericAndParams.2, @GenericAndParams.2(%T, %U.loc10_30.2)> [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.d96)]
// CHECK:STDOUT:   %Self.2: %GenericAndParams.type.d96 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.37a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.2.%GenericAndParams.type (%GenericAndParams.type.d96) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.37a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c96: %X.ref as %NotGenericNoParams.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.f59: %X.ref as %NotGenericButParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.87f: %X.ref.loc16_6 as %GenericAndParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3f8: %X.ref.loc17_6 as %GenericNoParams.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.15f: %X.ref.loc18_6 as %GenericAndParams.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.1: type) {
// CHECK:STDOUT:   %T.loc8_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_interface_type @GenericAndParams.2, @C(%T.loc8_9.2) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.597)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.597) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.2ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [concrete = constants.%GenericNoParams.type.f90] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.597) = interface_decl @GenericAndParams.2 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.2ec)] {
// CHECK:STDOUT:       %U.patt.loc10_30.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_30.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.loc10_30.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc10_30.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%T) {
// CHECK:STDOUT:   %T.loc6_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(%T.loc6_28.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc10_30.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc10_30.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(%T, %U.loc10_30.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc8_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%X) {
// CHECK:STDOUT:   %T.loc6_28.2 => constants.%X
// CHECK:STDOUT:   %T.patt.loc6_28.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.4b6
// CHECK:STDOUT:   %Self.2 => constants.%Self.2f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%X
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.425
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.b50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc10_30.2 => constants.%X
// CHECK:STDOUT:   %U.patt.loc10_30.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.79c
// CHECK:STDOUT:   %Self.2 => constants.%Self.37a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.c7f: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.c7f = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %A.type.c7f = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.c7f: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type.c7f = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %A.type.c7f = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar.type.982: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.982 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.ea3: type = facet_type <@Bar> [concrete]
// CHECK:STDOUT:   %Self: %Bar.type.ea3 = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %Bar.type.ea3 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Bar.type.982: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.982 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.7ed: type = facet_type <@Bar, @Bar(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Bar.type.7ed = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.982 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%T.loc8_15.2)> [symbolic = %Bar.type (constants.%Bar.type.7ed)]
// CHECK:STDOUT:   %Self.2: %Bar.type.7ed = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Bar.%Bar.type (%Bar.type.7ed) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(%T.loc8_15.2) {}
// CHECK:STDOUT:
