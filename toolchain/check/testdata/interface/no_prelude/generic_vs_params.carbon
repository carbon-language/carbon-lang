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

// CHECK:STDERR: fail_non_generic_implicit_params.carbon:[[@LINE+4]]:13: error: parameters of generic types must be constant
// CHECK:STDERR: interface A[T: type]() {}
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
interface A[T: type]() {}

// --- fail_non_generic_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_params.carbon:[[@LINE+3]]:13: error: parameters of generic types must be constant
// CHECK:STDERR: interface A(T: type) {}
// CHECK:STDERR:             ^~~~~~~
interface A(T: type) {}

// CHECK:STDOUT: --- params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @NotGenericNoParams [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type: type = generic_interface_type @NotGenericButParams [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %NotGenericButParams: %NotGenericButParams.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @NotGenericButParams [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.1: type = generic_interface_type @GenericAndParams.1 [template]
// CHECK:STDOUT:   %GenericAndParams.1: %GenericAndParams.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%T) [symbolic]
// CHECK:STDOUT:   %Self.3: %.4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = interface_type @GenericNoParams [template]
// CHECK:STDOUT:   %.6: type = interface_type @GenericNoParams, @GenericNoParams(%T) [symbolic]
// CHECK:STDOUT:   %Self.4: %.6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.2: type = generic_interface_type @GenericAndParams.2, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.2: %GenericAndParams.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%T, %U) [symbolic]
// CHECK:STDOUT:   %Self.5: %.7 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.11: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%X) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%X) [template]
// CHECK:STDOUT:   %GenericAndParams.type.3: type = generic_interface_type @GenericAndParams.2, @C(%X) [template]
// CHECK:STDOUT:   %GenericAndParams.3: %GenericAndParams.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.13: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%X, %X) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NotGenericNoParams = %NotGenericNoParams.decl
// CHECK:STDOUT:     .NotGenericButParams = %NotGenericButParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericNoParams.decl: type = interface_decl @NotGenericNoParams [template = constants.%.1] {} {}
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type = interface_decl @NotGenericButParams [template = constants.%NotGenericButParams] {} {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.1 = interface_decl @GenericAndParams.1 [template = constants.%GenericAndParams.1] {
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc6, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc8: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc8, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, file.%NotGenericNoParams.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type = name_ref NotGenericButParams, file.%NotGenericButParams.decl [template = constants.%NotGenericButParams]
// CHECK:STDOUT:     %.loc15_30: type = interface_type @NotGenericButParams [template = constants.%.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.1 = name_ref GenericAndParams, file.%GenericAndParams.decl [template = constants.%GenericAndParams.1]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc16_27: type = interface_type @GenericAndParams.1, @GenericAndParams.1(constants.%X) [template = constants.%.11]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [template = constants.%C.3]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, @C.%GenericNoParams.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%X) [template = constants.%C.3]
// CHECK:STDOUT:     %.loc18_15: %GenericAndParams.type.3 = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [template = constants.%GenericAndParams.3]
// CHECK:STDOUT:     %GenericAndParams.ref: %GenericAndParams.type.3 = name_ref GenericAndParams, %.loc18_15 [template = constants.%GenericAndParams.3]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc18_32: type = interface_type @GenericAndParams.2, @GenericAndParams.2(constants.%X, constants.%X) [template = constants.%.13]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.1(%T.loc6: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%T.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.1.%.1 (%.4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericNoParams(@C.%T.loc8: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @GenericNoParams, @GenericNoParams(%T) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericNoParams.%.1 (%.6) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.2(@C.%T.loc8: type, %U.loc10: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%T, %U.1) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self.2: %.7 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.2.%.1 (%.7) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %X as %.1 {
// CHECK:STDOUT:   %.loc14: <witness> = interface_witness () [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X as %.3 {
// CHECK:STDOUT:   %.loc15_33: <witness> = interface_witness () [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc15_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %X as %.11 {
// CHECK:STDOUT:   %.loc16_31: <witness> = interface_witness () [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc16_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %X as %.5 {
// CHECK:STDOUT:   %.loc17: <witness> = interface_witness () [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %X as %.13 {
// CHECK:STDOUT:   %.loc18_36: <witness> = interface_witness () [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc18_36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_interface_type @GenericAndParams.2, @C(%T.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.2)]
// CHECK:STDOUT:   %GenericAndParams: @C.%GenericAndParams.type (%GenericAndParams.type.2) = struct_value () [symbolic = %GenericAndParams (constants.%GenericAndParams.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [template = constants.%.5] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.2) = interface_decl @GenericAndParams.2 [symbolic = @C.%GenericAndParams (constants.%GenericAndParams.2)] {
// CHECK:STDOUT:       %U.patt.loc10: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = param_pattern %U.patt.loc10, runtime_param<invalid> [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %param: type = param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc10: type = bind_symbolic_name U, 1, %param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(@GenericAndParams.1.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(@GenericNoParams.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %U.patt.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(@GenericAndParams.2.%T, @GenericAndParams.2.%U.1) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %U.patt.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@C.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%X) {
// CHECK:STDOUT:   %T.1 => constants.%X
// CHECK:STDOUT:   %T.patt.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.1 => constants.%X
// CHECK:STDOUT:   %T.patt.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.3
// CHECK:STDOUT:   %GenericAndParams => constants.%GenericAndParams.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.1 => constants.%X
// CHECK:STDOUT:   %U.patt.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %Self.2 => constants.%Self.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = generic_interface_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = interface_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %T.patt: type = binding_pattern T
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt, runtime_param<invalid>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T: type = bind_name T, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = generic_interface_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = interface_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %T.patt: type = binding_pattern T
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt, runtime_param<invalid>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T: type = bind_name T, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
