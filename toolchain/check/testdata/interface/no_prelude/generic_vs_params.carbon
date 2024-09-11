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

// CHECK:STDERR: fail_non_generic_implicit_params.carbon:[[@LINE+4]]:13: ERROR: Parameters of generic types must be constant.
// CHECK:STDERR: interface A[T: type]() {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface A[T: type]() {}

// --- fail_non_generic_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_params.carbon:[[@LINE+3]]:13: ERROR: Parameters of generic types must be constant.
// CHECK:STDERR: interface A(T: type) {}
// CHECK:STDERR:             ^
interface A(T: type) {}

// CHECK:STDOUT: --- params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @NotGenericNoParams [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %NotGenericButParams.type: type = generic_interface_type @NotGenericButParams [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %NotGenericButParams: %NotGenericButParams.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @NotGenericButParams [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.1: type = generic_interface_type @GenericAndParams.1 [template]
// CHECK:STDOUT:   %GenericAndParams.1: %GenericAndParams.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%T) [symbolic]
// CHECK:STDOUT:   %Self.3: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = interface_type @GenericNoParams [template]
// CHECK:STDOUT:   %.6: type = interface_type @GenericNoParams, @GenericNoParams(%T) [symbolic]
// CHECK:STDOUT:   %Self.4: %.6 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.2: type = generic_interface_type @GenericAndParams.2 [template]
// CHECK:STDOUT:   %GenericAndParams.2: %GenericAndParams.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%T, %U) [symbolic]
// CHECK:STDOUT:   %Self.5: %.7 = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {} [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.10: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%X) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%X) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.12: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%X) [template]
// CHECK:STDOUT:   %.13: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%X, %U) [symbolic]
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
// CHECK:STDOUT:   %NotGenericNoParams.decl: type = interface_decl @NotGenericNoParams [template = constants.%.1] {}
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type = interface_decl @NotGenericButParams [template = constants.%NotGenericButParams] {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.1 = interface_decl @GenericAndParams.1 [template = constants.%GenericAndParams.1] {
// CHECK:STDOUT:     %T.loc6_28.1: type = param T
// CHECK:STDOUT:     %T.loc6_28.2: type = bind_symbolic_name T 0, %T.loc6_28.1 [symbolic = @GenericAndParams.1.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.loc8_9.1: type = param T
// CHECK:STDOUT:     %T.loc8_9.2: type = bind_symbolic_name T 0, %T.loc8_9.1 [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %X.ref.loc14: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, %NotGenericNoParams.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_31.1: type = value_of_initializer %.loc15_30 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_31.2: type = converted %.loc15_30, %.loc15_31.1 [template = constants.%.3]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %X.ref.loc15: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type = name_ref NotGenericButParams, %NotGenericButParams.decl [template = constants.%NotGenericButParams]
// CHECK:STDOUT:     %.loc15_30: init type = call %NotGenericButParams.ref() [template = constants.%.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_29.1: type = value_of_initializer %.loc16_27 [template = constants.%.10]
// CHECK:STDOUT:   %.loc16_29.2: type = converted %.loc16_27, %.loc16_29.1 [template = constants.%.10]
// CHECK:STDOUT:   impl_decl @impl.3 {
// CHECK:STDOUT:     %X.ref.loc16_6: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.ref.loc16: %GenericAndParams.type.1 = name_ref GenericAndParams, %GenericAndParams.decl [template = constants.%GenericAndParams.1]
// CHECK:STDOUT:     %X.ref.loc16_28: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc16_27: init type = call %GenericAndParams.ref.loc16(%X.ref.loc16_28) [template = constants.%.10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 {
// CHECK:STDOUT:     %X.ref.loc17_6: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %C.ref.loc17: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %X.ref.loc17_13: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc17: init type = call %C.ref.loc17(%X.ref.loc17_13) [template = constants.%C.3]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, @C.%GenericNoParams.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_34.1: type = value_of_initializer %.loc18_32 [template = constants.%.12]
// CHECK:STDOUT:   %.loc18_34.2: type = converted %.loc18_32, %.loc18_34.1 [template = constants.%.12]
// CHECK:STDOUT:   impl_decl @impl.5 {
// CHECK:STDOUT:     %X.ref.loc18_6: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %C.ref.loc18: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %X.ref.loc18_13: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc18_12: init type = call %C.ref.loc18(%X.ref.loc18_13) [template = constants.%C.3]
// CHECK:STDOUT:     %GenericAndParams.ref.loc18: %GenericAndParams.type.2 = name_ref GenericAndParams, @C.%GenericAndParams.decl [template = constants.%GenericAndParams.2]
// CHECK:STDOUT:     %X.ref.loc18_33: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc18_32: init type = call %GenericAndParams.ref.loc18(%X.ref.loc18_33) [template = constants.%.12]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericNoParams {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGenericButParams {
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.1(file.%T.loc6_28.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @GenericAndParams.1, @GenericAndParams.1(%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericNoParams(file.%T.loc8_9.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @GenericNoParams, @GenericNoParams(%T) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericNoParams.%.1 (%.6) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericAndParams.2(file.%T.loc8_9.2: type, @C.%U.loc10_30.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @GenericAndParams.2, @GenericAndParams.2(%T, %U) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self.2: %.7 = bind_symbolic_name Self 2 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericAndParams.2.%.1 (%.7) = bind_symbolic_name Self 2 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %X as %.1 {
// CHECK:STDOUT:   %.loc14: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %X as %.3 {
// CHECK:STDOUT:   %.loc15: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %X as %.10 {
// CHECK:STDOUT:   %.loc16: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %X as %.5 {
// CHECK:STDOUT:   %.loc17: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %X as %.12 {
// CHECK:STDOUT:   %.loc18: <witness> = interface_witness () [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(file.%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = interface_decl @GenericNoParams [template = constants.%.5] {}
// CHECK:STDOUT:     %GenericAndParams.decl: %GenericAndParams.type.2 = interface_decl @GenericAndParams.2 [template = constants.%GenericAndParams.2] {
// CHECK:STDOUT:       %U.loc10_30.1: type = param U
// CHECK:STDOUT:       %U.loc10_30.2: type = bind_symbolic_name U 1, %U.loc10_30.1 [symbolic = @GenericAndParams.2.%U (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(@GenericAndParams.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(@GenericNoParams.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(@GenericAndParams.2.%T, @GenericAndParams.2.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%X) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %Self.2 => constants.%Self.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%X, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = generic_interface_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = interface_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %T.loc8_13.1: type = param T
// CHECK:STDOUT:     %T.loc8_13.2: type = bind_name T, %T.loc8_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
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
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = interface_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_name T, %T.loc7_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
