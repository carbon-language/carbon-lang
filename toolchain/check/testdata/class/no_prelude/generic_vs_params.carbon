// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/generic_vs_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/generic_vs_params.carbon

class NotGenericNoParams {}
class NotGenericButParams() {}
class GenericAndParams(T:! type) {}

class C(T:! type) {
  class GenericNoParams {}
  class GenericAndParams(U:! type) {}
}

class X {}

var a: NotGenericNoParams = {};
var b: NotGenericButParams() = {};
var c: GenericAndParams(X) = {};
var d: C(X).GenericNoParams = {};
var e: C(X).GenericAndParams(X) = {};

// CHECK:STDOUT: --- generic_vs_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGenericNoParams: type = class_type @NotGenericNoParams [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %NotGenericButParams.type: type = generic_class_type @NotGenericButParams [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %NotGenericButParams.1: %NotGenericButParams.type = struct_value () [template]
// CHECK:STDOUT:   %NotGenericButParams.2: type = class_type @NotGenericButParams [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.1: type = generic_class_type @GenericAndParams.1 [template]
// CHECK:STDOUT:   %GenericAndParams.1: %GenericAndParams.type.1 = struct_value () [template]
// CHECK:STDOUT:   %GenericAndParams.2: type = class_type @GenericAndParams.1, @GenericAndParams.1(%T) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.1: type = class_type @GenericNoParams [template]
// CHECK:STDOUT:   %GenericNoParams.2: type = class_type @GenericNoParams, @GenericNoParams(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.2: type = generic_class_type @GenericAndParams.2 [template]
// CHECK:STDOUT:   %GenericAndParams.3: %GenericAndParams.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GenericAndParams.4: type = class_type @GenericAndParams.2, @GenericAndParams.2(%T, %U) [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %NotGenericNoParams = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %NotGenericButParams.2 = struct_value () [template]
// CHECK:STDOUT:   %GenericAndParams.5: type = class_type @GenericAndParams.1, @GenericAndParams.1(%X) [template]
// CHECK:STDOUT:   %struct.3: %GenericAndParams.5 = struct_value () [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%X) [template]
// CHECK:STDOUT:   %struct.4: %GenericNoParams.1 = struct_value () [template]
// CHECK:STDOUT:   %GenericAndParams.6: type = class_type @GenericAndParams.2, @GenericAndParams.2(%X) [template]
// CHECK:STDOUT:   %struct.5: %GenericAndParams.6 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NotGenericNoParams = %NotGenericNoParams.decl
// CHECK:STDOUT:     .NotGenericButParams = %NotGenericButParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericNoParams.decl: type = class_decl @NotGenericNoParams [template = constants.%NotGenericNoParams] {}
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type = class_decl @NotGenericButParams [template = constants.%NotGenericButParams.1] {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.1 = class_decl @GenericAndParams.1 [template = constants.%GenericAndParams.1] {
// CHECK:STDOUT:     %T.loc13_24.1: type = param T
// CHECK:STDOUT:     %T.loc13_24.2: type = bind_symbolic_name T 0, %T.loc13_24.1 [symbolic = @GenericAndParams.1.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.loc15_9.1: type = param T
// CHECK:STDOUT:     %T.loc15_9.2: type = bind_symbolic_name T 0, %T.loc15_9.1 [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, %NotGenericNoParams.decl [template = constants.%NotGenericNoParams]
// CHECK:STDOUT:   %a.var: ref %NotGenericNoParams = var a
// CHECK:STDOUT:   %a: ref %NotGenericNoParams = bind_name a, %a.var
// CHECK:STDOUT:   %NotGenericButParams.ref: %NotGenericButParams.type = name_ref NotGenericButParams, %NotGenericButParams.decl [template = constants.%NotGenericButParams.1]
// CHECK:STDOUT:   %.loc23_27: init type = call %NotGenericButParams.ref() [template = constants.%NotGenericButParams.2]
// CHECK:STDOUT:   %.loc23_28.1: type = value_of_initializer %.loc23_27 [template = constants.%NotGenericButParams.2]
// CHECK:STDOUT:   %.loc23_28.2: type = converted %.loc23_27, %.loc23_28.1 [template = constants.%NotGenericButParams.2]
// CHECK:STDOUT:   %b.var: ref %NotGenericButParams.2 = var b
// CHECK:STDOUT:   %b: ref %NotGenericButParams.2 = bind_name b, %b.var
// CHECK:STDOUT:   %GenericAndParams.ref.loc24: %GenericAndParams.type.1 = name_ref GenericAndParams, %GenericAndParams.decl [template = constants.%GenericAndParams.1]
// CHECK:STDOUT:   %X.ref.loc24: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc24_24: init type = call %GenericAndParams.ref.loc24(%X.ref.loc24) [template = constants.%GenericAndParams.5]
// CHECK:STDOUT:   %.loc24_26.1: type = value_of_initializer %.loc24_24 [template = constants.%GenericAndParams.5]
// CHECK:STDOUT:   %.loc24_26.2: type = converted %.loc24_24, %.loc24_26.1 [template = constants.%GenericAndParams.5]
// CHECK:STDOUT:   %c.var: ref %GenericAndParams.5 = var c
// CHECK:STDOUT:   %c: ref %GenericAndParams.5 = bind_name c, %c.var
// CHECK:STDOUT:   %C.ref.loc25: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:   %X.ref.loc25: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc25: init type = call %C.ref.loc25(%X.ref.loc25) [template = constants.%C.3]
// CHECK:STDOUT:   %GenericNoParams.ref: type = name_ref GenericNoParams, @C.%GenericNoParams.decl [template = constants.%GenericNoParams.1]
// CHECK:STDOUT:   %d.var: ref %GenericNoParams.1 = var d
// CHECK:STDOUT:   %d: ref %GenericNoParams.1 = bind_name d, %d.var
// CHECK:STDOUT:   %C.ref.loc26: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:   %X.ref.loc26_10: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc26_9: init type = call %C.ref.loc26(%X.ref.loc26_10) [template = constants.%C.3]
// CHECK:STDOUT:   %GenericAndParams.ref.loc26: %GenericAndParams.type.2 = name_ref GenericAndParams, @C.%GenericAndParams.decl [template = constants.%GenericAndParams.3]
// CHECK:STDOUT:   %X.ref.loc26_30: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc26_29: init type = call %GenericAndParams.ref.loc26(%X.ref.loc26_30) [template = constants.%GenericAndParams.6]
// CHECK:STDOUT:   %.loc26_31.1: type = value_of_initializer %.loc26_29 [template = constants.%GenericAndParams.6]
// CHECK:STDOUT:   %.loc26_31.2: type = converted %.loc26_29, %.loc26_31.1 [template = constants.%GenericAndParams.6]
// CHECK:STDOUT:   %e.var: ref %GenericAndParams.6 = var e
// CHECK:STDOUT:   %e: ref %GenericAndParams.6 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericNoParams {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericNoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericButParams {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericButParams.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.1(file.%T.loc13_24.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(file.%T.loc15_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = class_decl @GenericNoParams [template = constants.%GenericNoParams.1] {}
// CHECK:STDOUT:     %GenericAndParams.decl: %GenericAndParams.type.2 = class_decl @GenericAndParams.2 [template = constants.%GenericAndParams.3] {
// CHECK:STDOUT:       %U.loc17_26.1: type = param U
// CHECK:STDOUT:       %U.loc17_26.2: type = bind_symbolic_name U 1, %U.loc17_26.1 [symbolic = @GenericAndParams.2.%U (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericNoParams(file.%T.loc15_9.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericNoParams.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.2(file.%T.loc15_9.2: type, @C.%U.loc17_26.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_30.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc22_30.2: init %NotGenericNoParams = class_init (), file.%a.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc22_31: init %NotGenericNoParams = converted %.loc22_30.1, %.loc22_30.2 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc22_31
// CHECK:STDOUT:   %.loc23_33.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc23_33.2: init %NotGenericButParams.2 = class_init (), file.%b.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc23_34: init %NotGenericButParams.2 = converted %.loc23_33.1, %.loc23_33.2 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc23_34
// CHECK:STDOUT:   %.loc24_31.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc24_31.2: init %GenericAndParams.5 = class_init (), file.%c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc24_32: init %GenericAndParams.5 = converted %.loc24_31.1, %.loc24_31.2 [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%c.var, %.loc24_32
// CHECK:STDOUT:   %.loc25_32.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc25_32.2: init %GenericNoParams.1 = class_init (), file.%d.var [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc25_33: init %GenericNoParams.1 = converted %.loc25_32.1, %.loc25_32.2 [template = constants.%struct.4]
// CHECK:STDOUT:   assign file.%d.var, %.loc25_33
// CHECK:STDOUT:   %.loc26_36.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc26_36.2: init %GenericAndParams.6 = class_init (), file.%e.var [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc26_37: init %GenericAndParams.6 = converted %.loc26_36.1, %.loc26_36.2 [template = constants.%struct.5]
// CHECK:STDOUT:   assign file.%e.var, %.loc26_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.1(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
