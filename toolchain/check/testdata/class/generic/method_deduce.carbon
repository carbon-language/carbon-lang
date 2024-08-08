// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon

class A {}
class B {}

class Class(T:! type) {
  fn Get(U:! type) -> (T, U);
  fn GetNoDeduce(x: T, U:! type) -> (T, U);
}

fn CallGenericMethod(c: Class(A)) -> (A, B) {
  return c.Get(B);
}

fn CallGenericMethodWithNonDeducedParam(c: Class(A)) -> (A, B) {
  return c.GetNoDeduce({}, B);
}

// CHECK:STDOUT: --- method_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %Get.type.1: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.1: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.type.1: type = fn_type @GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.1: %GetNoDeduce.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%A) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%A, %B) [template]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [template]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.6, %.6) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %Get.type.2: type = fn_type @Get, @Class(%A) [template]
// CHECK:STDOUT:   %Get.2: %Get.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GetNoDeduce.type.2: type = fn_type @GetNoDeduce, @Class(%A) [template]
// CHECK:STDOUT:   %GetNoDeduce.2: %GetNoDeduce.type.2 = struct_value () [template]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [template]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .CallGenericMethodWithNonDeducedParam = %CallGenericMethodWithNonDeducedParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc14_13.1: type = param T
// CHECK:STDOUT:     %T.loc14_13.2: type = bind_symbolic_name T 0, %T.loc14_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [template = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %Class.ref.loc19: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %A.ref.loc19_31: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc19_30: init type = call %Class.ref.loc19(%A.ref.loc19_31) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc19_32.1: type = value_of_initializer %.loc19_30 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc19_32.2: type = converted %.loc19_30, %.loc19_32.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %c.loc19_22.1: %Class.3 = param c
// CHECK:STDOUT:     @CallGenericMethod.%c: %Class.3 = bind_name c, %c.loc19_22.1
// CHECK:STDOUT:     %A.ref.loc19_39: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %B.ref.loc19: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc19_43.1: %.3 = tuple_literal (%A.ref.loc19_39, %B.ref.loc19)
// CHECK:STDOUT:     %.loc19_43.2: type = converted %.loc19_43.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:     @CallGenericMethod.%return: ref %.5 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [template = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %Class.ref.loc23: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %A.ref.loc23_50: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc23_49: init type = call %Class.ref.loc23(%A.ref.loc23_50) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc23_51.1: type = value_of_initializer %.loc23_49 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc23_51.2: type = converted %.loc23_49, %.loc23_51.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %c.loc23_41.1: %Class.3 = param c
// CHECK:STDOUT:     @CallGenericMethodWithNonDeducedParam.%c: %Class.3 = bind_name c, %c.loc23_41.1
// CHECK:STDOUT:     %A.ref.loc23_58: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc23_62.1: %.3 = tuple_literal (%A.ref.loc23_58, %B.ref.loc23)
// CHECK:STDOUT:     %.loc23_62.2: type = converted %.loc23_62.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:     @CallGenericMethodWithNonDeducedParam.%return: ref %.5 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc14_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.1)]
// CHECK:STDOUT:   %GetNoDeduce: @Class.%GetNoDeduce.type (%GetNoDeduce.type.1) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.1) = fn_decl @Get [symbolic = %Get (constants.%Get.1)] {
// CHECK:STDOUT:       %U.loc15_10.1: type = param U
// CHECK:STDOUT:       %U.loc15_10.2: type = bind_symbolic_name U 1, %U.loc15_10.1 [symbolic = @Get.%U.1 (constants.%U)]
// CHECK:STDOUT:       %T.ref.loc15: type = name_ref T, file.%T.loc14_13.2 [symbolic = @Get.%T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc15: type = name_ref U, %U.loc15_10.2 [symbolic = @Get.%U.1 (constants.%U)]
// CHECK:STDOUT:       %.loc15_28.1: %.3 = tuple_literal (%T.ref.loc15, %U.ref.loc15)
// CHECK:STDOUT:       %.loc15_28.2: type = converted %.loc15_28.1, constants.%.4 [symbolic = @Get.%.1 (constants.%.4)]
// CHECK:STDOUT:       %return.var.loc15: ref @Get.%.1 (%.4) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetNoDeduce.decl: @Class.%GetNoDeduce.type (%GetNoDeduce.type.1) = fn_decl @GetNoDeduce [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)] {
// CHECK:STDOUT:       %T.ref.loc16_21: type = name_ref T, file.%T.loc14_13.2 [symbolic = @GetNoDeduce.%T (constants.%T)]
// CHECK:STDOUT:       %x.loc16_18.1: @GetNoDeduce.%T (%T) = param x
// CHECK:STDOUT:       %x.loc16_18.2: @GetNoDeduce.%T (%T) = bind_name x, %x.loc16_18.1
// CHECK:STDOUT:       %U.loc16_24.1: type = param U
// CHECK:STDOUT:       %U.loc16_24.2: type = bind_symbolic_name U 1, %U.loc16_24.1 [symbolic = @GetNoDeduce.%U.1 (constants.%U)]
// CHECK:STDOUT:       %T.ref.loc16_38: type = name_ref T, file.%T.loc14_13.2 [symbolic = @GetNoDeduce.%T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc16: type = name_ref U, %U.loc16_24.2 [symbolic = @GetNoDeduce.%U.1 (constants.%U)]
// CHECK:STDOUT:       %.loc16_42.1: %.3 = tuple_literal (%T.ref.loc16_38, %U.ref.loc16)
// CHECK:STDOUT:       %.loc16_42.2: type = converted %.loc16_42.1, constants.%.4 [symbolic = @GetNoDeduce.%.1 (constants.%.4)]
// CHECK:STDOUT:       %return.var.loc16: ref @GetNoDeduce.%.1 (%.4) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %GetNoDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(file.%T.loc14_13.2: type, @Class.%U.loc15_10.2: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = tuple_type (@Get.%T (%T), @Get.%U.1 (%U)) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@Class.%U.loc15_10.2: type) -> @Get.%.1 (%.4);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetNoDeduce(file.%T.loc14_13.2: type, @Class.%U.loc16_24.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %.1: type = tuple_type (@GetNoDeduce.%T (%T), @GetNoDeduce.%U.1 (%U)) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@Class.%x.loc16_18.2: @GetNoDeduce.%T (%T), @Class.%U.loc16_24.2: type) -> @GetNoDeduce.%.1 (%.4);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c: %Class.3) -> %return: %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %.loc20: %Get.type.2 = specific_constant @Class.%Get.decl, @Class(constants.%A) [template = constants.%Get.2]
// CHECK:STDOUT:   %Get.ref: %Get.type.2 = name_ref Get, %.loc20 [template = constants.%Get.2]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc19: ref %.5 = splice_block %return {}
// CHECK:STDOUT:   %Get.call: init %.5 = call %Get.ref(%B.ref) to %.loc19
// CHECK:STDOUT:   return %Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c: %Class.3) -> %return: %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %.loc24_11: %GetNoDeduce.type.2 = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%A) [template = constants.%GetNoDeduce.2]
// CHECK:STDOUT:   %GetNoDeduce.ref: %GetNoDeduce.type.2 = name_ref GetNoDeduce, %.loc24_11 [template = constants.%GetNoDeduce.2]
// CHECK:STDOUT:   %.loc24_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc23: ref %.5 = splice_block %return {}
// CHECK:STDOUT:   %.loc24_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc24_25.3: init %A = class_init (), %.loc24_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc24_25.4: ref %A = temporary %.loc24_25.2, %.loc24_25.3
// CHECK:STDOUT:   %.loc24_23.1: ref %A = converted %.loc24_25.1, %.loc24_25.4
// CHECK:STDOUT:   %.loc24_23.2: %A = bind_value %.loc24_23.1
// CHECK:STDOUT:   %GetNoDeduce.call: init %.5 = call %GetNoDeduce.ref(%.loc24_23.2, %B.ref) to %.loc23
// CHECK:STDOUT:   return %GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.2
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.1 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %U.1 => constants.%B
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
