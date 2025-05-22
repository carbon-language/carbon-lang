// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon

class A {}
class B {}

class Class(T:! type) {
  fn Get(U:! type) -> (T, U) { return Get(U); }
  fn GetNoDeduce(x: T, U:! type) -> (T, U) { return GetNoDeduce(x, U); }
}

fn CallGenericMethod(c: Class(A)) -> (A, B) {
  //@dump-sem-ir-begin
  return c.Get(B);
  //@dump-sem-ir-end
}

fn CallGenericMethodWithNonDeducedParam(c: Class(A)) -> (A, B) {
  //@dump-sem-ir-begin
  return c.GetNoDeduce({}, B);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- method_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Class.480: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %tuple.type.cc6: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %Get.type.501: type = fn_type @Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Get.f37: %Get.type.501 = struct_value () [concrete]
// CHECK:STDOUT:   %GetNoDeduce.type.5d6: type = fn_type @GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %GetNoDeduce.162: %GetNoDeduce.type.5d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn.213: <specific function> = specific_function %Get.f37, @Get(%A, %B) [concrete]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.438: <specific function> = specific_function %GetNoDeduce.162, @GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc23: %Get.type.501 = specific_constant @Class.%Get.decl, @Class(constants.%A) [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %Get.ref: %Get.type.501 = name_ref Get, %.loc23 [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%A, constants.%B) [concrete = constants.%Get.specific_fn.213]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Get.call: init %tuple.type.cc6 = call %Get.specific_fn() to %.loc21_35
// CHECK:STDOUT:   return %Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc29_11: %GetNoDeduce.type.5d6 = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %GetNoDeduce.ref: %GetNoDeduce.type.5d6 = name_ref GetNoDeduce, %.loc29_11 [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %.loc29_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %B.ref.loc29: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%GetNoDeduce.specific_fn.438]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc29_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc29_25.3: init %A = class_init (), %.loc29_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc29_25.4: ref %A = temporary %.loc29_25.2, %.loc29_25.3
// CHECK:STDOUT:   %.loc29_25.5: ref %A = converted %.loc29_25.1, %.loc29_25.4
// CHECK:STDOUT:   %.loc29_25.6: %A = bind_value %.loc29_25.5
// CHECK:STDOUT:   %GetNoDeduce.call: init %tuple.type.cc6 = call %GetNoDeduce.specific_fn(%.loc29_25.6) to %.loc27_54
// CHECK:STDOUT:   return %GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
