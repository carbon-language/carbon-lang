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
  fn Get(U:! type) -> (T, U) { return Get(U); }
  fn GetNoDeduce(x: T, U:! type) -> (T, U) { return GetNoDeduce(x, U); }
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
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %Get.type.fd9: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.cf9: %Get.type.fd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.type.766: type = fn_type @GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.c9a: %GetNoDeduce.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %Get.specific_fn.f73: <specific function> = specific_function %Get.cf9, @Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.536: <specific function> = specific_function %GetNoDeduce.c9a, @GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.480: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %tuple.type.cc6: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Get.type.501: type = fn_type @Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Get.f37: %Get.type.501 = struct_value () [concrete]
// CHECK:STDOUT:   %GetNoDeduce.type.5d6: type = fn_type @GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %GetNoDeduce.162: %GetNoDeduce.type.5d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn.213: <specific function> = specific_function %Get.f37, @Get(%A, %B) [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.438: <specific function> = specific_function %GetNoDeduce.162, @GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.56a: <witness> = complete_type_witness %tuple.type.cc6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .CallGenericMethodWithNonDeducedParam = %CallGenericMethodWithNonDeducedParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc14_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %c.patt: %Class.480 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.480 = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.cc6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.cc6 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc19_39: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc19: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc19_43.1: %tuple.type.24b = tuple_literal (%A.ref.loc19_39, %B.ref.loc19)
// CHECK:STDOUT:     %.loc19_43.2: type = converted %.loc19_43.1, constants.%tuple.type.cc6 [concrete = constants.%tuple.type.cc6]
// CHECK:STDOUT:     %c.param: %Class.480 = value_param call_param0
// CHECK:STDOUT:     %.loc19_32: type = splice_block %Class [concrete = constants.%Class.480] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc19_31: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.480]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.480 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.cc6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.cc6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [concrete = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %c.patt: %Class.480 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.480 = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.cc6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.cc6 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc23_58: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc23_62.1: %tuple.type.24b = tuple_literal (%A.ref.loc23_58, %B.ref.loc23)
// CHECK:STDOUT:     %.loc23_62.2: type = converted %.loc23_62.1, constants.%tuple.type.cc6 [concrete = constants.%tuple.type.cc6]
// CHECK:STDOUT:     %c.param: %Class.480 = value_param call_param0
// CHECK:STDOUT:     %.loc23_51: type = splice_block %Class [concrete = constants.%Class.480] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc23_50: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.480]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.480 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.cc6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.cc6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc14_13.1: type) {
// CHECK:STDOUT:   %T.loc14_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T.loc14_13.2) [symbolic = %Get.type (constants.%Get.type.fd9)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.fd9) = struct_value () [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T.loc14_13.2) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.766)]
// CHECK:STDOUT:   %GetNoDeduce: @Class.%GetNoDeduce.type (%GetNoDeduce.type.766) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.fd9) = fn_decl @Get [symbolic = @Class.%Get (constants.%Get.cf9)] {
// CHECK:STDOUT:       %U.patt.loc15_10.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_10.2 (constants.%U.patt)]
// CHECK:STDOUT:       %return.patt: @Get.%tuple.type (%tuple.type.30b) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Get.%tuple.type (%tuple.type.30b) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc14_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc15_27: type = name_ref U, %U.loc15_10.2 [symbolic = %U.loc15_10.1 (constants.%U)]
// CHECK:STDOUT:       %.loc15_28.1: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc15_27)
// CHECK:STDOUT:       %.loc15_28.2: type = converted %.loc15_28.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %U.loc15_10.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_10.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Get.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Get.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetNoDeduce.decl: @Class.%GetNoDeduce.type (%GetNoDeduce.type.766) = fn_decl @GetNoDeduce [symbolic = @Class.%GetNoDeduce (constants.%GetNoDeduce.c9a)] {
// CHECK:STDOUT:       %x.patt: @GetNoDeduce.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @GetNoDeduce.%T (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:       %U.patt.loc16_24.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc16_24.2 (constants.%U.patt)]
// CHECK:STDOUT:       %return.patt: @GetNoDeduce.%tuple.type (%tuple.type.30b) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetNoDeduce.%tuple.type (%tuple.type.30b) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc16_38: type = name_ref T, @Class.%T.loc14_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc16_41: type = name_ref U, %U.loc16_24.2 [symbolic = %U.loc16_24.1 (constants.%U)]
// CHECK:STDOUT:       %.loc16_42.1: %tuple.type.24b = tuple_literal (%T.ref.loc16_38, %U.ref.loc16_41)
// CHECK:STDOUT:       %.loc16_42.2: type = converted %.loc16_42.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %x.param: @GetNoDeduce.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc16_21: type = name_ref T, @Class.%T.loc14_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @GetNoDeduce.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:       %U.loc16_24.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc16_24.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %GetNoDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(@Class.%T.loc14_13.1: type, %U.loc15_10.2: type) {
// CHECK:STDOUT:   %U.loc15_10.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_10.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc15_10.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@Get.%T (%T), @Get.%U.loc15_10.1 (%U)) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_20: <witness> = require_complete_type @Get.%tuple.type (%tuple.type.30b) [symbolic = %require_complete.loc15_20 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.fd9)]
// CHECK:STDOUT:   %Get: @Get.%Get.type (%Get.type.fd9) = struct_value () [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:   %Get.specific_fn.loc15_39.2: <specific function> = specific_function %Get, @Get(%T, %U.loc15_10.1) [symbolic = %Get.specific_fn.loc15_39.2 (constants.%Get.specific_fn.f73)]
// CHECK:STDOUT:   %require_complete.loc15_44.1: <witness> = require_complete_type @Get.%T (%T) [symbolic = %require_complete.loc15_44.1 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc15_44.2: <witness> = require_complete_type @Get.%U.loc15_10.1 (%U) [symbolic = %require_complete.loc15_44.2 (constants.%require_complete.b54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc15_10.1: type) -> %return.param_patt: @Get.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_39: @Get.%Get.type (%Get.type.fd9) = specific_constant @Class.%Get.decl, @Class(constants.%T) [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:     %Get.ref: @Get.%Get.type (%Get.type.fd9) = name_ref Get, %.loc15_39 [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:     %U.ref.loc15_43: type = name_ref U, %U.loc15_10.2 [symbolic = %U.loc15_10.1 (constants.%U)]
// CHECK:STDOUT:     %Get.specific_fn.loc15_39.1: <specific function> = specific_function %Get.ref, @Get(constants.%T, constants.%U) [symbolic = %Get.specific_fn.loc15_39.2 (constants.%Get.specific_fn.f73)]
// CHECK:STDOUT:     %.loc15_44.1: ref @Get.%tuple.type (%tuple.type.30b) = temporary_storage
// CHECK:STDOUT:     %Get.call: init @Get.%tuple.type (%tuple.type.30b) = call %Get.specific_fn.loc15_39.1() to %.loc15_44.1
// CHECK:STDOUT:     %.loc15_44.2: ref @Get.%tuple.type (%tuple.type.30b) = temporary %.loc15_44.1, %Get.call
// CHECK:STDOUT:     %tuple.elem0.loc15_44.1: ref @Get.%T (%T) = tuple_access %.loc15_44.2, element0
// CHECK:STDOUT:     %.loc15_44.3: @Get.%T (%T) = bind_value %tuple.elem0.loc15_44.1
// CHECK:STDOUT:     %tuple.elem0.loc15_44.2: ref @Get.%T (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc15_44.4: init @Get.%T (%T) = initialize_from %.loc15_44.3 to %tuple.elem0.loc15_44.2
// CHECK:STDOUT:     %tuple.elem1.loc15_44.1: ref @Get.%U.loc15_10.1 (%U) = tuple_access %.loc15_44.2, element1
// CHECK:STDOUT:     %.loc15_44.5: @Get.%U.loc15_10.1 (%U) = bind_value %tuple.elem1.loc15_44.1
// CHECK:STDOUT:     %tuple.elem1.loc15_44.2: ref @Get.%U.loc15_10.1 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc15_44.6: init @Get.%U.loc15_10.1 (%U) = initialize_from %.loc15_44.5 to %tuple.elem1.loc15_44.2
// CHECK:STDOUT:     %.loc15_44.7: init @Get.%tuple.type (%tuple.type.30b) = tuple_init (%.loc15_44.4, %.loc15_44.6) to %return
// CHECK:STDOUT:     %.loc15_45: init @Get.%tuple.type (%tuple.type.30b) = converted %Get.call, %.loc15_44.7
// CHECK:STDOUT:     return %.loc15_45 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetNoDeduce(@Class.%T.loc14_13.1: type, %U.loc16_24.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U.loc16_24.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc16_24.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc16_24.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc16_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@GetNoDeduce.%T (%T), @GetNoDeduce.%U.loc16_24.1 (%U)) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type @GetNoDeduce.%T (%T) [symbolic = %require_complete.loc16_19 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.766)]
// CHECK:STDOUT:   %GetNoDeduce: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc16_53.2: <specific function> = specific_function %GetNoDeduce, @GetNoDeduce(%T, %U.loc16_24.1) [symbolic = %GetNoDeduce.specific_fn.loc16_53.2 (constants.%GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:   %require_complete.loc16_70: <witness> = require_complete_type @GetNoDeduce.%tuple.type (%tuple.type.30b) [symbolic = %require_complete.loc16_70 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %require_complete.loc16_69: <witness> = require_complete_type @GetNoDeduce.%U.loc16_24.1 (%U) [symbolic = %require_complete.loc16_69 (constants.%require_complete.b54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @GetNoDeduce.%T (%T), %U.patt.loc16_24.1: type) -> %return.param_patt: @GetNoDeduce.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_53: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%T) [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = name_ref GetNoDeduce, %.loc16_53 [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:     %x.ref: @GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc16_68: type = name_ref U, %U.loc16_24.2 [symbolic = %U.loc16_24.1 (constants.%U)]
// CHECK:STDOUT:     %GetNoDeduce.specific_fn.loc16_53.1: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%T, constants.%U) [symbolic = %GetNoDeduce.specific_fn.loc16_53.2 (constants.%GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:     %.loc16_69.1: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = temporary_storage
// CHECK:STDOUT:     %GetNoDeduce.call: init @GetNoDeduce.%tuple.type (%tuple.type.30b) = call %GetNoDeduce.specific_fn.loc16_53.1(%x.ref) to %.loc16_69.1
// CHECK:STDOUT:     %.loc16_69.2: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = temporary %.loc16_69.1, %GetNoDeduce.call
// CHECK:STDOUT:     %tuple.elem0.loc16_69.1: ref @GetNoDeduce.%T (%T) = tuple_access %.loc16_69.2, element0
// CHECK:STDOUT:     %.loc16_69.3: @GetNoDeduce.%T (%T) = bind_value %tuple.elem0.loc16_69.1
// CHECK:STDOUT:     %tuple.elem0.loc16_69.2: ref @GetNoDeduce.%T (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc16_69.4: init @GetNoDeduce.%T (%T) = initialize_from %.loc16_69.3 to %tuple.elem0.loc16_69.2
// CHECK:STDOUT:     %tuple.elem1.loc16_69.1: ref @GetNoDeduce.%U.loc16_24.1 (%U) = tuple_access %.loc16_69.2, element1
// CHECK:STDOUT:     %.loc16_69.5: @GetNoDeduce.%U.loc16_24.1 (%U) = bind_value %tuple.elem1.loc16_69.1
// CHECK:STDOUT:     %tuple.elem1.loc16_69.2: ref @GetNoDeduce.%U.loc16_24.1 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc16_69.6: init @GetNoDeduce.%U.loc16_24.1 (%U) = initialize_from %.loc16_69.5 to %tuple.elem1.loc16_69.2
// CHECK:STDOUT:     %.loc16_69.7: init @GetNoDeduce.%tuple.type (%tuple.type.30b) = tuple_init (%.loc16_69.4, %.loc16_69.6) to %return
// CHECK:STDOUT:     %.loc16_70: init @GetNoDeduce.%tuple.type (%tuple.type.30b) = converted %GetNoDeduce.call, %.loc16_69.7
// CHECK:STDOUT:     return %.loc16_70 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param_patt: %Class.480) -> %return.param_patt: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc20: %Get.type.501 = specific_constant @Class.%Get.decl, @Class(constants.%A) [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %Get.ref: %Get.type.501 = name_ref Get, %.loc20 [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %B.ref.loc20: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%A, constants.%B) [concrete = constants.%Get.specific_fn.213]
// CHECK:STDOUT:   %.loc19_35: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %Get.call: init %tuple.type.cc6 = call %Get.specific_fn() to %.loc19_35
// CHECK:STDOUT:   return %Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param_patt: %Class.480) -> %return.param_patt: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc24_11: %GetNoDeduce.type.5d6 = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %GetNoDeduce.ref: %GetNoDeduce.type.5d6 = name_ref GetNoDeduce, %.loc24_11 [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %.loc24_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%GetNoDeduce.specific_fn.438]
// CHECK:STDOUT:   %.loc23_54: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %.loc24_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc24_25.3: init %A = class_init (), %.loc24_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc24_25.4: ref %A = temporary %.loc24_25.2, %.loc24_25.3
// CHECK:STDOUT:   %.loc24_25.5: ref %A = converted %.loc24_25.1, %.loc24_25.4
// CHECK:STDOUT:   %.loc24_25.6: %A = bind_value %.loc24_25.5
// CHECK:STDOUT:   %GetNoDeduce.call: init %tuple.type.cc6 = call %GetNoDeduce.specific_fn(%.loc24_25.6) to %.loc23_54
// CHECK:STDOUT:   return %GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc14_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.fd9
// CHECK:STDOUT:   %Get => constants.%Get.cf9
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.766
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.c9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc15_10.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc15_10.2 => constants.%U.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_20 => constants.%require_complete.fe1
// CHECK:STDOUT:   %Get.type => constants.%Get.type.fd9
// CHECK:STDOUT:   %Get => constants.%Get.cf9
// CHECK:STDOUT:   %Get.specific_fn.loc15_39.2 => constants.%Get.specific_fn.f73
// CHECK:STDOUT:   %require_complete.loc15_44.1 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc15_44.2 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U.loc16_24.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc16_24.2 => constants.%U.patt
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%require_complete.4ae
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.766
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.c9a
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc16_53.2 => constants.%GetNoDeduce.specific_fn.536
// CHECK:STDOUT:   %require_complete.loc16_70 => constants.%require_complete.fe1
// CHECK:STDOUT:   %require_complete.loc16_69 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc14_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(%T, %U.loc15_10.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetNoDeduce.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(%T, %U.loc16_24.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.loc14_13.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc14_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.501
// CHECK:STDOUT:   %Get => constants.%Get.f37
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.5d6
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.loc15_10.1 => constants.%B
// CHECK:STDOUT:   %U.patt.loc15_10.2 => constants.%U.patt
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_20 => constants.%complete_type.56a
// CHECK:STDOUT:   %Get.type => constants.%Get.type.501
// CHECK:STDOUT:   %Get => constants.%Get.f37
// CHECK:STDOUT:   %Get.specific_fn.loc15_39.2 => constants.%Get.specific_fn.213
// CHECK:STDOUT:   %require_complete.loc15_44.1 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_44.2 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %U.loc16_24.1 => constants.%B
// CHECK:STDOUT:   %U.patt.loc16_24.2 => constants.%U.patt
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.357
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.5d6
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.162
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc16_53.2 => constants.%GetNoDeduce.specific_fn.438
// CHECK:STDOUT:   %require_complete.loc16_70 => constants.%complete_type.56a
// CHECK:STDOUT:   %require_complete.loc16_69 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
