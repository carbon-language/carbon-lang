// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.0db: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %.5b7: form = init_form %tuple.type.a5e, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %Class.Get.type.ab7: type = fn_type @Class.Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Get.ecd: %Class.Get.type.ab7 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %.541: form = init_form %tuple.type.a5e, call_param1 [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.cf2: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.1a5: %Class.GetNoDeduce.type.cf2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %Class.Get.specific_fn.f51: <specific function> = specific_function %Class.Get.ecd, @Class.Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.710: <specific function> = specific_function %Class.GetNoDeduce.1a5, @Class.GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.802: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %pattern_type.36c: type = pattern_type %Class.802 [concrete]
// CHECK:STDOUT:   %tuple.5bc: %tuple.type.24b = tuple_value (%A, %B) [concrete]
// CHECK:STDOUT:   %tuple.type.e87: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %.c61: form = init_form %tuple.type.e87, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.b74: type = pattern_type %tuple.type.e87 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.type.a01: type = fn_type @Class.Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.Get.5f3: %Class.Get.type.a01 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.902: type = fn_type @Class.GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.472: %Class.GetNoDeduce.type.902 = struct_value () [concrete]
// CHECK:STDOUT:   %.a4d: form = init_form %tuple.type.e87, call_param0 [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn.54d: <specific function> = specific_function %Class.Get.5f3, @Class.Get(%A, %B) [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.83b: <specific function> = specific_function %Class.GetNoDeduce.472, @Class.GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.f71: <witness> = complete_type_witness %tuple.type.e87 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc18_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %c.patt: %pattern_type.36c = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.36c = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b74 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b74 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc23_39: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc23_43.1: %tuple.type.24b = tuple_literal (%A.ref.loc23_39, %B.ref.loc23) [concrete = constants.%tuple.5bc]
// CHECK:STDOUT:     %.loc23_43.2: type = converted %.loc23_43.1, constants.%tuple.type.e87 [concrete = constants.%tuple.type.e87]
// CHECK:STDOUT:     %.loc23_43.3: form = init_form %.loc23_43.2, call_param1 [concrete = constants.%.c61]
// CHECK:STDOUT:     %c.param: %Class.802 = value_param call_param0
// CHECK:STDOUT:     %.loc23_32: type = splice_block %Class [concrete = constants.%Class.802] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc23_31: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.802]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.802 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.e87 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.e87 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [concrete = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %c.patt: %pattern_type.36c = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.36c = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b74 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b74 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc27_58: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc27: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc27_62.1: %tuple.type.24b = tuple_literal (%A.ref.loc27_58, %B.ref.loc27) [concrete = constants.%tuple.5bc]
// CHECK:STDOUT:     %.loc27_62.2: type = converted %.loc27_62.1, constants.%tuple.type.e87 [concrete = constants.%tuple.type.e87]
// CHECK:STDOUT:     %.loc27_62.3: form = init_form %.loc27_62.2, call_param1 [concrete = constants.%.c61]
// CHECK:STDOUT:     %c.param: %Class.802 = value_param call_param0
// CHECK:STDOUT:     %.loc27_51: type = splice_block %Class [concrete = constants.%Class.802] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc27_50: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.802]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.802 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.e87 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.e87 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc18_13.2: type) {
// CHECK:STDOUT:   %T.loc18_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T.loc18_13.1) [symbolic = %Class.Get.type (constants.%Class.Get.type.ab7)]
// CHECK:STDOUT:   %Class.Get: @Class.%Class.Get.type (%Class.Get.type.ab7) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.ecd)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T.loc18_13.1) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.cf2)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.cf2) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.1a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.Get.decl: @Class.%Class.Get.type (%Class.Get.type.ab7) = fn_decl @Class.Get [symbolic = @Class.%Class.Get (constants.%Class.Get.ecd)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.Get.%pattern_type (%pattern_type.eee) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.Get.%pattern_type (%pattern_type.eee) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc19_27: type = name_ref U, %U.loc19_10.2 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:       %.loc19_28.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc19_27) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc19_28.3: type = converted %.loc19_28.2, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc19_28.4: form = init_form %.loc19_28.3, call_param0 [symbolic = %.loc19_28.1 (constants.%.5b7)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc19_10.2: type = symbolic_binding U, 1 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.Get.%tuple.type (%tuple.type.a5e) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.Get.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetNoDeduce.decl: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.cf2) = fn_decl @Class.GetNoDeduce [symbolic = @Class.%Class.GetNoDeduce (constants.%Class.GetNoDeduce.1a5)] {
// CHECK:STDOUT:       %x.patt: @Class.GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.51d) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @Class.GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.51d) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.eee) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.eee) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc20_38: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc20_41: type = name_ref U, %U.loc20_24.2 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:       %.loc20_42.2: %tuple.type.24b = tuple_literal (%T.ref.loc20_38, %U.ref.loc20_41) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc20_42.3: type = converted %.loc20_42.2, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc20_42.4: form = init_form %.loc20_42.3, call_param1 [symbolic = %.loc20_42.1 (constants.%.541)]
// CHECK:STDOUT:       %x.param: @Class.GetNoDeduce.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc20_21: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @Class.GetNoDeduce.%T (%T) = value_binding x, %x.param
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc20_24.2: type = symbolic_binding U, 1 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.0db
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %Class.GetNoDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc18_13.2: type, %U.loc19_10.2: type) {
// CHECK:STDOUT:   %U.loc19_10.1: type = symbolic_binding U, 1 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U.loc19_10.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc19_10.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc19_28.1: form = init_form %tuple.type, call_param0 [symbolic = %.loc19_28.1 (constants.%.5b7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.220)]
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T) [symbolic = %Class.Get.type (constants.%Class.Get.type.ab7)]
// CHECK:STDOUT:   %Class.Get: @Class.Get.%Class.Get.type (%Class.Get.type.ab7) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.ecd)]
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2: <specific function> = specific_function %Class.Get, @Class.Get(%T, %U.loc19_10.1) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.f51)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Class.Get.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_39: @Class.Get.%Class.Get.type (%Class.Get.type.ab7) = specific_constant @Class.%Class.Get.decl, @Class(constants.%T) [symbolic = %Class.Get (constants.%Class.Get.ecd)]
// CHECK:STDOUT:     %Get.ref: @Class.Get.%Class.Get.type (%Class.Get.type.ab7) = name_ref Get, %.loc19_39 [symbolic = %Class.Get (constants.%Class.Get.ecd)]
// CHECK:STDOUT:     %U.ref.loc19_43: type = name_ref U, %U.loc19_10.2 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:     %Class.Get.specific_fn.loc19_39.1: <specific function> = specific_function %Get.ref, @Class.Get(constants.%T, constants.%U) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.f51)]
// CHECK:STDOUT:     %.loc19_20: ref @Class.Get.%tuple.type (%tuple.type.a5e) = splice_block %return {}
// CHECK:STDOUT:     %Class.Get.call: init @Class.Get.%tuple.type (%tuple.type.a5e) = call %Class.Get.specific_fn.loc19_39.1() to %.loc19_20
// CHECK:STDOUT:     return %Class.Get.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetNoDeduce(@Class.%T.loc18_13.2: type, %U.loc20_24.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc20_18: type = pattern_type %T [symbolic = %pattern_type.loc20_18 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %U.loc20_24.1: type = symbolic_binding U, 1 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U.loc20_24.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc20_24.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc20_42.1: form = init_form %tuple.type, call_param1 [symbolic = %.loc20_42.1 (constants.%.541)]
// CHECK:STDOUT:   %pattern_type.loc20_34: type = pattern_type %tuple.type [symbolic = %pattern_type.loc20_34 (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19: <witness> = require_complete_type %T [symbolic = %require_complete.loc20_19 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc20_42: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc20_42 (constants.%require_complete.220)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.cf2)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.cf2) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.1a5)]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2: <specific function> = specific_function %Class.GetNoDeduce, @Class.GetNoDeduce(%T, %U.loc20_24.1) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.710)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Class.GetNoDeduce.%T (%T)) -> %return.param: @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc20_53: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.cf2) = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%T) [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.1a5)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.cf2) = name_ref GetNoDeduce, %.loc20_53 [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.1a5)]
// CHECK:STDOUT:     %x.ref: @Class.GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc20_68: type = name_ref U, %U.loc20_24.2 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:     %Class.GetNoDeduce.specific_fn.loc20_53.1: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%T, constants.%U) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.710)]
// CHECK:STDOUT:     %.loc20_34: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = splice_block %return {}
// CHECK:STDOUT:     %Class.GetNoDeduce.call: init @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = call %Class.GetNoDeduce.specific_fn.loc20_53.1(%x.ref) to %.loc20_34
// CHECK:STDOUT:     return %Class.GetNoDeduce.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param: %Class.802) -> %return.param: %tuple.type.e87 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.802 = name_ref c, %c
// CHECK:STDOUT:   %.loc24: %Class.Get.type.a01 = specific_constant @Class.%Class.Get.decl, @Class(constants.%A) [concrete = constants.%Class.Get.5f3]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.a01 = name_ref Get, %.loc24 [concrete = constants.%Class.Get.5f3]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%A, constants.%B) [concrete = constants.%Class.Get.specific_fn.54d]
// CHECK:STDOUT:   %.loc23_35: ref %tuple.type.e87 = splice_block %return {}
// CHECK:STDOUT:   %Class.Get.call: init %tuple.type.e87 = call %Class.Get.specific_fn() to %.loc23_35
// CHECK:STDOUT:   return %Class.Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param: %Class.802) -> %return.param: %tuple.type.e87 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.802 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_11: %Class.GetNoDeduce.type.902 = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%Class.GetNoDeduce.472]
// CHECK:STDOUT:   %GetNoDeduce.ref: %Class.GetNoDeduce.type.902 = name_ref GetNoDeduce, %.loc28_11 [concrete = constants.%Class.GetNoDeduce.472]
// CHECK:STDOUT:   %.loc28_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %B.ref.loc28: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%Class.GetNoDeduce.specific_fn.83b]
// CHECK:STDOUT:   %.loc27_54: ref %tuple.type.e87 = splice_block %return {}
// CHECK:STDOUT:   %.loc28_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc28_25.3: init %A = class_init (), %.loc28_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc28_25.4: ref %A = temporary %.loc28_25.2, %.loc28_25.3
// CHECK:STDOUT:   %.loc28_25.5: ref %A = converted %.loc28_25.1, %.loc28_25.4
// CHECK:STDOUT:   %.loc28_25.6: %A = acquire_value %.loc28_25.5
// CHECK:STDOUT:   %Class.GetNoDeduce.call: init %tuple.type.e87 = call %Class.GetNoDeduce.specific_fn(%.loc28_25.6) to %.loc27_54
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc28_25.4, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc28_25.4)
// CHECK:STDOUT:   return %Class.GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.ab7
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.ecd
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.cf2
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.1a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc19_10.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc19_28.1 => constants.%.5b7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.220
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.ab7
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.ecd
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.f51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.51d
// CHECK:STDOUT:   %U.loc20_24.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc20_42.1 => constants.%.541
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.eee
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%require_complete.944
// CHECK:STDOUT:   %require_complete.loc20_42 => constants.%require_complete.220
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.cf2
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.1a5
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.710
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.a01
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.5f3
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.902
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.472
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.loc19_10.1 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %tuple => constants.%tuple.5bc
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e87
// CHECK:STDOUT:   %.loc19_28.1 => constants.%.a4d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b74
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f71
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.a01
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.5f3
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.54d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.1ab
// CHECK:STDOUT:   %U.loc20_24.1 => constants.%B
// CHECK:STDOUT:   %tuple => constants.%tuple.5bc
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e87
// CHECK:STDOUT:   %.loc20_42.1 => constants.%.c61
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.b74
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc20_42 => constants.%complete_type.f71
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.902
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.472
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.83b
// CHECK:STDOUT: }
// CHECK:STDOUT:
