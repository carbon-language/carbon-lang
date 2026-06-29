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
  fn F[U:! type](self: Class(U)) -> U { return self.F(); }
}

fn CallGenericMethod(c: Class(A)) -> (A, B) {
  return c.Get(B);
}

fn CallGenericMethodWithNonDeducedParam(c: Class(A)) -> (A, B) {
  return c.GetNoDeduce({}, B);
}

fn CallMethodSelfDeduce(c: Class(A)) -> A {
  return c.F();
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.7bc: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %return.param_patt.a41: %pattern_type.eee = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.453: %pattern_type.eee = return_slot_pattern %return.param_patt.a41, %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %Class.Get.type.49d: type = fn_type @Class.Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Get.ceb: %Class.Get.type.49d = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.800: %pattern_type.51d = at_binding_pattern x, %x.param_patt.91d [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.83e: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.a04: %Class.GetNoDeduce.type.83e = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.d74: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %pattern_type.cc0: type = pattern_type %Class.d74 [symbolic]
// CHECK:STDOUT:   %self.param_patt.814: %pattern_type.cc0 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.868: %pattern_type.cc0 = at_binding_pattern self, %self.param_patt.814 [symbolic]
// CHECK:STDOUT:   %.822: Core.Form = init_form %U [symbolic]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %return.param_patt.6dd: %pattern_type.946 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4dd: %pattern_type.946 = return_slot_pattern %return.param_patt.6dd, %U [symbolic]
// CHECK:STDOUT:   %Class.F.type.982: type = fn_type @Class.F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.F.c63: %Class.F.type.982 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %Class.Get.specific_fn.ba5: <specific function> = specific_function %Class.Get.ceb, @Class.Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.def: <specific function> = specific_function %Class.GetNoDeduce.a04, @Class.GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.Get.type.3dc: type = fn_type @Class.Get, @Class(%U) [symbolic]
// CHECK:STDOUT:   %Class.Get.d1f: %Class.Get.type.3dc = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.2cf: type = fn_type @Class.GetNoDeduce, @Class(%U) [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.0cf: %Class.GetNoDeduce.type.2cf = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.F.type.6df: type = fn_type @Class.F, @Class(%U) [symbolic]
// CHECK:STDOUT:   %Class.F.c29: %Class.F.type.6df = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.716: <witness> = require_complete_type %Class.d74 [symbolic]
// CHECK:STDOUT:   %Class.F.specific_fn.844: <specific function> = specific_function %Class.F.c29, @Class.F(%U, %U) [symbolic]
// CHECK:STDOUT:   %Class.30a: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %pattern_type.89e: type = pattern_type %Class.30a [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.89e = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.89e = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %tuple.9aa: %tuple.type.24b = tuple_value (%A, %B) [concrete]
// CHECK:STDOUT:   %tuple.type.1e7: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %.7f5: Core.Form = init_form %tuple.type.1e7 [concrete]
// CHECK:STDOUT:   %pattern_type.7c6: type = pattern_type %tuple.type.1e7 [concrete]
// CHECK:STDOUT:   %return.param_patt.1be: %pattern_type.7c6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.709: %pattern_type.7c6 = return_slot_pattern %return.param_patt.1be, %tuple.type.1e7 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.type.9a5: type = fn_type @Class.Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.Get.e47: %Class.Get.type.9a5 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.7e9: type = fn_type @Class.GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.c1e: %Class.GetNoDeduce.type.7e9 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.F.type.e68: type = fn_type @Class.F, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.F.805: %Class.F.type.e68 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn.0ec: <specific function> = specific_function %Class.Get.e47, @Class.Get(%A, %B) [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %x.param_patt.4c3: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.313: %pattern_type.9ef = at_binding_pattern x, %x.param_patt.4c3 [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.931: <specific function> = specific_function %Class.GetNoDeduce.c1e, @Class.GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc29_25.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.367, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT:   %.cbf: Core.Form = init_form %A [concrete]
// CHECK:STDOUT:   %return.param_patt.f0c: %pattern_type.9ef = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a6d: %pattern_type.9ef = return_slot_pattern %return.param_patt.f0c, %A [concrete]
// CHECK:STDOUT:   %CallMethodSelfDeduce.type: type = fn_type @CallMethodSelfDeduce [concrete]
// CHECK:STDOUT:   %CallMethodSelfDeduce: %CallMethodSelfDeduce.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.225: %pattern_type.89e = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.39c: %pattern_type.89e = at_binding_pattern self, %self.param_patt.225 [concrete]
// CHECK:STDOUT:   %Class.F.specific_fn.0bd: <specific function> = specific_function %Class.F.805, @Class.F(%A, %A) [concrete]
// CHECK:STDOUT:   %complete_type.c44: <witness> = complete_type_witness %tuple.type.1e7 [concrete]
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
// CHECK:STDOUT:     .CallMethodSelfDeduce = %CallMethodSelfDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc18_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc18_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.89e = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.89e = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c6 = out_param_pattern [concrete = constants.%return.param_patt.1be]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c6 = return_slot_pattern %return.param_patt, %.loc24_43.3 [concrete = constants.%return.patt.709]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc24_39: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc24_43.2: %tuple.type.24b = tuple_literal (%A.ref.loc24_39, %B.ref.loc24) [concrete = constants.%tuple.9aa]
// CHECK:STDOUT:     %.loc24_43.3: type = converted %.loc24_43.2, constants.%tuple.type.1e7 [concrete = constants.%tuple.type.1e7]
// CHECK:STDOUT:     %.loc24_43.4: Core.Form = init_form %.loc24_43.3 [concrete = constants.%.7f5]
// CHECK:STDOUT:     %c.param: %Class.30a = value_param call_param0
// CHECK:STDOUT:     %.loc24_32: type = splice_block %Class [concrete = constants.%Class.30a] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc24_31: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.30a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.30a = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.1e7 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.1e7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [concrete = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.89e = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.89e = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c6 = out_param_pattern [concrete = constants.%return.param_patt.1be]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c6 = return_slot_pattern %return.param_patt, %.loc28_62.3 [concrete = constants.%return.patt.709]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc28_58: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc28: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc28_62.2: %tuple.type.24b = tuple_literal (%A.ref.loc28_58, %B.ref.loc28) [concrete = constants.%tuple.9aa]
// CHECK:STDOUT:     %.loc28_62.3: type = converted %.loc28_62.2, constants.%tuple.type.1e7 [concrete = constants.%tuple.type.1e7]
// CHECK:STDOUT:     %.loc28_62.4: Core.Form = init_form %.loc28_62.3 [concrete = constants.%.7f5]
// CHECK:STDOUT:     %c.param: %Class.30a = value_param call_param0
// CHECK:STDOUT:     %.loc28_51: type = splice_block %Class [concrete = constants.%Class.30a] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc28_50: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.30a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.30a = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.1e7 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.1e7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethodSelfDeduce.decl: %CallMethodSelfDeduce.type = fn_decl @CallMethodSelfDeduce [concrete = constants.%CallMethodSelfDeduce] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.89e = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.89e = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ef = out_param_pattern [concrete = constants.%return.param_patt.f0c]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ef = return_slot_pattern %return.param_patt, %A.ref.loc32_41 [concrete = constants.%return.patt.a6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc32_41: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %.loc32_41.2: Core.Form = init_form %A.ref.loc32_41 [concrete = constants.%.cbf]
// CHECK:STDOUT:     %c.param: %Class.30a = value_param call_param0
// CHECK:STDOUT:     %.loc32_35: type = splice_block %Class [concrete = constants.%Class.30a] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc32_34: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.30a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.30a = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param1
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
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
// CHECK:STDOUT: generic class @Class(%T.loc18_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc18_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc18_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T.loc18_14.1) [symbolic = %Class.Get.type (constants.%Class.Get.type.49d)]
// CHECK:STDOUT:   %Class.Get: @Class.%Class.Get.type (%Class.Get.type.49d) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.ceb)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T.loc18_14.1) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.83e)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.83e) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.a04)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc18_14.1) [symbolic = %Class.F.type (constants.%Class.F.type.982)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type.982) = struct_value () [symbolic = %Class.F (constants.%Class.F.c63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.Get.decl: @Class.%Class.Get.type (%Class.Get.type.49d) = fn_decl @Class.Get [symbolic = @Class.%Class.Get (constants.%Class.Get.ceb)] {
// CHECK:STDOUT:       %U.patt.loc19_11.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc19_11.2 (constants.%U.patt)]
// CHECK:STDOUT:       %return.param_patt.loc19_28.1: @Class.Get.%pattern_type (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc19_28.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:       %return.patt.loc19_20.1: @Class.Get.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc19_28.1, %.loc19_28.4 [symbolic = %return.patt.loc19_20.2 (constants.%return.patt.453)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc18_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc19_27: type = name_ref U, %U.loc19_11.2 [symbolic = %U.loc19_11.1 (constants.%U)]
// CHECK:STDOUT:       %.loc19_28.3: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc19_27) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc19_28.4: type = converted %.loc19_28.3, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc19_28.5: Core.Form = init_form %.loc19_28.4 [symbolic = %.loc19_28.2 (constants.%.f18)]
// CHECK:STDOUT:       %.loc19_14.1: type = splice_block %.loc19_14.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc19_14.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc19_11.2: type = symbolic_binding U, 1 [symbolic = %U.loc19_11.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.Get.%tuple.type (%tuple.type.a5e) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.Get.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetNoDeduce.decl: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.83e) = fn_decl @Class.GetNoDeduce [symbolic = @Class.%Class.GetNoDeduce (constants.%Class.GetNoDeduce.a04)] {
// CHECK:STDOUT:       %x.param_patt.loc20_19.1: @Class.GetNoDeduce.%pattern_type.loc20_19 (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc20_19.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:       %x.patt.loc20_19.1: @Class.GetNoDeduce.%pattern_type.loc20_19 (%pattern_type.51d) = at_binding_pattern x, %x.param_patt.loc20_19.1 [symbolic = %x.patt.loc20_19.2 (constants.%x.patt.800)]
// CHECK:STDOUT:       %U.patt.loc20_25.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc20_25.2 (constants.%U.patt)]
// CHECK:STDOUT:       %return.param_patt.loc20_42.1: @Class.GetNoDeduce.%pattern_type.loc20_42 (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc20_42.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:       %return.patt.loc20_34.1: @Class.GetNoDeduce.%pattern_type.loc20_42 (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc20_42.1, %.loc20_42.4 [symbolic = %return.patt.loc20_34.2 (constants.%return.patt.453)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc20_38: type = name_ref T, @Class.%T.loc18_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc20_41: type = name_ref U, %U.loc20_25.2 [symbolic = %U.loc20_25.1 (constants.%U)]
// CHECK:STDOUT:       %.loc20_42.3: %tuple.type.24b = tuple_literal (%T.ref.loc20_38, %U.ref.loc20_41) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc20_42.4: type = converted %.loc20_42.3, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc20_42.5: Core.Form = init_form %.loc20_42.4 [symbolic = %.loc20_42.2 (constants.%.f18)]
// CHECK:STDOUT:       %x.param: @Class.GetNoDeduce.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc20_21: type = name_ref T, @Class.%T.loc18_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @Class.GetNoDeduce.%T (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:       %.loc20_28.1: type = splice_block %.loc20_28.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc20_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc20_25.2: type = symbolic_binding U, 1 [symbolic = %U.loc20_25.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type.982) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F.c63)] {
// CHECK:STDOUT:       %U.patt.loc21_9.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc21_9.2 (constants.%U.patt)]
// CHECK:STDOUT:       %self.param_patt.loc21_22.1: @Class.F.%pattern_type.loc21_22 (%pattern_type.cc0) = value_param_pattern [symbolic = %self.param_patt.loc21_22.2 (constants.%self.param_patt.814)]
// CHECK:STDOUT:       %self.patt.loc21_22.1: @Class.F.%pattern_type.loc21_22 (%pattern_type.cc0) = at_binding_pattern self, %self.param_patt.loc21_22.1 [symbolic = %self.patt.loc21_22.2 (constants.%self.patt.868)]
// CHECK:STDOUT:       %return.param_patt.loc21_37.1: @Class.F.%pattern_type.loc21_37 (%pattern_type.946) = out_param_pattern [symbolic = %return.param_patt.loc21_37.2 (constants.%return.param_patt.6dd)]
// CHECK:STDOUT:       %return.patt.loc21_34.1: @Class.F.%pattern_type.loc21_37 (%pattern_type.946) = return_slot_pattern %return.param_patt.loc21_37.1, %U.ref.loc21_37 [symbolic = %return.patt.loc21_34.2 (constants.%return.patt.4dd)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref.loc21_37: type = name_ref U, %U.loc21_9.2 [symbolic = %U.loc21_9.1 (constants.%U)]
// CHECK:STDOUT:       %.loc21_37.3: Core.Form = init_form %U.ref.loc21_37 [symbolic = %.loc21_37.2 (constants.%.822)]
// CHECK:STDOUT:       %.loc21_12.1: type = splice_block %.loc21_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc21_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc21_9.2: type = symbolic_binding U, 1 [symbolic = %U.loc21_9.1 (constants.%U)]
// CHECK:STDOUT:       %self.param: @Class.F.%Class.loc21_31.1 (%Class.d74) = value_param call_param0
// CHECK:STDOUT:       %.loc21_31: type = splice_block %Class.loc21_31.2 [symbolic = %Class.loc21_31.1 (constants.%Class.d74)] {
// CHECK:STDOUT:         %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:         %U.ref.loc21_30: type = name_ref U, %U.loc21_9.2 [symbolic = %U.loc21_9.1 (constants.%U)]
// CHECK:STDOUT:         %Class.loc21_31.2: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc21_31.1 (constants.%Class.d74)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.F.%Class.loc21_31.1 (%Class.d74) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.F.%U.loc21_9.1 (%U) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.%U.loc21_9.1 (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.7bc
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %Class.GetNoDeduce.decl
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc18_14.2: type, %U.loc19_11.2: type) {
// CHECK:STDOUT:   %U.patt.loc19_11.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc19_11.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc19_11.1: type = symbolic_binding U, 1 [symbolic = %U.loc19_11.1 (constants.%U)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U.loc19_11.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc19_11.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc19_28.2: Core.Form = init_form %tuple.type [symbolic = %.loc19_28.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc19_28.2: @Class.Get.%pattern_type (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc19_28.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:   %return.patt.loc19_20.2: @Class.Get.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc19_28.2, %tuple.type [symbolic = %return.patt.loc19_20.2 (constants.%return.patt.453)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.220)]
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T) [symbolic = %Class.Get.type (constants.%Class.Get.type.49d)]
// CHECK:STDOUT:   %Class.Get: @Class.Get.%Class.Get.type (%Class.Get.type.49d) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.ceb)]
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2: <specific function> = specific_function %Class.Get, @Class.Get(%T, %U.loc19_11.1) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.ba5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Class.Get.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_39: @Class.Get.%Class.Get.type (%Class.Get.type.49d) = specific_constant @Class.%Class.Get.decl, @Class(constants.%T) [symbolic = %Class.Get (constants.%Class.Get.ceb)]
// CHECK:STDOUT:     %Get.ref: @Class.Get.%Class.Get.type (%Class.Get.type.49d) = name_ref Get, %.loc19_39 [symbolic = %Class.Get (constants.%Class.Get.ceb)]
// CHECK:STDOUT:     %U.ref.loc19_43: type = name_ref U, %U.loc19_11.2 [symbolic = %U.loc19_11.1 (constants.%U)]
// CHECK:STDOUT:     %Class.Get.specific_fn.loc19_39.1: <specific function> = specific_function %Get.ref, @Class.Get(constants.%T, constants.%U) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.ba5)]
// CHECK:STDOUT:     %.loc19_28.1: ref @Class.Get.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %Class.Get.call: init @Class.Get.%tuple.type (%tuple.type.a5e) to %.loc19_28.1 = call %Class.Get.specific_fn.loc19_39.1()
// CHECK:STDOUT:     return %Class.Get.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetNoDeduce(@Class.%T.loc18_14.2: type, %U.loc20_25.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc20_19: type = pattern_type %T [symbolic = %pattern_type.loc20_19 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc20_19.2: @Class.GetNoDeduce.%pattern_type.loc20_19 (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc20_19.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc20_19.2: @Class.GetNoDeduce.%pattern_type.loc20_19 (%pattern_type.51d) = at_binding_pattern x, %x.param_patt.loc20_19.2 [symbolic = %x.patt.loc20_19.2 (constants.%x.patt.800)]
// CHECK:STDOUT:   %U.patt.loc20_25.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc20_25.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc20_25.1: type = symbolic_binding U, 1 [symbolic = %U.loc20_25.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U.loc20_25.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc20_25.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc20_42.2: Core.Form = init_form %tuple.type [symbolic = %.loc20_42.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type.loc20_42: type = pattern_type %tuple.type [symbolic = %pattern_type.loc20_42 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc20_42.2: @Class.GetNoDeduce.%pattern_type.loc20_42 (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc20_42.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:   %return.patt.loc20_34.2: @Class.GetNoDeduce.%pattern_type.loc20_42 (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc20_42.2, %tuple.type [symbolic = %return.patt.loc20_34.2 (constants.%return.patt.453)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.83e)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.83e) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.a04)]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2: <specific function> = specific_function %Class.GetNoDeduce, @Class.GetNoDeduce(%T, %U.loc20_25.1) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.def)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Class.GetNoDeduce.%T (%T)) -> out %return.param: @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc20_53: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.83e) = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%T) [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.a04)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.83e) = name_ref GetNoDeduce, %.loc20_53 [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.a04)]
// CHECK:STDOUT:     %x.ref: @Class.GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc20_68: type = name_ref U, %U.loc20_25.2 [symbolic = %U.loc20_25.1 (constants.%U)]
// CHECK:STDOUT:     %Class.GetNoDeduce.specific_fn.loc20_53.1: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%T, constants.%U) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.def)]
// CHECK:STDOUT:     %.loc20_42.1: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %Class.GetNoDeduce.call: init @Class.GetNoDeduce.%tuple.type (%tuple.type.a5e) to %.loc20_42.1 = call %Class.GetNoDeduce.specific_fn.loc20_53.1(%x.ref)
// CHECK:STDOUT:     return %Class.GetNoDeduce.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc18_14.2: type, %U.loc21_9.2: type) {
// CHECK:STDOUT:   %U.patt.loc21_9.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc21_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc21_9.1: type = symbolic_binding U, 1 [symbolic = %U.loc21_9.1 (constants.%U)]
// CHECK:STDOUT:   %Class.loc21_31.1: type = class_type @Class, @Class(%U.loc21_9.1) [symbolic = %Class.loc21_31.1 (constants.%Class.d74)]
// CHECK:STDOUT:   %pattern_type.loc21_22: type = pattern_type %Class.loc21_31.1 [symbolic = %pattern_type.loc21_22 (constants.%pattern_type.cc0)]
// CHECK:STDOUT:   %self.param_patt.loc21_22.2: @Class.F.%pattern_type.loc21_22 (%pattern_type.cc0) = value_param_pattern [symbolic = %self.param_patt.loc21_22.2 (constants.%self.param_patt.814)]
// CHECK:STDOUT:   %self.patt.loc21_22.2: @Class.F.%pattern_type.loc21_22 (%pattern_type.cc0) = at_binding_pattern self, %self.param_patt.loc21_22.2 [symbolic = %self.patt.loc21_22.2 (constants.%self.patt.868)]
// CHECK:STDOUT:   %.loc21_37.2: Core.Form = init_form %U.loc21_9.1 [symbolic = %.loc21_37.2 (constants.%.822)]
// CHECK:STDOUT:   %pattern_type.loc21_37: type = pattern_type %U.loc21_9.1 [symbolic = %pattern_type.loc21_37 (constants.%pattern_type.946)]
// CHECK:STDOUT:   %return.param_patt.loc21_37.2: @Class.F.%pattern_type.loc21_37 (%pattern_type.946) = out_param_pattern [symbolic = %return.param_patt.loc21_37.2 (constants.%return.param_patt.6dd)]
// CHECK:STDOUT:   %return.patt.loc21_34.2: @Class.F.%pattern_type.loc21_37 (%pattern_type.946) = return_slot_pattern %return.param_patt.loc21_37.2, %U.loc21_9.1 [symbolic = %return.patt.loc21_34.2 (constants.%return.patt.4dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc21_31.1 [symbolic = %require_complete (constants.%require_complete.716)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%U.loc21_9.1) [symbolic = %Class.F.type (constants.%Class.F.type.6df)]
// CHECK:STDOUT:   %Class.F: @Class.F.%Class.F.type (%Class.F.type.6df) = struct_value () [symbolic = %Class.F (constants.%Class.F.c29)]
// CHECK:STDOUT:   %Class.F.specific_fn.loc21_52.2: <specific function> = specific_function %Class.F, @Class.F(%U.loc21_9.1, %U.loc21_9.1) [symbolic = %Class.F.specific_fn.loc21_52.2 (constants.%Class.F.specific_fn.844)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.F.%Class.loc21_31.1 (%Class.d74)) -> out %return.param: @Class.F.%U.loc21_9.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.F.%Class.loc21_31.1 (%Class.d74) = name_ref self, %self
// CHECK:STDOUT:     %.loc21_52: @Class.F.%Class.F.type (%Class.F.type.6df) = specific_constant @Class.%Class.F.decl, @Class(constants.%U) [symbolic = %Class.F (constants.%Class.F.c29)]
// CHECK:STDOUT:     %F.ref: @Class.F.%Class.F.type (%Class.F.type.6df) = name_ref F, %.loc21_52 [symbolic = %Class.F (constants.%Class.F.c29)]
// CHECK:STDOUT:     %Class.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:     %Class.F.specific_fn.loc21_52.1: <specific function> = specific_function %F.ref, @Class.F(constants.%U, constants.%U) [symbolic = %Class.F.specific_fn.loc21_52.2 (constants.%Class.F.specific_fn.844)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.ref, %Class.F.specific_fn.loc21_52.1
// CHECK:STDOUT:     %.loc21_37.1: ref @Class.F.%U.loc21_9.1 (%U) = splice_block %return.param {}
// CHECK:STDOUT:     %Class.F.call: init @Class.F.%U.loc21_9.1 (%U) to %.loc21_37.1 = call %bound_method(%self.ref)
// CHECK:STDOUT:     return %Class.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param: %Class.30a) -> out %return.param: %tuple.type.1e7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.30a = name_ref c, %c
// CHECK:STDOUT:   %.loc25: %Class.Get.type.9a5 = specific_constant @Class.%Class.Get.decl, @Class(constants.%A) [concrete = constants.%Class.Get.e47]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.9a5 = name_ref Get, %.loc25 [concrete = constants.%Class.Get.e47]
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%A, constants.%B) [concrete = constants.%Class.Get.specific_fn.0ec]
// CHECK:STDOUT:   %.loc24_43.1: ref %tuple.type.1e7 = splice_block %return.param {}
// CHECK:STDOUT:   %Class.Get.call: init %tuple.type.1e7 to %.loc24_43.1 = call %Class.Get.specific_fn()
// CHECK:STDOUT:   return %Class.Get.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param: %Class.30a) -> out %return.param: %tuple.type.1e7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.30a = name_ref c, %c
// CHECK:STDOUT:   %.loc29_11: %Class.GetNoDeduce.type.7e9 = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%Class.GetNoDeduce.c1e]
// CHECK:STDOUT:   %GetNoDeduce.ref: %Class.GetNoDeduce.type.7e9 = name_ref GetNoDeduce, %.loc29_11 [concrete = constants.%Class.GetNoDeduce.c1e]
// CHECK:STDOUT:   %.loc29_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %B.ref.loc29: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%Class.GetNoDeduce.specific_fn.931]
// CHECK:STDOUT:   %.loc28_62.1: ref %tuple.type.1e7 = splice_block %return.param {}
// CHECK:STDOUT:   %.loc29_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc29_25.3: init %A to %.loc29_25.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc29_25.4: init %A = converted %.loc29_25.1, %.loc29_25.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc29_25.5: ref %A = temporary %.loc29_25.2, %.loc29_25.4 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc29_25.6: %A = acquire_value %.loc29_25.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   %Class.GetNoDeduce.call: init %tuple.type.1e7 to %.loc28_62.1 = call %Class.GetNoDeduce.specific_fn(%.loc29_25.6)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   return %Class.GetNoDeduce.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc29_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc29_25.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethodSelfDeduce(%c.param: %Class.30a) -> out %return.param: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.30a = name_ref c, %c
// CHECK:STDOUT:   %.loc33: %Class.F.type.e68 = specific_constant @Class.%Class.F.decl, @Class(constants.%A) [concrete = constants.%Class.F.805]
// CHECK:STDOUT:   %F.ref: %Class.F.type.e68 = name_ref F, %.loc33 [concrete = constants.%Class.F.805]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %Class.F.specific_fn: <specific function> = specific_function %F.ref, @Class.F(constants.%A, constants.%A) [concrete = constants.%Class.F.specific_fn.0bd]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %Class.F.specific_fn
// CHECK:STDOUT:   %.loc32_41.1: ref %A = splice_block %return.param {}
// CHECK:STDOUT:   %Class.F.call: init %A to %.loc32_41.1 = call %bound_method(%c.ref)
// CHECK:STDOUT:   return %Class.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc18_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc18_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.49d
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.ceb
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.83e
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.a04
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.982
// CHECK:STDOUT:   %Class.F => constants.%Class.F.c63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.patt.loc19_11.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc19_11.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc19_28.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc19_28.2 => constants.%return.param_patt.a41
// CHECK:STDOUT:   %return.patt.loc19_20.2 => constants.%return.patt.453
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.220
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.49d
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.ceb
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.ba5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type.loc20_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc20_19.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc20_19.2 => constants.%x.patt.800
// CHECK:STDOUT:   %U.patt.loc20_25.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc20_25.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc20_42.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type.loc20_42 => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc20_42.2 => constants.%return.param_patt.a41
// CHECK:STDOUT:   %return.patt.loc20_34.2 => constants.%return.patt.453
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.83e
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.a04
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.def
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.patt.loc18_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc18_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.3dc
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.d1f
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.2cf
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.0cf
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.6df
// CHECK:STDOUT:   %Class.F => constants.%Class.F.c29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.patt.loc21_9.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc21_9.1 => constants.%U
// CHECK:STDOUT:   %Class.loc21_31.1 => constants.%Class.d74
// CHECK:STDOUT:   %pattern_type.loc21_22 => constants.%pattern_type.cc0
// CHECK:STDOUT:   %self.param_patt.loc21_22.2 => constants.%self.param_patt.814
// CHECK:STDOUT:   %self.patt.loc21_22.2 => constants.%self.patt.868
// CHECK:STDOUT:   %.loc21_37.2 => constants.%.822
// CHECK:STDOUT:   %pattern_type.loc21_37 => constants.%pattern_type.946
// CHECK:STDOUT:   %return.param_patt.loc21_37.2 => constants.%return.param_patt.6dd
// CHECK:STDOUT:   %return.patt.loc21_34.2 => constants.%return.patt.4dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%U, constants.%U) {
// CHECK:STDOUT:   %U.patt.loc21_9.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc21_9.1 => constants.%U
// CHECK:STDOUT:   %Class.loc21_31.1 => constants.%Class.d74
// CHECK:STDOUT:   %pattern_type.loc21_22 => constants.%pattern_type.cc0
// CHECK:STDOUT:   %self.param_patt.loc21_22.2 => constants.%self.param_patt.814
// CHECK:STDOUT:   %self.patt.loc21_22.2 => constants.%self.patt.868
// CHECK:STDOUT:   %.loc21_37.2 => constants.%.822
// CHECK:STDOUT:   %pattern_type.loc21_37 => constants.%pattern_type.946
// CHECK:STDOUT:   %return.param_patt.loc21_37.2 => constants.%return.param_patt.6dd
// CHECK:STDOUT:   %return.patt.loc21_34.2 => constants.%return.patt.4dd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.716
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.6df
// CHECK:STDOUT:   %Class.F => constants.%Class.F.c29
// CHECK:STDOUT:   %Class.F.specific_fn.loc21_52.2 => constants.%Class.F.specific_fn.844
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.patt.loc18_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc18_14.1 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.9a5
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.e47
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.7e9
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.c1e
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.e68
// CHECK:STDOUT:   %Class.F => constants.%Class.F.805
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.patt.loc19_11.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc19_11.1 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %tuple => constants.%tuple.9aa
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.1e7
// CHECK:STDOUT:   %.loc19_28.2 => constants.%.7f5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c6
// CHECK:STDOUT:   %return.param_patt.loc19_28.2 => constants.%return.param_patt.1be
// CHECK:STDOUT:   %return.patt.loc19_20.2 => constants.%return.patt.709
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c44
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.9a5
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.e47
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.0ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %pattern_type.loc20_19 => constants.%pattern_type.9ef
// CHECK:STDOUT:   %x.param_patt.loc20_19.2 => constants.%x.param_patt.4c3
// CHECK:STDOUT:   %x.patt.loc20_19.2 => constants.%x.patt.313
// CHECK:STDOUT:   %U.patt.loc20_25.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc20_25.1 => constants.%B
// CHECK:STDOUT:   %tuple => constants.%tuple.9aa
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.1e7
// CHECK:STDOUT:   %.loc20_42.2 => constants.%.7f5
// CHECK:STDOUT:   %pattern_type.loc20_42 => constants.%pattern_type.7c6
// CHECK:STDOUT:   %return.param_patt.loc20_42.2 => constants.%return.param_patt.1be
// CHECK:STDOUT:   %return.patt.loc20_34.2 => constants.%return.patt.709
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.7e9
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.c1e
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.931
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%A, constants.%A) {
// CHECK:STDOUT:   %U.patt.loc21_9.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc21_9.1 => constants.%A
// CHECK:STDOUT:   %Class.loc21_31.1 => constants.%Class.30a
// CHECK:STDOUT:   %pattern_type.loc21_22 => constants.%pattern_type.89e
// CHECK:STDOUT:   %self.param_patt.loc21_22.2 => constants.%self.param_patt.225
// CHECK:STDOUT:   %self.patt.loc21_22.2 => constants.%self.patt.39c
// CHECK:STDOUT:   %.loc21_37.2 => constants.%.cbf
// CHECK:STDOUT:   %pattern_type.loc21_37 => constants.%pattern_type.9ef
// CHECK:STDOUT:   %return.param_patt.loc21_37.2 => constants.%return.param_patt.f0c
// CHECK:STDOUT:   %return.patt.loc21_34.2 => constants.%return.patt.a6d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.e68
// CHECK:STDOUT:   %Class.F => constants.%Class.F.805
// CHECK:STDOUT:   %Class.F.specific_fn.loc21_52.2 => constants.%Class.F.specific_fn.0bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
