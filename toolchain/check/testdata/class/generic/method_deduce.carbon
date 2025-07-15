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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %Get.type.fd9: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.cf9: %Get.type.fd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.type.766: type = fn_type @GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.c9a: %GetNoDeduce.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.937: type = ptr_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %Get.specific_fn.f73: <specific function> = specific_function %Get.cf9, @Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %ptr.937 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.30b, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1c6: %Destroy.type = facet_value %tuple.type.30b, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.1e9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.1c6 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.1e9 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.1c6) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.536: <specific function> = specific_function %GetNoDeduce.c9a, @GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.480: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %pattern_type.827: type = pattern_type %Class.480 [concrete]
// CHECK:STDOUT:   %tuple.type.cc6: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %pattern_type.edc: type = pattern_type %tuple.type.cc6 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Get.type.501: type = fn_type @Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Get.f37: %Get.type.501 = struct_value () [concrete]
// CHECK:STDOUT:   %GetNoDeduce.type.5d6: type = fn_type @GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %GetNoDeduce.162: %GetNoDeduce.type.5d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn.213: <specific function> = specific_function %Get.f37, @Get(%A, %B) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae3: <witness> = impl_witness imports.%Destroy.impl_witness_table, @Destroy.impl(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %Op.type.87c: type = fn_type @Op.2, @Destroy.impl(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %Op.f13: %Op.type.87c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3b5: type = ptr_type %tuple.type.cc6 [concrete]
// CHECK:STDOUT:   %complete_type.a4a: <witness> = complete_type_witness %ptr.3b5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.fae: %Destroy.type = facet_value %tuple.type.cc6, (%Destroy.impl_witness.ae3) [concrete]
// CHECK:STDOUT:   %.c9f: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.fae [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8d7: <specific function> = specific_function %Op.f13, @Op.2(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.438: <specific function> = specific_function %GetNoDeduce.162, @GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.b96: type = fn_type @Op.2, @Destroy.impl(%A) [concrete]
// CHECK:STDOUT:   %Op.885: %Op.type.b96 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %Op.specific_fn.716: <specific function> = specific_function %Op.885, @Op.2(%A) [concrete]
// CHECK:STDOUT:   %complete_type.56a: <witness> = complete_type_witness %tuple.type.cc6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Destroy.impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @Destroy.impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @Destroy.impl [concrete]
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
// CHECK:STDOUT:     %T.loc18_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %c.patt: %pattern_type.827 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.827 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.edc = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.edc = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc23_39: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc23_43.1: %tuple.type.24b = tuple_literal (%A.ref.loc23_39, %B.ref.loc23)
// CHECK:STDOUT:     %.loc23_43.2: type = converted %.loc23_43.1, constants.%tuple.type.cc6 [concrete = constants.%tuple.type.cc6]
// CHECK:STDOUT:     %c.param: %Class.480 = value_param call_param0
// CHECK:STDOUT:     %.loc23_32: type = splice_block %Class [concrete = constants.%Class.480] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc23_31: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.480]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.480 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.cc6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.cc6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [concrete = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %c.patt: %pattern_type.827 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.827 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.edc = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.edc = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc27_58: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %B.ref.loc27: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc27_62.1: %tuple.type.24b = tuple_literal (%A.ref.loc27_58, %B.ref.loc27)
// CHECK:STDOUT:     %.loc27_62.2: type = converted %.loc27_62.1, constants.%tuple.type.cc6 [concrete = constants.%tuple.type.cc6]
// CHECK:STDOUT:     %c.param: %Class.480 = value_param call_param0
// CHECK:STDOUT:     %.loc27_51: type = splice_block %Class [concrete = constants.%Class.480] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %A.ref.loc27_50: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%A) [concrete = constants.%Class.480]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.480 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.cc6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.cc6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc18_13.2: type) {
// CHECK:STDOUT:   %T.loc18_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T.loc18_13.1) [symbolic = %Get.type (constants.%Get.type.fd9)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.fd9) = struct_value () [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T.loc18_13.1) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.766)]
// CHECK:STDOUT:   %GetNoDeduce: @Class.%GetNoDeduce.type (%GetNoDeduce.type.766) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.fd9) = fn_decl @Get [symbolic = @Class.%Get (constants.%Get.cf9)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @Get.%pattern_type (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Get.%pattern_type (%pattern_type.65c) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc19_27: type = name_ref U, %U.loc19_10.1 [symbolic = %U.loc19_10.2 (constants.%U)]
// CHECK:STDOUT:       %.loc19_28.1: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc19_27)
// CHECK:STDOUT:       %.loc19_28.2: type = converted %.loc19_28.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %U.loc19_10.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc19_10.2 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Get.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Get.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetNoDeduce.decl: @Class.%GetNoDeduce.type (%GetNoDeduce.type.766) = fn_decl @GetNoDeduce [symbolic = @Class.%GetNoDeduce (constants.%GetNoDeduce.c9a)] {
// CHECK:STDOUT:       %x.patt: @GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.65c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc20_38: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc20_41: type = name_ref U, %U.loc20_24.1 [symbolic = %U.loc20_24.2 (constants.%U)]
// CHECK:STDOUT:       %.loc20_42.1: %tuple.type.24b = tuple_literal (%T.ref.loc20_38, %U.ref.loc20_41)
// CHECK:STDOUT:       %.loc20_42.2: type = converted %.loc20_42.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %x.param: @GetNoDeduce.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc20_21: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @GetNoDeduce.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:       %U.loc20_24.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc20_24.2 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: generic fn @Get(@Class.%T.loc18_13.2: type, %U.loc19_10.1: type) {
// CHECK:STDOUT:   %U.loc19_10.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc19_10.2 (constants.%U)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc19_10.2) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc19_20.1 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.fd9)]
// CHECK:STDOUT:   %Get: @Get.%Get.type (%Get.type.fd9) = struct_value () [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:   %Get.specific_fn.loc19_39.2: <specific function> = specific_function %Get, @Get(%T, %U.loc19_10.2) [symbolic = %Get.specific_fn.loc19_39.2 (constants.%Get.specific_fn.f73)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1c6)]
// CHECK:STDOUT:   %.loc19_20.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc19_20.3 (constants.%.1e9)]
// CHECK:STDOUT:   %impl.elem0.loc19_20.2: @Get.%.loc19_20.3 (%.1e9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_20.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2: <specific function> = specific_impl_function %impl.elem0.loc19_20.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc19_20.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.937)]
// CHECK:STDOUT:   %require_complete.loc19_20.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc19_20.2 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Get.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_39: @Get.%Get.type (%Get.type.fd9) = specific_constant @Class.%Get.decl, @Class(constants.%T) [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:     %Get.ref: @Get.%Get.type (%Get.type.fd9) = name_ref Get, %.loc19_39 [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:     %U.ref.loc19_43: type = name_ref U, %U.loc19_10.1 [symbolic = %U.loc19_10.2 (constants.%U)]
// CHECK:STDOUT:     %Get.specific_fn.loc19_39.1: <specific function> = specific_function %Get.ref, @Get(constants.%T, constants.%U) [symbolic = %Get.specific_fn.loc19_39.2 (constants.%Get.specific_fn.f73)]
// CHECK:STDOUT:     %.loc19_20.1: ref @Get.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %Get.call: init @Get.%tuple.type (%tuple.type.30b) = call %Get.specific_fn.loc19_39.1() to %.loc19_20.1
// CHECK:STDOUT:     %impl.elem0.loc19_20.1: @Get.%.loc19_20.3 (%.1e9) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_20.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc19_20.1: <bound method> = bound_method %.loc19_20.1, %impl.elem0.loc19_20.1
// CHECK:STDOUT:     %specific_impl_fn.loc19_20.1: <specific function> = specific_impl_function %impl.elem0.loc19_20.1, @Op.1(constants.%Destroy.facet.1c6) [symbolic = %specific_impl_fn.loc19_20.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc19_20.2: <bound method> = bound_method %.loc19_20.1, %specific_impl_fn.loc19_20.1
// CHECK:STDOUT:     %addr: @Get.%ptr (%ptr.937) = addr_of %.loc19_20.1
// CHECK:STDOUT:     %.loc19_20.2: init %empty_tuple.type = call %bound_method.loc19_20.2(%addr)
// CHECK:STDOUT:     return %Get.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetNoDeduce(@Class.%T.loc18_13.2: type, %U.loc20_24.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc20_18: type = pattern_type %T [symbolic = %pattern_type.loc20_18 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %U.loc20_24.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc20_24.2 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc20_24.2) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type.loc20_34: type = pattern_type %tuple.type [symbolic = %pattern_type.loc20_34 (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19: <witness> = require_complete_type %T [symbolic = %require_complete.loc20_19 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.766)]
// CHECK:STDOUT:   %GetNoDeduce: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc20_53.2: <specific function> = specific_function %GetNoDeduce, @GetNoDeduce(%T, %U.loc20_24.2) [symbolic = %GetNoDeduce.specific_fn.loc20_53.2 (constants.%GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:   %require_complete.loc20_70: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc20_70 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1c6)]
// CHECK:STDOUT:   %.loc20_34.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc20_34.3 (constants.%.1e9)]
// CHECK:STDOUT:   %impl.elem0.loc20_34.2: @GetNoDeduce.%.loc20_34.3 (%.1e9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_34.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_34.2: <specific function> = specific_impl_function %impl.elem0.loc20_34.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc20_34.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.937)]
// CHECK:STDOUT:   %require_complete.loc20_34: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc20_34 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @GetNoDeduce.%T (%T)) -> %return.param: @GetNoDeduce.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc20_53: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%T) [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.766) = name_ref GetNoDeduce, %.loc20_53 [symbolic = %GetNoDeduce (constants.%GetNoDeduce.c9a)]
// CHECK:STDOUT:     %x.ref: @GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc20_68: type = name_ref U, %U.loc20_24.1 [symbolic = %U.loc20_24.2 (constants.%U)]
// CHECK:STDOUT:     %GetNoDeduce.specific_fn.loc20_53.1: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%T, constants.%U) [symbolic = %GetNoDeduce.specific_fn.loc20_53.2 (constants.%GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:     %.loc20_34.1: ref @GetNoDeduce.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %GetNoDeduce.call: init @GetNoDeduce.%tuple.type (%tuple.type.30b) = call %GetNoDeduce.specific_fn.loc20_53.1(%x.ref) to %.loc20_34.1
// CHECK:STDOUT:     %impl.elem0.loc20_34.1: @GetNoDeduce.%.loc20_34.3 (%.1e9) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_34.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc20_34.1: <bound method> = bound_method %.loc20_34.1, %impl.elem0.loc20_34.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_34.1: <specific function> = specific_impl_function %impl.elem0.loc20_34.1, @Op.1(constants.%Destroy.facet.1c6) [symbolic = %specific_impl_fn.loc20_34.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc20_34.2: <bound method> = bound_method %.loc20_34.1, %specific_impl_fn.loc20_34.1
// CHECK:STDOUT:     %addr: @GetNoDeduce.%ptr (%ptr.937) = addr_of %.loc20_34.1
// CHECK:STDOUT:     %.loc20_34.2: init %empty_tuple.type = call %bound_method.loc20_34.2(%addr)
// CHECK:STDOUT:     return %GetNoDeduce.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc24: %Get.type.501 = specific_constant @Class.%Get.decl, @Class(constants.%A) [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %Get.ref: %Get.type.501 = name_ref Get, %.loc24 [concrete = constants.%Get.f37]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%A, constants.%B) [concrete = constants.%Get.specific_fn.213]
// CHECK:STDOUT:   %.loc23_35: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %Get.call: init %tuple.type.cc6 = call %Get.specific_fn() to %.loc23_35
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc23_35, constants.%Op.f13
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.f13, @Op.2(constants.%tuple.type.cc6) [concrete = constants.%Op.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc23_35, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.3b5 = addr_of %.loc23_35
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_11: %GetNoDeduce.type.5d6 = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %GetNoDeduce.ref: %GetNoDeduce.type.5d6 = name_ref GetNoDeduce, %.loc28_11 [concrete = constants.%GetNoDeduce.162]
// CHECK:STDOUT:   %.loc28_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %B.ref.loc28: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%GetNoDeduce.specific_fn.438]
// CHECK:STDOUT:   %.loc27_54: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %.loc28_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc28_25.3: init %A = class_init (), %.loc28_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc28_25.4: ref %A = temporary %.loc28_25.2, %.loc28_25.3
// CHECK:STDOUT:   %.loc28_25.5: ref %A = converted %.loc28_25.1, %.loc28_25.4
// CHECK:STDOUT:   %.loc28_25.6: %A = bind_value %.loc28_25.5
// CHECK:STDOUT:   %GetNoDeduce.call: init %tuple.type.cc6 = call %GetNoDeduce.specific_fn(%.loc28_25.6) to %.loc27_54
// CHECK:STDOUT:   %Op.bound.loc28: <bound method> = bound_method %.loc28_25.2, constants.%Op.885
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.885, @Op.2(constants.%A) [concrete = constants.%Op.specific_fn.716]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %.loc28_25.2, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc28: %ptr.6db = addr_of %.loc28_25.2
// CHECK:STDOUT:   %no_op.loc28: init %empty_tuple.type = call %bound_method.loc28(%addr.loc28)
// CHECK:STDOUT:   %Op.bound.loc27: <bound method> = bound_method %.loc27_54, constants.%Op.f13
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.f13, @Op.2(constants.%tuple.type.cc6) [concrete = constants.%Op.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method.loc27: <bound method> = bound_method %.loc27_54, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc27: %ptr.3b5 = addr_of %.loc27_54
// CHECK:STDOUT:   %no_op.loc27: init %empty_tuple.type = call %bound_method.loc27(%addr.loc27)
// CHECK:STDOUT:   return %GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.fd9
// CHECK:STDOUT:   %Get => constants.%Get.cf9
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.766
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.c9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc19_10.2 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1 => constants.%require_complete.fe1
// CHECK:STDOUT:   %Get.type => constants.%Get.type.fd9
// CHECK:STDOUT:   %Get => constants.%Get.cf9
// CHECK:STDOUT:   %Get.specific_fn.loc19_39.2 => constants.%Get.specific_fn.f73
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.1c6
// CHECK:STDOUT:   %.loc19_20.3 => constants.%.1e9
// CHECK:STDOUT:   %impl.elem0.loc19_20.2 => constants.%impl.elem0
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2 => constants.%specific_impl_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.937
// CHECK:STDOUT:   %require_complete.loc19_20.2 => constants.%require_complete.8fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %U.loc20_24.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%require_complete.4ae
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.766
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.c9a
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc20_53.2 => constants.%GetNoDeduce.specific_fn.536
// CHECK:STDOUT:   %require_complete.loc20_70 => constants.%require_complete.fe1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.1c6
// CHECK:STDOUT:   %.loc20_34.3 => constants.%.1e9
// CHECK:STDOUT:   %impl.elem0.loc20_34.2 => constants.%impl.elem0
// CHECK:STDOUT:   %specific_impl_fn.loc20_34.2 => constants.%specific_impl_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.937
// CHECK:STDOUT:   %require_complete.loc20_34 => constants.%require_complete.8fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.501
// CHECK:STDOUT:   %Get => constants.%Get.f37
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.5d6
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.loc19_10.2 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.edc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1 => constants.%complete_type.56a
// CHECK:STDOUT:   %Get.type => constants.%Get.type.501
// CHECK:STDOUT:   %Get => constants.%Get.f37
// CHECK:STDOUT:   %Get.specific_fn.loc19_39.2 => constants.%Get.specific_fn.213
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.ae3
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.fae
// CHECK:STDOUT:   %.loc19_20.3 => constants.%.c9f
// CHECK:STDOUT:   %impl.elem0.loc19_20.2 => constants.%Op.f13
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2 => constants.%Op.specific_fn.8d7
// CHECK:STDOUT:   %ptr => constants.%ptr.3b5
// CHECK:STDOUT:   %require_complete.loc19_20.2 => constants.%complete_type.a4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.c10
// CHECK:STDOUT:   %U.loc20_24.2 => constants.%B
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.edc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%complete_type.357
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.5d6
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.162
// CHECK:STDOUT:   %GetNoDeduce.specific_fn.loc20_53.2 => constants.%GetNoDeduce.specific_fn.438
// CHECK:STDOUT:   %require_complete.loc20_70 => constants.%complete_type.56a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.ae3
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.fae
// CHECK:STDOUT:   %.loc20_34.3 => constants.%.c9f
// CHECK:STDOUT:   %impl.elem0.loc20_34.2 => constants.%Op.f13
// CHECK:STDOUT:   %specific_impl_fn.loc20_34.2 => constants.%Op.specific_fn.8d7
// CHECK:STDOUT:   %ptr => constants.%ptr.3b5
// CHECK:STDOUT:   %require_complete.loc20_34 => constants.%complete_type.a4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
