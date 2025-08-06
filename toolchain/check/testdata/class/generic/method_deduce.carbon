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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b44: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %Class.Get.type.fd9: type = fn_type @Class.Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Get.cf9: %Class.Get.type.fd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.766: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.c9a: %Class.GetNoDeduce.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.95a: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.937: type = ptr_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %Class.Get.specific_fn.f73: <specific function> = specific_function %Class.Get.cf9, @Class.Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.bc9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.46f: %T.as.Destroy.impl.Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %ptr.937 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.30b, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1c6: %Destroy.type = facet_value %tuple.type.30b, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.1e9: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1c6 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.1e9 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.Op(%Destroy.facet.1c6) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.536: <specific function> = specific_function %Class.GetNoDeduce.c9a, @Class.GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.480: type = class_type @Class, @Class(%A) [concrete]
// CHECK:STDOUT:   %pattern_type.827: type = pattern_type %Class.480 [concrete]
// CHECK:STDOUT:   %tuple.type.cc6: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %pattern_type.edc: type = pattern_type %tuple.type.cc6 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.type.501: type = fn_type @Class.Get, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.Get.f37: %Class.Get.type.501 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.type.5d6: type = fn_type @Class.GetNoDeduce, @Class(%A) [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.162: %Class.GetNoDeduce.type.5d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn.213: <specific function> = specific_function %Class.Get.f37, @Class.Get(%A, %B) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae3: <witness> = impl_witness imports.%Destroy.impl_witness_table, @T.as.Destroy.impl(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.87c: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.f13: %T.as.Destroy.impl.Op.type.87c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3b5: type = ptr_type %tuple.type.cc6 [concrete]
// CHECK:STDOUT:   %complete_type.a4a: <witness> = complete_type_witness %ptr.3b5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.fae: %Destroy.type = facet_value %tuple.type.cc6, (%Destroy.impl_witness.ae3) [concrete]
// CHECK:STDOUT:   %.c9f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.fae [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.f13, @T.as.Destroy.impl.Op(%tuple.type.cc6) [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [concrete]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.438: <specific function> = specific_function %Class.GetNoDeduce.162, @Class.GetNoDeduce(%A, %B) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0b9: @T.as.Destroy.impl.%T.as.Destroy.impl.Op.type (%T.as.Destroy.impl.Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @T.as.Destroy.impl.%T.as.Destroy.impl.Op (constants.%T.as.Destroy.impl.Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @T.as.Destroy.impl [concrete]
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
// CHECK:STDOUT: impl @A.as.Destroy.impl: constants.%A as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: constants.%B as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc18_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Class.fe1 as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc18_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc18_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1)] {
// CHECK:STDOUT:         %.loc18_23.3: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc18_23.3 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.b44]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.40d]
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
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T.loc18_13.1) [symbolic = %Class.Get.type (constants.%Class.Get.type.fd9)]
// CHECK:STDOUT:   %Class.Get: @Class.%Class.Get.type (%Class.Get.type.fd9) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.cf9)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T.loc18_13.1) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.766)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.766) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.c9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.Get.decl: @Class.%Class.Get.type (%Class.Get.type.fd9) = fn_decl @Class.Get [symbolic = @Class.%Class.Get (constants.%Class.Get.cf9)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.Get.%pattern_type (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.Get.%pattern_type (%pattern_type.65c) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc19_27: type = name_ref U, %U.loc19_10.2 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:       %.loc19_28.1: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc19_27)
// CHECK:STDOUT:       %.loc19_28.2: type = converted %.loc19_28.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %U.loc19_10.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.Get.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.Get.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetNoDeduce.decl: @Class.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.766) = fn_decl @Class.GetNoDeduce [symbolic = @Class.%Class.GetNoDeduce (constants.%Class.GetNoDeduce.c9a)] {
// CHECK:STDOUT:       %x.patt: @Class.GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @Class.GetNoDeduce.%pattern_type.loc20_18 (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetNoDeduce.%pattern_type.loc20_34 (%pattern_type.65c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc20_38: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc20_41: type = name_ref U, %U.loc20_24.2 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:       %.loc20_42.1: %tuple.type.24b = tuple_literal (%T.ref.loc20_38, %U.ref.loc20_41)
// CHECK:STDOUT:       %.loc20_42.2: type = converted %.loc20_42.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %x.param: @Class.GetNoDeduce.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc20_21: type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @Class.GetNoDeduce.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:       %U.loc20_24.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.30b) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %Class.GetNoDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc18_13.2: type, %U.loc19_10.2: type) {
// CHECK:STDOUT:   %U.loc19_10.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc19_10.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc19_20.1 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T) [symbolic = %Class.Get.type (constants.%Class.Get.type.fd9)]
// CHECK:STDOUT:   %Class.Get: @Class.Get.%Class.Get.type (%Class.Get.type.fd9) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.cf9)]
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2: <specific function> = specific_function %Class.Get, @Class.Get(%T, %U.loc19_10.1) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.f73)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1c6)]
// CHECK:STDOUT:   %.loc19_20.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc19_20.3 (constants.%.1e9)]
// CHECK:STDOUT:   %impl.elem0.loc19_20.2: @Class.Get.%.loc19_20.3 (%.1e9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_20.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2: <specific function> = specific_impl_function %impl.elem0.loc19_20.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc19_20.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.937)]
// CHECK:STDOUT:   %require_complete.loc19_20.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc19_20.2 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Class.Get.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_39: @Class.Get.%Class.Get.type (%Class.Get.type.fd9) = specific_constant @Class.%Class.Get.decl, @Class(constants.%T) [symbolic = %Class.Get (constants.%Class.Get.cf9)]
// CHECK:STDOUT:     %Get.ref: @Class.Get.%Class.Get.type (%Class.Get.type.fd9) = name_ref Get, %.loc19_39 [symbolic = %Class.Get (constants.%Class.Get.cf9)]
// CHECK:STDOUT:     %U.ref.loc19_43: type = name_ref U, %U.loc19_10.2 [symbolic = %U.loc19_10.1 (constants.%U)]
// CHECK:STDOUT:     %Class.Get.specific_fn.loc19_39.1: <specific function> = specific_function %Get.ref, @Class.Get(constants.%T, constants.%U) [symbolic = %Class.Get.specific_fn.loc19_39.2 (constants.%Class.Get.specific_fn.f73)]
// CHECK:STDOUT:     %.loc19_20.1: ref @Class.Get.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %Class.Get.call: init @Class.Get.%tuple.type (%tuple.type.30b) = call %Class.Get.specific_fn.loc19_39.1() to %.loc19_20.1
// CHECK:STDOUT:     %impl.elem0.loc19_20.1: @Class.Get.%.loc19_20.3 (%.1e9) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_20.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc19_20.1: <bound method> = bound_method %.loc19_20.1, %impl.elem0.loc19_20.1
// CHECK:STDOUT:     %specific_impl_fn.loc19_20.1: <specific function> = specific_impl_function %impl.elem0.loc19_20.1, @Destroy.Op(constants.%Destroy.facet.1c6) [symbolic = %specific_impl_fn.loc19_20.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc19_20.2: <bound method> = bound_method %.loc19_20.1, %specific_impl_fn.loc19_20.1
// CHECK:STDOUT:     %addr: @Class.Get.%ptr (%ptr.937) = addr_of %.loc19_20.1
// CHECK:STDOUT:     %.loc19_20.2: init %empty_tuple.type = call %bound_method.loc19_20.2(%addr)
// CHECK:STDOUT:     return %Class.Get.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetNoDeduce(@Class.%T.loc18_13.2: type, %U.loc20_24.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc20_18: type = pattern_type %T [symbolic = %pattern_type.loc20_18 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %U.loc20_24.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T, %U.loc20_24.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type.loc20_34: type = pattern_type %tuple.type [symbolic = %pattern_type.loc20_34 (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19: <witness> = require_complete_type %T [symbolic = %require_complete.loc20_19 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class.GetNoDeduce.type: type = fn_type @Class.GetNoDeduce, @Class(%T) [symbolic = %Class.GetNoDeduce.type (constants.%Class.GetNoDeduce.type.766)]
// CHECK:STDOUT:   %Class.GetNoDeduce: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.766) = struct_value () [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.c9a)]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2: <specific function> = specific_function %Class.GetNoDeduce, @Class.GetNoDeduce(%T, %U.loc20_24.1) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:   %require_complete.loc20_70: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc20_70 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1c6)]
// CHECK:STDOUT:   %.loc20_34.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc20_34.3 (constants.%.1e9)]
// CHECK:STDOUT:   %impl.elem0.loc20_34.2: @Class.GetNoDeduce.%.loc20_34.3 (%.1e9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_34.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_34.2: <specific function> = specific_impl_function %impl.elem0.loc20_34.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc20_34.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.937)]
// CHECK:STDOUT:   %require_complete.loc20_34: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc20_34 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Class.GetNoDeduce.%T (%T)) -> %return.param: @Class.GetNoDeduce.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc20_53: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.766) = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%T) [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.c9a)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @Class.GetNoDeduce.%Class.GetNoDeduce.type (%Class.GetNoDeduce.type.766) = name_ref GetNoDeduce, %.loc20_53 [symbolic = %Class.GetNoDeduce (constants.%Class.GetNoDeduce.c9a)]
// CHECK:STDOUT:     %x.ref: @Class.GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc20_68: type = name_ref U, %U.loc20_24.2 [symbolic = %U.loc20_24.1 (constants.%U)]
// CHECK:STDOUT:     %Class.GetNoDeduce.specific_fn.loc20_53.1: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%T, constants.%U) [symbolic = %Class.GetNoDeduce.specific_fn.loc20_53.2 (constants.%Class.GetNoDeduce.specific_fn.536)]
// CHECK:STDOUT:     %.loc20_34.1: ref @Class.GetNoDeduce.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %Class.GetNoDeduce.call: init @Class.GetNoDeduce.%tuple.type (%tuple.type.30b) = call %Class.GetNoDeduce.specific_fn.loc20_53.1(%x.ref) to %.loc20_34.1
// CHECK:STDOUT:     %impl.elem0.loc20_34.1: @Class.GetNoDeduce.%.loc20_34.3 (%.1e9) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_34.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc20_34.1: <bound method> = bound_method %.loc20_34.1, %impl.elem0.loc20_34.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_34.1: <specific function> = specific_impl_function %impl.elem0.loc20_34.1, @Destroy.Op(constants.%Destroy.facet.1c6) [symbolic = %specific_impl_fn.loc20_34.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc20_34.2: <bound method> = bound_method %.loc20_34.1, %specific_impl_fn.loc20_34.1
// CHECK:STDOUT:     %addr: @Class.GetNoDeduce.%ptr (%ptr.937) = addr_of %.loc20_34.1
// CHECK:STDOUT:     %.loc20_34.2: init %empty_tuple.type = call %bound_method.loc20_34.2(%addr)
// CHECK:STDOUT:     return %Class.GetNoDeduce.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc18_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc24: %Class.Get.type.501 = specific_constant @Class.%Class.Get.decl, @Class(constants.%A) [concrete = constants.%Class.Get.f37]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.501 = name_ref Get, %.loc24 [concrete = constants.%Class.Get.f37]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%A, constants.%B) [concrete = constants.%Class.Get.specific_fn.213]
// CHECK:STDOUT:   %.loc23_35: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %Class.Get.call: init %tuple.type.cc6 = call %Class.Get.specific_fn() to %.loc23_35
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc23_35, constants.%T.as.Destroy.impl.Op.f13
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f13, @T.as.Destroy.impl.Op(constants.%tuple.type.cc6) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc23_35, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.3b5 = addr_of %.loc23_35
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Class.Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param: %Class.480) -> %return.param: %tuple.type.cc6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.480 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_11: %Class.GetNoDeduce.type.5d6 = specific_constant @Class.%Class.GetNoDeduce.decl, @Class(constants.%A) [concrete = constants.%Class.GetNoDeduce.162]
// CHECK:STDOUT:   %GetNoDeduce.ref: %Class.GetNoDeduce.type.5d6 = name_ref GetNoDeduce, %.loc28_11 [concrete = constants.%Class.GetNoDeduce.162]
// CHECK:STDOUT:   %.loc28_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %B.ref.loc28: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn: <specific function> = specific_function %GetNoDeduce.ref, @Class.GetNoDeduce(constants.%A, constants.%B) [concrete = constants.%Class.GetNoDeduce.specific_fn.438]
// CHECK:STDOUT:   %.loc27_54: ref %tuple.type.cc6 = splice_block %return {}
// CHECK:STDOUT:   %.loc28_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc28_25.3: init %A = class_init (), %.loc28_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc28_25.4: ref %A = temporary %.loc28_25.2, %.loc28_25.3
// CHECK:STDOUT:   %.loc28_25.5: ref %A = converted %.loc28_25.1, %.loc28_25.4
// CHECK:STDOUT:   %.loc28_25.6: %A = bind_value %.loc28_25.5
// CHECK:STDOUT:   %Class.GetNoDeduce.call: init %tuple.type.cc6 = call %Class.GetNoDeduce.specific_fn(%.loc28_25.6) to %.loc27_54
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_25.2, constants.%A.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc28: %ptr.6db = addr_of %.loc28_25.2
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.call: init %empty_tuple.type = call %A.as.Destroy.impl.Op.bound(%addr.loc28)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc27_54, constants.%T.as.Destroy.impl.Op.f13
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f13, @T.as.Destroy.impl.Op(constants.%tuple.type.cc6) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc27_54, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc27: %ptr.3b5 = addr_of %.loc27_54
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc27)
// CHECK:STDOUT:   return %Class.GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.fd9
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.cf9
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.766
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.c9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc19_10.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1 => constants.%require_complete.fe1
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.fd9
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.cf9
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.f73
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.1c6
// CHECK:STDOUT:   %.loc19_20.3 => constants.%.1e9
// CHECK:STDOUT:   %impl.elem0.loc19_20.2 => constants.%impl.elem0
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2 => constants.%specific_impl_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.937
// CHECK:STDOUT:   %require_complete.loc19_20.2 => constants.%require_complete.8fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %U.loc20_24.1 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.766
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.c9a
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.536
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
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.95a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.501
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.f37
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.5d6
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.loc19_10.1 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.edc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19_20.1 => constants.%complete_type.56a
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.501
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.f37
// CHECK:STDOUT:   %Class.Get.specific_fn.loc19_39.2 => constants.%Class.Get.specific_fn.213
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.ae3
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.fae
// CHECK:STDOUT:   %.loc19_20.3 => constants.%.c9f
// CHECK:STDOUT:   %impl.elem0.loc19_20.2 => constants.%T.as.Destroy.impl.Op.f13
// CHECK:STDOUT:   %specific_impl_fn.loc19_20.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.3b5
// CHECK:STDOUT:   %require_complete.loc19_20.2 => constants.%complete_type.a4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.c10
// CHECK:STDOUT:   %U.loc20_24.1 => constants.%B
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cc6
// CHECK:STDOUT:   %pattern_type.loc20_34 => constants.%pattern_type.edc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_19 => constants.%complete_type.357
// CHECK:STDOUT:   %Class.GetNoDeduce.type => constants.%Class.GetNoDeduce.type.5d6
// CHECK:STDOUT:   %Class.GetNoDeduce => constants.%Class.GetNoDeduce.162
// CHECK:STDOUT:   %Class.GetNoDeduce.specific_fn.loc20_53.2 => constants.%Class.GetNoDeduce.specific_fn.438
// CHECK:STDOUT:   %require_complete.loc20_70 => constants.%complete_type.56a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.ae3
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.fae
// CHECK:STDOUT:   %.loc20_34.3 => constants.%.c9f
// CHECK:STDOUT:   %impl.elem0.loc20_34.2 => constants.%T.as.Destroy.impl.Op.f13
// CHECK:STDOUT:   %specific_impl_fn.loc20_34.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.3b5
// CHECK:STDOUT:   %require_complete.loc20_34 => constants.%complete_type.a4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
