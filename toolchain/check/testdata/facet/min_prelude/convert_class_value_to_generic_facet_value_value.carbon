// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_class_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_class_value_to_generic_facet_value_value.carbon

// --- convert_class_value_to_generic_facet_value_value.carbon

library "[[@TEST_NAME]]";

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod[T:! type, U:! Generic(T)](a: U, s: T) {
  U.F();
}

fn G() {
  CallGenericMethod({} as ImplsGeneric, {} as GenericParam);
}

// --- multiple_generic_params_one_fixed_one_deduced.carbon
library "[[@TEST_NAME]]";

interface I(V:! type, W:! type) {}

class C {}

impl forall [T:! type] C as I(T, ()) {}

fn A[T:! I({}, ())](t: T) {}

fn B() {
  A({} as C);
}

// --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I(V:! type, W:! type) {}

class C {}

impl forall [T:! type] C as I(T, ()) {}

fn A[T:! I({}, {})](t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+10]]:3: error: cannot implicitly convert type `C` into type implementing `I({}, {})` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(I({}, {}))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I({}, {})](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C);
}

// --- fail_mismatch_impl_self_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I {}

class C(V:! type, W:! type) {}

impl forall [T:! type] C(T, ()) as I {}

fn A[T:! I](t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+10]]:3: error: cannot implicitly convert type `C({}, {})` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C({}, {}));
}

// CHECK:STDOUT: --- convert_class_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Scalar.patt: type = symbolic_binding_pattern Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.1: type = fn_type @F.1, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.1: %F.type.f439a9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.de973d.1: type = assoc_entity_type %Generic.type.91ccba.1 [symbolic]
// CHECK:STDOUT:   %assoc0.29ce53.1: %Generic.assoc_type.de973d.1 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %F.type.4cf: type = fn_type @F.1, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %F.118: %F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.9f1: type = assoc_entity_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.9f1 = assoc_entity element0, @Generic.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.17b: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.a56: %F.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet.8bd: %Generic.type.769 = facet_value %ImplsGeneric, (%impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7b2: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.2: type = fn_type @F.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.2: %F.type.f439a9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.de973d.2: type = assoc_entity_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %assoc0.29ce53.2: %Generic.assoc_type.de973d.2 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.02a: <witness> = require_complete_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %U.as_wit.iface0: <witness> = facet_access_witness %U, element0 [symbolic]
// CHECK:STDOUT:   %Generic.facet.0aa: %Generic.type.91ccba.2 = facet_value %U.as_type, (%U.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.10b: type = fn_type_with_self_type %F.type.f439a9.2, %Generic.facet.0aa [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.10b = impl_witness_access %U.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.1(%T, %Generic.facet.0aa) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet.8bd) [concrete]
// CHECK:STDOUT:   %complete_type.997: <witness> = complete_type_witness %Generic.type.769 [concrete]
// CHECK:STDOUT:   %.1eb: type = fn_type_with_self_type %F.type.4cf, %Generic.facet.8bd [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.a56, @F.1(%GenericParam, %Generic.facet.8bd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .GenericParam = %GenericParam.decl
// CHECK:STDOUT:     .ImplsGeneric = %ImplsGeneric.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt.loc4_19.1: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc4_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.loc4_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt.loc15_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_32.2 (constants.%U.patt)]
// CHECK:STDOUT:     %a.patt: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %s.patt: @CallGenericMethod.%T.loc15_22.2 (%T) = binding_pattern s
// CHECK:STDOUT:     %s.param_patt: @CallGenericMethod.%T.loc15_22.2 (%T) = value_param_pattern %s.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:     %.loc15_45: type = splice_block %Generic.type.loc15_45.1 [symbolic = %Generic.type.loc15_45.2 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_44: type = name_ref T, %T.loc15_22.1 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_45.1: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:     %a.param: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_51.1: type = splice_block %.loc15_51.2 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref.loc15: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc15_32.1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc15_51.1: type = facet_access_type %U.ref.loc15 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc15_51.2: type = converted %U.ref.loc15, %U.as_type.loc15_51.1 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type) = bind_name a, %a.param
// CHECK:STDOUT:     %s.param: @CallGenericMethod.%T.loc15_22.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc15_57: type = name_ref T, %T.loc15_22.1 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:     %s: @CallGenericMethod.%T.loc15_22.2 (%T) = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_19.1: type) {
// CHECK:STDOUT:   %Scalar.loc4_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   %Scalar.patt.loc4_19.2: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc4_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%Scalar.loc4_19.2) [symbolic = %F.type (constants.%F.type.f439a9.1)]
// CHECK:STDOUT:   %F: @Generic.%F.type (%F.type.f439a9.1) = struct_value () [symbolic = %F (constants.%F.8a2d67.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic.%Generic.type (%Generic.type.91ccba.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.de973d.1)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.de973d.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.29ce53.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @Generic.%F.type (%F.type.f439a9.1) = fn_decl @F.1 [symbolic = @Generic.%F (constants.%F.8a2d67.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.de973d.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.29ce53.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %F.decl: %F.type.17b = fn_decl @F.2 [concrete = constants.%F.a56] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @GenericParam {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Generic.%Scalar.loc4_19.1: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_22.1: type, %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(%T.loc15_22.2)> [symbolic = %Generic.type.loc15_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc15_32.2: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc15_32.2: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_32.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc15_51.2: type = facet_access_type %U.loc15_32.2 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49: <witness> = require_complete_type @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type) [symbolic = %require_complete.loc15_49 (constants.%require_complete.7b2)]
// CHECK:STDOUT:   %require_complete.loc15_55: <witness> = require_complete_type @CallGenericMethod.%T.loc15_22.2 (%T) [symbolic = %require_complete.loc15_55 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) [symbolic = %require_complete.loc16 (constants.%require_complete.02a)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.de973d.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = assoc_entity element0, @Generic.%F.decl [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:   %U.as_wit.iface0.loc16_4.2: <witness> = facet_access_witness %U.loc15_32.2, element0 [symbolic = %U.as_wit.iface0.loc16_4.2 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%T.loc15_22.2) [symbolic = %F.type (constants.%F.type.f439a9.2)]
// CHECK:STDOUT:   %Generic.facet: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = facet_value %U.as_type.loc15_51.2, (%U.as_wit.iface0.loc16_4.2) [symbolic = %Generic.facet (constants.%Generic.facet.0aa)]
// CHECK:STDOUT:   %.loc16_4.3: type = fn_type_with_self_type %F.type, %Generic.facet [symbolic = %.loc16_4.3 (constants.%.10b)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @CallGenericMethod.%.loc16_4.3 (%.10b) = impl_witness_access %U.as_wit.iface0.loc16_4.2, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc16_4.2: <specific function> = specific_function %impl.elem0.loc16_4.2, @F.1(%T.loc15_22.2, %Generic.facet) [symbolic = %specific_fn.loc16_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc15_22.1: type, %U.patt.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2)](%a.param_patt: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type), %s.param_patt: @CallGenericMethod.%T.loc15_22.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc16: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc15_32.1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:     %.loc16_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = specific_constant @Generic.%assoc0.loc5_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = name_ref F, %.loc16_4.1 [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:     %U.as_type.loc16: type = facet_access_type %U.ref.loc16 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc16_4.2: type = converted %U.ref.loc16, %U.as_type.loc16 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %U.as_wit.iface0.loc16_4.1: <witness> = facet_access_witness %U.ref.loc16, element0 [symbolic = %U.as_wit.iface0.loc16_4.2 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @CallGenericMethod.%.loc16_4.3 (%.10b) = impl_witness_access %U.as_wit.iface0.loc16_4.1, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc16_4.1: <specific function> = specific_function %impl.elem0.loc16_4.1, @F.1(constants.%T, constants.%Generic.facet.0aa) [symbolic = %specific_fn.loc16_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %specific_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %.loc20_22.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %.loc20_22.2: ref %ImplsGeneric = temporary_storage
// CHECK:STDOUT:   %.loc20_22.3: init %ImplsGeneric = class_init (), %.loc20_22.2 [concrete = constants.%ImplsGeneric.val]
// CHECK:STDOUT:   %.loc20_22.4: ref %ImplsGeneric = temporary %.loc20_22.2, %.loc20_22.3
// CHECK:STDOUT:   %.loc20_24.1: ref %ImplsGeneric = converted %.loc20_22.1, %.loc20_22.4
// CHECK:STDOUT:   %.loc20_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc20_42.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc20_42.3: init %GenericParam = class_init (), %.loc20_42.2 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc20_42.4: ref %GenericParam = temporary %.loc20_42.2, %.loc20_42.3
// CHECK:STDOUT:   %.loc20_44.1: ref %GenericParam = converted %.loc20_42.1, %.loc20_42.4
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%impl_witness) [concrete = constants.%Generic.facet.8bd]
// CHECK:STDOUT:   %.loc20_59: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet.8bd]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8bd) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc20_24.2: %ImplsGeneric = bind_value %.loc20_24.1
// CHECK:STDOUT:   %.loc20_44.2: %GenericParam = bind_value %.loc20_44.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc20_24.2, %.loc20_44.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%Scalar
// CHECK:STDOUT:   %Scalar.patt.loc4_19.2 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Scalar, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%Scalar.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%GenericParam
// CHECK:STDOUT:   %Scalar.patt.loc4_19.2 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %F => constants.%F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.9f1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%GenericParam, constants.%Generic.facet.8bd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %Scalar.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.f439a9.2
// CHECK:STDOUT:   %F => constants.%F.8a2d67.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.de973d.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.29ce53.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc15_32.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc15_32.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc15_51.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@CallGenericMethod.%T.loc15_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Generic.facet.0aa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(@CallGenericMethod.%T.loc15_22.2, @CallGenericMethod.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8bd) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc15_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.patt.loc15_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.as_type.loc15_51.2 => constants.%ImplsGeneric
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.997
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.9f1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9b7
// CHECK:STDOUT:   %U.as_wit.iface0.loc16_4.2 => constants.%impl_witness
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %Generic.facet => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %.loc16_4.3 => constants.%.1eb
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%F.a56
// CHECK:STDOUT:   %specific_fn.loc16_4.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %impl_witness.298: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.202: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %T.e96: %I.type.202 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.5b9: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.e96 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c8b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.3d9: <witness> = complete_type_witness %I.type.202 [concrete]
// CHECK:STDOUT:   %impl_witness.e3d: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.202 = facet_value %C, (%impl_witness.e3d) [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc3_13.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc3_23.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.5b9)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc9_24.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc9_24.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.202] {
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_tuple.type)> [concrete = constants.%I.type.202]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.1: %I.type.202 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.e96)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_24.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.202 = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T.e96)]
// CHECK:STDOUT:       %T.as_type.loc9_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_24.2: type = converted %T.ref, %T.as_type.loc9_24.1 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_24.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.1: type, %W.loc3_23.1: type) {
// CHECK:STDOUT:   %V.loc3_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc3_13.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_13.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc3_23.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.2, %W.loc3_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc7_36.2 (%I.type.bea) [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.1: %I.type.202) {
// CHECK:STDOUT:   %T.loc9_6.2: %I.type.202 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.e96)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.5b9)]
// CHECK:STDOUT:   %T.as_type.loc9_24.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc9_24.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.c8b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: %I.type.202](%t.param_patt: @A.%T.as_type.loc9_24.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc12_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_6.3: init %C = class_init (), %.loc12_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_6.4: ref %C = temporary %.loc12_6.2, %.loc12_6.3
// CHECK:STDOUT:   %.loc12_8.1: ref %C = converted %.loc12_6.1, %.loc12_6.4
// CHECK:STDOUT:   %I.facet.loc12_12.1: %I.type.202 = facet_value constants.%C, (constants.%impl_witness.e3d) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.1: %I.type.202 = converted constants.%C, %I.facet.loc12_12.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %I.facet.loc12_12.2: %I.type.202 = facet_value constants.%C, (constants.%impl_witness.e3d) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.2: %I.type.202 = converted constants.%C, %I.facet.loc12_12.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc12_8.2: %C = bind_value %.loc12_8.1
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc12_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%V.loc3_13.2, %W.loc3_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%T.loc7_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.202
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.e96) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.e96
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.e96
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.202
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.e3d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%I.facet
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%I.facet
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.770: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %impl_witness.298: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.906: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %T.4b2: %I.type.906 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6fc: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.4b2 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d5b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.202: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %complete_type.3d9: <witness> = complete_type_witness %I.type.202 [concrete]
// CHECK:STDOUT:   %impl_witness.e3d: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.442: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type.906)> [concrete]
// CHECK:STDOUT:   %Convert.type.b43: type = fn_type @Convert, @ImplicitAs(%I.type.906) [concrete]
// CHECK:STDOUT:   %Convert.2fc: %Convert.type.b43 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.128: type = assoc_entity_type %ImplicitAs.type.442 [concrete]
// CHECK:STDOUT:   %assoc0.e59: %ImplicitAs.assoc_type.128 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc12_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc12_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst65 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc3_13.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc3_23.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.6fc)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc9_24.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc9_24.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.906] {
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_17: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_struct_type)> [concrete = constants.%I.type.906]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.1: %I.type.906 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_24.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.906 = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:       %T.as_type.loc9_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_24.2: type = converted %T.ref, %T.as_type.loc9_24.1 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_24.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.1: type, %W.loc3_23.1: type) {
// CHECK:STDOUT:   %V.loc3_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc3_13.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_13.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc3_23.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.2, %W.loc3_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "toolchain/testing/min_prelude/convert.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc7_36.2 (%I.type.bea) [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.1: %I.type.906) {
// CHECK:STDOUT:   %T.loc9_6.2: %I.type.906 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.6fc)]
// CHECK:STDOUT:   %T.as_type.loc9_24.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc9_24.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.d5b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: %I.type.906](%t.param_patt: @A.%T.as_type.loc9_24.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc22_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc22_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_6.3: init %C = class_init (), %.loc22_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_6.4: ref %C = temporary %.loc22_6.2, %.loc22_6.3
// CHECK:STDOUT:   %.loc22_8: ref %C = converted %.loc22_6.1, %.loc22_6.4
// CHECK:STDOUT:   %.loc22_12: %I.type.906 = converted constants.%C, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "toolchain/testing/min_prelude/convert.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%V.loc3_13.2, %W.loc3_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%T.loc7_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.4b2) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.4b2
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.4b2
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.202
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.e3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.202
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type.906) {
// CHECK:STDOUT:   %Dest => constants.%I.type.906
// CHECK:STDOUT:   %Dest.patt => constants.%I.type.906
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.442
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.b43
// CHECK:STDOUT:   %Convert => constants.%Convert.2fc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.128
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c6b: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.463: type = class_type @C, @C(%T.8b3, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %impl_witness.1a1: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.826: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3ad: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.826 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c74: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.c74 = struct_value () [concrete]
// CHECK:STDOUT:   %C.83b: type = class_type @C, @C(%empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %impl_witness.366: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c42: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.7ef: type = fn_type @Convert, @ImplicitAs(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.c77: %Convert.type.7ef = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.167: type = assoc_entity_type %ImplicitAs.type.c42 [concrete]
// CHECK:STDOUT:   %assoc0.0e0: %ImplicitAs.assoc_type.167 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc12_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc12_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst65 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %V.patt.loc5_9.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_9.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc5_19.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_19.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc5_9.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc5_19.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_31: type = converted %.loc7_30, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc7_31.1: type = class_type @C, @C(constants.%T.8b3, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.1a1)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc9_16.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc9_16.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc9_6.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.826)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_16.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %T.as_type.loc9_16.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T.826)]
// CHECK:STDOUT:       %T.as_type.loc9_16.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_16.2: type = converted %T.ref, %T.as_type.loc9_16.1 [symbolic = %T.as_type.loc9_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_16.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "toolchain/testing/min_prelude/convert.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %C.loc7_31.2: type = class_type @C, @C(%T.loc7_14.2, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness.1a1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_31.1 as %I.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc5_9.1: type, %W.loc5_19.1: type) {
// CHECK:STDOUT:   %V.loc5_9.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc5_9.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_9.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc5_19.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc5_19.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_19.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.c6b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.1: %I.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.826)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   %T.as_type.loc9_16.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc9_16.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: %I.type](%t.param_patt: @A.%T.as_type.loc9_16.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc22_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc22_14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_19.1: type = converted %.loc22_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc22_19.2: type = converted %.loc22_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_struct_type, constants.%empty_struct_type) [concrete = constants.%C.c74]
// CHECK:STDOUT:   %.loc22_6.2: ref %C.c74 = temporary_storage
// CHECK:STDOUT:   %.loc22_6.3: init %C.c74 = class_init (), %.loc22_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_6.4: ref %C.c74 = temporary %.loc22_6.2, %.loc22_6.3
// CHECK:STDOUT:   %.loc22_8: ref %C.c74 = converted %.loc22_6.1, %.loc22_6.4
// CHECK:STDOUT:   %.loc22_20: %I.type = converted constants.%C.c74, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "toolchain/testing/min_prelude/convert.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc5_9.2 => constants.%V
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc5_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %C.loc7_31.2 => constants.%C.463
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc7_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.826) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.826
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.826
// CHECK:STDOUT:   %T.as_type.loc9_16.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc5_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc7_31.2 => constants.%C.83b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc5_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type) {
// CHECK:STDOUT:   %Dest => constants.%I.type
// CHECK:STDOUT:   %Dest.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c42
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7ef
// CHECK:STDOUT:   %Convert => constants.%Convert.c77
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.167
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- toolchain/testing/min_prelude/convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc8_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc8_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc8_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc8_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc12_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc12_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc12_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc12_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc8_14.1: type) {
// CHECK:STDOUT:   %Dest.loc8_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc8_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc8_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc8_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc8_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc8_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc9_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc8_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc9_20.1: type = splice_block %.loc9_20.3 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc9_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc9_20.3: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc9_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc12_22.1: type) {
// CHECK:STDOUT:   %Dest.loc12_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc12_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc12_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc12_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc12_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc12_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc12_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc8_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc12_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc8_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc8_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc12_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc12_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc12_22.2) {}
// CHECK:STDOUT:
