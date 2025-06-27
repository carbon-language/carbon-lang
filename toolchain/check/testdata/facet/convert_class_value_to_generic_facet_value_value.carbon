// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_class_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_class_value_to_generic_facet_value_value.carbon

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
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I({}, {})` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+7]]:3: error: cannot convert type `C({}, {})` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C({}, {}));
}

// CHECK:STDOUT: --- convert_class_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.dee8d8.1: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.1: type = fn_type @F.1, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.1: %F.type.f439a9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd877.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.2966cb.1: %Generic.assoc_type.0fd877.1 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Self.f84: %Generic.type.769 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.4cf: type = fn_type @F.1, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %F.118: %F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.713: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.713 = assoc_entity element0, @Generic.%F.decl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness file.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type.17b: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.a56: %F.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet.8ff: %Generic.type.769 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.f48: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7b2: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Self.dee8d8.2: %Generic.type.91ccba.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.2: type = fn_type @F.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.2: %F.type.f439a9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd877.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.2966cb.2: %Generic.assoc_type.0fd877.2 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.02a: <witness> = require_complete_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.facet.680: %Generic.type.91ccba.2 = facet_value %U.as_type, (%Generic.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.177: type = fn_type_with_self_type %F.type.f439a9.2, %Generic.facet.680 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.177 = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F.1(%T, %Generic.facet.680) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ded: type = pattern_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %pattern_type.c8d: type = pattern_type %ImplsGeneric [concrete]
// CHECK:STDOUT:   %pattern_type.589: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet.8ff) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.abd: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%GenericParam) [concrete]
// CHECK:STDOUT:   %Op.type.ae1: type = fn_type @Op.2, @impl.49c(%GenericParam) [concrete]
// CHECK:STDOUT:   %Op.0cc: %Op.type.ae1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.8f4: %Destroy.type = facet_value %GenericParam, (%Destroy.impl_witness.abd) [concrete]
// CHECK:STDOUT:   %.4c3: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.8f4 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.934: <specific function> = specific_function %Op.0cc, @Op.2(%GenericParam) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.32c: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%ImplsGeneric) [concrete]
// CHECK:STDOUT:   %Op.type.e98: type = fn_type @Op.2, @impl.49c(%ImplsGeneric) [concrete]
// CHECK:STDOUT:   %Op.51a: %Op.type.e98 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ec7: %Destroy.type = facet_value %ImplsGeneric, (%Destroy.impl_witness.32c) [concrete]
// CHECK:STDOUT:   %.de2: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ec7 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.7b0: <specific function> = specific_function %Op.51a, @Op.2(%ImplsGeneric) [concrete]
// CHECK:STDOUT:   %complete_type.997: <witness> = complete_type_witness %Generic.type.769 [concrete]
// CHECK:STDOUT:   %.3e6: type = fn_type_with_self_type %F.type.4cf, %Generic.facet.8ff [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.loc4_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @impl.046 [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.impl_witness_table = impl_witness_table (@impl.046.%F.decl), @impl.046 [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness %Generic.impl_witness_table [concrete = constants.%Generic.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.80f) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.f48) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.f48) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %s.patt: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.7dc) = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.7dc) = value_param_pattern %s.patt, call_param1 [concrete]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee8d8.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%Scalar.loc4_19.2) [symbolic = %F.type (constants.%F.type.f439a9.1)]
// CHECK:STDOUT:   %F: @Generic.%F.type (%F.type.f439a9.1) = struct_value () [symbolic = %F (constants.%F.8a2d67.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc4_19.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.1)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee8d8.1)]
// CHECK:STDOUT:     %F.decl: @Generic.%F.type (%F.type.f439a9.1) = fn_decl @F.1 [symbolic = @Generic.%F (constants.%F.8a2d67.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.046: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %F.decl: %F.type.17b = fn_decl @F.2 [concrete = constants.%F.a56] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%Generic.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @GenericParam {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:   %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(%T.loc15_22.2)> [symbolic = %Generic.type.loc15_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc15_32: type = pattern_type %Generic.type.loc15_45.2 [symbolic = %pattern_type.loc15_32 (constants.%pattern_type.80f)]
// CHECK:STDOUT:   %U.as_type.loc15_51.2: type = facet_access_type %U.loc15_32.2 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type.loc15_48: type = pattern_type %U.as_type.loc15_51.2 [symbolic = %pattern_type.loc15_48 (constants.%pattern_type.f48)]
// CHECK:STDOUT:   %pattern_type.loc15_54: type = pattern_type %T.loc15_22.2 [symbolic = %pattern_type.loc15_54 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49: <witness> = require_complete_type %U.as_type.loc15_51.2 [symbolic = %require_complete.loc15_49 (constants.%require_complete.7b2)]
// CHECK:STDOUT:   %require_complete.loc15_55: <witness> = require_complete_type %T.loc15_22.2 [symbolic = %require_complete.loc15_55 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %Generic.type.loc15_45.2 [symbolic = %require_complete.loc16 (constants.%require_complete.02a)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc15_22.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = assoc_entity element0, @Generic.%F.decl [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc15_32.2, @Generic, @Generic(%T.loc15_22.2) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%T.loc15_22.2) [symbolic = %F.type (constants.%F.type.f439a9.2)]
// CHECK:STDOUT:   %Generic.facet: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = facet_value %U.as_type.loc15_51.2, (%Generic.lookup_impl_witness) [symbolic = %Generic.facet (constants.%Generic.facet.680)]
// CHECK:STDOUT:   %.loc16_4.3: type = fn_type_with_self_type %F.type, %Generic.facet [symbolic = %.loc16_4.3 (constants.%.177)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @CallGenericMethod.%.loc16_4.3 (%.177) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @F.1(%T.loc15_22.2, %Generic.facet) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @CallGenericMethod.%U.as_type.loc15_51.2 (%U.as_type), %s.param: @CallGenericMethod.%T.loc15_22.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc16: @CallGenericMethod.%Generic.type.loc15_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc15_32.1 [symbolic = %U.loc15_32.2 (constants.%U)]
// CHECK:STDOUT:     %.loc16_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = specific_constant @Generic.%assoc0.loc5_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = name_ref F, %.loc16_4.1 [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %U.as_type.loc16: type = facet_access_type %U.ref.loc16 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc16_4.2: type = converted %U.ref.loc16, %U.as_type.loc16 [symbolic = %U.as_type.loc15_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @CallGenericMethod.%.loc16_4.3 (%.177) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @F.1(constants.%T, constants.%Generic.facet.680) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc16_7: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
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
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet.8ff]
// CHECK:STDOUT:   %.loc20_59: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet.8ff]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8ff) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc20_24.2: %ImplsGeneric = bind_value %.loc20_24.1
// CHECK:STDOUT:   %.loc20_44.2: %GenericParam = bind_value %.loc20_44.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc20_24.2, %.loc20_44.2)
// CHECK:STDOUT:   %impl.elem0.loc20_42: %.4c3 = impl_witness_access constants.%Destroy.impl_witness.abd, element0 [concrete = constants.%Op.0cc]
// CHECK:STDOUT:   %bound_method.loc20_42.1: <bound method> = bound_method %.loc20_42.2, %impl.elem0.loc20_42
// CHECK:STDOUT:   %specific_fn.loc20_42: <specific function> = specific_function %impl.elem0.loc20_42, @Op.2(constants.%GenericParam) [concrete = constants.%Op.specific_fn.934]
// CHECK:STDOUT:   %bound_method.loc20_42.2: <bound method> = bound_method %.loc20_42.2, %specific_fn.loc20_42
// CHECK:STDOUT:   %.loc20_42.5: %GenericParam = bind_value %.loc20_42.2
// CHECK:STDOUT:   %no_op.loc20_42: init %empty_tuple.type = call %bound_method.loc20_42.2(%.loc20_42.5)
// CHECK:STDOUT:   %impl.elem0.loc20_22: %.de2 = impl_witness_access constants.%Destroy.impl_witness.32c, element0 [concrete = constants.%Op.51a]
// CHECK:STDOUT:   %bound_method.loc20_22.1: <bound method> = bound_method %.loc20_22.2, %impl.elem0.loc20_22
// CHECK:STDOUT:   %specific_fn.loc20_22: <specific function> = specific_function %impl.elem0.loc20_22, @Op.2(constants.%ImplsGeneric) [concrete = constants.%Op.specific_fn.7b0]
// CHECK:STDOUT:   %bound_method.loc20_22.2: <bound method> = bound_method %.loc20_22.2, %specific_fn.loc20_22
// CHECK:STDOUT:   %.loc20_22.5: %ImplsGeneric = bind_value %.loc20_22.2
// CHECK:STDOUT:   %no_op.loc20_22: init %empty_tuple.type = call %bound_method.loc20_22.2(%.loc20_22.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Scalar, constants.%Self.dee8d8.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self.f84
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %F => constants.%F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%GenericParam, constants.%Generic.facet.8ff) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_19.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.dee8d8.2
// CHECK:STDOUT:   %F.type => constants.%F.type.f439a9.2
// CHECK:STDOUT:   %F => constants.%F.8a2d67.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.0fd877.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.2966cb.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc15_32.2 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.80f
// CHECK:STDOUT:   %U.as_type.loc15_51.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.f48
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Generic.facet.680) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8ff) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc15_32.2 => constants.%Generic.facet.8ff
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.ded
// CHECK:STDOUT:   %U.as_type.loc15_51.2 => constants.%ImplsGeneric
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.c8d
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.589
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.997
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9b7
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %Generic.facet => constants.%Generic.facet.8ff
// CHECK:STDOUT:   %.loc16_4.3 => constants.%.3e6
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%F.a56
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%F.a56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.770: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %Self.ec5: %I.type.bea = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %I.impl_witness.989: <witness> = impl_witness file.%I.impl_witness_table, @impl.0dc(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.202: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %T.e96: %I.type.202 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a84: type = pattern_type %I.type.202 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.e96 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ff: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c8b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Self.38c: %I.type.202 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %complete_type.3d9: <witness> = complete_type_witness %I.type.202 [concrete]
// CHECK:STDOUT:   %I.impl_witness.806: <witness> = impl_witness file.%I.impl_witness_table, @impl.0dc(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.202 = facet_value %C, (%I.impl_witness.806) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.906: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.2, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.impl_witness.906) [concrete]
// CHECK:STDOUT:   %.b80: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.5d7, @Op.2(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.0dc [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @impl.0dc [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @impl.0dc(constants.%T.8b3) [symbolic = @impl.0dc.%I.impl_witness (constants.%I.impl_witness.989)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a84 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.3ff) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.3ff) = value_param_pattern %t.patt, call_param0 [concrete]
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
// CHECK:STDOUT:   %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.2, %W.loc3_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
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
// CHECK:STDOUT: generic impl @impl.0dc(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl.0dc(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.989)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.1: %I.type.202) {
// CHECK:STDOUT:   %T.loc9_6.2: %I.type.202 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.e96)]
// CHECK:STDOUT:   %T.as_type.loc9_24.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_24.2 [symbolic = %pattern_type (constants.%pattern_type.3ff)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_24.2 [symbolic = %require_complete (constants.%require_complete.c8b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_24.2 (%T.as_type)) {
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
// CHECK:STDOUT:   %I.facet.loc12_12.1: %I.type.202 = facet_value constants.%C, (constants.%I.impl_witness.806) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.1: %I.type.202 = converted constants.%C, %I.facet.loc12_12.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %I.facet.loc12_12.2: %I.type.202 = facet_value constants.%C, (constants.%I.impl_witness.806) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.2: %I.type.202 = converted constants.%C, %I.facet.loc12_12.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc12_8.2: %C = bind_value %.loc12_8.1
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc12_8.2)
// CHECK:STDOUT:   %impl.elem0: %.b80 = impl_witness_access constants.%Destroy.impl_witness.906, element0 [concrete = constants.%Op.5d7]
// CHECK:STDOUT:   %bound_method.loc12_6.1: <bound method> = bound_method %.loc12_6.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%C) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_6.2: <bound method> = bound_method %.loc12_6.2, %specific_fn
// CHECK:STDOUT:   %.loc12_6.5: %C = bind_value %.loc12_6.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc12_6.2(%.loc12_6.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self.ec5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.0dc(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.989
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.202
// CHECK:STDOUT:   %Self.2 => constants.%Self.38c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.e96) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.e96
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.0dc(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.202
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d9
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.806
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%I.facet
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.770: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %Self.ec5: %I.type.bea = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl.0dc(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.906: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %T.4b2: %I.type.906 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a62: type = pattern_type %I.type.906 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.4b2 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1c: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d5b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.906: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.2, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.impl_witness.906) [concrete]
// CHECK:STDOUT:   %.b80: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.5d7, @Op.2(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.0dc [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @impl.0dc [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @impl.0dc(constants.%T.8b3) [symbolic = @impl.0dc.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a62 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.f1c) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.f1c) = value_param_pattern %t.patt, call_param0 [concrete]
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
// CHECK:STDOUT:   %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.2 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.2, %W.loc3_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
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
// CHECK:STDOUT: generic impl @impl.0dc(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl.0dc(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.1: %I.type.906) {
// CHECK:STDOUT:   %T.loc9_6.2: %I.type.906 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:   %T.as_type.loc9_24.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_24.2 [symbolic = %pattern_type (constants.%pattern_type.f1c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_24.2 [symbolic = %require_complete (constants.%require_complete.d5b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_24.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc19_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_6.3: init %C = class_init (), %.loc19_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_6.4: ref %C = temporary %.loc19_6.2, %.loc19_6.3
// CHECK:STDOUT:   %.loc19_8: ref %C = converted %.loc19_6.1, %.loc19_6.4
// CHECK:STDOUT:   %impl.elem0: %.b80 = impl_witness_access constants.%Destroy.impl_witness.906, element0 [concrete = constants.%Op.5d7]
// CHECK:STDOUT:   %bound_method.loc19_6.1: <bound method> = bound_method %.loc19_6.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%C) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_6.2: <bound method> = bound_method %.loc19_6.2, %specific_fn
// CHECK:STDOUT:   %.loc19_6.5: %C = bind_value %.loc19_6.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc19_6.2(%.loc19_6.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self.ec5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.0dc(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc3_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.4b2) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.4b2
// CHECK:STDOUT:   %T.as_type.loc9_24.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f1c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c6b: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.463: type = class_type @C, @C(%T.8b3, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl.a1a(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.826: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c74: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.c74 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.efa: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C.c74) [concrete]
// CHECK:STDOUT:   %Op.type.53e: type = fn_type @Op.2, @impl.49c(%C.c74) [concrete]
// CHECK:STDOUT:   %Op.2b8: %Op.type.53e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.c74, (%Destroy.impl_witness.efa) [concrete]
// CHECK:STDOUT:   %.77a: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.2b8, @Op.2(%C.c74) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc5_9.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc5_19.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.a1a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_31: type = converted %.loc7_30, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc7_31.1: type = class_type @C, @C(constants.%T.8b3, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @impl.a1a [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @impl.a1a(constants.%T.8b3) [symbolic = @impl.a1a.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.6de) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.6de) = value_param_pattern %t.patt, call_param0 [concrete]
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
// CHECK:STDOUT: generic impl @impl.a1a(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %C.loc7_31.2: type = class_type @C, @C(%T.loc7_14.2, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl.a1a(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_31.1 as %I.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc5_9.1: type, %W.loc5_19.1: type) {
// CHECK:STDOUT:   %V.loc5_9.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.2 (constants.%V)]
// CHECK:STDOUT:   %W.loc5_19.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.2 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:   %T.as_type.loc9_16.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_16.2 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_16.2 [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_16.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc19_14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_19.1: type = converted %.loc19_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc19_19.2: type = converted %.loc19_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_struct_type, constants.%empty_struct_type) [concrete = constants.%C.c74]
// CHECK:STDOUT:   %.loc19_6.2: ref %C.c74 = temporary_storage
// CHECK:STDOUT:   %.loc19_6.3: init %C.c74 = class_init (), %.loc19_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_6.4: ref %C.c74 = temporary %.loc19_6.2, %.loc19_6.3
// CHECK:STDOUT:   %.loc19_8: ref %C.c74 = converted %.loc19_6.1, %.loc19_6.4
// CHECK:STDOUT:   %impl.elem0: %.77a = impl_witness_access constants.%Destroy.impl_witness.efa, element0 [concrete = constants.%Op.2b8]
// CHECK:STDOUT:   %bound_method.loc19_6.1: <bound method> = bound_method %.loc19_6.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%C.c74) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_6.2: <bound method> = bound_method %.loc19_6.2, %specific_fn
// CHECK:STDOUT:   %.loc19_6.5: %C.c74 = bind_value %.loc19_6.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc19_6.2(%.loc19_6.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%V
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.a1a(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %C.loc7_31.2 => constants.%C.463
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.826) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.826
// CHECK:STDOUT:   %T.as_type.loc9_16.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc5_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_19.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
