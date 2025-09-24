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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.2db63e.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.8997e7.1: %Generic.type.2db63e.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.f439a9.1: type = fn_type @Generic.F, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %Generic.F.8a2d67.1: %Generic.F.type.f439a9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd877.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.2966cb.1: %Generic.assoc_type.0fd877.1 = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.621: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Self.93d: %Generic.type.621 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.4cf: type = fn_type @Generic.F, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %Generic.F.118: %Generic.F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.713: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.713 = assoc_entity element0, @Generic.%Generic.F.decl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness file.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.621 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.2db63e.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.2db63e.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e93: type = pattern_type %Generic.type.2db63e.2 [symbolic]
// CHECK:STDOUT:   %Self.8997e7.2: %Generic.type.2db63e.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.f439a9.2: type = fn_type @Generic.F, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.F.8a2d67.2: %Generic.F.type.f439a9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd877.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.2966cb.2: %Generic.assoc_type.0fd877.2 = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.70f: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.26c: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.9a6: <witness> = require_complete_type %Generic.type.2db63e.2 [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %.468: type = fn_type_with_self_type %Generic.F.type.f439a9.2, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0.31b: %.468 = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6af: <specific function> = specific_impl_function %impl.elem0.31b, @Generic.F(%T, %U) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c4b: type = pattern_type %Generic.type.621 [concrete]
// CHECK:STDOUT:   %pattern_type.c8d: type = pattern_type %ImplsGeneric [concrete]
// CHECK:STDOUT:   %pattern_type.589: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.6cb: %type_where = facet_value %GenericParam, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.13c: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.6cb) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.a0a: %AggregateT.as_type.as.Destroy.impl.Op.type.13c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f73: type = ptr_type %GenericParam [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.994: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.a0a, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.6cb) [concrete]
// CHECK:STDOUT:   %facet_value.004: %type_where = facet_value %ImplsGeneric, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.028: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.004) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.4ac: %AggregateT.as_type.as.Destroy.impl.Op.type.028 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.011: type = ptr_type %ImplsGeneric [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e4a: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.4ac, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.004) [concrete]
// CHECK:STDOUT:   %complete_type.ca2: <witness> = complete_type_witness %Generic.type.621 [concrete]
// CHECK:STDOUT:   %.573: type = fn_type_with_self_type %Generic.F.type.4cf, %Generic.facet [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Scalar.loc4_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.621]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.impl_witness_table = impl_witness_table (@ImplsGeneric.as.Generic.impl.%ImplsGeneric.as.Generic.impl.F.decl), @ImplsGeneric.as.Generic.impl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness %Generic.impl_witness_table [concrete = constants.%Generic.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.e93) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.70f) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.70f) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %s.patt: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.7dc) = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.7dc) = value_param_pattern %s.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_45: type = splice_block %Generic.type.loc15_45.2 [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.2db63e.2)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_44: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.2db63e.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.2db63e.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %a.param: @CallGenericMethod.%U.as_type.loc15_51.1 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_51.1: type = splice_block %.loc15_51.2 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref.loc15: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.2db63e.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc15_51.2: type = facet_access_type %U.ref.loc15 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc15_51.2: type = converted %U.ref.loc15, %U.as_type.loc15_51.2 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @CallGenericMethod.%U.as_type.loc15_51.1 (%U.as_type) = bind_name a, %a.param
// CHECK:STDOUT:     %s.param: @CallGenericMethod.%T.loc15_22.1 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc15_57: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %s: @CallGenericMethod.%T.loc15_22.1 (%T) = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.2db63e.1)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.2db63e.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8997e7.1)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.f439a9.1)]
// CHECK:STDOUT:   %Generic.F: @Generic.%Generic.F.type (%Generic.F.type.f439a9.1) = struct_value () [symbolic = %Generic.F (constants.%Generic.F.8a2d67.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.1)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.2db63e.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8997e7.1)]
// CHECK:STDOUT:     %Generic.F.decl: @Generic.%Generic.F.type (%Generic.F.type.f439a9.1) = fn_decl @Generic.F [symbolic = @Generic.%Generic.F (constants.%Generic.F.8a2d67.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%Generic.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplsGeneric.as.Generic.impl: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.decl: %ImplsGeneric.as.Generic.impl.F.type = fn_decl @ImplsGeneric.as.Generic.impl.F [concrete = constants.%ImplsGeneric.as.Generic.impl.F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %ImplsGeneric.as.Generic.impl.F.decl
// CHECK:STDOUT:   witness = file.%Generic.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @GenericParam {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.F(@Generic.%Scalar.loc4_19.2: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.2db63e.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_22.2: type, %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.2db63e.2)) {
// CHECK:STDOUT:   %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_45.1: type = facet_type <@Generic, @Generic(%T.loc15_22.1)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.2db63e.2)]
// CHECK:STDOUT:   %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.2db63e.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc15_32: type = pattern_type %Generic.type.loc15_45.1 [symbolic = %pattern_type.loc15_32 (constants.%pattern_type.e93)]
// CHECK:STDOUT:   %U.as_type.loc15_51.1: type = facet_access_type %U.loc15_32.1 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type.loc15_48: type = pattern_type %U.as_type.loc15_51.1 [symbolic = %pattern_type.loc15_48 (constants.%pattern_type.70f)]
// CHECK:STDOUT:   %pattern_type.loc15_54: type = pattern_type %T.loc15_22.1 [symbolic = %pattern_type.loc15_54 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49: <witness> = require_complete_type %U.as_type.loc15_51.1 [symbolic = %require_complete.loc15_49 (constants.%require_complete.26c)]
// CHECK:STDOUT:   %require_complete.loc15_55: <witness> = require_complete_type %T.loc15_22.1 [symbolic = %require_complete.loc15_55 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %Generic.type.loc15_45.1 [symbolic = %require_complete.loc16 (constants.%require_complete.9a6)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc15_22.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = assoc_entity element0, @Generic.%Generic.F.decl [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc15_32.1, @Generic, @Generic(%T.loc15_22.1) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%T.loc15_22.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.f439a9.2)]
// CHECK:STDOUT:   %.loc16_4.3: type = fn_type_with_self_type %Generic.F.type, %U.loc15_32.1 [symbolic = %.loc16_4.3 (constants.%.468)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @CallGenericMethod.%.loc16_4.3 (%.468) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.31b)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Generic.F(%T.loc15_22.1, %U.loc15_32.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.6af)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @CallGenericMethod.%U.as_type.loc15_51.1 (%U.as_type), %s.param: @CallGenericMethod.%T.loc15_22.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc16: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.2db63e.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %.loc16_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = specific_constant @Generic.%assoc0.loc5_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = name_ref F, %.loc16_4.1 [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %U.as_type.loc16: type = facet_access_type %U.ref.loc16 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc16_4.2: type = converted %U.ref.loc16, %U.as_type.loc16 [symbolic = %U.as_type.loc15_51.1 (constants.%U.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @CallGenericMethod.%.loc16_4.3 (%.468) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.31b)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Generic.F(constants.%T, constants.%U) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.6af)]
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
// CHECK:STDOUT:   %Generic.facet: %Generic.type.621 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc20_59: %Generic.type.621 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc20_24.2: %ImplsGeneric = bind_value %.loc20_24.1
// CHECK:STDOUT:   %.loc20_44.2: %GenericParam = bind_value %.loc20_44.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc20_24.2, %.loc20_44.2)
// CHECK:STDOUT:   %facet_value.loc20_42: %type_where = facet_value constants.%GenericParam, () [concrete = constants.%facet_value.6cb]
// CHECK:STDOUT:   %.loc20_42.5: %type_where = converted constants.%GenericParam, %facet_value.loc20_42 [concrete = constants.%facet_value.6cb]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc20_42: <bound method> = bound_method %.loc20_42.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.a0a
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.a0a, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.6cb) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.994]
// CHECK:STDOUT:   %bound_method.loc20_42: <bound method> = bound_method %.loc20_42.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc20_42: %ptr.f73 = addr_of %.loc20_42.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc20_42: init %empty_tuple.type = call %bound_method.loc20_42(%addr.loc20_42)
// CHECK:STDOUT:   %facet_value.loc20_22: %type_where = facet_value constants.%ImplsGeneric, () [concrete = constants.%facet_value.004]
// CHECK:STDOUT:   %.loc20_22.5: %type_where = converted constants.%ImplsGeneric, %facet_value.loc20_22 [concrete = constants.%facet_value.004]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc20_22: <bound method> = bound_method %.loc20_22.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.4ac
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.4ac, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.004) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e4a]
// CHECK:STDOUT:   %bound_method.loc20_22: <bound method> = bound_method %.loc20_22.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc20_22: %ptr.011 = addr_of %.loc20_22.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc20_22: init %empty_tuple.type = call %bound_method.loc20_22(%addr.loc20_22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%Scalar, constants.%Self.8997e7.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.621
// CHECK:STDOUT:   %Self.2 => constants.%Self.93d
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.4cf
// CHECK:STDOUT:   %Generic.F => constants.%Generic.F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.2db63e.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.8997e7.2
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.f439a9.2
// CHECK:STDOUT:   %Generic.F => constants.%Generic.F.8a2d67.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.0fd877.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.2966cb.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.2db63e.2
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.e93
// CHECK:STDOUT:   %U.as_type.loc15_51.1 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.70f
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.621
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.c4b
// CHECK:STDOUT:   %U.as_type.loc15_51.1 => constants.%ImplsGeneric
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.c8d
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.589
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_49 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.ca2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9b7
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.4cf
// CHECK:STDOUT:   %.loc16_4.3 => constants.%.573
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.892: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.942: %I.type.892 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.5cc: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %Self.4c1: %I.type.5cc = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.0ba: <witness> = require_complete_type %I.type.5cc [symbolic]
// CHECK:STDOUT:   %I.impl_witness.9c7: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.302: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %T.305: %I.type.302 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.610: type = pattern_type %I.type.302 [concrete]
// CHECK:STDOUT:   %Self.16c: %I.type.302 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.305 [symbolic]
// CHECK:STDOUT:   %pattern_type.8a1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.720: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.ab3: <witness> = complete_type_witness %I.type.302 [concrete]
// CHECK:STDOUT:   %I.impl_witness.6f6: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.302 = facet_value %C, (%I.impl_witness.6f6) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.075: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.b80: %AggregateT.as_type.as.Destroy.impl.Op.type.075 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.b80, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.loc3_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.5cc)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.8b3) [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.9c7)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.610 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.8a1) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.8a1) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.302] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_tuple.type)> [concrete = constants.%I.type.302]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type.302 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.305)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_24.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.2 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.302 = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.305)]
// CHECK:STDOUT:       %T.as_type.loc9_24.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_24.2: type = converted %T.ref, %T.as_type.loc9_24.2 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_24.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.892)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.892) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.942)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.892) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.942)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.5cc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.0ba)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.9c7)]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type.302) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type.302 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.305)]
// CHECK:STDOUT:   %T.as_type.loc9_24.1: type = facet_access_type %T.loc9_6.1 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_24.1 [symbolic = %pattern_type (constants.%pattern_type.8a1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_24.1 [symbolic = %require_complete (constants.%require_complete.720)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_24.1 (%T.as_type)) {
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
// CHECK:STDOUT:   %I.facet.loc12_12.1: %I.type.302 = facet_value constants.%C, (constants.%I.impl_witness.6f6) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.1: %I.type.302 = converted constants.%C, %I.facet.loc12_12.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %I.facet.loc12_12.2: %I.type.302 = facet_value constants.%C, (constants.%I.impl_witness.6f6) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc12_12.2: %I.type.302 = converted constants.%C, %I.facet.loc12_12.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc12_8.2: %C = bind_value %.loc12_8.1
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc12_8.2)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc12_6.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.b80
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.b80, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc12_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5cc
// CHECK:STDOUT:   %Self.2 => constants.%Self.4c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.5cc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0ba
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.9c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.302
// CHECK:STDOUT:   %Self.2 => constants.%Self.16c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.305) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.305
// CHECK:STDOUT:   %T.as_type.loc9_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.302
// CHECK:STDOUT:   %require_complete => constants.%complete_type.ab3
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%I.facet
// CHECK:STDOUT:   %T.as_type.loc9_24.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.892: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.942: %I.type.892 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.5cc: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %Self.4c1: %I.type.5cc = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.0ba: <witness> = require_complete_type %I.type.5cc [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.1c6: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %T.034: %I.type.1c6 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.1f1: type = pattern_type %I.type.1c6 [concrete]
// CHECK:STDOUT:   %Self.757: %I.type.1c6 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.034 [symbolic]
// CHECK:STDOUT:   %pattern_type.ca7: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7a0: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.075: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.b80: %AggregateT.as_type.as.Destroy.impl.Op.type.075 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.b80, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.loc3_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc3_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.5cc)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.8b3) [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.ca7) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.ca7) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.1c6] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_17: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_struct_type)> [concrete = constants.%I.type.1c6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type.1c6 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.034)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_24.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.2 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.1c6 = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.034)]
// CHECK:STDOUT:       %T.as_type.loc9_24.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_24.2: type = converted %T.ref, %T.as_type.loc9_24.2 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_24.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.loc3_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc3_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.892)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.892) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.942)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.892) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.942)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.5cc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.0ba)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type.1c6) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type.1c6 = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.034)]
// CHECK:STDOUT:   %T.as_type.loc9_24.1: type = facet_access_type %T.loc9_6.1 [symbolic = %T.as_type.loc9_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_24.1 [symbolic = %pattern_type (constants.%pattern_type.ca7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_24.1 [symbolic = %require_complete (constants.%require_complete.7a0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_24.1 (%T.as_type)) {
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
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc19_6.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc19_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.b80
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.b80, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc19_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5cc
// CHECK:STDOUT:   %Self.2 => constants.%Self.4c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.5cc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0ba
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1c6
// CHECK:STDOUT:   %Self.2 => constants.%Self.757
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.034) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.034
// CHECK:STDOUT:   %T.as_type.loc9_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ca7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c6b: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.463: type = class_type @C, @C(%T.8b3, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.7ee: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.7ee [symbolic]
// CHECK:STDOUT:   %pattern_type.d22: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.742: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c74: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.c74 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C.c74, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.26f: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.64f: %AggregateT.as_type.as.Destroy.impl.Op.type.26f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %C.c74 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.64f, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.loc5_9.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.1 (constants.%V)]
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc5_19.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc7_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_31: type = converted %.loc7_30, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc7_31.1: type = class_type @C, @C(constants.%T.8b3, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.8b3) [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.09a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.d22) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.d22) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_10: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.7ee)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_16.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %T.as_type.loc9_16.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.7ee)]
// CHECK:STDOUT:       %T.as_type.loc9_16.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_16.2: type = converted %T.ref, %T.as_type.loc9_16.2 [symbolic = %T.as_type.loc9_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_16.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %C.loc7_31.2: type = class_type @C, @C(%T.loc7_14.2, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.463)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_31.1 as %I.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc5_9.2: type, %W.loc5_19.2: type) {
// CHECK:STDOUT:   %V.loc5_9.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_9.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc5_19.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.c6b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.7ee)]
// CHECK:STDOUT:   %T.as_type.loc9_16.1: type = facet_access_type %T.loc9_6.1 [symbolic = %T.as_type.loc9_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_16.1 [symbolic = %pattern_type (constants.%pattern_type.d22)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_16.1 [symbolic = %require_complete (constants.%require_complete.742)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_16.1 (%T.as_type)) {
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
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C.c74, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc19_6.5: %type_where = converted constants.%C.c74, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc19_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.64f
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.64f, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.128 = addr_of %.loc19_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%V
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %C.loc7_31.2 => constants.%C.463
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.7ee) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.7ee
// CHECK:STDOUT:   %T.as_type.loc9_16.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
