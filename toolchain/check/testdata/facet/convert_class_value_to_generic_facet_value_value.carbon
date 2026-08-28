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

interface Generic(Scalar: type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod[T: type, U: Generic(T)](unused a: U, unused s: T) {
  U.F();
}

fn G() {
  CallGenericMethod({} as ImplsGeneric, {} as GenericParam);
}

// --- multiple_generic_params_one_fixed_one_deduced.carbon
library "[[@TEST_NAME]]";

interface I(V: type, W: type) {}

class C {}

impl forall [T: type] C as I(T, ()) {}

fn A[T: I({}, ())](unused t: T) {}

fn B() {
  A({} as C);
}

// --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I(V: type, W: type) {}

class C {}

impl forall [T: type] C as I(T, ()) {}

fn A[T: I({}, {})](unused t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+3]]:3-12: error: cannot convert type `C` into type implementing `I({}, {})` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE-4]]:1-33: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  A({} as C);
}

// --- fail_mismatch_impl_self_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I {}

class C(V: type, W: type) {}

impl forall [T: type] C(T, ()) as I {}

fn A[T: I](unused t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+3]]:3-20: error: cannot convert type `C({}, {})` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE-4]]:1-25: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  A({} as C({}, {}));
}

// CHECK:STDOUT: --- convert_class_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.30d: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.30d = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.ee14e9.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.8ea1ea.1: %Generic.type.ee14e9.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.390775.1: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%Scalar, %Self.8ea1ea.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.b85a66.1: %Generic.WithSelf.F.type.390775.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.d7021d.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.31052a.1: %Generic.assoc_type.d7021d.1 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.b40: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %.e53: <witness> = impl_self_witness %ImplsGeneric, @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.685: %Generic.type.b40 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.e39: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Self.8ea1ea.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.d19: %Generic.WithSelf.F.type.e39 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.c13: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.b4a: %Generic.assoc_type.c13 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.b40 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.7ef: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.e1b: %Generic.WithSelf.F.type.7ef = struct_value () [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.ee14e9.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.4e0: type = pattern_type %Generic.type.ee14e9.2 [symbolic]
// CHECK:STDOUT:   %U.patt.8a6: %pattern_type.4e0 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.ee14e9.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.70f: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %a.param_patt.98a: %pattern_type.70f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.593: %pattern_type.70f = wrapper_binding_pattern a, %a.param_patt.98a [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %s.param_patt.cb5: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %s.patt.9d4: %pattern_type.51d = wrapper_binding_pattern s, %s.param_patt.cb5 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.096: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Self.8ea1ea.2: %Generic.type.ee14e9.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.390775.2: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T, %Self.8ea1ea.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.b85a66.2: %Generic.WithSelf.F.type.390775.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.d7021d.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.31052a.2: %Generic.assoc_type.d7021d.2 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.13f: <witness> = require_complete_type %Generic.type.ee14e9.2 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.a70: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T, %U) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.5e8: %Generic.WithSelf.F.type.a70 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %.aca: type = fn_type_with_self_type %Generic.WithSelf.F.type.a70, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.aca = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Generic.WithSelf.F(%T, %U) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.de1: type = pattern_type %Generic.type.b40 [concrete]
// CHECK:STDOUT:   %U.patt.e09: %pattern_type.de1 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.488: type = pattern_type %ImplsGeneric [concrete]
// CHECK:STDOUT:   %a.param_patt.150: %pattern_type.488 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.d11: %pattern_type.488 = wrapper_binding_pattern a, %a.param_patt.150 [concrete]
// CHECK:STDOUT:   %pattern_type.485: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %s.param_patt.12f: %pattern_type.485 = value_param_pattern [concrete]
// CHECK:STDOUT:   %s.patt.a1f: %pattern_type.485 = wrapper_binding_pattern s, %s.param_patt.12f [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %.cd4: ref %ImplsGeneric = temporary invalid, %ImplsGeneric.val [concrete]
// CHECK:STDOUT:   %.b65: ref %GenericParam = temporary invalid, %GenericParam.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc20_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.852: <bound method> = bound_method %.b65, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc20_24 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.047: <bound method> = bound_method %.cd4, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT:   %complete_type.35b: <witness> = complete_type_witness %Generic.type.b40 [concrete]
// CHECK:STDOUT:   %.636: type = fn_type_with_self_type %Generic.WithSelf.F.type.7ef, %Generic.facet [concrete]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.30d = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt.loc4_25.1: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc4_25.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Scalar.loc4_25.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.30d = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.b40]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @ImplsGeneric.as.Generic.impl.%ImplsGeneric.ref, @Generic, @Generic(constants.%GenericParam) [concrete = constants.%.e53]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt.loc15_23.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_23.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc15_32.1: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.4e0) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_32.2 (constants.%U.patt.8a6)]
// CHECK:STDOUT:     %a.param_patt.loc15_54.1: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.70f) = value_param_pattern [symbolic = %a.param_patt.loc15_54.2 (constants.%a.param_patt.98a)]
// CHECK:STDOUT:     %a.patt.loc15_54.1: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.70f) = wrapper_binding_pattern a, %a.param_patt.loc15_54.1 [symbolic = %a.patt.loc15_54.2 (constants.%a.patt.593)]
// CHECK:STDOUT:     %s.param_patt.loc15_67.1: @CallGenericMethod.%pattern_type.loc15_67 (%pattern_type.51d) = value_param_pattern [symbolic = %s.param_patt.loc15_67.2 (constants.%s.param_patt.cb5)]
// CHECK:STDOUT:     %s.patt.loc15_67.1: @CallGenericMethod.%pattern_type.loc15_67 (%pattern_type.51d) = wrapper_binding_pattern s, %s.param_patt.loc15_67.1 [symbolic = %s.patt.loc15_67.2 (constants.%s.patt.9d4)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_25.1: type = splice_block %.loc15_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc15_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_43: type = splice_block %Generic.type.loc15_43.2 [symbolic = %Generic.type.loc15_43.1 (constants.%Generic.type.ee14e9.2)] {
// CHECK:STDOUT:       %.Self.frozen.loc15_32: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.30d = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_42: type = name_ref T, %T.loc15_23.2 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_43.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_43.1 (constants.%Generic.type.ee14e9.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_43.1 (%Generic.type.ee14e9.2) = symbolic_binding U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %a.param: @CallGenericMethod.%U.as_type.loc15_56.1 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_56.1: type = splice_block %.loc15_56.2 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref.loc15: @CallGenericMethod.%Generic.type.loc15_43.1 (%Generic.type.ee14e9.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc15_56.2: type = facet_access_type %U.ref.loc15 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc15_56.2: type = converted %U.ref.loc15, %U.as_type.loc15_56.2 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @CallGenericMethod.%U.as_type.loc15_56.1 (%U.as_type) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %s.param: @CallGenericMethod.%T.loc15_23.1 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc15_69: type = name_ref T, %T.loc15_23.2 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:     %s: @CallGenericMethod.%T.loc15_23.1 (%T) = wrapper_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_25.2: type) {
// CHECK:STDOUT:   %Scalar.patt.loc4_25.2: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc4_25.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:   %Scalar.loc4_25.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_25.1)> [symbolic = %Generic.type (constants.%Generic.type.ee14e9.1)]
// CHECK:STDOUT:   %Self.loc4_33.2: @Generic.%Generic.type (%Generic.type.ee14e9.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_33.2 (constants.%Self.8ea1ea.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_33.1: @Generic.%Generic.type (%Generic.type.ee14e9.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_33.2 (constants.%Self.8ea1ea.1)]
// CHECK:STDOUT:     %Generic.WithSelf.decl = interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.WithSelf.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.390775.1) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.WithSelf.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.b85a66.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.WithSelf.%Generic.assoc_type (%Generic.assoc_type.d7021d.1) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.31052a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_33.1
// CHECK:STDOUT:     .F = @Generic.WithSelf.%assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (@Generic.WithSelf.%Generic.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplsGeneric.as.Generic.impl: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.decl: %ImplsGeneric.as.Generic.impl.F.type = fn_decl @ImplsGeneric.as.Generic.impl.F [concrete = constants.%ImplsGeneric.as.Generic.impl.F] {} {}
// CHECK:STDOUT:   %Generic.impl_witness_table = impl_witness_table (%ImplsGeneric.as.Generic.impl.F.decl), @ImplsGeneric.as.Generic.impl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness %Generic.impl_witness_table [concrete = constants.%Generic.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %ImplsGeneric.as.Generic.impl.F.decl
// CHECK:STDOUT:   extend %Generic.type
// CHECK:STDOUT:   witness = %Generic.impl_witness
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
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc4_25.2: type, @Generic.%Self.loc4_33.1: @Generic.%Generic.type (%Generic.type.ee14e9.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_23.2: type, %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_43.1 (%Generic.type.ee14e9.2)) {
// CHECK:STDOUT:   %T.patt.loc15_23.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_43.1: type = facet_type <@Generic, @Generic(%T.loc15_23.1)> [symbolic = %Generic.type.loc15_43.1 (constants.%Generic.type.ee14e9.2)]
// CHECK:STDOUT:   %pattern_type.loc15_32: type = pattern_type %Generic.type.loc15_43.1 [symbolic = %pattern_type.loc15_32 (constants.%pattern_type.4e0)]
// CHECK:STDOUT:   %U.patt.loc15_32.2: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.4e0) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_32.2 (constants.%U.patt.8a6)]
// CHECK:STDOUT:   %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_43.1 (%Generic.type.ee14e9.2) = symbolic_binding U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc15_56.1: type = facet_access_type %U.loc15_32.1 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type.loc15_54: type = pattern_type %U.as_type.loc15_56.1 [symbolic = %pattern_type.loc15_54 (constants.%pattern_type.70f)]
// CHECK:STDOUT:   %a.param_patt.loc15_54.2: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.70f) = value_param_pattern [symbolic = %a.param_patt.loc15_54.2 (constants.%a.param_patt.98a)]
// CHECK:STDOUT:   %a.patt.loc15_54.2: @CallGenericMethod.%pattern_type.loc15_54 (%pattern_type.70f) = wrapper_binding_pattern a, %a.param_patt.loc15_54.2 [symbolic = %a.patt.loc15_54.2 (constants.%a.patt.593)]
// CHECK:STDOUT:   %pattern_type.loc15_67: type = pattern_type %T.loc15_23.1 [symbolic = %pattern_type.loc15_67 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %s.param_patt.loc15_67.2: @CallGenericMethod.%pattern_type.loc15_67 (%pattern_type.51d) = value_param_pattern [symbolic = %s.param_patt.loc15_67.2 (constants.%s.param_patt.cb5)]
// CHECK:STDOUT:   %s.patt.loc15_67.2: @CallGenericMethod.%pattern_type.loc15_67 (%pattern_type.51d) = wrapper_binding_pattern s, %s.param_patt.loc15_67.2 [symbolic = %s.patt.loc15_67.2 (constants.%s.patt.9d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_54: <witness> = require_complete_type %U.as_type.loc15_56.1 [symbolic = %require_complete.loc15_54 (constants.%require_complete.096)]
// CHECK:STDOUT:   %require_complete.loc15_67: <witness> = require_complete_type %T.loc15_23.1 [symbolic = %require_complete.loc15_67 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %Generic.type.loc15_43.1 [symbolic = %require_complete.loc16 (constants.%require_complete.13f)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc15_23.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.d7021d.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.d7021d.2) = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.31052a.2)]
// CHECK:STDOUT:   %Generic.WithSelf.F.type: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T.loc15_23.1, %U.loc15_32.1) [symbolic = %Generic.WithSelf.F.type (constants.%Generic.WithSelf.F.type.a70)]
// CHECK:STDOUT:   %.loc16_4.3: type = fn_type_with_self_type %Generic.WithSelf.F.type, %U.loc15_32.1 [symbolic = %.loc16_4.3 (constants.%.aca)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc15_32.1, @Generic, @Generic(%T.loc15_23.1) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @CallGenericMethod.%.loc16_4.3 (%.aca) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Generic.WithSelf.F(%T.loc15_23.1, %U.loc15_32.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @CallGenericMethod.%U.as_type.loc15_56.1 (%U.as_type), %s.param: @CallGenericMethod.%T.loc15_23.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc16: @CallGenericMethod.%Generic.type.loc15_43.1 (%Generic.type.ee14e9.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc16: type = facet_access_type %U.ref.loc16 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc16_4.1: type = converted %U.ref.loc16, %U.as_type.loc16 [symbolic = %U.as_type.loc15_56.1 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc16_4.2: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.d7021d.2) = specific_constant @Generic.WithSelf.%assoc0.loc5_9.1, @Generic.WithSelf(constants.%T, constants.%U) [symbolic = %assoc0 (constants.%assoc0.31052a.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.d7021d.2) = name_ref F, %.loc16_4.2 [symbolic = %assoc0 (constants.%assoc0.31052a.2)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @CallGenericMethod.%.loc16_4.3 (%.aca) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Generic.WithSelf.F(constants.%T, constants.%U) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Generic.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %.loc20_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %.loc20_22.2: ref %ImplsGeneric = temporary_storage
// CHECK:STDOUT:   %.loc20_22.3: init %ImplsGeneric to %.loc20_22.2 = class_init () [concrete = constants.%ImplsGeneric.val]
// CHECK:STDOUT:   %.loc20_24.1: init %ImplsGeneric = converted %.loc20_22.1, %.loc20_22.3 [concrete = constants.%ImplsGeneric.val]
// CHECK:STDOUT:   %.loc20_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc20_42.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc20_42.3: init %GenericParam to %.loc20_42.2 = class_init () [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc20_44.1: init %GenericParam = converted %.loc20_42.1, %.loc20_42.3 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.b40 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc20_59: %Generic.type.b40 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc20_24.2: ref %ImplsGeneric = temporary %.loc20_22.2, %.loc20_24.1 [concrete = constants.%.cd4]
// CHECK:STDOUT:   %.loc20_24.3: %ImplsGeneric = acquire_value %.loc20_24.2 [concrete = constants.%ImplsGeneric.val]
// CHECK:STDOUT:   %.loc20_44.2: ref %GenericParam = temporary %.loc20_42.2, %.loc20_44.1 [concrete = constants.%.b65]
// CHECK:STDOUT:   %.loc20_44.3: %GenericParam = acquire_value %.loc20_44.2 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc20_24.3, %.loc20_44.3)
// CHECK:STDOUT:   %Destroy.Op.call.loc20_44: init %empty_tuple.type = call constants.%Destroy.Op.bound.852(constants.%.b65)
// CHECK:STDOUT:   %Destroy.Op.call.loc20_24: init %empty_tuple.type = call constants.%Destroy.Op.bound.047(constants.%.cd4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_44.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_44.2(%self.param: ref %GenericParam) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_24(%self.param: ref %ImplsGeneric) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.patt.loc4_25.2 => constants.%Scalar.patt
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Scalar, constants.%Self.8ea1ea.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.8ea1ea.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.patt.loc4_25.2 => constants.%Scalar.patt
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.b40
// CHECK:STDOUT:   %Self.loc4_33.2 => constants.%Self.685
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Self.8ea1ea.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.b40
// CHECK:STDOUT:   %Self => constants.%Self.8ea1ea.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.e39
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.d19
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.c13
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.b4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.b40
// CHECK:STDOUT:   %Self => constants.%Generic.facet
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.7ef
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.e1b
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.c13
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.b4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.patt.loc4_25.2 => constants.%Scalar.patt
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.ee14e9.2
// CHECK:STDOUT:   %Self.loc4_33.2 => constants.%Self.8ea1ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc15_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_43.1 => constants.%Generic.type.ee14e9.2
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.4e0
// CHECK:STDOUT:   %U.patt.loc15_32.2 => constants.%U.patt.8a6
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc15_56.1 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.70f
// CHECK:STDOUT:   %a.param_patt.loc15_54.2 => constants.%a.param_patt.98a
// CHECK:STDOUT:   %a.patt.loc15_54.2 => constants.%a.patt.593
// CHECK:STDOUT:   %pattern_type.loc15_67 => constants.%pattern_type.51d
// CHECK:STDOUT:   %s.param_patt.loc15_67.2 => constants.%s.param_patt.cb5
// CHECK:STDOUT:   %s.patt.loc15_67.2 => constants.%s.patt.9d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%T, constants.%Self.8ea1ea.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%T
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.ee14e9.2
// CHECK:STDOUT:   %Self => constants.%Self.8ea1ea.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.390775.2
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.b85a66.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.d7021d.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.31052a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%T, constants.%U) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%T
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.ee14e9.2
// CHECK:STDOUT:   %Self => constants.%U
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.a70
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.5e8
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.d7021d.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.31052a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.patt.loc15_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_43.1 => constants.%Generic.type.b40
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.de1
// CHECK:STDOUT:   %U.patt.loc15_32.2 => constants.%U.patt.e09
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %U.as_type.loc15_56.1 => constants.%ImplsGeneric
// CHECK:STDOUT:   %pattern_type.loc15_54 => constants.%pattern_type.488
// CHECK:STDOUT:   %a.param_patt.loc15_54.2 => constants.%a.param_patt.150
// CHECK:STDOUT:   %a.patt.loc15_54.2 => constants.%a.patt.d11
// CHECK:STDOUT:   %pattern_type.loc15_67 => constants.%pattern_type.485
// CHECK:STDOUT:   %s.param_patt.loc15_67.2 => constants.%s.param_patt.12f
// CHECK:STDOUT:   %s.patt.loc15_67.2 => constants.%s.patt.a1f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_54 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_67 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.35b
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.c13
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b4a
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.7ef
// CHECK:STDOUT:   %.loc16_4.3 => constants.%.636
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.d96: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.299: %I.type.d96 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.type.269: type = facet_type <@I, @I(%T.67d, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %C, @I, @I(%T.67d, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e6a: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Self.7df: %I.type.269 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.2dc: <witness> = require_complete_type %I.type.269 [symbolic]
// CHECK:STDOUT:   %I.facet.dd0: %I.type.269 = facet_value %C, (%I.impl_witness.e6a) [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.722: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %pattern_type.bd4: type = pattern_type %I.type.722 [concrete]
// CHECK:STDOUT:   %T.patt.be3: %pattern_type.bd4 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.a08: %I.type.722 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.a08 [symbolic]
// CHECK:STDOUT:   %pattern_type.d7b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.be8: %pattern_type.d7b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.05d: %pattern_type.d7b = wrapper_binding_pattern t, %t.param_patt.be8 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d9e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.793: <witness> = impl_self_witness %C, @I, @I(%empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %I.impl_witness.7f0: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Self.9e2: %I.type.722 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %complete_type.dfa: <witness> = complete_type_witness %I.type.722 [concrete]
// CHECK:STDOUT:   %I.facet.dfc: %I.type.722 = facet_value %C, (%I.impl_witness.7f0) [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %t.param_patt.5f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.429: %pattern_type.98b = wrapper_binding_pattern t, %t.param_patt.5f9 [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet.dfc) [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc12_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
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
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc3_14.1: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc3_23.1: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc3_14.2: type = symbolic_binding V, 0 [symbolic = %V.loc3_14.1 (constants.%V)]
// CHECK:STDOUT:     %.loc3_25.1: type = splice_block %.loc3_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc3_23.2: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_34: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_35: type = converted %.loc7_34, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_35.1: type = facet_type <@I, @I(constants.%T.67d, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_35.2 (constants.%I.type.269)]
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I, @I(constants.%T.67d, constants.%empty_tuple.type) [symbolic = @C.as.I.impl.%.loc7_37 (constants.%.78e)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.bd4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.be3)]
// CHECK:STDOUT:     %t.param_patt.loc9_28.1: @A.%pattern_type (%pattern_type.d7b) = value_param_pattern [symbolic = %t.param_patt.loc9_28.2 (constants.%t.param_patt.be8)]
// CHECK:STDOUT:     %t.patt.loc9_28.1: @A.%pattern_type (%pattern_type.d7b) = wrapper_binding_pattern t, %t.param_patt.loc9_28.1 [symbolic = %t.patt.loc9_28.2 (constants.%t.patt.05d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %I.type [concrete = constants.%I.type.722] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_16: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_17.2: type = converted %.loc9_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_16, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_tuple.type)> [concrete = constants.%I.type.722]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %I.type.722 = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.a08)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_30.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_30.1: type = splice_block %.loc9_30.2 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.722 = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.a08)]
// CHECK:STDOUT:       %T.as_type.loc9_30.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_30.2: type = converted %T.ref, %T.as_type.loc9_30.2 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_30.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_14.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc3_14.1: type = symbolic_binding V, 0 [symbolic = %V.loc3_14.1 (constants.%V)]
// CHECK:STDOUT:   %W.patt.loc3_23.2: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc3_23.1: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_14.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.d96)]
// CHECK:STDOUT:   %Self.loc3_31.2: @I.%I.type (%I.type.d96) = symbolic_binding Self, 2 [symbolic = %Self.loc3_31.2 (constants.%Self.299)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_31.1: @I.%I.type (%I.type.d96) = symbolic_binding Self, 2 [symbolic = %Self.loc3_31.2 (constants.%Self.299)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_31.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %I.type.loc7_35.2: type = facet_type <@I, @I(%T.loc7_15.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_35.2 (constants.%I.type.269)]
// CHECK:STDOUT:   %.loc7_37: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc7_15.2, constants.%empty_tuple.type) [symbolic = %.loc7_37 (constants.%.78e)]
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_15.2) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness.e6a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_35.2 [symbolic = %require_complete (constants.%require_complete.2dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_35.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_37.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness.e6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc7_35.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_37.1
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
// CHECK:STDOUT: generic fn @A(%T.loc9_7.2: %I.type.722) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.bd4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.be3)]
// CHECK:STDOUT:   %T.loc9_7.1: %I.type.722 = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.a08)]
// CHECK:STDOUT:   %T.as_type.loc9_30.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_30.1 [symbolic = %pattern_type (constants.%pattern_type.d7b)]
// CHECK:STDOUT:   %t.param_patt.loc9_28.2: @A.%pattern_type (%pattern_type.d7b) = value_param_pattern [symbolic = %t.param_patt.loc9_28.2 (constants.%t.param_patt.be8)]
// CHECK:STDOUT:   %t.patt.loc9_28.2: @A.%pattern_type (%pattern_type.d7b) = wrapper_binding_pattern t, %t.param_patt.loc9_28.2 [symbolic = %t.patt.loc9_28.2 (constants.%t.patt.05d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_30.1 [symbolic = %require_complete (constants.%require_complete.d9e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_30.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc12_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_6.3: init %C to %.loc12_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_8.1: init %C = converted %.loc12_6.1, %.loc12_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %I.facet.loc12_12.1: %I.type.722 = facet_value constants.%C, (constants.%I.impl_witness.7f0) [concrete = constants.%I.facet.dfc]
// CHECK:STDOUT:   %.loc12_12.1: %I.type.722 = converted constants.%C, %I.facet.loc12_12.1 [concrete = constants.%I.facet.dfc]
// CHECK:STDOUT:   %I.facet.loc12_12.2: %I.type.722 = facet_value constants.%C, (constants.%I.impl_witness.7f0) [concrete = constants.%I.facet.dfc]
// CHECK:STDOUT:   %.loc12_12.2: %I.type.722 = converted constants.%C, %I.facet.loc12_12.2 [concrete = constants.%I.facet.dfc]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet.dfc) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc12_8.2: ref %C = temporary %.loc12_6.2, %.loc12_8.1 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc12_8.3: %C = acquire_value %.loc12_8.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc12_8.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_8.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%V
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%V, constants.%W, constants.%Self.299) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.269
// CHECK:STDOUT:   %Self.loc3_31.2 => constants.%Self.7df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T.67d
// CHECK:STDOUT:   %I.type.loc7_35.2 => constants.%I.type.269
// CHECK:STDOUT:   %.loc7_37 => constants.%.78e
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2 => constants.%I.impl_witness.e6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%Self.299) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%I.facet.dd0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.722
// CHECK:STDOUT:   %Self.loc3_31.2 => constants.%Self.9e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.a08) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.be3
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.a08
// CHECK:STDOUT:   %T.as_type.loc9_30.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d7b
// CHECK:STDOUT:   %t.param_patt.loc9_28.2 => constants.%t.param_patt.be8
// CHECK:STDOUT:   %t.patt.loc9_28.2 => constants.%t.patt.05d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_35.2 => constants.%I.type.722
// CHECK:STDOUT:   %.loc7_37 => constants.%.793
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2 => constants.%I.impl_witness.7f0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dfa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%empty_tuple.type, constants.%Self.299) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet.dfc) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.be3
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%I.facet.dfc
// CHECK:STDOUT:   %T.as_type.loc9_30.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %t.param_patt.loc9_28.2 => constants.%t.param_patt.5f9
// CHECK:STDOUT:   %t.patt.loc9_28.2 => constants.%t.patt.429
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.d96: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.299: %I.type.d96 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.type.269: type = facet_type <@I, @I(%T.67d, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %C, @I, @I(%T.67d, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Self.7df: %I.type.269 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.2dc: <witness> = require_complete_type %I.type.269 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.269 = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.6c0: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %pattern_type.5c6: type = pattern_type %I.type.6c0 [concrete]
// CHECK:STDOUT:   %T.patt.c4c: %pattern_type.5c6 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.b75: %I.type.6c0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.b75 [symbolic]
// CHECK:STDOUT:   %pattern_type.67d: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.67d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.67d = wrapper_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.401: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc3_14.1: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc3_23.1: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc3_14.2: type = symbolic_binding V, 0 [symbolic = %V.loc3_14.1 (constants.%V)]
// CHECK:STDOUT:     %.loc3_25.1: type = splice_block %.loc3_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc3_23.2: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_34: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_35: type = converted %.loc7_34, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_35.1: type = facet_type <@I, @I(constants.%T.67d, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_35.2 (constants.%I.type.269)]
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I, @I(constants.%T.67d, constants.%empty_tuple.type) [symbolic = @C.as.I.impl.%.loc7_37 (constants.%.78e)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.5c6 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.c4c)]
// CHECK:STDOUT:     %t.param_patt.loc9_28.1: @A.%pattern_type (%pattern_type.67d) = value_param_pattern [symbolic = %t.param_patt.loc9_28.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc9_28.1: @A.%pattern_type (%pattern_type.67d) = wrapper_binding_pattern t, %t.param_patt.loc9_28.1 [symbolic = %t.patt.loc9_28.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %I.type [concrete = constants.%I.type.6c0] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_16: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_17.2: type = converted %.loc9_12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_struct_type)> [concrete = constants.%I.type.6c0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %I.type.6c0 = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.b75)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_30.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_30.1: type = splice_block %.loc9_30.2 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.6c0 = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.b75)]
// CHECK:STDOUT:       %T.as_type.loc9_30.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_30.2: type = converted %T.ref, %T.as_type.loc9_30.2 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_30.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_14.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc3_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc3_14.1: type = symbolic_binding V, 0 [symbolic = %V.loc3_14.1 (constants.%V)]
// CHECK:STDOUT:   %W.patt.loc3_23.2: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc3_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc3_23.1: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_14.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.d96)]
// CHECK:STDOUT:   %Self.loc3_31.2: @I.%I.type (%I.type.d96) = symbolic_binding Self, 2 [symbolic = %Self.loc3_31.2 (constants.%Self.299)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_31.1: @I.%I.type (%I.type.d96) = symbolic_binding Self, 2 [symbolic = %Self.loc3_31.2 (constants.%Self.299)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_31.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %I.type.loc7_35.2: type = facet_type <@I, @I(%T.loc7_15.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_35.2 (constants.%I.type.269)]
// CHECK:STDOUT:   %.loc7_37: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc7_15.2, constants.%empty_tuple.type) [symbolic = %.loc7_37 (constants.%.78e)]
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_15.2) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_35.2 [symbolic = %require_complete (constants.%require_complete.2dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_35.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_37.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc7_35.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_7.2: %I.type.6c0) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.5c6 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.c4c)]
// CHECK:STDOUT:   %T.loc9_7.1: %I.type.6c0 = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.b75)]
// CHECK:STDOUT:   %T.as_type.loc9_30.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_30.1 [symbolic = %pattern_type (constants.%pattern_type.67d)]
// CHECK:STDOUT:   %t.param_patt.loc9_28.2: @A.%pattern_type (%pattern_type.67d) = value_param_pattern [symbolic = %t.param_patt.loc9_28.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc9_28.2: @A.%pattern_type (%pattern_type.67d) = wrapper_binding_pattern t, %t.param_patt.loc9_28.2 [symbolic = %t.patt.loc9_28.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_30.1 [symbolic = %require_complete (constants.%require_complete.401)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_30.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc15_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_6.3: init %C to %.loc15_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_8: init %C = converted %.loc15_6.1, %.loc15_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%V
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%V, constants.%W, constants.%Self.299) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.269
// CHECK:STDOUT:   %Self.loc3_31.2 => constants.%Self.7df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T.67d
// CHECK:STDOUT:   %I.type.loc7_35.2 => constants.%I.type.269
// CHECK:STDOUT:   %.loc7_37 => constants.%.78e
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%Self.299) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.patt.loc3_14.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc3_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc3_23.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.b75) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.c4c
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.b75
// CHECK:STDOUT:   %T.as_type.loc9_30.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.67d
// CHECK:STDOUT:   %t.param_patt.loc9_28.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc9_28.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.47a: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.ab0: type = class_type @C, @C(%T.67d, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %.c42: <witness> = impl_self_witness %C.ab0, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ab0, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.dda [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.b3a = wrapper_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.b48: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.b48 = struct_value () [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %V.patt.loc5_10.1: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_10.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc5_19.1: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_19.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc5_10.2: type = symbolic_binding V, 0 [symbolic = %V.loc5_10.1 (constants.%V)]
// CHECK:STDOUT:     %.loc5_21.1: type = splice_block %.loc5_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc5_19.2: type = symbolic_binding W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_29: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_30: type = converted %.loc7_29, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc7_30.1: type = class_type @C, @C(constants.%T.67d, constants.%empty_tuple.type) [symbolic = %C.loc7_30.2 (constants.%C.ab0)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.loc7_30.1, @I [symbolic = @C.as.I.impl.%.loc7_37 (constants.%.c42)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:     %t.param_patt.loc9_20.1: @A.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc9_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc9_20.1: @A.%pattern_type (%pattern_type.b3a) = wrapper_binding_pattern t, %t.param_patt.loc9_20.1 [symbolic = %t.patt.loc9_20.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_9: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.dda)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc9_22.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.dda)]
// CHECK:STDOUT:       %T.as_type.loc9_22.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref, %T.as_type.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc9_22.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %C.loc7_30.2: type = class_type @C, @C(%T.loc7_15.2, constants.%empty_tuple.type) [symbolic = %C.loc7_30.2 (constants.%C.ab0)]
// CHECK:STDOUT:   %.loc7_37: <witness> = impl_self_witness %C.loc7_30.2, @I [symbolic = %.loc7_37 (constants.%.c42)]
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_15.2) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_30.1 as %I.ref {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_37.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_37.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc5_10.2: type, %W.loc5_19.2: type) {
// CHECK:STDOUT:   %V.patt.loc5_10.2: %pattern_type.98f = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc5_10.1: type = symbolic_binding V, 0 [symbolic = %V.loc5_10.1 (constants.%V)]
// CHECK:STDOUT:   %W.patt.loc5_19.2: %pattern_type.98f = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_19.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc5_19.1: type = symbolic_binding W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.47a
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_7.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc9_7.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.dda)]
// CHECK:STDOUT:   %T.as_type.loc9_22.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_22.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %t.param_patt.loc9_20.2: @A.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc9_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc9_20.2: @A.%pattern_type (%pattern_type.b3a) = wrapper_binding_pattern t, %t.param_patt.loc9_20.2 [symbolic = %t.patt.loc9_20.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_22.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.as_type.loc9_22.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc15_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc15_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_18: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_19.1: type = converted %.loc15_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc15_19.2: type = converted %.loc15_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_struct_type, constants.%empty_struct_type) [concrete = constants.%C.b48]
// CHECK:STDOUT:   %.loc15_6.2: ref %C.b48 = temporary_storage
// CHECK:STDOUT:   %.loc15_6.3: init %C.b48 to %.loc15_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_8: init %C.b48 = converted %.loc15_6.1, %.loc15_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.patt.loc5_10.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc5_10.1 => constants.%V
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.patt.loc5_10.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc5_10.1 => constants.%T.67d
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T.67d
// CHECK:STDOUT:   %C.loc7_30.2 => constants.%C.ab0
// CHECK:STDOUT:   %.loc7_37 => constants.%.c42
// CHECK:STDOUT:   %I.impl_witness.loc7_37.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.dda
// CHECK:STDOUT:   %T.as_type.loc9_22.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %t.param_patt.loc9_20.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc9_20.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.patt.loc5_10.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc5_10.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc5_19.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
