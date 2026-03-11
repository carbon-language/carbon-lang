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

fn CallGenericMethod[T:! type, U:! Generic(T)](unused a: U, unused s: T) {
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

fn A[T:! I({}, ())](unused t: T) {}

fn B() {
  A({} as C);
}

// --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I(V:! type, W:! type) {}

class C {}

impl forall [T:! type] C as I(T, ()) {}

fn A[T:! I({}, {})](unused t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `I({}, {})` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I({}, {})](unused t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C);
}

// --- fail_mismatch_impl_self_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

interface I {}

class C(V:! type, W:! type) {}

impl forall [T:! type] C(T, ()) as I {}

fn A[T:! I](unused t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+7]]:3: error: cannot convert type `C({}, {})` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I](unused t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C({}, {}));
}

// CHECK:STDOUT: --- convert_class_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.835: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.835 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.03dff7.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.15d2d5.1: %Generic.type.03dff7.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.74390a.1: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%Scalar, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.baf95a.1: %Generic.WithSelf.F.type.74390a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.22afda.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.e0dc00.1: %Generic.assoc_type.22afda.1 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.498: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.1f5: %Generic.type.498 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.a23: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.fe0: %Generic.WithSelf.F.type.a23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.6dc: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.717: %Generic.assoc_type.6dc = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.498 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.c86: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.1fd: %Generic.WithSelf.F.type.c86 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.03dff7.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.03dff7.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c49: type = pattern_type %Generic.type.03dff7.2 [symbolic]
// CHECK:STDOUT:   %Self.15d2d5.2: %Generic.type.03dff7.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.74390a.2: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.baf95a.2: %Generic.WithSelf.F.type.74390a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.22afda.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.e0dc00.2: %Generic.assoc_type.22afda.2 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.46f: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.186: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.7da: <witness> = require_complete_type %Generic.type.03dff7.2 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.56e: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T, %U) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.a99: %Generic.WithSelf.F.type.56e = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %.ab3: type = fn_type_with_self_type %Generic.WithSelf.F.type.56e, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ab3 = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Generic.WithSelf.F(%T, %U) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cba: type = pattern_type %Generic.type.498 [concrete]
// CHECK:STDOUT:   %pattern_type.4da: type = pattern_type %ImplsGeneric [concrete]
// CHECK:STDOUT:   %pattern_type.27a: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc20_42 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc20_22 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.57a: <witness> = complete_type_witness %Generic.type.498 [concrete]
// CHECK:STDOUT:   %.fcf: type = fn_type_with_self_type %Generic.WithSelf.F.type.c86, %Generic.facet [concrete]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.835 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Scalar.loc4_19.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.498]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.c49) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @CallGenericMethod.%pattern_type.loc15_55 (%pattern_type.46f) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @CallGenericMethod.%pattern_type.loc15_55 (%pattern_type.46f) = value_param_pattern %a.patt [concrete]
// CHECK:STDOUT:     %s.patt: @CallGenericMethod.%pattern_type.loc15_68 (%pattern_type.51d) = value_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: @CallGenericMethod.%pattern_type.loc15_68 (%pattern_type.51d) = value_param_pattern %s.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_26.1: type = splice_block %.loc15_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_45: type = splice_block %Generic.type.loc15_45.2 [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.03dff7.2)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_44: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %a.param: @CallGenericMethod.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_58.1: type = splice_block %.loc15_58.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref.loc15: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.03dff7.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc15: type = facet_access_type %U.ref.loc15 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc15_58.2: type = converted %U.ref.loc15, %U.as_type.loc15 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @CallGenericMethod.%U.binding.as_type (%U.binding.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %s.param: @CallGenericMethod.%T.loc15_22.1 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc15_71: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %s: @CallGenericMethod.%T.loc15_22.1 (%T) = value_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_19.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.03dff7.1)]
// CHECK:STDOUT:   %Self.loc4_34.2: @Generic.%Generic.type (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.15d2d5.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.15d2d5.1)]
// CHECK:STDOUT:     %Generic.WithSelf.decl = interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.WithSelf.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.74390a.1) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.WithSelf.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.baf95a.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.WithSelf.%Generic.assoc_type (%Generic.assoc_type.22afda.1) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.e0dc00.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_34.1
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
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc4_19.2: type, @Generic.%Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.03dff7.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_22.2: type, %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.03dff7.2)) {
// CHECK:STDOUT:   %T.loc15_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_45.1: type = facet_type <@Generic, @Generic(%T.loc15_22.1)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:   %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc15_32: type = pattern_type %Generic.type.loc15_45.1 [symbolic = %pattern_type.loc15_32 (constants.%pattern_type.c49)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc15_32.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc15_55: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc15_55 (constants.%pattern_type.46f)]
// CHECK:STDOUT:   %pattern_type.loc15_68: type = pattern_type %T.loc15_22.1 [symbolic = %pattern_type.loc15_68 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_56: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc15_56 (constants.%require_complete.186)]
// CHECK:STDOUT:   %require_complete.loc15_69: <witness> = require_complete_type %T.loc15_22.1 [symbolic = %require_complete.loc15_69 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %Generic.type.loc15_45.1 [symbolic = %require_complete.loc16 (constants.%require_complete.7da)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc15_22.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.22afda.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc15_32.1, @Generic, @Generic(%T.loc15_22.1) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %Generic.WithSelf.F.type: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T.loc15_22.1, %U.loc15_32.1) [symbolic = %Generic.WithSelf.F.type (constants.%Generic.WithSelf.F.type.56e)]
// CHECK:STDOUT:   %.loc16_4.3: type = fn_type_with_self_type %Generic.WithSelf.F.type, %U.loc15_32.1 [symbolic = %.loc16_4.3 (constants.%.ab3)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @CallGenericMethod.%.loc16_4.3 (%.ab3) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Generic.WithSelf.F(%T.loc15_22.1, %U.loc15_32.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @CallGenericMethod.%U.binding.as_type (%U.binding.as_type), %s.param: @CallGenericMethod.%T.loc15_22.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc16: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.03dff7.2) = name_ref U, %U.loc15_32.2 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc16: type = facet_access_type %U.ref.loc16 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc16_4.1: type = converted %U.ref.loc16, %U.as_type.loc16 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc16_4.2: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = specific_constant @Generic.WithSelf.%assoc0.loc5_9.1, @Generic.WithSelf(constants.%T, constants.%U) [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = name_ref F, %.loc16_4.2 [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @CallGenericMethod.%.loc16_4.3 (%.ab3) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
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
// CHECK:STDOUT:   %.loc20_22.4: ref %ImplsGeneric = temporary %.loc20_22.2, %.loc20_22.3
// CHECK:STDOUT:   %.loc20_24.1: ref %ImplsGeneric = converted %.loc20_22.1, %.loc20_22.4
// CHECK:STDOUT:   %.loc20_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc20_42.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc20_42.3: init %GenericParam to %.loc20_42.2 = class_init () [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc20_42.4: ref %GenericParam = temporary %.loc20_42.2, %.loc20_42.3
// CHECK:STDOUT:   %.loc20_44.1: ref %GenericParam = converted %.loc20_42.1, %.loc20_42.4
// CHECK:STDOUT:   %Generic.facet: %Generic.type.498 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc20_59: %Generic.type.498 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc20_24.2: %ImplsGeneric = acquire_value %.loc20_24.1
// CHECK:STDOUT:   %.loc20_44.2: %GenericParam = acquire_value %.loc20_44.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc20_24.2, %.loc20_44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc20_42: <bound method> = bound_method %.loc20_42.4, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc20_42: init %empty_tuple.type = call %Destroy.Op.bound.loc20_42(%.loc20_42.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc20_22: <bound method> = bound_method %.loc20_22.4, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc20_22: init %empty_tuple.type = call %Destroy.Op.bound.loc20_22(%.loc20_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_42(%self.param: ref %GenericParam) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_22(%self.param: ref %ImplsGeneric) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Scalar, constants.%Self.15d2d5.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.15d2d5.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.498
// CHECK:STDOUT:   %Self.loc4_34.2 => constants.%Self.1f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Self.15d2d5.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.498
// CHECK:STDOUT:   %Self => constants.%Self.15d2d5.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.a23
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.fe0
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.498
// CHECK:STDOUT:   %Self => constants.%Generic.facet
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.c86
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.1fd
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self.loc4_34.2 => constants.%Self.15d2d5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%T, constants.%Self.15d2d5.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%T
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self => constants.%Self.15d2d5.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.74390a.2
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.baf95a.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.22afda.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.e0dc00.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.c49
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc15_55 => constants.%pattern_type.46f
// CHECK:STDOUT:   %pattern_type.loc15_68 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%T, constants.%U) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%T
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self => constants.%U
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.56e
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.a99
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.22afda.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.e0dc00.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.498
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.cba
// CHECK:STDOUT:   %U.binding.as_type => constants.%ImplsGeneric
// CHECK:STDOUT:   %pattern_type.loc15_55 => constants.%pattern_type.4da
// CHECK:STDOUT:   %pattern_type.loc15_68 => constants.%pattern_type.27a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_69 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.57a
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.717
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.c86
// CHECK:STDOUT:   %.loc16_4.3 => constants.%.fcf
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.54f: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.8fa: %I.type.54f = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.type.eb9: type = facet_type <@I, @I(%T.67d, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.1cb: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Self.061: %I.type.eb9 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.d0a: <witness> = require_complete_type %I.type.eb9 [symbolic]
// CHECK:STDOUT:   %I.facet.a6e: %I.type.eb9 = facet_value %C, (%I.impl_witness.1cb) [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.5f7: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %T.bc3: %I.type.5f7 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %I.type.5f7 [concrete]
// CHECK:STDOUT:   %Self.2a6: %I.type.5f7 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.bc3 [symbolic]
// CHECK:STDOUT:   %pattern_type.826: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d69: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.1c4: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.1c0: <witness> = complete_type_witness %I.type.5f7 [concrete]
// CHECK:STDOUT:   %I.facet.0e1: %I.type.5f7 = facet_value %C, (%I.impl_witness.1c4) [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet.0e1) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc3_13.2: type = symbolic_binding V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:     %.loc3_27.1: type = splice_block %.loc3_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc3_23.2: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.67d, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.eb9)]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.eee = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.826) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.826) = value_param_pattern %t.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.5f7] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_17: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_tuple.type)> [concrete = constants.%I.type.5f7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type.5f7 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.bc3)]
// CHECK:STDOUT:     %t.param: @A.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_31.1: type = splice_block %.loc9_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.5f7 = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.bc3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.loc3_13.1: type = symbolic_binding V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc3_23.1: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.54f)]
// CHECK:STDOUT:   %Self.loc3_33.2: @I.%I.type (%I.type.54f) = symbolic_binding Self, 2 [symbolic = %Self.loc3_33.2 (constants.%Self.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_33.1: @I.%I.type (%I.type.54f) = symbolic_binding Self, 2 [symbolic = %Self.loc3_33.2 (constants.%Self.8fa)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_33.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.eb9)]
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness.1cb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.d0a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_38.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness.1cb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_38.1
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
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type.5f7) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type.5f7 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.bc3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.826)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.d69)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.binding.as_type (%T.binding.as_type)) {
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
// CHECK:STDOUT:   %.loc12_6.4: ref %C = temporary %.loc12_6.2, %.loc12_6.3
// CHECK:STDOUT:   %.loc12_8.1: ref %C = converted %.loc12_6.1, %.loc12_6.4
// CHECK:STDOUT:   %I.facet: %I.type.5f7 = facet_value constants.%C, (constants.%I.impl_witness.1c4) [concrete = constants.%I.facet.0e1]
// CHECK:STDOUT:   %.loc12_12: %I.type.5f7 = converted constants.%C, %I.facet [concrete = constants.%I.facet.0e1]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet.0e1) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc12_8.2: %C = acquire_value %.loc12_8.1
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc12_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc12_6.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc12_6.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%V, constants.%W, constants.%Self.8fa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%T.67d
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.eb9
// CHECK:STDOUT:   %Self.loc3_33.2 => constants.%Self.061
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.67d
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.eb9
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2 => constants.%I.impl_witness.1cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%Self.8fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%I.facet.a6e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5f7
// CHECK:STDOUT:   %Self.loc3_33.2 => constants.%Self.2a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%empty_tuple.type, constants.%Self.8fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.bc3) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.bc3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.826
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.5f7
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2 => constants.%I.impl_witness.1c4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet.0e1) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%I.facet.0e1
// CHECK:STDOUT:   %T.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.54f: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.8fa: %I.type.54f = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.type.eb9: type = facet_type <@I, @I(%T.67d, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Self.061: %I.type.eb9 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %require_complete.d0a: <witness> = require_complete_type %I.type.eb9 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.eb9 = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.7b9: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %T.274: %I.type.7b9 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.24a: type = pattern_type %I.type.7b9 [concrete]
// CHECK:STDOUT:   %Self.c70: %I.type.7b9 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.274 [symbolic]
// CHECK:STDOUT:   %pattern_type.ced: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b50: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc3_13.2: type = symbolic_binding V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:     %.loc3_27.1: type = splice_block %.loc3_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc3_23.2: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_36: type = converted %.loc7_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc7_36.1: type = facet_type <@I, @I(constants.%T.67d, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.eb9)]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.24a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.ced) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.ced) = value_param_pattern %t.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %I.type [concrete = constants.%I.type.7b9] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc9_13: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_18.3: type = converted %.loc9_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_struct_type)> [concrete = constants.%I.type.7b9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type.7b9 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.274)]
// CHECK:STDOUT:     %t.param: @A.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_31.1: type = splice_block %.loc9_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.7b9 = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.274)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc3_13.2: type, %W.loc3_23.2: type) {
// CHECK:STDOUT:   %V.loc3_13.1: type = symbolic_binding V, 0 [symbolic = %V.loc3_13.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc3_23.1: type = symbolic_binding W, 1 [symbolic = %W.loc3_23.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc3_13.1, %W.loc3_23.1)> [symbolic = %I.type (constants.%I.type.54f)]
// CHECK:STDOUT:   %Self.loc3_33.2: @I.%I.type (%I.type.54f) = symbolic_binding Self, 2 [symbolic = %Self.loc3_33.2 (constants.%Self.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_33.1: @I.%I.type (%I.type.54f) = symbolic_binding Self, 2 [symbolic = %Self.loc3_33.2 (constants.%Self.8fa)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_33.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %I.type.loc7_36.2: type = facet_type <@I, @I(%T.loc7_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc7_36.2 (constants.%I.type.eb9)]
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_36.2 [symbolic = %require_complete (constants.%require_complete.d0a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_36.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_38.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_38.1
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
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type.7b9) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type.7b9 = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.274)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.ced)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.b50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc19_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_6.3: init %C to %.loc19_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_6.4: ref %C = temporary %.loc19_6.2, %.loc19_6.3
// CHECK:STDOUT:   %.loc19_8: ref %C = converted %.loc19_6.1, %.loc19_6.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc19_6.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc19_6.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%V
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%V, constants.%W, constants.%Self.8fa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%T.67d
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.eb9
// CHECK:STDOUT:   %Self.loc3_33.2 => constants.%Self.061
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.67d
// CHECK:STDOUT:   %I.type.loc7_36.2 => constants.%I.type.eb9
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%Self.8fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%empty_tuple.type, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc3_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc3_23.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.7b9
// CHECK:STDOUT:   %Self.loc3_33.2 => constants.%Self.c70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%empty_struct_type, constants.%Self.8fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.274) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.274
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ced
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.2f8: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.13c: type = class_type @C, @C(%T.67d, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.13c, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %T.651: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.651 [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.9f1: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.9f1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc5_13.1: type = splice_block %.loc5_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc5_9.2: type = symbolic_binding V, 0 [symbolic = %V.loc5_9.1 (constants.%V)]
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc5_19.2: type = symbolic_binding W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc7_30: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_31: type = converted %.loc7_30, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc7_31.1: type = class_type @C, @C(constants.%T.67d, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.13c)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @A.%pattern_type (%pattern_type.422) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @A.%pattern_type (%pattern_type.422) = value_param_pattern %t.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_10: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.651)]
// CHECK:STDOUT:     %t.param: @A.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %.loc9_23.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.651)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_23.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %C.loc7_31.2: type = class_type @C, @C(%T.loc7_14.2, constants.%empty_tuple.type) [symbolic = %C.loc7_31.2 (constants.%C.13c)]
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_31.1 as %I.ref {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_38.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T.67d) [symbolic = %I.impl_witness.loc7_38.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_38.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc5_9.2: type, %W.loc5_19.2: type) {
// CHECK:STDOUT:   %V.loc5_9.1: type = symbolic_binding V, 0 [symbolic = %V.loc5_9.1 (constants.%V)]
// CHECK:STDOUT:   %W.loc5_19.1: type = symbolic_binding W, 1 [symbolic = %W.loc5_19.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2f8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc9_6.2: %I.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.651)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @A.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc19_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_18: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_19.1: type = converted %.loc19_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc19_19.2: type = converted %.loc19_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_struct_type, constants.%empty_struct_type) [concrete = constants.%C.9f1]
// CHECK:STDOUT:   %.loc19_6.2: ref %C.9f1 = temporary_storage
// CHECK:STDOUT:   %.loc19_6.3: init %C.9f1 to %.loc19_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_6.4: ref %C.9f1 = temporary %.loc19_6.2, %.loc19_6.3
// CHECK:STDOUT:   %.loc19_8: ref %C.9f1 = converted %.loc19_6.1, %.loc19_6.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc19_6.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc19_6.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %C.9f1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%V
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%T.67d
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T.67d
// CHECK:STDOUT:   %C.loc7_31.2 => constants.%C.13c
// CHECK:STDOUT:   %I.impl_witness.loc7_38.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.651) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.651
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc5_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_19.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
