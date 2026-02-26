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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_convert_class_type_to_generic_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_convert_class_type_to_generic_facet_value.carbon

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}
class WrongGenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod(T:! type, unused U:! Generic(T)) {}

fn G() {
  // CHECK:STDERR: fail_convert_class_type_to_generic_facet_value.carbon:[[@LINE+7]]:3: error: cannot convert type `ImplsGeneric` into type implementing `Generic(WrongGenericParam)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CallGenericMethod(WrongGenericParam, ImplsGeneric);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_convert_class_type_to_generic_facet_value.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CallGenericMethod(T:! type, unused U:! Generic(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CallGenericMethod(WrongGenericParam, ImplsGeneric);
}

// CHECK:STDOUT: --- fail_convert_class_type_to_generic_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.835: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.835 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.03dff7.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.15d: %Generic.type.03dff7.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.743: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%Scalar, %Self.15d) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.baf: %Generic.WithSelf.F.type.743 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.22a: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.e0d: %Generic.assoc_type.22a = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %WrongGenericParam: type = class_type @WrongGenericParam [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.498: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.1f5: %Generic.type.498 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.a23: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Self.15d) [symbolic]
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
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .GenericParam = %GenericParam.decl
// CHECK:STDOUT:     .WrongGenericParam = %WrongGenericParam.decl
// CHECK:STDOUT:     .ImplsGeneric = %ImplsGeneric.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type.835 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_28.1: type = splice_block %.loc15_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Scalar.loc15_19.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %WrongGenericParam.decl: type = class_decl @WrongGenericParam [concrete = constants.%WrongGenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.498]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.c49) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_26.1: type = splice_block %.loc27_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc27_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc27_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc27_52: type = splice_block %Generic.type.loc27_52.2 [symbolic = %Generic.type.loc27_52.1 (constants.%Generic.type.03dff7.2)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc27_22.2 [symbolic = %T.loc27_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc27_52.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc27_52.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc27_39.2: @CallGenericMethod.%Generic.type.loc27_52.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc27_39.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc15_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc15_19.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc15_19.1)> [symbolic = %Generic.type (constants.%Generic.type.03dff7.1)]
// CHECK:STDOUT:   %Self.loc15_34.2: @Generic.%Generic.type (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.15d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_34.1: @Generic.%Generic.type (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.15d)]
// CHECK:STDOUT:     %Generic.WithSelf.decl = interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.WithSelf.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.743) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.WithSelf.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.baf)] {} {}
// CHECK:STDOUT:     %assoc0.loc16_9.1: @Generic.WithSelf.%Generic.assoc_type (%Generic.assoc_type.22a) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.e0d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_34.1
// CHECK:STDOUT:     .F = @Generic.WithSelf.%assoc0.loc16_9.1
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongGenericParam {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WrongGenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc15_19.2: type, @Generic.%Self.loc15_34.1: @Generic.%Generic.type (%Generic.type.03dff7.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc27_22.2: type, %U.loc27_39.2: @CallGenericMethod.%Generic.type.loc27_52.1 (%Generic.type.03dff7.2)) {
// CHECK:STDOUT:   %T.loc27_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc27_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc27_52.1: type = facet_type <@Generic, @Generic(%T.loc27_22.1)> [symbolic = %Generic.type.loc27_52.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:   %U.loc27_39.1: @CallGenericMethod.%Generic.type.loc27_52.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc27_39.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc27_52.1 [symbolic = %pattern_type (constants.%pattern_type.c49)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %WrongGenericParam.ref: type = name_ref WrongGenericParam, file.%WrongGenericParam.decl [concrete = constants.%WrongGenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Scalar, constants.%Self.15d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.15d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.498
// CHECK:STDOUT:   %Self.loc15_34.2 => constants.%Self.1f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Self.15d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.498
// CHECK:STDOUT:   %Self => constants.%Self.15d
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.a23
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.fe0
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.717
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
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc27_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc27_52.1 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %U.loc27_39.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%WrongGenericParam) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%WrongGenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
