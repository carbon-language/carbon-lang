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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_class_type_to_generic_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_class_type_to_generic_facet_value.carbon

// --- generic_facet_type.carbon

library "[[@TEST_NAME]]";

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod(T:! type, unused U:! Generic(T)) {}

fn G() {
  CallGenericMethod(GenericParam, ImplsGeneric);
}

fn PassThroughToGenericMethod(T:! type, U:! Generic(T)) {
  CallGenericMethod(T, U);
}

fn H() {
  PassThroughToGenericMethod(GenericParam, ImplsGeneric);
}

// --- generic_facet_type_from_implicit_param.carbon

library "[[@TEST_NAME]]";

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod[T:! type](unused U:! Generic(T), unused t: T) {}

fn G() {
  CallGenericMethod(ImplsGeneric, {} as GenericParam);
}

// CHECK:STDOUT: --- generic_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.600: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.600 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.6c910d.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.5722d4.1: %Generic.type.6c910d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.e3a452.1: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%Scalar, %Self.5722d4.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.5f20bc.1: %Generic.WithSelf.F.type.e3a452.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.4e2e58.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.d7d4e4.1: %Generic.assoc_type.4e2e58.1 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.5c0: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.d5f: %Generic.type.5c0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.3f6: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Self.5722d4.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.0ab: %Generic.WithSelf.F.type.3f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.7b1: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.f08: %Generic.assoc_type.7b1 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.5c0 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.d8d: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.708: %Generic.WithSelf.F.type.d8d = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.6c910d.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.5a6: type = pattern_type %Generic.type.6c910d.2 [symbolic]
// CHECK:STDOUT:   %U.ae489d.1: %Generic.type.6c910d.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c96: type = pattern_type %Generic.type.5c0 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.3ac: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %U.ae489d.2: %Generic.type.6c910d.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %PassThroughToGenericMethod.type: type = fn_type @PassThroughToGenericMethod [concrete]
// CHECK:STDOUT:   %PassThroughToGenericMethod: %PassThroughToGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.5722d4.2: %Generic.type.6c910d.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.e3a452.2: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%T, %Self.5722d4.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.5f20bc.2: %Generic.WithSelf.F.type.e3a452.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.4e2e58.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.d7d4e4.2: %Generic.assoc_type.4e2e58.2 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.fb8: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%T, %U.ae489d.2) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassThroughToGenericMethod.specific_fn: <specific function> = specific_function %PassThroughToGenericMethod, @PassThroughToGenericMethod(%GenericParam, %Generic.facet) [concrete]
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
// CHECK:STDOUT:     .ImplsGeneric = %ImplsGeneric.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .PassThroughToGenericMethod = %PassThroughToGenericMethod.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type.600 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Scalar.loc4_25.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.600 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.5c0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.5a6) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_26.1: type = splice_block %.loc15_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc15_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_52: type = splice_block %Generic.type.loc15_52.2 [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)] {
// CHECK:STDOUT:       %.Self.loc15_40: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.600 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15_23.2 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_52.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_40.2: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc15_40.1 (constants.%U.ae489d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %PassThroughToGenericMethod.decl: %PassThroughToGenericMethod.type = fn_decl @PassThroughToGenericMethod [concrete = constants.%PassThroughToGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @PassThroughToGenericMethod.%pattern_type (%pattern_type.5a6) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_35.1: type = splice_block %.loc21_35.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc21_32: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_35.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_32.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_32.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21_54: type = splice_block %Generic.type.loc21_54.2 [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.6c910d.2)] {
// CHECK:STDOUT:       %.Self.loc21_42: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.600 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc21: type = name_ref T, %T.loc21_32.2 [symbolic = %T.loc21_32.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc21_54.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc21_42.2: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc21_42.1 (constants.%U.ae489d.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_25.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_25.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_25.1)> [symbolic = %Generic.type (constants.%Generic.type.6c910d.1)]
// CHECK:STDOUT:   %Self.loc4_34.2: @Generic.%Generic.type (%Generic.type.6c910d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.5722d4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.6c910d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.5722d4.1)]
// CHECK:STDOUT:     %Generic.WithSelf.decl = interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.WithSelf.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.e3a452.1) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.WithSelf.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.5f20bc.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.WithSelf.%Generic.assoc_type (%Generic.assoc_type.4e2e58.1) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.d7d4e4.1)]
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
// CHECK:STDOUT:   extend %Generic.type
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
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc4_25.2: type, @Generic.%Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.6c910d.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_23.2: type, %U.loc15_40.2: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2)) {
// CHECK:STDOUT:   %T.loc15_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_52.1: type = facet_type <@Generic, @Generic(%T.loc15_23.1)> [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:   %U.loc15_40.1: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc15_40.1 (constants.%U.ae489d.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc15_52.1 [symbolic = %pattern_type (constants.%pattern_type.5a6)]
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
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.5c0 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc18: %Generic.type.5c0 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn.3ac]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PassThroughToGenericMethod(%T.loc21_32.2: type, %U.loc21_42.2: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.6c910d.2)) {
// CHECK:STDOUT:   %T.loc21_32.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_32.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc21_54.1: type = facet_type <@Generic, @Generic(%T.loc21_32.1)> [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:   %U.loc21_42.1: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc21_42.1 (constants.%U.ae489d.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc21_54.1 [symbolic = %pattern_type (constants.%pattern_type.5a6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc22_3.2: <specific function> = specific_function constants.%CallGenericMethod, @CallGenericMethod(%T.loc21_32.1, %U.loc21_42.1) [symbolic = %CallGenericMethod.specific_fn.loc22_3.2 (constants.%CallGenericMethod.specific_fn.fb8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:     %T.ref.loc22: type = name_ref T, %T.loc21_32.2 [symbolic = %T.loc21_32.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.6c910d.2) = name_ref U, %U.loc21_42.2 [symbolic = %U.loc21_42.1 (constants.%U.ae489d.2)]
// CHECK:STDOUT:     %CallGenericMethod.specific_fn.loc22_3.1: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%T, constants.%U.ae489d.2) [symbolic = %CallGenericMethod.specific_fn.loc22_3.2 (constants.%CallGenericMethod.specific_fn.fb8)]
// CHECK:STDOUT:     %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn.loc22_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PassThroughToGenericMethod.ref: %PassThroughToGenericMethod.type = name_ref PassThroughToGenericMethod, file.%PassThroughToGenericMethod.decl [concrete = constants.%PassThroughToGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.5c0 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc26: %Generic.type.5c0 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %PassThroughToGenericMethod.specific_fn: <specific function> = specific_function %PassThroughToGenericMethod.ref, @PassThroughToGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%PassThroughToGenericMethod.specific_fn]
// CHECK:STDOUT:   %PassThroughToGenericMethod.call: init %empty_tuple.type = call %PassThroughToGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Scalar, constants.%Self.5722d4.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.5722d4.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self.loc4_34.2 => constants.%Self.d5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Self.5722d4.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self => constants.%Self.5722d4.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.3f6
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.0ab
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.7b1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self => constants.%Generic.facet
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.d8d
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.708
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.7b1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %Self.loc4_34.2 => constants.%Self.5722d4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U.ae489d.1) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_52.1 => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %U.loc15_40.1 => constants.%U.ae489d.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_52.1 => constants.%Generic.type.5c0
// CHECK:STDOUT:   %U.loc15_40.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%T, constants.%U.ae489d.2) {
// CHECK:STDOUT:   %T.loc21_32.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc21_54.1 => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %U.loc21_42.1 => constants.%U.ae489d.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%T, constants.%Self.5722d4.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%T
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %Self => constants.%Self.5722d4.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.e3a452.2
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.5f20bc.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.4e2e58.2
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.d7d4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U.ae489d.2) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_52.1 => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %U.loc15_40.1 => constants.%U.ae489d.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5a6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc21_32.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc21_54.1 => constants.%Generic.type.5c0
// CHECK:STDOUT:   %U.loc21_42.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc22_3.2 => constants.%CallGenericMethod.specific_fn.3ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_facet_type_from_implicit_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.600: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.600 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.6c910d.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.572: %Generic.type.6c910d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.e3a: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%Scalar, %Self.572) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.5f2: %Generic.WithSelf.F.type.e3a = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.4e2: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.d7d: %Generic.assoc_type.4e2 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.5c0: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.d5f: %Generic.type.5c0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.3f6: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Self.572) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.0ab: %Generic.WithSelf.F.type.3f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.7b1: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.f08: %Generic.assoc_type.7b1 = assoc_entity element0, @Generic.WithSelf.%Generic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.5c0 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.d8d: type = fn_type @Generic.WithSelf.F, @Generic.WithSelf(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.708: %Generic.WithSelf.F.type.d8d = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.6c910d.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type.5a6: type = pattern_type %Generic.type.6c910d.2 [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.6c910d.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c96: type = pattern_type %Generic.type.5c0 [concrete]
// CHECK:STDOUT:   %pattern_type.71e: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %.9f5: ref %GenericParam = temporary invalid, %GenericParam.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.2: type = fn_type @Destroy.Op.loc18_38.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.2: %Destroy.Op.type.3f5f78.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.9f5, %Destroy.Op.f0ba5f.2 [concrete]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.600 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Scalar.loc4_25.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.600 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.5c0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type.loc15_40 (%pattern_type.5a6) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %t.param_patt: @CallGenericMethod.%pattern_type.loc15_63 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @CallGenericMethod.%pattern_type.loc15_63 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_26.1: type = splice_block %.loc15_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc15_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_52: type = splice_block %Generic.type.loc15_52.2 [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)] {
// CHECK:STDOUT:       %.Self.loc15_40: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.600 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_51: type = name_ref T, %T.loc15_23.2 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_52.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_40.2: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc15_40.1 (constants.%U)]
// CHECK:STDOUT:     %t.param: @CallGenericMethod.%T.loc15_23.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc15_65: type = name_ref T, %T.loc15_23.2 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:     %t: @CallGenericMethod.%T.loc15_23.1 (%T) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_25.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_25.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc4_25.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_25.1)> [symbolic = %Generic.type (constants.%Generic.type.6c910d.1)]
// CHECK:STDOUT:   %Self.loc4_34.2: @Generic.%Generic.type (%Generic.type.6c910d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.572)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.6c910d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.572)]
// CHECK:STDOUT:     %Generic.WithSelf.decl = interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.WithSelf.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.e3a) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.WithSelf.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.5f2)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.WithSelf.%Generic.assoc_type (%Generic.assoc_type.4e2) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.d7d)]
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
// CHECK:STDOUT:   extend %Generic.type
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
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc4_25.2: type, @Generic.%Self.loc4_34.1: @Generic.%Generic.type (%Generic.type.6c910d.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_23.2: type, %U.loc15_40.2: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2)) {
// CHECK:STDOUT:   %T.loc15_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_52.1: type = facet_type <@Generic, @Generic(%T.loc15_23.1)> [symbolic = %Generic.type.loc15_52.1 (constants.%Generic.type.6c910d.2)]
// CHECK:STDOUT:   %U.loc15_40.1: @CallGenericMethod.%Generic.type.loc15_52.1 (%Generic.type.6c910d.2) = symbolic_binding U, 1 [symbolic = %U.loc15_40.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc15_40: type = pattern_type %Generic.type.loc15_52.1 [symbolic = %pattern_type.loc15_40 (constants.%pattern_type.5a6)]
// CHECK:STDOUT:   %pattern_type.loc15_63: type = pattern_type %T.loc15_23.1 [symbolic = %pattern_type.loc15_63 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_23.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @CallGenericMethod.%T.loc15_23.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %.loc18_36.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc18_36.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc18_36.3: init %GenericParam to %.loc18_36.2 = class_init () [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc18_38.1: init %GenericParam = converted %.loc18_36.1, %.loc18_36.3 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.5c0 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc18_53: %Generic.type.5c0 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc18_38.2: ref %GenericParam = temporary %.loc18_36.2, %.loc18_38.1 [concrete = constants.%.9f5]
// CHECK:STDOUT:   %.loc18_38.3: %GenericParam = acquire_value %.loc18_38.2 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc18_38.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.9f5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_38.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_38.2(%self.param: ref %GenericParam) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Scalar, constants.%Self.572) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.572) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self.loc4_34.2 => constants.%Self.d5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Self.572) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self => constants.%Self.572
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.3f6
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.0ab
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.7b1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.5c0
// CHECK:STDOUT:   %Self => constants.%Generic.facet
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.d8d
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.708
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.7b1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc4_25.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_52.1 => constants.%Generic.type.6c910d.2
// CHECK:STDOUT:   %U.loc15_40.1 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc15_40 => constants.%pattern_type.5a6
// CHECK:STDOUT:   %pattern_type.loc15_63 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_52.1 => constants.%Generic.type.5c0
// CHECK:STDOUT:   %U.loc15_40.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type.loc15_40 => constants.%pattern_type.c96
// CHECK:STDOUT:   %pattern_type.loc15_63 => constants.%pattern_type.71e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
