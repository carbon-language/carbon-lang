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

fn CallGenericMethod(T:! type, U:! Generic(T)) {}

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

fn CallGenericMethod[T:! type](U:! Generic(T), t: T) {}

fn G() {
  CallGenericMethod(ImplsGeneric, {} as GenericParam);
}

// CHECK:STDOUT: --- generic_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.dee: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.f43: type = fn_type @Generic.F, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %Generic.F.8a2: %Generic.F.type.f43 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.296: %Generic.assoc_type.0fd = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Self.f84: %Generic.type.769 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.4cf: type = fn_type @Generic.F, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %Generic.F.118: %Generic.F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.713: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.713 = assoc_entity element0, @Generic.%Generic.F.decl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness file.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ded: type = pattern_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.2ff: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %PassThroughToGenericMethod.type: type = fn_type @PassThroughToGenericMethod [concrete]
// CHECK:STDOUT:   %PassThroughToGenericMethod: %PassThroughToGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.a24: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%T, %U) [symbolic]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.loc4_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.impl_witness_table = impl_witness_table (@ImplsGeneric.as.Generic.impl.%ImplsGeneric.as.Generic.impl.F.decl), @ImplsGeneric.as.Generic.impl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness %Generic.impl_witness_table [concrete = constants.%Generic.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.80f) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15: type = splice_block %Generic.type.loc15_45.2 [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %PassThroughToGenericMethod.decl: %PassThroughToGenericMethod.type = fn_decl @PassThroughToGenericMethod [concrete = constants.%PassThroughToGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @PassThroughToGenericMethod.%pattern_type (%pattern_type.80f) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc21_31.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_31.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21: type = splice_block %Generic.type.loc21_54.2 [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc21: type = name_ref T, %T.loc21_31.2 [symbolic = %T.loc21_31.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc21_54.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc21_41.2: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc21_41.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.f43)]
// CHECK:STDOUT:   %Generic.F: @Generic.%Generic.F.type (%Generic.F.type.f43) = struct_value () [symbolic = %Generic.F (constants.%Generic.F.8a2)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.296)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:     %Generic.F.decl: @Generic.%Generic.F.type (%Generic.F.type.f43) = fn_decl @Generic.F [symbolic = @Generic.%Generic.F (constants.%Generic.F.8a2)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.296)]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.F(@Generic.%Scalar.loc4_19.2: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_22.2: type, %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_45.1: type = facet_type <@Generic, @Generic(%T.loc15_22.1)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc15_45.1 [symbolic = %pattern_type (constants.%pattern_type.80f)]
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
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc18: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn.2ff]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PassThroughToGenericMethod(%T.loc21_31.2: type, %U.loc21_41.2: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc21_31.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_31.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc21_54.1: type = facet_type <@Generic, @Generic(%T.loc21_31.1)> [symbolic = %Generic.type.loc21_54.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc21_41.1: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc21_41.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc21_54.1 [symbolic = %pattern_type (constants.%pattern_type.80f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc22_3.2: <specific function> = specific_function constants.%CallGenericMethod, @CallGenericMethod(%T.loc21_31.1, %U.loc21_41.1) [symbolic = %CallGenericMethod.specific_fn.loc22_3.2 (constants.%CallGenericMethod.specific_fn.a24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:     %T.ref.loc22: type = name_ref T, %T.loc21_31.2 [symbolic = %T.loc21_31.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: @PassThroughToGenericMethod.%Generic.type.loc21_54.1 (%Generic.type.91ccba.2) = name_ref U, %U.loc21_41.2 [symbolic = %U.loc21_41.1 (constants.%U)]
// CHECK:STDOUT:     %CallGenericMethod.specific_fn.loc22_3.1: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%T, constants.%U) [symbolic = %CallGenericMethod.specific_fn.loc22_3.2 (constants.%CallGenericMethod.specific_fn.a24)]
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
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc26: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %PassThroughToGenericMethod.specific_fn: <specific function> = specific_function %PassThroughToGenericMethod.ref, @PassThroughToGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%PassThroughToGenericMethod.specific_fn]
// CHECK:STDOUT:   %PassThroughToGenericMethod.call: init %empty_tuple.type = call %PassThroughToGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%Scalar, constants.%Self.dee) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self.f84
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ded
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc21_31.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc21_54.1 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc21_41.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc21_31.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc21_54.1 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc21_41.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ded
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc22_3.2 => constants.%CallGenericMethod.specific_fn.2ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_facet_type_from_implicit_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.dee: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.f43: type = fn_type @Generic.F, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %Generic.F.8a2: %Generic.F.type.f43 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.296: %Generic.assoc_type.0fd = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Self.f84: %Generic.type.769 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.4cf: type = fn_type @Generic.F, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %Generic.F.118: %Generic.F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.713: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.713 = assoc_entity element0, @Generic.%Generic.F.decl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness file.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ded: type = pattern_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %pattern_type.589: type = pattern_type %GenericParam [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ae1: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%GenericParam) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.0cc: %T.as.Destroy.impl.Op.type.ae1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f73: type = ptr_type %GenericParam [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.0cc, @T.as.Destroy.impl.Op(%GenericParam) [concrete]
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
// CHECK:STDOUT:     %Scalar.loc4_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.impl_witness_table = impl_witness_table (@ImplsGeneric.as.Generic.impl.%ImplsGeneric.as.Generic.impl.F.decl), @ImplsGeneric.as.Generic.impl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness %Generic.impl_witness_table [concrete = constants.%Generic.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type.loc15_32 (%pattern_type.80f) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %t.patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @CallGenericMethod.%pattern_type.loc15_48 (%pattern_type.7dc) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15: type = splice_block %Generic.type.loc15_45.2 [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc15_44: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc15_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:     %t.param: @CallGenericMethod.%T.loc15_22.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc15_51: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:     %t: @CallGenericMethod.%T.loc15_22.1 (%T) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc4_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc4_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc4_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc4_19.1)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.f43)]
// CHECK:STDOUT:   %Generic.F: @Generic.%Generic.F.type (%Generic.F.type.f43) = struct_value () [symbolic = %Generic.F (constants.%Generic.F.8a2)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc4_19.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.296)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:     %Generic.F.decl: @Generic.%Generic.F.type (%Generic.F.type.f43) = fn_decl @Generic.F [symbolic = @Generic.%Generic.F (constants.%Generic.F.8a2)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.296)]
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
// CHECK:STDOUT: generic fn @Generic.F(@Generic.%Scalar.loc4_19.2: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc15_22.2: type, %U.loc15_32.2: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc15_45.1: type = facet_type <@Generic, @Generic(%T.loc15_22.1)> [symbolic = %Generic.type.loc15_45.1 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc15_32.1: @CallGenericMethod.%Generic.type.loc15_45.1 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc15_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc15_32: type = pattern_type %Generic.type.loc15_45.1 [symbolic = %pattern_type.loc15_32 (constants.%pattern_type.80f)]
// CHECK:STDOUT:   %pattern_type.loc15_48: type = pattern_type %T.loc15_22.1 [symbolic = %pattern_type.loc15_48 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_22.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @CallGenericMethod.%T.loc15_22.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %.loc18_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc18_36.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc18_36.3: init %GenericParam = class_init (), %.loc18_36.2 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc18_36.4: ref %GenericParam = temporary %.loc18_36.2, %.loc18_36.3
// CHECK:STDOUT:   %.loc18_38.1: ref %GenericParam = converted %.loc18_36.1, %.loc18_36.4
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc18_53: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc18_38.2: %GenericParam = bind_value %.loc18_38.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc18_38.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc18_36.2, constants.%T.as.Destroy.impl.Op.0cc
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.0cc, @T.as.Destroy.impl.Op(constants.%GenericParam) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc18_36.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f73 = addr_of %.loc18_36.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%Scalar, constants.%Self.dee) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc4_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self.f84
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.80f
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc15_45.1 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc15_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type.loc15_32 => constants.%pattern_type.ded
// CHECK:STDOUT:   %pattern_type.loc15_48 => constants.%pattern_type.589
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
