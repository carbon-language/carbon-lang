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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/call_method_on_generic_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/call_method_on_generic_facet.carbon

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

interface Other {
  fn G();
}
impl ImplsGeneric as Other {
  fn G();
}

fn CallGenericMethod(T:! type, U:! Generic(T)) {
  U.F();
}

fn G() {
  CallGenericMethod(GenericParam, ImplsGeneric);
}

// CHECK:STDOUT: --- call_method_on_generic_facet.carbon
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
// CHECK:STDOUT:   %Other.type: type = facet_type <@Other> [concrete]
// CHECK:STDOUT:   %Self.807: %Other.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.0c6: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.17f: %G.type.0c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Other.assoc_type: type = assoc_entity_type @Other [concrete]
// CHECK:STDOUT:   %assoc0.25d: %Other.assoc_type = assoc_entity element0, @Other.%G.decl [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness file.%Other.impl_witness_table [concrete]
// CHECK:STDOUT:   %G.type.58d: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.b67: %G.type.58d = struct_value () [concrete]
// CHECK:STDOUT:   %Other.facet: %Other.type = facet_value %ImplsGeneric, (%Other.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.dee8d8.2: %Generic.type.91ccba.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.2: type = fn_type @F.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.2: %F.type.f439a9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.0fd877.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.2966cb.2: %Generic.assoc_type.0fd877.2 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.facet.680: %Generic.type.91ccba.2 = facet_value %U.as_type, (%Generic.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.177: type = fn_type_with_self_type %F.type.f439a9.2, %Generic.facet.680 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.177 = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F.1(%T, %Generic.facet.680) [symbolic]
// CHECK:STDOUT:   %G.type.9f9: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.57b: %G.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ded: type = pattern_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet.8ff) [concrete]
// CHECK:STDOUT:   %complete_type.997: <witness> = complete_type_witness %Generic.type.769 [concrete]
// CHECK:STDOUT:   %.3e6: type = fn_type_with_self_type %F.type.4cf, %Generic.facet.8ff [concrete]
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
// CHECK:STDOUT:     .Other = %Other.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.loc15_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc15_19.2 (constants.%Scalar)]
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
// CHECK:STDOUT:   %Other.decl: type = interface_decl @Other [concrete = constants.%Other.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.728 [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Other.ref: type = name_ref Other, file.%Other.decl [concrete = constants.%Other.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.impl_witness_table = impl_witness_table (@impl.728.%G.decl), @impl.728 [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness %Other.impl_witness_table [concrete = constants.%Other.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.80f) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc33_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc33_22.2 (constants.%T)]
// CHECK:STDOUT:     %.loc33: type = splice_block %Generic.type.loc33_45.1 [symbolic = %Generic.type.loc33_45.2 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc33_22.1 [symbolic = %T.loc33_22.2 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc33_45.1: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc33_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc33_32.1: @CallGenericMethod.%Generic.type.loc33_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc33_32.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.9f9 = fn_decl @G.3 [concrete = constants.%G.57b] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc15_19.1: type) {
// CHECK:STDOUT:   %Scalar.loc15_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc15_19.2 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc15_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee8d8.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%Scalar.loc15_19.2) [symbolic = %F.type (constants.%F.type.f439a9.1)]
// CHECK:STDOUT:   %F: @Generic.%F.type (%F.type.f439a9.1) = struct_value () [symbolic = %F (constants.%F.8a2d67.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc15_19.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.1)]
// CHECK:STDOUT:   %assoc0.loc16_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee8d8.1)]
// CHECK:STDOUT:     %F.decl: @Generic.%F.type (%F.type.f439a9.1) = fn_decl @F.1 [symbolic = @Generic.%F (constants.%F.8a2d67.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc16_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.0fd877.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.2966cb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc16_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Other {
// CHECK:STDOUT:   %Self: %Other.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.807]
// CHECK:STDOUT:   %G.decl: %G.type.0c6 = fn_decl @G.1 [concrete = constants.%G.17f] {} {}
// CHECK:STDOUT:   %assoc0: %Other.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.25d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
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
// CHECK:STDOUT: impl @impl.728: %ImplsGeneric.ref as %Other.ref {
// CHECK:STDOUT:   %G.decl: %G.type.58d = fn_decl @G.2 [concrete = constants.%G.b67] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%Other.impl_witness
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
// CHECK:STDOUT: generic fn @F.1(@Generic.%Scalar.loc15_19.1: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@Other.%Self: %Other.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc33_22.1: type, %U.loc33_32.1: @CallGenericMethod.%Generic.type.loc33_45.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc33_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc33_22.2 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc33_45.2: type = facet_type <@Generic, @Generic(%T.loc33_22.2)> [symbolic = %Generic.type.loc33_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc33_32.2: @CallGenericMethod.%Generic.type.loc33_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc33_32.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc33_45.2 [symbolic = %pattern_type (constants.%pattern_type.80f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.loc33_45.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc33_22.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.0fd877.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = assoc_entity element0, @Generic.%F.decl [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:   %U.as_type.loc34_4.2: type = facet_access_type %U.loc33_32.2 [symbolic = %U.as_type.loc34_4.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc33_32.2, @Generic, @Generic(%T.loc33_22.2) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%T.loc33_22.2) [symbolic = %F.type (constants.%F.type.f439a9.2)]
// CHECK:STDOUT:   %Generic.facet: @CallGenericMethod.%Generic.type.loc33_45.2 (%Generic.type.91ccba.2) = facet_value %U.as_type.loc34_4.2, (%Generic.lookup_impl_witness) [symbolic = %Generic.facet (constants.%Generic.facet.680)]
// CHECK:STDOUT:   %.loc34_4.3: type = fn_type_with_self_type %F.type, %Generic.facet [symbolic = %.loc34_4.3 (constants.%.177)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @CallGenericMethod.%.loc34_4.3 (%.177) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @F.1(%T.loc33_22.2, %Generic.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @CallGenericMethod.%Generic.type.loc33_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc33_32.1 [symbolic = %U.loc33_32.2 (constants.%U)]
// CHECK:STDOUT:     %.loc34_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = specific_constant @Generic.%assoc0.loc16_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.0fd877.2) = name_ref F, %.loc34_4.1 [symbolic = %assoc0 (constants.%assoc0.2966cb.2)]
// CHECK:STDOUT:     %U.as_type.loc34_4.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc34_4.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc34_4.2: type = converted %U.ref, %U.as_type.loc34_4.1 [symbolic = %U.as_type.loc34_4.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @CallGenericMethod.%.loc34_4.3 (%.177) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @F.1(constants.%T, constants.%Generic.facet.680) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc34_7: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet.8ff]
// CHECK:STDOUT:   %.loc38: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet.8ff]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8ff) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc15_19.2 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Scalar, constants.%Self.dee8d8.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc15_19.2 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self.f84
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %F => constants.%F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%GenericParam, constants.%Generic.facet.8ff) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.807) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Other.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc15_19.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.dee8d8.2
// CHECK:STDOUT:   %F.type => constants.%F.type.f439a9.2
// CHECK:STDOUT:   %F => constants.%F.8a2d67.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.0fd877.2
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.2966cb.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc33_22.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc33_45.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc33_32.2 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Generic.facet.680) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8ff) {
// CHECK:STDOUT:   %T.loc33_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc33_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc33_32.2 => constants.%Generic.facet.8ff
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ded
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.997
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9b7
// CHECK:STDOUT:   %U.as_type.loc34_4.2 => constants.%ImplsGeneric
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %Generic.facet => constants.%Generic.facet.8ff
// CHECK:STDOUT:   %.loc34_4.3 => constants.%.3e6
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%F.a56
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%F.a56
// CHECK:STDOUT: }
// CHECK:STDOUT:
