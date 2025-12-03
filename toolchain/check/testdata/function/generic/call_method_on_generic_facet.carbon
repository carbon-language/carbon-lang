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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Scalar: type = symbolic_binding Scalar, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.68bb8d.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.fc1c73.1: %Generic.type.68bb8d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.fc1c8a.1: type = fn_type @Generic.F, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %Generic.F.d952d5.1: %Generic.F.type.fc1c8a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.414a90.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.3122a2.1: %Generic.assoc_type.414a90.1 = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.621: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Self.ae2: %Generic.type.621 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.4cf: type = fn_type @Generic.F, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %Generic.F.118: %Generic.F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.713: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.815: %Generic.assoc_type.713 = assoc_entity element0, @Generic.%Generic.F.decl [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness file.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.621 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Other.type: type = facet_type <@Other> [concrete]
// CHECK:STDOUT:   %Self.5b9: %Other.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Other.G.type: type = fn_type @Other.G [concrete]
// CHECK:STDOUT:   %Other.G: %Other.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Other.assoc_type: type = assoc_entity_type @Other [concrete]
// CHECK:STDOUT:   %assoc0.25d: %Other.assoc_type = assoc_entity element0, @Other.%Other.G.decl [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness file.%Other.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G.type: type = fn_type @ImplsGeneric.as.Other.impl.G [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G: %ImplsGeneric.as.Other.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Other.facet: %Other.type = facet_value %ImplsGeneric, (%Other.impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.68bb8d.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.68bb8d.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e11: type = pattern_type %Generic.type.68bb8d.2 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.fc1c73.2: %Generic.type.68bb8d.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.F.type.fc1c8a.2: type = fn_type @Generic.F, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.F.d952d5.2: %Generic.F.type.fc1c8a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.414a90.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.3122a2.2: %Generic.assoc_type.414a90.2 = assoc_entity element0, @Generic.%Generic.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.68bb8d.2 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %.9ee: type = fn_type_with_self_type %Generic.F.type.fc1c8a.2, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.9ee = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Generic.F(%T, %U) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c4b: type = pattern_type %Generic.type.621 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %complete_type.ca2: <witness> = complete_type_witness %Generic.type.621 [concrete]
// CHECK:STDOUT:   %.573: type = fn_type_with_self_type %Generic.F.type.4cf, %Generic.facet [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Scalar.loc15_19.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
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
// CHECK:STDOUT:   %Other.decl: type = interface_decl @Other [concrete = constants.%Other.type] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Other.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Other.ref: type = name_ref Other, file.%Other.decl [concrete = constants.%Other.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.impl_witness_table = impl_witness_table (@ImplsGeneric.as.Other.impl.%ImplsGeneric.as.Other.impl.G.decl), @ImplsGeneric.as.Other.impl [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness %Other.impl_witness_table [concrete = constants.%Other.impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.e11) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc33_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc33: type = splice_block %Generic.type.loc33_45.2 [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.68bb8d.2)] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc33_22.2 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc33_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.68bb8d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc33_32.2: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.68bb8d.2) = symbolic_binding U, 1 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc15_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc15_19.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc15_19.1)> [symbolic = %Generic.type (constants.%Generic.type.68bb8d.1)]
// CHECK:STDOUT:   %Self.loc15_34.2: @Generic.%Generic.type (%Generic.type.68bb8d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.fc1c73.1)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%Scalar.loc15_19.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.fc1c8a.1)]
// CHECK:STDOUT:   %Generic.F: @Generic.%Generic.F.type (%Generic.F.type.fc1c8a.1) = struct_value () [symbolic = %Generic.F (constants.%Generic.F.d952d5.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%Scalar.loc15_19.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.414a90.1)]
// CHECK:STDOUT:   %assoc0.loc16_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.414a90.1) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.3122a2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_34.1: @Generic.%Generic.type (%Generic.type.68bb8d.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.fc1c73.1)]
// CHECK:STDOUT:     %Generic.F.decl: @Generic.%Generic.F.type (%Generic.F.type.fc1c8a.1) = fn_decl @Generic.F [symbolic = @Generic.%Generic.F (constants.%Generic.F.d952d5.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc16_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.414a90.1) = assoc_entity element0, %Generic.F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.3122a2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_34.1
// CHECK:STDOUT:     .F = %assoc0.loc16_9.1
// CHECK:STDOUT:     witness = (%Generic.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Other {
// CHECK:STDOUT:   %Self: %Other.type = symbolic_binding Self, 0 [symbolic = constants.%Self.5b9]
// CHECK:STDOUT:   %Other.G.decl: %Other.G.type = fn_decl @Other.G [concrete = constants.%Other.G] {} {}
// CHECK:STDOUT:   %assoc0: %Other.assoc_type = assoc_entity element0, %Other.G.decl [concrete = constants.%assoc0.25d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%Other.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT: impl @ImplsGeneric.as.Other.impl: %ImplsGeneric.ref as %Other.ref {
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G.decl: %ImplsGeneric.as.Other.impl.G.type = fn_decl @ImplsGeneric.as.Other.impl.G [concrete = constants.%ImplsGeneric.as.Other.impl.G] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %ImplsGeneric.as.Other.impl.G.decl
// CHECK:STDOUT:   witness = file.%Other.impl_witness
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
// CHECK:STDOUT: generic fn @Generic.F(@Generic.%Scalar.loc15_19.2: type, @Generic.%Self.loc15_34.1: @Generic.%Generic.type (%Generic.type.68bb8d.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Other.G(@Other.%Self: %Other.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Other.impl.G();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc33_22.2: type, %U.loc33_32.2: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.68bb8d.2)) {
// CHECK:STDOUT:   %T.loc33_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc33_45.1: type = facet_type <@Generic, @Generic(%T.loc33_22.1)> [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.68bb8d.2)]
// CHECK:STDOUT:   %U.loc33_32.1: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.68bb8d.2) = symbolic_binding U, 1 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc33_45.1 [symbolic = %pattern_type (constants.%pattern_type.e11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.loc33_45.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc33_22.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.414a90.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.414a90.2) = assoc_entity element0, @Generic.%Generic.F.decl [symbolic = %assoc0 (constants.%assoc0.3122a2.2)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc33_32.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc33_32.1, @Generic, @Generic(%T.loc33_22.1) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %Generic.F.type: type = fn_type @Generic.F, @Generic(%T.loc33_22.1) [symbolic = %Generic.F.type (constants.%Generic.F.type.fc1c8a.2)]
// CHECK:STDOUT:   %.loc34_4.3: type = fn_type_with_self_type %Generic.F.type, %U.loc33_32.1 [symbolic = %.loc34_4.3 (constants.%.9ee)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @CallGenericMethod.%.loc34_4.3 (%.9ee) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @Generic.F(%T.loc33_22.1, %U.loc33_32.1) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.68bb8d.2) = name_ref U, %U.loc33_32.2 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:     %.loc34_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.414a90.2) = specific_constant @Generic.%assoc0.loc16_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.3122a2.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.414a90.2) = name_ref F, %.loc34_4.1 [symbolic = %assoc0 (constants.%assoc0.3122a2.2)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc34_4.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @CallGenericMethod.%.loc34_4.3 (%.9ee) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @Generic.F(constants.%T, constants.%U) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc34_7: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.621 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc38: %Generic.type.621 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%Scalar, constants.%Self.fc1c73.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.621
// CHECK:STDOUT:   %Self.loc15_34.2 => constants.%Self.ae2
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.4cf
// CHECK:STDOUT:   %Generic.F => constants.%Generic.F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.815
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other.G(constants.%Self.5b9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other.G(constants.%Other.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.68bb8d.2
// CHECK:STDOUT:   %Self.loc15_34.2 => constants.%Self.fc1c73.2
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.fc1c8a.2
// CHECK:STDOUT:   %Generic.F => constants.%Generic.F.d952d5.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.414a90.2
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.3122a2.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc33_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc33_45.1 => constants.%Generic.type.68bb8d.2
// CHECK:STDOUT:   %U.loc33_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.F(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc33_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc33_45.1 => constants.%Generic.type.621
// CHECK:STDOUT:   %U.loc33_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.ca2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.713
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.815
// CHECK:STDOUT:   %U.binding.as_type => constants.%ImplsGeneric
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %Generic.F.type => constants.%Generic.F.type.4cf
// CHECK:STDOUT:   %.loc34_4.3 => constants.%.573
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
