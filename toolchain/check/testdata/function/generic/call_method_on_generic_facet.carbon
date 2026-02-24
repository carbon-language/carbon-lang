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
// CHECK:STDOUT:   %Generic.type.835: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.835 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.03dff7.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.15d2d5.1: %Generic.type.03dff7.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.74390a.1: type = fn_type @Generic.WithSelf.F, @Generic(%Scalar, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.baf95a.1: %Generic.WithSelf.F.type.74390a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.22afda.1: type = assoc_entity_type @Generic, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %assoc0.e0dc00.1: %Generic.assoc_type.22afda.1 = assoc_entity element0, @Generic.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.498: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %Generic.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Generic.impl.%Generic.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.1f5: %Generic.type.498 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.a23: type = fn_type @Generic.WithSelf.F, @Generic(%GenericParam, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.fe0: %Generic.WithSelf.F.type.a23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.6dc: type = assoc_entity_type @Generic, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %assoc0.717: %Generic.assoc_type.6dc = assoc_entity element0, @Generic.%Generic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F.type: type = fn_type @ImplsGeneric.as.Generic.impl.F [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Generic.impl.F: %ImplsGeneric.as.Generic.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.498 = facet_value %ImplsGeneric, (%Generic.impl_witness) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.c86: type = fn_type @Generic.WithSelf.F, @Generic(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %Generic.WithSelf.F.1fd: %Generic.WithSelf.F.type.c86 = struct_value () [concrete]
// CHECK:STDOUT:   %Other.type: type = facet_type <@Other> [concrete]
// CHECK:STDOUT:   %Self.8c4: %Other.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Other.WithSelf.G.type.acd: type = fn_type @Other.WithSelf.G, @Other(%Self.8c4) [symbolic]
// CHECK:STDOUT:   %Other.WithSelf.G.63d: %Other.WithSelf.G.type.acd = struct_value () [symbolic]
// CHECK:STDOUT:   %Other.assoc_type: type = assoc_entity_type @Other [concrete]
// CHECK:STDOUT:   %assoc0.205: %Other.assoc_type = assoc_entity element0, @Other.%Other.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness @ImplsGeneric.as.Other.impl.%Other.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G.type: type = fn_type @ImplsGeneric.as.Other.impl.G [concrete]
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G: %ImplsGeneric.as.Other.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Other.facet: %Other.type = facet_value %ImplsGeneric, (%Other.impl_witness) [concrete]
// CHECK:STDOUT:   %Other.WithSelf.G.type.1df: type = fn_type @Other.WithSelf.G, @Other(%Other.facet) [concrete]
// CHECK:STDOUT:   %Other.WithSelf.G.13d: %Other.WithSelf.G.type.1df = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.03dff7.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.03dff7.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c49: type = pattern_type %Generic.type.03dff7.2 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.15d2d5.2: %Generic.type.03dff7.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.74390a.2: type = fn_type @Generic.WithSelf.F, @Generic(%T, %Self.15d2d5.1) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.baf95a.2: %Generic.WithSelf.F.type.74390a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.22afda.2: type = assoc_entity_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.e0dc00.2: %Generic.assoc_type.22afda.2 = assoc_entity element0, @Generic.%Generic.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.03dff7.2 [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.type.56e: type = fn_type @Generic.WithSelf.F, @Generic(%T, %U) [symbolic]
// CHECK:STDOUT:   %Generic.WithSelf.F.a99: %Generic.WithSelf.F.type.56e = struct_value () [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %.ab3: type = fn_type_with_self_type %Generic.WithSelf.F.type.56e, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ab3 = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Generic.WithSelf.F(%T, %U) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cba: type = pattern_type %Generic.type.498 [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet) [concrete]
// CHECK:STDOUT:   %complete_type.57a: <witness> = complete_type_witness %Generic.type.498 [concrete]
// CHECK:STDOUT:   %.fcf: type = fn_type_with_self_type %Generic.WithSelf.F.type.c86, %Generic.facet [concrete]
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
// CHECK:STDOUT:   %Generic.decl: %Generic.type.835 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt: %pattern_type.98f = symbolic_binding_pattern Scalar, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Scalar.loc15_19.2: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Generic.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.498]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.decl: type = interface_decl @Other [concrete = constants.%Other.type] {} {}
// CHECK:STDOUT:   impl_decl @ImplsGeneric.as.Other.impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Other.ref: type = name_ref Other, file.%Other.decl [concrete = constants.%Other.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @CallGenericMethod.%pattern_type (%pattern_type.c49) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc33_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc33: type = splice_block %Generic.type.loc33_45.2 [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.03dff7.2)] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Generic.ref: %Generic.type.835 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc33_22.2 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc33_45.2: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc33_32.2: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc15_19.2: type) {
// CHECK:STDOUT:   %Scalar.loc15_19.1: type = symbolic_binding Scalar, 0 [symbolic = %Scalar.loc15_19.1 (constants.%Scalar)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type.loc15: type = facet_type <@Generic, @Generic(%Scalar.loc15_19.1)> [symbolic = %Generic.type.loc15 (constants.%Generic.type.03dff7.1)]
// CHECK:STDOUT:   %Self.loc15_34.2: @Generic.%Generic.type.loc15 (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.15d2d5.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_34.1: @Generic.%Generic.type.loc15 (%Generic.type.03dff7.1) = symbolic_binding Self, 1 [symbolic = %Self.loc15_34.2 (constants.%Self.15d2d5.1)]
// CHECK:STDOUT:     interface_with_self_decl @Generic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Generic.WithSelf.F.decl: @Generic.%Generic.WithSelf.F.type (%Generic.WithSelf.F.type.74390a.1) = fn_decl @Generic.WithSelf.F [symbolic = @Generic.%Generic.WithSelf.F (constants.%Generic.WithSelf.F.baf95a.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc16_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.22afda.1) = assoc_entity element0, %Generic.WithSelf.F.decl [symbolic = %assoc0.loc16_9.2 (constants.%assoc0.e0dc00.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_34.1
// CHECK:STDOUT:     .F = %assoc0.loc16_9.1
// CHECK:STDOUT:     witness = (%Generic.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Other {
// CHECK:STDOUT:   %Self.loc26: %Other.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8c4]
// CHECK:STDOUT:   interface_with_self_decl @Other [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Other.WithSelf.G.decl: @Other.%Other.WithSelf.G.type (%Other.WithSelf.G.type.acd) = fn_decl @Other.WithSelf.G [symbolic = @Other.%Other.WithSelf.G (constants.%Other.WithSelf.G.63d)] {} {}
// CHECK:STDOUT:   %assoc0: %Other.assoc_type = assoc_entity element0, %Other.WithSelf.G.decl [concrete = constants.%assoc0.205]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc26
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%Other.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT: impl @ImplsGeneric.as.Other.impl: %ImplsGeneric.ref as %Other.ref {
// CHECK:STDOUT:   %ImplsGeneric.as.Other.impl.G.decl: %ImplsGeneric.as.Other.impl.G.type = fn_decl @ImplsGeneric.as.Other.impl.G [concrete = constants.%ImplsGeneric.as.Other.impl.G] {} {}
// CHECK:STDOUT:   %Other.impl_witness_table = impl_witness_table (%ImplsGeneric.as.Other.impl.G.decl), @ImplsGeneric.as.Other.impl [concrete]
// CHECK:STDOUT:   %Other.impl_witness: <witness> = impl_witness %Other.impl_witness_table [concrete = constants.%Other.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %ImplsGeneric.as.Other.impl.G.decl
// CHECK:STDOUT:   witness = %Other.impl_witness
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
// CHECK:STDOUT: generic fn @Generic.WithSelf.F(@Generic.%Scalar.loc15_19.2: type, @Generic.%Self.loc15_34.1: @Generic.%Generic.type.loc15 (%Generic.type.03dff7.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Generic.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Other.WithSelf.G(@Other.%Self.loc26: %Other.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplsGeneric.as.Other.impl.G();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc33_22.2: type, %U.loc33_32.2: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.03dff7.2)) {
// CHECK:STDOUT:   %T.loc33_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc33_22.1 (constants.%T)]
// CHECK:STDOUT:   %Generic.type.loc33_45.1: type = facet_type <@Generic, @Generic(%T.loc33_22.1)> [symbolic = %Generic.type.loc33_45.1 (constants.%Generic.type.03dff7.2)]
// CHECK:STDOUT:   %U.loc33_32.1: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.03dff7.2) = symbolic_binding U, 1 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Generic.type.loc33_45.1 [symbolic = %pattern_type (constants.%pattern_type.c49)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Generic.type.loc33_45.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc33_32.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic, @Generic(%T.loc33_22.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.22afda.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = assoc_entity element0, @Generic.%Generic.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:   %Generic.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc33_32.1, @Generic, @Generic(%T.loc33_22.1) [symbolic = %Generic.lookup_impl_witness (constants.%Generic.lookup_impl_witness)]
// CHECK:STDOUT:   %Generic.WithSelf.F.type: type = fn_type @Generic.WithSelf.F, @Generic(%T.loc33_22.1, %U.loc33_32.1) [symbolic = %Generic.WithSelf.F.type (constants.%Generic.WithSelf.F.type.56e)]
// CHECK:STDOUT:   %.loc34_4.3: type = fn_type_with_self_type %Generic.WithSelf.F.type, %U.loc33_32.1 [symbolic = %.loc34_4.3 (constants.%.ab3)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @CallGenericMethod.%.loc34_4.3 (%.ab3) = impl_witness_access %Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @Generic.WithSelf.F(%T.loc33_22.1, %U.loc33_32.1) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @CallGenericMethod.%Generic.type.loc33_45.1 (%Generic.type.03dff7.2) = name_ref U, %U.loc33_32.2 [symbolic = %U.loc33_32.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc34_4.1: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc34_4.2: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = specific_constant @Generic.%assoc0.loc16_9.1, @Generic(constants.%T, constants.%U) [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.22afda.2) = name_ref F, %.loc34_4.2 [symbolic = %assoc0 (constants.%assoc0.e0dc00.2)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @CallGenericMethod.%.loc34_4.3 (%.ab3) = impl_witness_access constants.%Generic.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @Generic.WithSelf.F(constants.%T, constants.%U) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Generic.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.498 = facet_value constants.%ImplsGeneric, (constants.%Generic.impl_witness) [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %.loc38: %Generic.type.498 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar, constants.%Self.15d2d5.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%Scalar, constants.%Self.15d2d5.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type.loc15 => constants.%Generic.type.498
// CHECK:STDOUT:   %Self.loc15_34.2 => constants.%Self.1f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam, constants.%Self.15d2d5.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar.loc16 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc16 => constants.%Generic.type.498
// CHECK:STDOUT:   %Self.loc16 => constants.%Self.15d2d5.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.a23
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.fe0
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar.loc16 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc16 => constants.%Generic.type.498
// CHECK:STDOUT:   %Self.loc16 => constants.%Generic.facet
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.c86
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.1fd
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%GenericParam, constants.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other(constants.%Self.8c4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc27 => constants.%Self.8c4
// CHECK:STDOUT:   %Other.WithSelf.G.type => constants.%Other.WithSelf.G.type.acd
// CHECK:STDOUT:   %Other.WithSelf.G => constants.%Other.WithSelf.G.63d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other.WithSelf.G(constants.%Self.8c4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other(constants.%Other.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc27 => constants.%Other.facet
// CHECK:STDOUT:   %Other.WithSelf.G.type => constants.%Other.WithSelf.G.type.1df
// CHECK:STDOUT:   %Other.WithSelf.G => constants.%Other.WithSelf.G.13d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Other.WithSelf.G(constants.%Other.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc15_19.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type.loc15 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self.loc15_34.2 => constants.%Self.15d2d5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc33_22.1 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc33_45.1 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %U.loc33_32.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T, constants.%Self.15d2d5.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar.loc16 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc16 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self.loc16 => constants.%Self.15d2d5.1
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.74390a.2
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.baf95a.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.22afda.2
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.e0dc00.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T, constants.%U) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Scalar.loc16 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc16 => constants.%Generic.type.03dff7.2
// CHECK:STDOUT:   %Self.loc16 => constants.%U
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.56e
// CHECK:STDOUT:   %Generic.WithSelf.F => constants.%Generic.WithSelf.F.a99
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.22afda.2
// CHECK:STDOUT:   %assoc0.loc16_9.2 => constants.%assoc0.e0dc00.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf.F(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet) {
// CHECK:STDOUT:   %T.loc33_22.1 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc33_45.1 => constants.%Generic.type.498
// CHECK:STDOUT:   %U.loc33_32.1 => constants.%Generic.facet
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.57a
// CHECK:STDOUT:   %U.binding.as_type => constants.%ImplsGeneric
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.6dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.717
// CHECK:STDOUT:   %Generic.lookup_impl_witness => constants.%Generic.impl_witness
// CHECK:STDOUT:   %Generic.WithSelf.F.type => constants.%Generic.WithSelf.F.type.c86
// CHECK:STDOUT:   %.loc34_4.3 => constants.%.fcf
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%ImplsGeneric.as.Generic.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
