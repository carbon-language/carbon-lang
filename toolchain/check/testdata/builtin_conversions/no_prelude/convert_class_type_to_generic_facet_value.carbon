// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_generic_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_type_to_generic_facet_value.carbon

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

// CHECK:STDOUT: --- convert_class_type_to_generic_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Scalar.patt: type = symbolic_binding_pattern Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [template]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self.dee: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f43: type = fn_type @F.1, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %F.8a2: %F.type.f43 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.de9: type = assoc_entity_type %Generic.type.91ccba.1 [symbolic]
// CHECK:STDOUT:   %assoc0.29c: %Generic.assoc_type.de9 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [template]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [template]
// CHECK:STDOUT:   %F.type.4cf: type = fn_type @F.1, @Generic(%GenericParam) [template]
// CHECK:STDOUT:   %F.118: %F.type.4cf = struct_value () [template]
// CHECK:STDOUT:   %Generic.assoc_type.9f1: type = assoc_entity_type %Generic.type.769 [template]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.9f1 = assoc_entity element0, @Generic.%F.decl [template]
// CHECK:STDOUT:   %impl_witness.b42: <witness> = impl_witness (@impl.1.%F.decl) [template]
// CHECK:STDOUT:   %F.type.17b: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.a56: %F.type.17b = struct_value () [template]
// CHECK:STDOUT:   %Generic.facet.1e5: %Generic.type.91ccba.1 = facet_value %ImplsGeneric, %impl_witness.b42 [symbolic]
// CHECK:STDOUT:   %Other.type: type = facet_type <@Other> [template]
// CHECK:STDOUT:   %Self.807: %Other.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.0c6: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.17f: %G.type.0c6 = struct_value () [template]
// CHECK:STDOUT:   %Other.assoc_type: type = assoc_entity_type %Other.type [template]
// CHECK:STDOUT:   %assoc0.5ce: %Other.assoc_type = assoc_entity element0, @Other.%G.decl [template]
// CHECK:STDOUT:   %impl_witness.51c: <witness> = impl_witness (@impl.2.%G.decl) [template]
// CHECK:STDOUT:   %G.type.58d: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.b67: %G.type.58d = struct_value () [template]
// CHECK:STDOUT:   %Other.facet: %Other.type = facet_value %ImplsGeneric, %impl_witness.51c [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [template]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [template]
// CHECK:STDOUT:   %G.type.9f9: type = fn_type @G.3 [template]
// CHECK:STDOUT:   %G.57b: %G.type.9f9 = struct_value () [template]
// CHECK:STDOUT:   %Generic.facet.b0a: %Generic.type.769 = facet_value %ImplsGeneric, %impl_witness.b42 [template]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.d8c: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet.b0a) [template]
// CHECK:STDOUT:   %PassThroughToGenericMethod.type: type = fn_type @PassThroughToGenericMethod [template]
// CHECK:STDOUT:   %PassThroughToGenericMethod: %PassThroughToGenericMethod.type = struct_value () [template]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.a24: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%T, %U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %PassThroughToGenericMethod.specific_fn: <specific function> = specific_function %PassThroughToGenericMethod, @PassThroughToGenericMethod(%GenericParam, %Generic.facet.b0a) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .GenericParam = %GenericParam.decl
// CHECK:STDOUT:     .ImplsGeneric = %ImplsGeneric.decl
// CHECK:STDOUT:     .Other = %Other.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .PassThroughToGenericMethod = %PassThroughToGenericMethod.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [template = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt.loc11_19.1: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc11_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:     %Scalar.param_patt: type = value_param_pattern %Scalar.patt.loc11_19.1, runtime_param<none> [symbolic = %Scalar.patt.loc11_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Scalar.loc11_19.1: type = bind_symbolic_name Scalar, 0, %Scalar.param [symbolic = %Scalar.loc11_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [template = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [template = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [template = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [template = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [template = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [template = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.1.%F.decl) [template = constants.%impl_witness.b42]
// CHECK:STDOUT:   %Other.decl: type = interface_decl @Other [template = constants.%Other.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [template = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Other.ref: type = name_ref Other, file.%Other.decl [template = constants.%Other.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (@impl.2.%G.decl) [template = constants.%impl_witness.51c]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [template = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt.loc29_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc29_22.1, runtime_param<none> [symbolic = %T.patt.loc29_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc29_32.1: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc29_32.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2) = value_param_pattern %U.patt.loc29_32.1, runtime_param<none> [symbolic = %U.patt.loc29_32.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc29_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc29_22.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc29: type = splice_block %Generic.type.loc29_45.1 [symbolic = %Generic.type.loc29_45.2 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [template = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc29_22.1 [symbolic = %T.loc29_22.2 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc29_45.1: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc29_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc29_32.1: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc29_32.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.9f9 = fn_decl @G.3 [template = constants.%G.57b] {} {}
// CHECK:STDOUT:   %PassThroughToGenericMethod.decl: %PassThroughToGenericMethod.type = fn_decl @PassThroughToGenericMethod [template = constants.%PassThroughToGenericMethod] {
// CHECK:STDOUT:     %T.patt.loc35_31.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc35_31.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc35_31.1, runtime_param<none> [symbolic = %T.patt.loc35_31.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc35_41.1: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc35_41.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2) = value_param_pattern %U.patt.loc35_41.1, runtime_param<none> [symbolic = %U.patt.loc35_41.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc35_31.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc35_31.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc35: type = splice_block %Generic.type.loc35_54.1 [symbolic = %Generic.type.loc35_54.2 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [template = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc35: type = name_ref T, %T.loc35_31.1 [symbolic = %T.loc35_31.2 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc35_54.1: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc35_54.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc35_41.1: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc35_41.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc11_19.1: type) {
// CHECK:STDOUT:   %Scalar.loc11_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc11_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   %Scalar.patt.loc11_19.2: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc11_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc11_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%Scalar.loc11_19.2) [symbolic = %F.type (constants.%F.type.f43)]
// CHECK:STDOUT:   %F: @Generic.%F.type (%F.type.f43) = struct_value () [symbolic = %F (constants.%F.8a2)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic.%Generic.type (%Generic.type.91ccba.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.de9)]
// CHECK:STDOUT:   %assoc0.loc12_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.de9) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_9.2 (constants.%assoc0.29c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.dee)]
// CHECK:STDOUT:     %F.decl: @Generic.%F.type (%F.type.f43) = fn_decl @F.1 [symbolic = @Generic.%F (constants.%F.8a2)] {} {}
// CHECK:STDOUT:     %assoc0.loc12_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.de9) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_9.2 (constants.%assoc0.29c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc12_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Other {
// CHECK:STDOUT:   %Self: %Other.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.807]
// CHECK:STDOUT:   %G.decl: %G.type.0c6 = fn_decl @G.1 [template = constants.%G.17f] {} {}
// CHECK:STDOUT:   %assoc0: %Other.assoc_type = assoc_entity element0, %G.decl [template = constants.%assoc0.5ce]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %F.decl: %F.type.17b = fn_decl @F.2 [template = constants.%F.a56] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %ImplsGeneric.ref as %Other.ref {
// CHECK:STDOUT:   %G.decl: %G.type.58d = fn_decl @G.2 [template = constants.%G.b67] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @GenericParam {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Generic.%Scalar.loc11_19.1: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
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
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc29_22.1: type, %U.loc29_32.1: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc29_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc29_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc29_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Generic.type.loc29_45.2: type = facet_type <@Generic, @Generic(%T.loc29_22.2)> [symbolic = %Generic.type.loc29_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc29_32.2: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic = %U.loc29_32.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc29_32.2: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc29_32.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %U.param_patt: @CallGenericMethod.%Generic.type.loc29_45.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [template = constants.%CallGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [template = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [template = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, constants.%impl_witness.b42 [template = constants.%Generic.facet.b0a]
// CHECK:STDOUT:   %.loc32: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [template = constants.%Generic.facet.b0a]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, %.loc32) [template = constants.%CallGenericMethod.specific_fn.d8c]
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PassThroughToGenericMethod(%T.loc35_31.1: type, %U.loc35_41.1: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc35_31.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc35_31.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc35_31.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc35_31.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Generic.type.loc35_54.2: type = facet_type <@Generic, @Generic(%T.loc35_31.2)> [symbolic = %Generic.type.loc35_54.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc35_41.2: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic = %U.loc35_41.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc35_41.2: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc35_41.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc36_3.2: <specific function> = specific_function constants.%CallGenericMethod, @CallGenericMethod(%T.loc35_31.2, %U.loc35_41.2) [symbolic = %CallGenericMethod.specific_fn.loc36_3.2 (constants.%CallGenericMethod.specific_fn.a24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %U.param_patt: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [template = constants.%CallGenericMethod]
// CHECK:STDOUT:     %T.ref.loc36: type = name_ref T, %T.loc35_31.1 [symbolic = %T.loc35_31.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref: @PassThroughToGenericMethod.%Generic.type.loc35_54.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc35_41.1 [symbolic = %U.loc35_41.2 (constants.%U)]
// CHECK:STDOUT:     %CallGenericMethod.specific_fn.loc36_3.1: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%T, constants.%U) [symbolic = %CallGenericMethod.specific_fn.loc36_3.2 (constants.%CallGenericMethod.specific_fn.a24)]
// CHECK:STDOUT:     %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn.loc36_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PassThroughToGenericMethod.ref: %PassThroughToGenericMethod.type = name_ref PassThroughToGenericMethod, file.%PassThroughToGenericMethod.decl [template = constants.%PassThroughToGenericMethod]
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [template = constants.%GenericParam]
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [template = constants.%ImplsGeneric]
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, constants.%impl_witness.b42 [template = constants.%Generic.facet.b0a]
// CHECK:STDOUT:   %.loc40: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [template = constants.%Generic.facet.b0a]
// CHECK:STDOUT:   %PassThroughToGenericMethod.specific_fn: <specific function> = specific_function %PassThroughToGenericMethod.ref, @PassThroughToGenericMethod(constants.%GenericParam, %.loc40) [template = constants.%PassThroughToGenericMethod.specific_fn]
// CHECK:STDOUT:   %PassThroughToGenericMethod.call: init %empty_tuple.type = call %PassThroughToGenericMethod.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc11_19.2 => constants.%Scalar
// CHECK:STDOUT:   %Scalar.patt.loc11_19.2 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Scalar, constants.%Self.dee) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%Scalar.loc11_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc11_19.2 => constants.%GenericParam
// CHECK:STDOUT:   %Scalar.patt.loc11_19.2 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self.dee
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %F => constants.%F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.9f1
// CHECK:STDOUT:   %assoc0.loc12_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%GenericParam, constants.%Generic.facet.1e5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.807) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Other.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc11_19.2 => constants.%T
// CHECK:STDOUT:   %Scalar.patt.loc11_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc29_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc29_22.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc29_45.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc29_32.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc29_32.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@CallGenericMethod.%T.loc29_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, @G.3.%.loc32) {
// CHECK:STDOUT:   %T.loc29_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc29_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc29_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc29_32.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:   %U.patt.loc29_32.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.b0a) {
// CHECK:STDOUT:   %T.loc29_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc29_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc29_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc29_32.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:   %U.patt.loc29_32.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc35_31.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc35_31.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc35_54.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc35_41.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc35_41.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@PassThroughToGenericMethod.%T.loc35_31.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(@PassThroughToGenericMethod.%T.loc35_31.2, @PassThroughToGenericMethod.%U.loc35_41.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%GenericParam, @H.%.loc40) {
// CHECK:STDOUT:   %T.loc35_31.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc35_31.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc35_54.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc35_41.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:   %U.patt.loc35_41.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CallGenericMethod.specific_fn.loc36_3.2 => constants.%CallGenericMethod.specific_fn.d8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PassThroughToGenericMethod(constants.%GenericParam, constants.%Generic.facet.b0a) {
// CHECK:STDOUT:   %T.loc35_31.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc35_31.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc35_54.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc35_41.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT:   %U.patt.loc35_41.2 => constants.%Generic.facet.b0a
// CHECK:STDOUT: }
// CHECK:STDOUT:
