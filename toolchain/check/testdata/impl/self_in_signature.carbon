// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/self_in_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/self_in_signature.carbon

interface UseSelf {
  fn F[self: Self](x: Self) -> Self;
}

class C {}

class D {}

impl C as UseSelf {
  fn F[self: C](x: C) -> C { return {}; }
}

impl D as UseSelf {
  fn F[self: Self](x: Self) -> Self { return {}; }
}

interface SelfNested {
  fn F(x: (Self*, {.x: Self, .y: ()}));
}

impl C as SelfNested {
  fn F(x: (C*, {.x: C, .y: ()}));
}

impl D as SelfNested {
  fn F(x: (Self*, {.x: Self, .y: ()}));
}

// CHECK:STDOUT: --- self_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UseSelf.type: type = facet_type <@UseSelf> [concrete]
// CHECK:STDOUT:   %Self.f7b: %UseSelf.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.e3d: type = symbolic_binding_type Self, 0, %Self.f7b [symbolic]
// CHECK:STDOUT:   %pattern_type.431: type = pattern_type %Self.binding.as_type.e3d [symbolic]
// CHECK:STDOUT:   %.7fd: Core.Form = init_form %Self.binding.as_type.e3d, call_param2 [symbolic]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.ad8: type = fn_type @UseSelf.WithSelf.F, @UseSelf(%Self.f7b) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.de0: %UseSelf.WithSelf.F.type.ad8 = struct_value () [symbolic]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type @UseSelf [concrete]
// CHECK:STDOUT:   %assoc0.44b: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.%UseSelf.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.3c4: <witness> = impl_witness @C.as.UseSelf.impl.%UseSelf.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.92f: Core.Form = init_form %C, call_param2 [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.type: type = fn_type @C.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F: %C.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.d4d: %UseSelf.type = facet_value %C, (%UseSelf.impl_witness.3c4) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.a7f: type = fn_type @UseSelf.WithSelf.F, @UseSelf(%UseSelf.facet.d4d) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.67c: %UseSelf.WithSelf.F.type.a7f = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.22d: <witness> = impl_witness @D.as.UseSelf.impl.%UseSelf.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.513: Core.Form = init_form %D, call_param2 [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.type: type = fn_type @D.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F: %D.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.6f8: %UseSelf.type = facet_value %D, (%UseSelf.impl_witness.22d) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.37f: type = fn_type @UseSelf.WithSelf.F, @UseSelf(%UseSelf.facet.6f8) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.633: %UseSelf.WithSelf.F.type.37f = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.aed: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.07b: type = symbolic_binding_type Self, 0, %Self.aed [symbolic]
// CHECK:STDOUT:   %ptr.d99: type = ptr_type %Self.binding.as_type.07b [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.b54: type = struct_type {.x: %Self.binding.as_type.07b, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.ceb: %tuple.type.24b = tuple_value (%ptr.d99, %struct_type.x.y.b54) [symbolic]
// CHECK:STDOUT:   %tuple.type.cbf: type = tuple_type (%ptr.d99, %struct_type.x.y.b54) [symbolic]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %tuple.type.cbf [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.430: type = fn_type @SelfNested.WithSelf.F, @SelfNested(%Self.aed) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.257: %SelfNested.WithSelf.F.type.430 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.da4: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.a03: <witness> = impl_witness @C.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %struct_type.x.y.710: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.509: %tuple.type.24b = tuple_value (%ptr.31e, %struct_type.x.y.710) [concrete]
// CHECK:STDOUT:   %tuple.type.2f1: type = tuple_type (%ptr.31e, %struct_type.x.y.710) [concrete]
// CHECK:STDOUT:   %pattern_type.137: type = pattern_type %tuple.type.2f1 [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.type: type = fn_type @C.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F: %C.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.f22: %SelfNested.type = facet_value %C, (%SelfNested.impl_witness.a03) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.f28: type = fn_type @SelfNested.WithSelf.F, @SelfNested(%SelfNested.facet.f22) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.45f: %SelfNested.WithSelf.F.type.f28 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.130: <witness> = impl_witness @D.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.805: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %struct_type.x.y.3e1: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.21a: %tuple.type.24b = tuple_value (%ptr.805, %struct_type.x.y.3e1) [concrete]
// CHECK:STDOUT:   %tuple.type.e0d: type = tuple_type (%ptr.805, %struct_type.x.y.3e1) [concrete]
// CHECK:STDOUT:   %pattern_type.d05: type = pattern_type %tuple.type.e0d [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.type: type = fn_type @D.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F: %D.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.372: %SelfNested.type = facet_value %D, (%SelfNested.impl_witness.130) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.88c: type = fn_type @SelfNested.WithSelf.F, @SelfNested(%SelfNested.facet.372) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.02f: %SelfNested.WithSelf.F.type.88c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [concrete = constants.%UseSelf.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @C.as.UseSelf.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.UseSelf.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [concrete = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self.loc15: %UseSelf.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f7b]
// CHECK:STDOUT:   interface_with_self_decl @UseSelf [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %UseSelf.WithSelf.F.decl: @UseSelf.%UseSelf.WithSelf.F.type (%UseSelf.WithSelf.F.type.ad8) = fn_decl @UseSelf.WithSelf.F [symbolic = @UseSelf.%UseSelf.WithSelf.F (constants.%UseSelf.WithSelf.F.de0)] {
// CHECK:STDOUT:     %self.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.431) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_32: %UseSelf.type = name_ref Self, @UseSelf.%Self.loc15 [symbolic = %Self (constants.%Self.f7b)]
// CHECK:STDOUT:     %Self.as_type.loc16_32: type = facet_access_type %Self.ref.loc16_32 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:     %.loc16_32.2: type = converted %Self.ref.loc16_32, %Self.as_type.loc16_32 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:     %.loc16_32.3: Core.Form = init_form %.loc16_32.2, call_param2 [symbolic = %.loc16_32.1 (constants.%.7fd)]
// CHECK:STDOUT:     %self.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %UseSelf.type = name_ref Self, @UseSelf.%Self.loc15 [symbolic = %Self (constants.%Self.f7b)]
// CHECK:STDOUT:       %Self.as_type.loc16_14: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = value_param call_param1
// CHECK:STDOUT:     %.loc16_23.1: type = splice_block %.loc16_23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)] {
// CHECK:STDOUT:       %Self.ref.loc16_23: %UseSelf.type = name_ref Self, @UseSelf.%Self.loc15 [symbolic = %Self (constants.%Self.f7b)]
// CHECK:STDOUT:       %Self.as_type.loc16_23: type = facet_access_type %Self.ref.loc16_23 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:       %.loc16_23.2: type = converted %Self.ref.loc16_23, %Self.as_type.loc16_23 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = out_param call_param2
// CHECK:STDOUT:     %return: ref @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %UseSelf.WithSelf.F.decl [concrete = constants.%assoc0.44b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc15
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%UseSelf.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self.loc31: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.aed]
// CHECK:STDOUT:   interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.430) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.257)] {
// CHECK:STDOUT:     %x.patt: @SelfNested.WithSelf.F.%pattern_type (%pattern_type.fb7) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @SelfNested.WithSelf.F.%pattern_type (%pattern_type.fb7) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.cbf) = value_param call_param0
// CHECK:STDOUT:     %.loc32_37.1: type = splice_block %.loc32_37.3 [symbolic = %tuple.type (constants.%tuple.type.cbf)] {
// CHECK:STDOUT:       %Self.ref.loc32_12: %SelfNested.type = name_ref Self, @SelfNested.%Self.loc31 [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:       %Self.as_type.loc32_16: type = facet_access_type %Self.ref.loc32_12 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:       %.loc32_16: type = converted %Self.ref.loc32_12, %Self.as_type.loc32_16 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:       %ptr.loc32_16.2: type = ptr_type %.loc32_16 [symbolic = %ptr.loc32_16.1 (constants.%ptr.d99)]
// CHECK:STDOUT:       %Self.ref.loc32_24: %SelfNested.type = name_ref Self, @SelfNested.%Self.loc31 [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:       %Self.as_type.loc32_24: type = facet_access_type %Self.ref.loc32_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:       %.loc32_24: type = converted %Self.ref.loc32_24, %Self.as_type.loc32_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:       %.loc32_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc32_35.2: type = converted %.loc32_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc32_36.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.07b), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.b54)]
// CHECK:STDOUT:       %.loc32_37.2: %tuple.type.24b = tuple_literal (%ptr.loc32_16.2, %struct_type.x.y.loc32_36.2) [symbolic = %tuple (constants.%tuple.ceb)]
// CHECK:STDOUT:       %.loc32_37.3: type = converted %.loc32_37.2, constants.%tuple.type.cbf [symbolic = %tuple.type (constants.%tuple.type.cbf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.cbf) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.da4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc31
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%SelfNested.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.UseSelf.impl: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.decl: %C.as.UseSelf.impl.F.type = fn_decl @C.as.UseSelf.impl.F [concrete = constants.%C.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7c7 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7c7 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc24_26: Core.Form = init_form %C.ref.loc24_26, call_param2 [concrete = constants.%.92f]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc24_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc24_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table = impl_witness_table (%C.as.UseSelf.impl.F.decl), @C.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness: <witness> = impl_witness %UseSelf.impl_witness_table [concrete = constants.%UseSelf.impl_witness.3c4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   witness = %UseSelf.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.UseSelf.impl: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.decl: %D.as.UseSelf.impl.F.type = fn_decl @D.as.UseSelf.impl.F [concrete = constants.%D.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.9c8 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9c8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.9c8 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.9c8 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9c8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9c8 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %.loc28_32: Core.Form = init_form %Self.ref.loc28_32, call_param2 [concrete = constants.%.513]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_14: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_23: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %x: %D = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table = impl_witness_table (%D.as.UseSelf.impl.F.decl), @D.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness: <witness> = impl_witness %UseSelf.impl_witness_table [concrete = constants.%UseSelf.impl_witness.22d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   witness = %UseSelf.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.SelfNested.impl: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.decl: %C.as.SelfNested.impl.F.type = fn_decl @C.as.SelfNested.impl.F [concrete = constants.%C.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.137 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.137 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.2f1 = value_param call_param0
// CHECK:STDOUT:     %.loc36_31.1: type = splice_block %.loc36_31.3 [concrete = constants.%tuple.type.2f1] {
// CHECK:STDOUT:       %C.ref.loc36_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc36_12 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:       %C.ref.loc36_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc36_29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc36_29.2: type = converted %.loc36_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.710]
// CHECK:STDOUT:       %.loc36_31.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.509]
// CHECK:STDOUT:       %.loc36_31.3: type = converted %.loc36_31.2, constants.%tuple.type.2f1 [concrete = constants.%tuple.type.2f1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.2f1 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%C.as.SelfNested.impl.F.decl), @C.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.a03]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.SelfNested.impl: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.decl: %D.as.SelfNested.impl.F.type = fn_decl @D.as.SelfNested.impl.F [concrete = constants.%D.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.d05 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d05 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.e0d = value_param call_param0
// CHECK:STDOUT:     %.loc40_37.1: type = splice_block %.loc40_37.3 [concrete = constants.%tuple.type.e0d] {
// CHECK:STDOUT:       %Self.ref.loc40_12: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref.loc40_12 [concrete = constants.%ptr.805]
// CHECK:STDOUT:       %Self.ref.loc40_24: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %.loc40_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc40_35.2: type = converted %.loc40_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.3e1]
// CHECK:STDOUT:       %.loc40_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.21a]
// CHECK:STDOUT:       %.loc40_37.3: type = converted %.loc40_37.2, constants.%tuple.type.e0d [concrete = constants.%tuple.type.e0d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.e0d = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%D.as.SelfNested.impl.F.decl), @D.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.130]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseSelf.WithSelf.F(@UseSelf.%Self.loc15: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f7b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e3d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.431)]
// CHECK:STDOUT:   %.loc16_32.1: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic = %.loc16_32.1 (constants.%.7fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d), %x.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d)) -> out %return.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.e3d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.UseSelf.impl.F(%self.param: %C, %x.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_38.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_39: init %C = converted %.loc24_38.1, %.loc24_38.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc24_39 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.UseSelf.impl.F(%self.param: %D, %x.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_47.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc28_47.2: init %D to %return.param = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc28_48: init %D = converted %.loc28_47.1, %.loc28_47.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc28_48 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self.loc31: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.07b)]
// CHECK:STDOUT:   %ptr.loc32_16.1: type = ptr_type %Self.binding.as_type [symbolic = %ptr.loc32_16.1 (constants.%ptr.d99)]
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.07b), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.b54)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1) [symbolic = %tuple (constants.%tuple.ceb)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1) [symbolic = %tuple.type (constants.%tuple.type.cbf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.cbf));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.SelfNested.impl.F(%x.param: %tuple.type.2f1);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.SelfNested.impl.F(%x.param: %tuple.type.e0d);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf(constants.%Self.f7b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc16 => constants.%Self.f7b
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.ad8
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.de0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%Self.f7b) {
// CHECK:STDOUT:   %Self => constants.%Self.f7b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e3d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.431
// CHECK:STDOUT:   %.loc16_32.1 => constants.%.7fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf(constants.%UseSelf.facet.d4d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc16 => constants.%UseSelf.facet.d4d
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.a7f
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.67c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%UseSelf.facet.d4d) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.d4d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT:   %.loc16_32.1 => constants.%.92f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf(constants.%UseSelf.facet.6f8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc16 => constants.%UseSelf.facet.6f8
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.37f
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.633
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%UseSelf.facet.6f8) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.6f8
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c8
// CHECK:STDOUT:   %.loc16_32.1 => constants.%.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested(constants.%Self.aed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc32 => constants.%Self.aed
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.430
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.aed) {
// CHECK:STDOUT:   %Self => constants.%Self.aed
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.07b
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.d99
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.b54
// CHECK:STDOUT:   %tuple => constants.%tuple.ceb
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cbf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested(constants.%SelfNested.facet.f22) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc32 => constants.%SelfNested.facet.f22
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.f28
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.45f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.f22) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.f22
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.31e
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.710
// CHECK:STDOUT:   %tuple => constants.%tuple.509
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2f1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested(constants.%SelfNested.facet.372) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc32 => constants.%SelfNested.facet.372
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.88c
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.02f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.372) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.372
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.805
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.3e1
// CHECK:STDOUT:   %tuple => constants.%tuple.21a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e0d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d05
// CHECK:STDOUT: }
// CHECK:STDOUT:
