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
  fn F(self, x: Self) -> Self;
}

class C {}

class D {}

impl C as UseSelf {
  fn F(unused self: C, unused x: C) -> C { return {}; }
}

impl D as UseSelf {
  fn F(unused self, unused x: Self) -> Self { return {}; }
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
// CHECK:STDOUT:   %Self.bde: %UseSelf.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.968: type = facet_access_type %Self.bde [symbolic]
// CHECK:STDOUT:   %pattern_type.04e: type = pattern_type %Self.as_type.968 [symbolic]
// CHECK:STDOUT:   %.326: Core.Form = init_form %Self.as_type.968 [symbolic]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.74d: type = fn_type @UseSelf.WithSelf.F, @UseSelf.WithSelf(%Self.bde) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.8c4: %UseSelf.WithSelf.F.type.74d = struct_value () [symbolic]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type @UseSelf [concrete]
// CHECK:STDOUT:   %assoc0.f2e: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.WithSelf.%UseSelf.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.046: <witness> = impl_witness @C.as.UseSelf.impl.%UseSelf.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.type: type = fn_type @C.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F: %C.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.14b: %UseSelf.type = facet_value %C, (%UseSelf.impl_witness.046) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.103: type = fn_type @UseSelf.WithSelf.F, @UseSelf.WithSelf(%UseSelf.facet.14b) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.e1d: %UseSelf.WithSelf.F.type.103 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.047: <witness> = impl_witness @D.as.UseSelf.impl.%UseSelf.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.type: type = fn_type @D.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F: %D.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.64f: %UseSelf.type = facet_value %D, (%UseSelf.impl_witness.047) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type.bbe: type = fn_type @UseSelf.WithSelf.F, @UseSelf.WithSelf(%UseSelf.facet.64f) [concrete]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.8b5: %UseSelf.WithSelf.F.type.bbe = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.e81: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.8ab: type = facet_access_type %Self.e81 [symbolic]
// CHECK:STDOUT:   %ptr.a16: type = ptr_type %Self.as_type.8ab [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.749: type = struct_type {.x: %Self.as_type.8ab, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.876: %tuple.type.24b = tuple_value (%ptr.a16, %struct_type.x.y.749) [symbolic]
// CHECK:STDOUT:   %tuple.type.d83: type = tuple_type (%ptr.a16, %struct_type.x.y.749) [symbolic]
// CHECK:STDOUT:   %pattern_type.6eb: type = pattern_type %tuple.type.d83 [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.702: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.e81) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.eec: %SelfNested.WithSelf.F.type.702 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.219: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.d40: <witness> = impl_witness @C.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ee7: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.562: %tuple.type.24b = tuple_value (%ptr.6b6, %struct_type.x.y.ee7) [concrete]
// CHECK:STDOUT:   %tuple.type.746: type = tuple_type (%ptr.6b6, %struct_type.x.y.ee7) [concrete]
// CHECK:STDOUT:   %pattern_type.79c: type = pattern_type %tuple.type.746 [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.type: type = fn_type @C.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F: %C.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.749: %SelfNested.type = facet_value %C, (%SelfNested.impl_witness.d40) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.369: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.749) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.0c3: %SelfNested.WithSelf.F.type.369 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.ad8: <witness> = impl_witness @D.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d57: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %struct_type.x.y.e52: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.c73: %tuple.type.24b = tuple_value (%ptr.d57, %struct_type.x.y.e52) [concrete]
// CHECK:STDOUT:   %tuple.type.aaa: type = tuple_type (%ptr.d57, %struct_type.x.y.e52) [concrete]
// CHECK:STDOUT:   %pattern_type.338: type = pattern_type %tuple.type.aaa [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.type: type = fn_type @D.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F: %D.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.d22: %SelfNested.type = facet_value %D, (%SelfNested.impl_witness.ad8) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.f5f: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.d22) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.ece: %SelfNested.WithSelf.F.type.f5f = struct_value () [concrete]
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
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bde]
// CHECK:STDOUT:   %UseSelf.WithSelf.decl = interface_with_self_decl @UseSelf [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %UseSelf.WithSelf.F.decl: @UseSelf.WithSelf.%UseSelf.WithSelf.F.type (%UseSelf.WithSelf.F.type.74d) = fn_decl @UseSelf.WithSelf.F [symbolic = @UseSelf.WithSelf.%UseSelf.WithSelf.F (constants.%UseSelf.WithSelf.F.8c4)] {
// CHECK:STDOUT:     %self.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %x.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type.04e) = return_slot_pattern %return.param_patt, %.loc16_26.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_26: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.bde)]
// CHECK:STDOUT:     %Self.as_type.loc16_26: type = facet_access_type %Self.ref.loc16_26 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %Self.ref.loc16_26, %Self.as_type.loc16_26 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:     %.loc16_26.3: Core.Form = init_form %.loc16_26.2 [symbolic = %.loc16_26.1 (constants.%.326)]
// CHECK:STDOUT:     %self.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = value_param call_param0
// CHECK:STDOUT:     %.loc16_8.1: type = splice_block %.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)] {
// CHECK:STDOUT:       %Self.ref.loc16_8: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.bde)]
// CHECK:STDOUT:       %Self.as_type.loc16_8.2: type = facet_access_type %Self.ref.loc16_8 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:       %.loc16_8.2: type = converted %Self.ref.loc16_8, %Self.as_type.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = value_param call_param1
// CHECK:STDOUT:     %.loc16_17.1: type = splice_block %.loc16_17.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)] {
// CHECK:STDOUT:       %Self.ref.loc16_17: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.bde)]
// CHECK:STDOUT:       %Self.as_type.loc16_17: type = facet_access_type %Self.ref.loc16_17 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:       %.loc16_17.2: type = converted %Self.ref.loc16_17, %Self.as_type.loc16_17 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = out_param call_param2
// CHECK:STDOUT:     %return: ref @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %UseSelf.WithSelf.F.decl [concrete = constants.%assoc0.f2e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @UseSelf.WithSelf.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   witness = (@UseSelf.WithSelf.%UseSelf.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e81]
// CHECK:STDOUT:   %SelfNested.WithSelf.decl = interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.702) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.eec)] {
// CHECK:STDOUT:     %x.param_patt: @SelfNested.WithSelf.F.%pattern_type (%pattern_type.6eb) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @SelfNested.WithSelf.F.%pattern_type (%pattern_type.6eb) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.d83) = value_param call_param0
// CHECK:STDOUT:     %.loc32_37.1: type = splice_block %.loc32_37.3 [symbolic = %tuple.type (constants.%tuple.type.d83)] {
// CHECK:STDOUT:       %Self.ref.loc32_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc32_16.2: type = facet_access_type %Self.ref.loc32_12 [symbolic = %Self.as_type.loc32_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc32_16: type = converted %Self.ref.loc32_12, %Self.as_type.loc32_16.2 [symbolic = %Self.as_type.loc32_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %ptr.loc32_16.2: type = ptr_type %.loc32_16 [symbolic = %ptr.loc32_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:       %Self.ref.loc32_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc32_24: type = facet_access_type %Self.ref.loc32_24 [symbolic = %Self.as_type.loc32_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc32_24: type = converted %Self.ref.loc32_24, %Self.as_type.loc32_24 [symbolic = %Self.as_type.loc32_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc32_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc32_35.2: type = converted %.loc32_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc32_36.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc32_16.1 (%Self.as_type.8ab), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.749)]
// CHECK:STDOUT:       %.loc32_37.2: %tuple.type.24b = tuple_literal (%ptr.loc32_16.2, %struct_type.x.y.loc32_36.2) [symbolic = %tuple (constants.%tuple.876)]
// CHECK:STDOUT:       %.loc32_37.3: type = converted %.loc32_37.2, constants.%tuple.type.d83 [symbolic = %tuple.type (constants.%tuple.type.d83)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.d83) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.219]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @SelfNested.WithSelf.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   witness = (@SelfNested.WithSelf.%SelfNested.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.UseSelf.impl: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.decl: %C.as.UseSelf.impl.F.type = fn_decl @C.as.UseSelf.impl.F [concrete = constants.%C.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.98b = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc24_40 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24_40: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc24_40: Core.Form = init_form %C.ref.loc24_40 [concrete = constants.%.a44]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc24_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc24_34: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table = impl_witness_table (%C.as.UseSelf.impl.F.decl), @C.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness: <witness> = impl_witness %UseSelf.impl_witness_table [concrete = constants.%UseSelf.impl_witness.046]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   extend %UseSelf.ref
// CHECK:STDOUT:   witness = %UseSelf.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.UseSelf.impl: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.decl: %D.as.UseSelf.impl.F.type = fn_decl @D.as.UseSelf.impl.F [concrete = constants.%D.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.d8d = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %Self.ref.loc28_40 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_40: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %.loc28_40: Core.Form = init_form %Self.ref.loc28_40 [concrete = constants.%.592]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_15: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_31: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %x: %D = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table = impl_witness_table (%D.as.UseSelf.impl.F.decl), @D.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness: <witness> = impl_witness %UseSelf.impl_witness_table [concrete = constants.%UseSelf.impl_witness.047]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   extend %UseSelf.ref
// CHECK:STDOUT:   witness = %UseSelf.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.SelfNested.impl: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.decl: %C.as.SelfNested.impl.F.type = fn_decl @C.as.SelfNested.impl.F [concrete = constants.%C.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.79c = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.79c = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.746 = value_param call_param0
// CHECK:STDOUT:     %.loc36_31.1: type = splice_block %.loc36_31.3 [concrete = constants.%tuple.type.746] {
// CHECK:STDOUT:       %C.ref.loc36_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc36_12 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:       %C.ref.loc36_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc36_29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc36_29.2: type = converted %.loc36_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.ee7]
// CHECK:STDOUT:       %.loc36_31.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.562]
// CHECK:STDOUT:       %.loc36_31.3: type = converted %.loc36_31.2, constants.%tuple.type.746 [concrete = constants.%tuple.type.746]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.746 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%C.as.SelfNested.impl.F.decl), @C.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.d40]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.SelfNested.impl: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.decl: %D.as.SelfNested.impl.F.type = fn_decl @D.as.SelfNested.impl.F [concrete = constants.%D.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.338 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.338 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.aaa = value_param call_param0
// CHECK:STDOUT:     %.loc40_37.1: type = splice_block %.loc40_37.3 [concrete = constants.%tuple.type.aaa] {
// CHECK:STDOUT:       %Self.ref.loc40_12: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref.loc40_12 [concrete = constants.%ptr.d57]
// CHECK:STDOUT:       %Self.ref.loc40_24: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %.loc40_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc40_35.2: type = converted %.loc40_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.e52]
// CHECK:STDOUT:       %.loc40_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.c73]
// CHECK:STDOUT:       %.loc40_37.3: type = converted %.loc40_37.2, constants.%tuple.type.aaa [concrete = constants.%tuple.type.aaa]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.aaa = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%D.as.SelfNested.impl.F.decl), @D.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.ad8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   extend %SelfNested.ref
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
// CHECK:STDOUT: generic fn @UseSelf.WithSelf.F(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bde)]
// CHECK:STDOUT:   %Self.as_type.loc16_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.968)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_8.1 [symbolic = %pattern_type (constants.%pattern_type.04e)]
// CHECK:STDOUT:   %.loc16_26.1: Core.Form = init_form %Self.as_type.loc16_8.1 [symbolic = %.loc16_26.1 (constants.%.326)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968), %x.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968)) -> out %return.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.968);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.UseSelf.impl.F(%self.param: %C, %x.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_52.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_52.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_53: init %C = converted %.loc24_52.1, %.loc24_52.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc24_53 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.UseSelf.impl.F(%self.param: %D, %x.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc28_55.2: init %D to %return.param = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc28_56: init %D = converted %.loc28_55.1, %.loc28_55.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc28_56 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:   %Self.as_type.loc32_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc32_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:   %ptr.loc32_16.1: type = ptr_type %Self.as_type.loc32_16.1 [symbolic = %ptr.loc32_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc32_16.1 (%Self.as_type.8ab), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.749)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1) [symbolic = %tuple (constants.%tuple.876)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1) [symbolic = %tuple.type (constants.%tuple.type.d83)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.6eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.d83));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.SelfNested.impl.F(%x.param: %tuple.type.746);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.SelfNested.impl.F(%x.param: %tuple.type.aaa);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf(constants.%Self.bde) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.bde
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.74d
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%Self.bde) {
// CHECK:STDOUT:   %Self => constants.%Self.bde
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%Self.as_type.968
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.04e
// CHECK:STDOUT:   %.loc16_26.1 => constants.%.326
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf(constants.%UseSelf.facet.14b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.14b
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.103
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.e1d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%UseSelf.facet.14b) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.14b
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %.loc16_26.1 => constants.%.a44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf(constants.%UseSelf.facet.64f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.64f
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type => constants.%UseSelf.WithSelf.F.type.bbe
// CHECK:STDOUT:   %UseSelf.WithSelf.F => constants.%UseSelf.WithSelf.F.8b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%UseSelf.facet.64f) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.64f
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8d
// CHECK:STDOUT:   %.loc16_26.1 => constants.%.592
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.e81) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.702
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.eec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.e81) {
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %Self.as_type.loc32_16.1 => constants.%Self.as_type.8ab
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.a16
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.749
// CHECK:STDOUT:   %tuple => constants.%tuple.876
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.749) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.749
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.369
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.0c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.749) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.749
// CHECK:STDOUT:   %Self.as_type.loc32_16.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.6b6
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.ee7
// CHECK:STDOUT:   %tuple => constants.%tuple.562
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.746
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.79c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.d22) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.d22
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.f5f
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.d22) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.d22
// CHECK:STDOUT:   %Self.as_type.loc32_16.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.d57
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.e52
// CHECK:STDOUT:   %tuple => constants.%tuple.c73
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.aaa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.338
// CHECK:STDOUT: }
// CHECK:STDOUT:
