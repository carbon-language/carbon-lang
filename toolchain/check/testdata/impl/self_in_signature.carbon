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
// CHECK:STDOUT:   %Self.9ee: %UseSelf.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.2b4: type = symbolic_binding_type Self, 0, %Self.9ee [symbolic]
// CHECK:STDOUT:   %pattern_type.308: type = pattern_type %Self.binding.as_type.2b4 [symbolic]
// CHECK:STDOUT:   %UseSelf.F.type: type = fn_type @UseSelf.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %UseSelf.F: %UseSelf.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type @UseSelf [concrete]
// CHECK:STDOUT:   %assoc0.24c: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.%UseSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.34b: <witness> = impl_witness file.%UseSelf.impl_witness_table.loc23 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.type: type = fn_type @C.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %C.as.UseSelf.impl.F: %C.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.0a1: %UseSelf.type = facet_value %C, (%UseSelf.impl_witness.34b) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.2dc: <witness> = impl_witness file.%UseSelf.impl_witness_table.loc27 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.type: type = fn_type @D.as.UseSelf.impl.F [concrete]
// CHECK:STDOUT:   %D.as.UseSelf.impl.F: %D.as.UseSelf.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.85d: %UseSelf.type = facet_value %D, (%UseSelf.impl_witness.2dc) [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.708: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.49d: type = symbolic_binding_type Self, 0, %Self.708 [symbolic]
// CHECK:STDOUT:   %ptr.e7a: type = ptr_type %Self.binding.as_type.49d [symbolic]
// CHECK:STDOUT:   %struct_type.x.y.b2f: type = struct_type {.x: %Self.binding.as_type.49d, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.b03: type = tuple_type (%ptr.e7a, %struct_type.x.y.b2f) [symbolic]
// CHECK:STDOUT:   %pattern_type.bc7: type = pattern_type %tuple.type.b03 [symbolic]
// CHECK:STDOUT:   %SelfNested.F.type: type = fn_type @SelfNested.F [concrete]
// CHECK:STDOUT:   %SelfNested.F: %SelfNested.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.beb: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%SelfNested.F.decl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.e41: <witness> = impl_witness file.%SelfNested.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %struct_type.x.y.2f0: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a17: type = tuple_type (%ptr.019, %struct_type.x.y.2f0) [concrete]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %tuple.type.a17 [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.type: type = fn_type @C.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %C.as.SelfNested.impl.F: %C.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.60c: %SelfNested.type = facet_value %C, (%SelfNested.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.641: <witness> = impl_witness file.%SelfNested.impl_witness_table.loc39 [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %struct_type.x.y.527: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a5f: type = tuple_type (%ptr.19c, %struct_type.x.y.527) [concrete]
// CHECK:STDOUT:   %pattern_type.b43: type = pattern_type %tuple.type.a5f [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.type: type = fn_type @D.as.SelfNested.impl.F [concrete]
// CHECK:STDOUT:   %D.as.SelfNested.impl.F: %D.as.SelfNested.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.86f: %SelfNested.type = facet_value %D, (%SelfNested.impl_witness.641) [concrete]
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
// CHECK:STDOUT:   %UseSelf.impl_witness_table.loc23 = impl_witness_table (@C.as.UseSelf.impl.%C.as.UseSelf.impl.F.decl), @C.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.loc23: <witness> = impl_witness %UseSelf.impl_witness_table.loc23 [concrete = constants.%UseSelf.impl_witness.34b]
// CHECK:STDOUT:   impl_decl @D.as.UseSelf.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table.loc27 = impl_witness_table (@D.as.UseSelf.impl.%D.as.UseSelf.impl.F.decl), @D.as.UseSelf.impl [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.loc27: <witness> = impl_witness %UseSelf.impl_witness_table.loc27 [concrete = constants.%UseSelf.impl_witness.2dc]
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [concrete = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table.loc35 = impl_witness_table (@C.as.SelfNested.impl.%C.as.SelfNested.impl.F.decl), @C.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.loc35: <witness> = impl_witness %SelfNested.impl_witness_table.loc35 [concrete = constants.%SelfNested.impl_witness.e41]
// CHECK:STDOUT:   impl_decl @D.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table.loc39 = impl_witness_table (@D.as.SelfNested.impl.%D.as.SelfNested.impl.F.decl), @D.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.loc39: <witness> = impl_witness %SelfNested.impl_witness_table.loc39 [concrete = constants.%SelfNested.impl_witness.641]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9ee]
// CHECK:STDOUT:   %UseSelf.F.decl: %UseSelf.F.type = fn_decl @UseSelf.F [concrete = constants.%UseSelf.F] {
// CHECK:STDOUT:     %self.patt: @UseSelf.F.%pattern_type (%pattern_type.308) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @UseSelf.F.%pattern_type (%pattern_type.308) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: @UseSelf.F.%pattern_type (%pattern_type.308) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @UseSelf.F.%pattern_type (%pattern_type.308) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @UseSelf.F.%pattern_type (%pattern_type.308) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseSelf.F.%pattern_type (%pattern_type.308) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_32: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.9ee)]
// CHECK:STDOUT:     %Self.as_type.loc16_32: type = facet_access_type %Self.ref.loc16_32 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:     %.loc16_32: type = converted %Self.ref.loc16_32, %Self.as_type.loc16_32 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:     %self.param: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.9ee)]
// CHECK:STDOUT:       %Self.as_type.loc16_14: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = value_param call_param1
// CHECK:STDOUT:     %.loc16_23.1: type = splice_block %.loc16_23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)] {
// CHECK:STDOUT:       %Self.ref.loc16_23: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.9ee)]
// CHECK:STDOUT:       %Self.as_type.loc16_23: type = facet_access_type %Self.ref.loc16_23 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:       %.loc16_23.2: type = converted %Self.ref.loc16_23, %Self.as_type.loc16_23 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = out_param call_param2
// CHECK:STDOUT:     %return: ref @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %UseSelf.F.decl [concrete = constants.%assoc0.24c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%UseSelf.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.708]
// CHECK:STDOUT:   %SelfNested.F.decl: %SelfNested.F.type = fn_decl @SelfNested.F [concrete = constants.%SelfNested.F] {
// CHECK:STDOUT:     %x.patt: @SelfNested.F.%pattern_type (%pattern_type.bc7) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @SelfNested.F.%pattern_type (%pattern_type.bc7) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @SelfNested.F.%tuple.type (%tuple.type.b03) = value_param call_param0
// CHECK:STDOUT:     %.loc32_37.1: type = splice_block %.loc32_37.3 [symbolic = %tuple.type (constants.%tuple.type.b03)] {
// CHECK:STDOUT:       %Self.ref.loc32_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.708)]
// CHECK:STDOUT:       %Self.as_type.loc32_16: type = facet_access_type %Self.ref.loc32_12 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.49d)]
// CHECK:STDOUT:       %.loc32_16: type = converted %Self.ref.loc32_12, %Self.as_type.loc32_16 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.49d)]
// CHECK:STDOUT:       %ptr.loc32_16.2: type = ptr_type %.loc32_16 [symbolic = %ptr.loc32_16.1 (constants.%ptr.e7a)]
// CHECK:STDOUT:       %Self.ref.loc32_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.708)]
// CHECK:STDOUT:       %Self.as_type.loc32_24: type = facet_access_type %Self.ref.loc32_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.49d)]
// CHECK:STDOUT:       %.loc32_24: type = converted %Self.ref.loc32_24, %Self.as_type.loc32_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.49d)]
// CHECK:STDOUT:       %.loc32_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc32_35.2: type = converted %.loc32_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc32_36.2: type = struct_type {.x: @SelfNested.F.%Self.binding.as_type (%Self.binding.as_type.49d), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.b2f)]
// CHECK:STDOUT:       %.loc32_37.2: %tuple.type.24b = tuple_literal (%ptr.loc32_16.2, %struct_type.x.y.loc32_36.2)
// CHECK:STDOUT:       %.loc32_37.3: type = converted %.loc32_37.2, constants.%tuple.type.b03 [symbolic = %tuple.type (constants.%tuple.type.b03)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.F.%tuple.type (%tuple.type.b03) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.F.decl [concrete = constants.%assoc0.beb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%SelfNested.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.UseSelf.impl: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %C.as.UseSelf.impl.F.decl: %C.as.UseSelf.impl.F.type = fn_decl @C.as.UseSelf.impl.F [concrete = constants.%C.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.c48 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c48 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc24_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc24_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   witness = file.%UseSelf.impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.UseSelf.impl: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %D.as.UseSelf.impl.F.decl: %D.as.UseSelf.impl.F.type = fn_decl @D.as.UseSelf.impl.F [concrete = constants.%D.as.UseSelf.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.510 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.510 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_14: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_23: type = name_ref Self, @D.as.UseSelf.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %x: %D = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.UseSelf.impl.F.decl
// CHECK:STDOUT:   witness = file.%UseSelf.impl_witness.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.SelfNested.impl: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %C.as.SelfNested.impl.F.decl: %C.as.SelfNested.impl.F.type = fn_decl @C.as.SelfNested.impl.F [concrete = constants.%C.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.809 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.809 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a17 = value_param call_param0
// CHECK:STDOUT:     %.loc36_31.1: type = splice_block %.loc36_31.3 [concrete = constants.%tuple.type.a17] {
// CHECK:STDOUT:       %C.ref.loc36_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc36_12 [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %C.ref.loc36_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc36_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc36_29.2: type = converted %.loc36_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.2f0]
// CHECK:STDOUT:       %.loc36_31.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc36_31.3: type = converted %.loc36_31.2, constants.%tuple.type.a17 [concrete = constants.%tuple.type.a17]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a17 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   witness = file.%SelfNested.impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.SelfNested.impl: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %D.as.SelfNested.impl.F.decl: %D.as.SelfNested.impl.F.type = fn_decl @D.as.SelfNested.impl.F [concrete = constants.%D.as.SelfNested.impl.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.b43 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.b43 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a5f = value_param call_param0
// CHECK:STDOUT:     %.loc40_37.1: type = splice_block %.loc40_37.3 [concrete = constants.%tuple.type.a5f] {
// CHECK:STDOUT:       %Self.ref.loc40_12: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref.loc40_12 [concrete = constants.%ptr.19c]
// CHECK:STDOUT:       %Self.ref.loc40_24: type = name_ref Self, @D.as.SelfNested.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %.loc40_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc40_35.2: type = converted %.loc40_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.527]
// CHECK:STDOUT:       %.loc40_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc40_37.3: type = converted %.loc40_37.2, constants.%tuple.type.a5f [concrete = constants.%tuple.type.a5f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a5f = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.SelfNested.impl.F.decl
// CHECK:STDOUT:   witness = file.%SelfNested.impl_witness.loc39
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
// CHECK:STDOUT: generic fn @UseSelf.F(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9ee)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.2b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.308)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4), %x.param: @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4)) -> @UseSelf.F.%Self.binding.as_type (%Self.binding.as_type.2b4);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.UseSelf.impl.F(%self.param: %C, %x.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_38.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_39: init %C = converted %.loc24_38.1, %.loc24_38.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc24_39 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.UseSelf.impl.F(%self.param: %D, %x.param: %D) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_47.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc28_47.2: init %D = class_init (), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc28_48: init %D = converted %.loc28_47.1, %.loc28_47.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc28_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.708)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.49d)]
// CHECK:STDOUT:   %ptr.loc32_16.1: type = ptr_type %Self.binding.as_type [symbolic = %ptr.loc32_16.1 (constants.%ptr.e7a)]
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1: type = struct_type {.x: @SelfNested.F.%Self.binding.as_type (%Self.binding.as_type.49d), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.1 (constants.%struct_type.x.y.b2f)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1) [symbolic = %tuple.type (constants.%tuple.type.b03)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.bc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.F.%tuple.type (%tuple.type.b03));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.SelfNested.impl.F(%x.param: %tuple.type.a17);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.SelfNested.impl.F(%x.param: %tuple.type.a5f);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.F(constants.%Self.9ee) {
// CHECK:STDOUT:   %Self => constants.%Self.9ee
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.2b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.F(constants.%UseSelf.facet.0a1) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.0a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.F(constants.%UseSelf.facet.85d) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.85d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%Self.708) {
// CHECK:STDOUT:   %Self => constants.%Self.708
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.49d
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.e7a
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.b2f
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.b03
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%SelfNested.facet.60c) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.60c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.019
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.2f0
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a17
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.809
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%SelfNested.facet.86f) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.86f
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %ptr.loc32_16.1 => constants.%ptr.19c
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.1 => constants.%struct_type.x.y.527
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b43
// CHECK:STDOUT: }
// CHECK:STDOUT:
