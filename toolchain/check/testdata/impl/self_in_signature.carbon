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
// CHECK:STDOUT:   %Self.085: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.599: type = facet_access_type %Self.085 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ef: type = pattern_type %Self.as_type.599 [symbolic]
// CHECK:STDOUT:   %F.type.86a: type = fn_type @F.loc16 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.408: %F.type.86a = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type @UseSelf [concrete]
// CHECK:STDOUT:   %assoc0.24c: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.%F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.b25: <witness> = impl_witness file.%UseSelf.impl_witness_table.loc23 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type.fc6: type = fn_type @F.loc24 [concrete]
// CHECK:STDOUT:   %F.5e2: %F.type.fc6 = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.727: %UseSelf.type = facet_value %C, (%UseSelf.impl_witness.b25) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.f90: <witness> = impl_witness file.%UseSelf.impl_witness_table.loc27 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.loc28 [concrete]
// CHECK:STDOUT:   %F.f71: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.d11: %UseSelf.type = facet_value %D, (%UseSelf.impl_witness.f90) [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.2ff: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.e1e: type = facet_access_type %Self.2ff [symbolic]
// CHECK:STDOUT:   %ptr.e87: type = ptr_type %Self.as_type.e1e [symbolic]
// CHECK:STDOUT:   %struct_type.x.y.270: type = struct_type {.x: %Self.as_type.e1e, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.46b: type = tuple_type (%ptr.e87, %struct_type.x.y.270) [symbolic]
// CHECK:STDOUT:   %pattern_type.5a1: type = pattern_type %tuple.type.46b [symbolic]
// CHECK:STDOUT:   %F.type.6ed: type = fn_type @F.loc32 [concrete]
// CHECK:STDOUT:   %F.998: %F.type.6ed = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.beb: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%F.decl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.5d5: <witness> = impl_witness file.%SelfNested.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %struct_type.x.y.2f0: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a17: type = tuple_type (%ptr.019, %struct_type.x.y.2f0) [concrete]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %tuple.type.a17 [concrete]
// CHECK:STDOUT:   %F.type.ef0: type = fn_type @F.loc36 [concrete]
// CHECK:STDOUT:   %F.9a9: %F.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.2b2: %SelfNested.type = facet_value %C, (%SelfNested.impl_witness.5d5) [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.16b: <witness> = impl_witness file.%SelfNested.impl_witness_table.loc39 [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %struct_type.x.y.527: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a5f: type = tuple_type (%ptr.19c, %struct_type.x.y.527) [concrete]
// CHECK:STDOUT:   %pattern_type.b43: type = pattern_type %tuple.type.a5f [concrete]
// CHECK:STDOUT:   %F.type.a9d: type = fn_type @F.loc40 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.a9d = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.f2c: %SelfNested.type = facet_value %D, (%SelfNested.impl_witness.16b) [concrete]
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
// CHECK:STDOUT:   impl_decl @UseSelf.impl.c4a [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table.loc23 = impl_witness_table (@UseSelf.impl.c4a.%F.decl), @UseSelf.impl.c4a [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.loc23: <witness> = impl_witness %UseSelf.impl_witness_table.loc23 [concrete = constants.%UseSelf.impl_witness.b25]
// CHECK:STDOUT:   impl_decl @UseSelf.impl.659 [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.impl_witness_table.loc27 = impl_witness_table (@UseSelf.impl.659.%F.decl), @UseSelf.impl.659 [concrete]
// CHECK:STDOUT:   %UseSelf.impl_witness.loc27: <witness> = impl_witness %UseSelf.impl_witness_table.loc27 [concrete = constants.%UseSelf.impl_witness.f90]
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [concrete = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @SelfNested.impl.730 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table.loc35 = impl_witness_table (@SelfNested.impl.730.%F.decl), @SelfNested.impl.730 [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.loc35: <witness> = impl_witness %SelfNested.impl_witness_table.loc35 [concrete = constants.%SelfNested.impl_witness.5d5]
// CHECK:STDOUT:   impl_decl @SelfNested.impl.82d [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table.loc39 = impl_witness_table (@SelfNested.impl.82d.%F.decl), @SelfNested.impl.82d [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.loc39: <witness> = impl_witness %SelfNested.impl_witness_table.loc39 [concrete = constants.%SelfNested.impl_witness.16b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.085]
// CHECK:STDOUT:   %F.decl: %F.type.86a = fn_decl @F.loc16 [concrete = constants.%F.408] {
// CHECK:STDOUT:     %self.patt: @F.loc16.%pattern_type (%pattern_type.3ef) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.loc16.%pattern_type (%pattern_type.3ef) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.loc16.%pattern_type (%pattern_type.3ef) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.loc16.%pattern_type (%pattern_type.3ef) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc16.%pattern_type (%pattern_type.3ef) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc16.%pattern_type (%pattern_type.3ef) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_32: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:     %Self.as_type.loc16_32: type = facet_access_type %Self.ref.loc16_32 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     %.loc16_32: type = converted %Self.ref.loc16_32, %Self.as_type.loc16_32 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     %self.param: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:       %Self.as_type.loc16_14.1: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14.1 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = value_param call_param1
// CHECK:STDOUT:     %.loc16_23.1: type = splice_block %.loc16_23.2 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)] {
// CHECK:STDOUT:       %Self.ref.loc16_23: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:       %Self.as_type.loc16_23: type = facet_access_type %Self.ref.loc16_23 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:       %.loc16_23.2: type = converted %Self.ref.loc16_23, %Self.as_type.loc16_23 [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.24c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2ff]
// CHECK:STDOUT:   %F.decl: %F.type.6ed = fn_decl @F.loc32 [concrete = constants.%F.998] {
// CHECK:STDOUT:     %x.patt: @F.loc32.%pattern_type (%pattern_type.5a1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.loc32.%pattern_type (%pattern_type.5a1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @F.loc32.%tuple.type (%tuple.type.46b) = value_param call_param0
// CHECK:STDOUT:     %.loc32_37.1: type = splice_block %.loc32_37.3 [symbolic = %tuple.type (constants.%tuple.type.46b)] {
// CHECK:STDOUT:       %Self.ref.loc32_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc32_16.1: type = facet_access_type %Self.ref.loc32_12 [symbolic = %Self.as_type.loc32_16.2 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc32_16: type = converted %Self.ref.loc32_12, %Self.as_type.loc32_16.1 [symbolic = %Self.as_type.loc32_16.2 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %ptr.loc32_16.1: type = ptr_type %.loc32_16 [symbolic = %ptr.loc32_16.2 (constants.%ptr.e87)]
// CHECK:STDOUT:       %Self.ref.loc32_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc32_24: type = facet_access_type %Self.ref.loc32_24 [symbolic = %Self.as_type.loc32_16.2 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc32_24: type = converted %Self.ref.loc32_24, %Self.as_type.loc32_24 [symbolic = %Self.as_type.loc32_16.2 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc32_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc32_35.2: type = converted %.loc32_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc32_36.1: type = struct_type {.x: @F.loc32.%Self.as_type.loc32_16.2 (%Self.as_type.e1e), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.2 (constants.%struct_type.x.y.270)]
// CHECK:STDOUT:       %.loc32_37.2: %tuple.type.24b = tuple_literal (%ptr.loc32_16.1, %struct_type.x.y.loc32_36.1)
// CHECK:STDOUT:       %.loc32_37.3: type = converted %.loc32_37.2, constants.%tuple.type.46b [symbolic = %tuple.type (constants.%tuple.type.46b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.loc32.%tuple.type (%tuple.type.46b) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.beb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @UseSelf.impl.c4a: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.fc6 = fn_decl @F.loc24 [concrete = constants.%F.5e2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.c48 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c48 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc24_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc24_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%UseSelf.impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @UseSelf.impl.659: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.0aa = fn_decl @F.loc28 [concrete = constants.%F.f71] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.510 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.510 = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_32: type = name_ref Self, @UseSelf.impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc28_14: type = name_ref Self, @UseSelf.impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc28_23: type = name_ref Self, @UseSelf.impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %x: %D = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%UseSelf.impl_witness.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNested.impl.730: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.ef0 = fn_decl @F.loc36 [concrete = constants.%F.9a9] {
// CHECK:STDOUT:     %x.patt: %pattern_type.809 = binding_pattern x [concrete]
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
// CHECK:STDOUT:     %x: %tuple.type.a17 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%SelfNested.impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNested.impl.82d: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.a9d = fn_decl @F.loc40 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %x.patt: %pattern_type.b43 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.b43 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a5f = value_param call_param0
// CHECK:STDOUT:     %.loc40_37.1: type = splice_block %.loc40_37.3 [concrete = constants.%tuple.type.a5f] {
// CHECK:STDOUT:       %Self.ref.loc40_12: type = name_ref Self, @SelfNested.impl.82d.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref.loc40_12 [concrete = constants.%ptr.19c]
// CHECK:STDOUT:       %Self.ref.loc40_24: type = name_ref Self, @SelfNested.impl.82d.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %.loc40_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc40_35.2: type = converted %.loc40_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.527]
// CHECK:STDOUT:       %.loc40_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc40_37.3: type = converted %.loc40_37.2, constants.%tuple.type.a5f [concrete = constants.%tuple.type.a5f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a5f = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%SelfNested.impl_witness.loc39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc16(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:   %Self.as_type.loc16_14.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_14.2 (constants.%Self.as_type.599)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_14.2 [symbolic = %pattern_type (constants.%pattern_type.3ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599), %x.param: @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599)) -> @F.loc16.%Self.as_type.loc16_14.2 (%Self.as_type.599);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc24(%self.param: %C, %x.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_38.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_39: init %C = converted %.loc24_38.1, %.loc24_38.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc24_39 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc28(%self.param: %D, %x.param: %D) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_47.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc28_47.2: init %D = class_init (), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc28_48: init %D = converted %.loc28_47.1, %.loc28_47.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc28_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc32(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:   %Self.as_type.loc32_16.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc32_16.2 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:   %ptr.loc32_16.2: type = ptr_type %Self.as_type.loc32_16.2 [symbolic = %ptr.loc32_16.2 (constants.%ptr.e87)]
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.2: type = struct_type {.x: @F.loc32.%Self.as_type.loc32_16.2 (%Self.as_type.e1e), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc32_36.2 (constants.%struct_type.x.y.270)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc32_16.2, %struct_type.x.y.loc32_36.2) [symbolic = %tuple.type (constants.%tuple.type.46b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.5a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.loc32.%tuple.type (%tuple.type.46b));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc36(%x.param: %tuple.type.a17);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc40(%x.param: %tuple.type.a5f);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%Self.085) {
// CHECK:STDOUT:   %Self => constants.%Self.085
// CHECK:STDOUT:   %Self.as_type.loc16_14.2 => constants.%Self.as_type.599
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%UseSelf.facet.727) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.727
// CHECK:STDOUT:   %Self.as_type.loc16_14.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%UseSelf.facet.d11) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.d11
// CHECK:STDOUT:   %Self.as_type.loc16_14.2 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc32(constants.%Self.2ff) {
// CHECK:STDOUT:   %Self => constants.%Self.2ff
// CHECK:STDOUT:   %Self.as_type.loc32_16.2 => constants.%Self.as_type.e1e
// CHECK:STDOUT:   %ptr.loc32_16.2 => constants.%ptr.e87
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.2 => constants.%struct_type.x.y.270
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.46b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc32(constants.%SelfNested.facet.2b2) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.2b2
// CHECK:STDOUT:   %Self.as_type.loc32_16.2 => constants.%C
// CHECK:STDOUT:   %ptr.loc32_16.2 => constants.%ptr.019
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.2 => constants.%struct_type.x.y.2f0
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a17
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.809
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc32(constants.%SelfNested.facet.f2c) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.f2c
// CHECK:STDOUT:   %Self.as_type.loc32_16.2 => constants.%D
// CHECK:STDOUT:   %ptr.loc32_16.2 => constants.%ptr.19c
// CHECK:STDOUT:   %struct_type.x.y.loc32_36.2 => constants.%struct_type.x.y.527
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b43
// CHECK:STDOUT: }
// CHECK:STDOUT:
