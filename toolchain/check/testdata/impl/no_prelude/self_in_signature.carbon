// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/self_in_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/self_in_signature.carbon

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
// CHECK:STDOUT:   %F.type.86a: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.408: %F.type.86a = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type %UseSelf.type [concrete]
// CHECK:STDOUT:   %assoc0.774: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.%F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %impl_witness.9ff: <witness> = impl_witness (@impl.c4a.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.fc6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.5e2: %F.type.fc6 = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.692: %UseSelf.type = facet_value %C, (%impl_witness.9ff) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.320: <witness> = impl_witness (@impl.659.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.f71: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %UseSelf.facet.5ea: %UseSelf.type = facet_value %D, (%impl_witness.320) [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.2ff: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.e1e: type = facet_access_type %Self.2ff [symbolic]
// CHECK:STDOUT:   %ptr.e87: type = ptr_type %Self.as_type.e1e [symbolic]
// CHECK:STDOUT:   %struct_type.x.y.270: type = struct_type {.x: %Self.as_type.e1e, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.46b: type = tuple_type (%ptr.e87, %struct_type.x.y.270) [symbolic]
// CHECK:STDOUT:   %F.type.6ed: type = fn_type @F.4 [concrete]
// CHECK:STDOUT:   %F.998: %F.type.6ed = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type %SelfNested.type [concrete]
// CHECK:STDOUT:   %assoc0.a58: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness.f4e: <witness> = impl_witness (@impl.730.%F.decl) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %struct_type.x.y.2f0: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a17: type = tuple_type (%ptr.019, %struct_type.x.y.2f0) [concrete]
// CHECK:STDOUT:   %F.type.ef0: type = fn_type @F.5 [concrete]
// CHECK:STDOUT:   %F.9a9: %F.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.5ec: %SelfNested.type = facet_value %C, (%impl_witness.f4e) [concrete]
// CHECK:STDOUT:   %impl_witness.4b5: <witness> = impl_witness (@impl.82d.%F.decl) [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %struct_type.x.y.527: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %tuple.type.a5f: type = tuple_type (%ptr.19c, %struct_type.x.y.527) [concrete]
// CHECK:STDOUT:   %F.type.a9d: type = fn_type @F.6 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.a9d = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.821: %SelfNested.type = facet_value %D, (%impl_witness.4b5) [concrete]
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
// CHECK:STDOUT:   impl_decl @impl.c4a [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.c4a.%F.decl) [concrete = constants.%impl_witness.9ff]
// CHECK:STDOUT:   impl_decl @impl.659 [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [concrete = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.659.%F.decl) [concrete = constants.%impl_witness.320]
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [concrete = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.730 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (@impl.730.%F.decl) [concrete = constants.%impl_witness.f4e]
// CHECK:STDOUT:   impl_decl @impl.82d [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc35: <witness> = impl_witness (@impl.82d.%F.decl) [concrete = constants.%impl_witness.4b5]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.085]
// CHECK:STDOUT:   %F.decl: %F.type.86a = fn_decl @F.1 [concrete = constants.%F.408] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %x.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = value_param_pattern %x.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_32: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:     %Self.as_type.loc12_32: type = facet_access_type %Self.ref.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     %.loc12_32: type = converted %Self.ref.loc12_32, %Self.as_type.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = value_param call_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)] {
// CHECK:STDOUT:       %Self.ref.loc12_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref.loc12_14 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref.loc12_14, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = value_param call_param1
// CHECK:STDOUT:     %.loc12_23.1: type = splice_block %.loc12_23.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)] {
// CHECK:STDOUT:       %Self.ref.loc12_23: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:       %Self.as_type.loc12_23: type = facet_access_type %Self.ref.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:       %.loc12_23.2: type = converted %Self.ref.loc12_23, %Self.as_type.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.774]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2ff]
// CHECK:STDOUT:   %F.decl: %F.type.6ed = fn_decl @F.4 [concrete = constants.%F.998] {
// CHECK:STDOUT:     %x.patt: @F.4.%tuple.type (%tuple.type.46b) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.4.%tuple.type (%tuple.type.46b) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @F.4.%tuple.type (%tuple.type.46b) = value_param call_param0
// CHECK:STDOUT:     %.loc28_37.1: type = splice_block %.loc28_37.3 [symbolic = %tuple.type (constants.%tuple.type.46b)] {
// CHECK:STDOUT:       %Self.ref.loc28_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc28_16.2: type = facet_access_type %Self.ref.loc28_12 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc28_16: type = converted %Self.ref.loc28_12, %Self.as_type.loc28_16.2 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %ptr.loc28_16.2: type = ptr_type %Self.as_type.e1e [symbolic = %ptr.loc28_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:       %Self.ref.loc28_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc28_24: type = facet_access_type %Self.ref.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc28_24: type = converted %Self.ref.loc28_24, %Self.as_type.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc28_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc28_35.2: type = converted %.loc28_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc28_36.2: type = struct_type {.x: %Self.as_type.e1e, .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.270)]
// CHECK:STDOUT:       %.loc28_37.2: %tuple.type.24b = tuple_literal (%ptr.loc28_16.2, %struct_type.x.y.loc28_36.2)
// CHECK:STDOUT:       %.loc28_37.3: type = converted %.loc28_37.2, constants.%tuple.type.46b [symbolic = %tuple.type (constants.%tuple.type.46b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.4.%tuple.type (%tuple.type.46b) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.a58]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c4a: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.fc6 = fn_decl @F.2 [concrete = constants.%F.5e2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.659: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.0aa = fn_decl @F.3 [concrete = constants.%F.f71] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %D = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %x.patt: %D = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %D = value_param_pattern %x.patt, call_param1
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc24_32: type = name_ref Self, @impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc24_14: type = name_ref Self, @impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc24_23: type = name_ref Self, @impl.659.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %x: %D = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.730: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.ef0 = fn_decl @F.5 [concrete = constants.%F.9a9] {
// CHECK:STDOUT:     %x.patt: %tuple.type.a17 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.a17 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a17 = value_param call_param0
// CHECK:STDOUT:     %.loc32_31.1: type = splice_block %.loc32_31.3 [concrete = constants.%tuple.type.a17] {
// CHECK:STDOUT:       %C.ref.loc32_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %C.ref.loc32_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc32_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc32_29.2: type = converted %.loc32_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.2f0]
// CHECK:STDOUT:       %.loc32_31.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc32_31.3: type = converted %.loc32_31.2, constants.%tuple.type.a17 [concrete = constants.%tuple.type.a17]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a17 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.82d: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.a9d = fn_decl @F.6 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %x.patt: %tuple.type.a5f = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.a5f = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a5f = value_param call_param0
// CHECK:STDOUT:     %.loc36_37.1: type = splice_block %.loc36_37.3 [concrete = constants.%tuple.type.a5f] {
// CHECK:STDOUT:       %Self.ref.loc36_12: type = name_ref Self, @impl.82d.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %D [concrete = constants.%ptr.19c]
// CHECK:STDOUT:       %Self.ref.loc36_24: type = name_ref Self, @impl.82d.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:       %.loc36_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc36_35.2: type = converted %.loc36_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y.527]
// CHECK:STDOUT:       %.loc36_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc36_37.3: type = converted %.loc36_37.2, constants.%tuple.type.a5f [concrete = constants.%tuple.type.a5f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a5f = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.085)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.599)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599)](%x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599)) -> @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.599);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C](%x.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc20_38.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc20_39: init %C = converted %.loc20_38.1, %.loc20_38.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc20_39 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %D](%x.param_patt: %D) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_47.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_47.2: init %D = class_init (), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc24_48: init %D = converted %.loc24_47.1, %.loc24_47.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc24_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.4(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:   %Self.as_type.loc28_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:   %ptr.loc28_16.1: type = ptr_type @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.e1e) [symbolic = %ptr.loc28_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1: type = struct_type {.x: @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.e1e), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.270)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.4.%ptr.loc28_16.1 (%ptr.e87), @F.4.%struct_type.x.y.loc28_36.1 (%struct_type.x.y.270)) [symbolic = %tuple.type (constants.%tuple.type.46b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.4.%tuple.type (%tuple.type.46b));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(%x.param_patt: %tuple.type.a17);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(%x.param_patt: %tuple.type.a5f);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.085) {
// CHECK:STDOUT:   %Self => constants.%Self.085
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.599
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.692) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.692
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.5ea) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.5ea
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%Self.2ff) {
// CHECK:STDOUT:   %Self => constants.%Self.2ff
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%Self.as_type.e1e
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.e87
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.270
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.46b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.5ec) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.5ec
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.019
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.2f0
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.821) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.821
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.19c
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.527
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
