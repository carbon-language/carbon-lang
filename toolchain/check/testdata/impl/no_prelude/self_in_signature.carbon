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
// CHECK:STDOUT:   %UseSelf.type: type = facet_type <@UseSelf> [template]
// CHECK:STDOUT:   %Self.aee: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.3e2: type = facet_access_type %Self.aee [symbolic]
// CHECK:STDOUT:   %F.type.a69: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.e5a: %F.type.a69 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.c96: type = assoc_entity_type %UseSelf.type, %F.type.a69 [template]
// CHECK:STDOUT:   %assoc0.8e9: %F.assoc_type.c96 = assoc_entity element0, @UseSelf.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %impl_witness.c6f: <witness> = impl_witness (@impl.1.%F.decl) [template]
// CHECK:STDOUT:   %F.type.261: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.63a: %F.type.261 = struct_value () [template]
// CHECK:STDOUT:   %UseSelf.facet.c5b: %UseSelf.type = facet_value %C, %impl_witness.c6f [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.02b: <witness> = impl_witness (@impl.2.%F.decl) [template]
// CHECK:STDOUT:   %F.type.5ab: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.d6a: %F.type.5ab = struct_value () [template]
// CHECK:STDOUT:   %UseSelf.facet.af5: %UseSelf.type = facet_value %D, %impl_witness.02b [template]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [template]
// CHECK:STDOUT:   %Self.36c: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.966: type = facet_access_type %Self.36c [symbolic]
// CHECK:STDOUT:   %ptr.af7: type = ptr_type %Self.as_type.966 [symbolic]
// CHECK:STDOUT:   %struct_type.x.y.0dc: type = struct_type {.x: %Self.as_type.966, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.7e5: type = tuple_type (%ptr.af7, %struct_type.x.y.0dc) [symbolic]
// CHECK:STDOUT:   %F.type.f33: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.04f: %F.type.f33 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.c02: type = assoc_entity_type %SelfNested.type, %F.type.f33 [template]
// CHECK:STDOUT:   %assoc0.982: %F.assoc_type.c02 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %impl_witness.d90: <witness> = impl_witness (@impl.3.%F.decl) [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT:   %struct_type.x.y.2f0: type = struct_type {.x: %C, .y: %empty_tuple.type} [template]
// CHECK:STDOUT:   %tuple.type.a17: type = tuple_type (%ptr.019, %struct_type.x.y.2f0) [template]
// CHECK:STDOUT:   %F.type.9b5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.c45: %F.type.9b5 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.d94: %SelfNested.type = facet_value %C, %impl_witness.d90 [template]
// CHECK:STDOUT:   %impl_witness.55c: <witness> = impl_witness (@impl.4.%F.decl) [template]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [template]
// CHECK:STDOUT:   %struct_type.x.y.527: type = struct_type {.x: %D, .y: %empty_tuple.type} [template]
// CHECK:STDOUT:   %tuple.type.a5f: type = tuple_type (%ptr.19c, %struct_type.x.y.527) [template]
// CHECK:STDOUT:   %F.type.3cb: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.3b6: %F.type.3cb = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.2f0: %SelfNested.type = facet_value %D, %impl_witness.55c [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%UseSelf.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [template = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.1.%F.decl) [template = constants.%impl_witness.c6f]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [template = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.2.%F.decl) [template = constants.%impl_witness.02b]
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (@impl.3.%F.decl) [template = constants.%impl_witness.d90]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc35: <witness> = impl_witness (@impl.4.%F.decl) [template = constants.%impl_witness.55c]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.aee]
// CHECK:STDOUT:   %F.decl: %F.type.a69 = fn_decl @F.1 [template = constants.%F.e5a] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_32: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.aee)]
// CHECK:STDOUT:     %Self.as_type.loc12_32: type = facet_access_type %Self.ref.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:     %.loc12_32: type = converted %Self.ref.loc12_32, %Self.as_type.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)] {
// CHECK:STDOUT:       %Self.ref.loc12_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.aee)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref.loc12_14 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref.loc12_14, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = value_param runtime_param1
// CHECK:STDOUT:     %.loc12_23.1: type = splice_block %.loc12_23.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)] {
// CHECK:STDOUT:       %Self.ref.loc12_23: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.aee)]
// CHECK:STDOUT:       %Self.as_type.loc12_23: type = facet_access_type %Self.ref.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:       %.loc12_23.2: type = converted %Self.ref.loc12_23, %Self.as_type.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.c96 = assoc_entity element0, %F.decl [template = constants.%assoc0.8e9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.36c]
// CHECK:STDOUT:   %F.decl: %F.type.f33 = fn_decl @F.4 [template = constants.%F.04f] {
// CHECK:STDOUT:     %x.patt: @F.4.%tuple.type (%tuple.type.7e5) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.4.%tuple.type (%tuple.type.7e5) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @F.4.%tuple.type (%tuple.type.7e5) = value_param runtime_param0
// CHECK:STDOUT:     %.loc28_37.1: type = splice_block %.loc28_37.3 [symbolic = %tuple.type (constants.%tuple.type.7e5)] {
// CHECK:STDOUT:       %Self.ref.loc28_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.36c)]
// CHECK:STDOUT:       %Self.as_type.loc28_16.2: type = facet_access_type %Self.ref.loc28_12 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.966)]
// CHECK:STDOUT:       %.loc28_16: type = converted %Self.ref.loc28_12, %Self.as_type.loc28_16.2 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.966)]
// CHECK:STDOUT:       %ptr.loc28_16.2: type = ptr_type %Self.as_type.966 [symbolic = %ptr.loc28_16.1 (constants.%ptr.af7)]
// CHECK:STDOUT:       %Self.ref.loc28_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.36c)]
// CHECK:STDOUT:       %Self.as_type.loc28_24: type = facet_access_type %Self.ref.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.966)]
// CHECK:STDOUT:       %.loc28_24: type = converted %Self.ref.loc28_24, %Self.as_type.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.966)]
// CHECK:STDOUT:       %.loc28_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc28_35.2: type = converted %.loc28_35.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y.loc28_36.2: type = struct_type {.x: %Self.as_type.966, .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.0dc)]
// CHECK:STDOUT:       %.loc28_37.2: %tuple.type.24b = tuple_literal (%ptr.loc28_16.2, %struct_type.x.y.loc28_36.2)
// CHECK:STDOUT:       %.loc28_37.3: type = converted %.loc28_37.2, constants.%tuple.type.7e5 [symbolic = %tuple.type (constants.%tuple.type.7e5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.4.%tuple.type (%tuple.type.7e5) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.c02 = assoc_entity element0, %F.decl [template = constants.%assoc0.982]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.261 = fn_decl @F.2 [template = constants.%F.63a] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref.loc20_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.5ab = fn_decl @F.3 [template = constants.%F.d6a] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %D = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: %D = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %D = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc24_32: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %self.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref.loc24_14: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %Self.ref.loc24_23: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %x: %D = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.9b5 = fn_decl @F.5 [template = constants.%F.c45] {
// CHECK:STDOUT:     %x.patt: %tuple.type.a17 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.a17 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a17 = value_param runtime_param0
// CHECK:STDOUT:     %.loc32_31.1: type = splice_block %.loc32_31.3 [template = constants.%tuple.type.a17] {
// CHECK:STDOUT:       %C.ref.loc32_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:       %C.ref.loc32_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %.loc32_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc32_29.2: type = converted %.loc32_29.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [template = constants.%struct_type.x.y.2f0]
// CHECK:STDOUT:       %.loc32_31.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc32_31.3: type = converted %.loc32_31.2, constants.%tuple.type.a17 [template = constants.%tuple.type.a17]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a17 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.3cb = fn_decl @F.6 [template = constants.%F.3b6] {
// CHECK:STDOUT:     %x.patt: %tuple.type.a5f = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.a5f = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.a5f = value_param runtime_param0
// CHECK:STDOUT:     %.loc36_37.1: type = splice_block %.loc36_37.3 [template = constants.%tuple.type.a5f] {
// CHECK:STDOUT:       %Self.ref.loc36_12: type = name_ref Self, @impl.4.%D.ref [template = constants.%D]
// CHECK:STDOUT:       %ptr: type = ptr_type %D [template = constants.%ptr.19c]
// CHECK:STDOUT:       %Self.ref.loc36_24: type = name_ref Self, @impl.4.%D.ref [template = constants.%D]
// CHECK:STDOUT:       %.loc36_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc36_35.2: type = converted %.loc36_35.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [template = constants.%struct_type.x.y.527]
// CHECK:STDOUT:       %.loc36_37.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc36_37.3: type = converted %.loc36_37.2, constants.%tuple.type.a5f [template = constants.%tuple.type.a5f]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.aee)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.3e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2)](%x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2)) -> @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.3e2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C](%x.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc20_38.2: init %C = class_init (), %return [template = constants.%C.val]
// CHECK:STDOUT:   %.loc20_39: init %C = converted %.loc20_38.1, %.loc20_38.2 [template = constants.%C.val]
// CHECK:STDOUT:   return %.loc20_39 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %D](%x.param_patt: %D) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_47.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_47.2: init %D = class_init (), %return [template = constants.%D.val]
// CHECK:STDOUT:   %.loc24_48: init %D = converted %.loc24_47.1, %.loc24_47.2 [template = constants.%D.val]
// CHECK:STDOUT:   return %.loc24_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.4(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.36c)]
// CHECK:STDOUT:   %Self.as_type.loc28_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.966)]
// CHECK:STDOUT:   %ptr.loc28_16.1: type = ptr_type @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.966) [symbolic = %ptr.loc28_16.1 (constants.%ptr.af7)]
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1: type = struct_type {.x: @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.966), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.0dc)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.4.%ptr.loc28_16.1 (%ptr.af7), @F.4.%struct_type.x.y.loc28_36.1 (%struct_type.x.y.0dc)) [symbolic = %tuple.type (constants.%tuple.type.7e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.4.%tuple.type (%tuple.type.7e5));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(%x.param_patt: %tuple.type.a17);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(%x.param_patt: %tuple.type.a5f);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.aee) {
// CHECK:STDOUT:   %Self => constants.%Self.aee
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.3e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.c5b) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.c5b
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.af5) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.af5
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%Self.36c) {
// CHECK:STDOUT:   %Self => constants.%Self.36c
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%Self.as_type.966
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.af7
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.0dc
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.7e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.d94) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.d94
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.019
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.2f0
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.2f0) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.2f0
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.19c
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.527
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
