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
// CHECK:STDOUT:   %Self.1: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.1: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %UseSelf.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @UseSelf.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %UseSelf.facet.1: %UseSelf.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %UseSelf.facet.2: %UseSelf.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [template]
// CHECK:STDOUT:   %Self.2: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.2: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Self.as_type.2 [symbolic]
// CHECK:STDOUT:   %struct_type.x.y.1: type = struct_type {.x: %Self.as_type.2, .y: %empty_tuple.type} [symbolic]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%ptr.2, %struct_type.x.y.1) [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %SelfNested.type, %F.type.4 [template]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %C [template]
// CHECK:STDOUT:   %struct_type.x.y.2: type = struct_type {.x: %C, .y: %empty_tuple.type} [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%ptr.3, %struct_type.x.y.2) [template]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.1: %SelfNested.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %interface.3: <witness> = interface_witness (%F.5) [template]
// CHECK:STDOUT:   %ptr.4: type = ptr_type %D [template]
// CHECK:STDOUT:   %struct_type.x.y.3: type = struct_type {.x: %D, .y: %empty_tuple.type} [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%ptr.4, %struct_type.x.y.3) [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.6: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.2: %SelfNested.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %interface.4: <witness> = interface_witness (%F.6) [template]
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
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [template = constants.%UseSelf.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref.loc12_14 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_14: type = converted %Self.ref.loc12_14, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %Self.ref.loc12_23: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_23: type = facet_access_type %Self.ref.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_23: type = converted %Self.ref.loc12_23, %Self.as_type.loc12_23 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %Self.ref.loc12_32: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_32: type = facet_access_type %Self.ref.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_32: type = converted %Self.ref.loc12_32, %Self.as_type.loc12_32 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param runtime_param1
// CHECK:STDOUT:     %x: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.1 = assoc_entity element0, %F.decl [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.4] {
// CHECK:STDOUT:     %x.patt: @F.4.%tuple.type (%tuple.type.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.4.%tuple.type (%tuple.type.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %Self.as_type.loc28_16.2: type = facet_access_type %Self.ref.loc28_12 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.2)]
// CHECK:STDOUT:     %.loc28_16: type = converted %Self.ref.loc28_12, %Self.as_type.loc28_16.2 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.2)]
// CHECK:STDOUT:     %ptr.loc28_16.2: type = ptr_type %Self.as_type.2 [symbolic = %ptr.loc28_16.1 (constants.%ptr.2)]
// CHECK:STDOUT:     %Self.ref.loc28_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %Self.as_type.loc28_24: type = facet_access_type %Self.ref.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.2)]
// CHECK:STDOUT:     %.loc28_24: type = converted %Self.ref.loc28_24, %Self.as_type.loc28_24 [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.2)]
// CHECK:STDOUT:     %.loc28_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc28_35.2: type = converted %.loc28_35.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.y.loc28_36.2: type = struct_type {.x: %Self.as_type.2, .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.1)]
// CHECK:STDOUT:     %.loc28_37.1: %tuple.type.1 = tuple_literal (%ptr.loc28_16.2, %struct_type.x.y.loc28_36.2)
// CHECK:STDOUT:     %.loc28_37.2: type = converted %.loc28_37.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %x.param: @F.4.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @F.4.%tuple.type (%tuple.type.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.2 = assoc_entity element0, %F.decl [template = constants.%assoc0.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc20_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D.ref as %UseSelf.ref {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %D = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %x.patt: %D = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %D = value_param_pattern %x.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc24_14: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %Self.ref.loc24_23: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %Self.ref.loc24_32: type = name_ref Self, @impl.2.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %self.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %D = value_param runtime_param1
// CHECK:STDOUT:     %x: %D = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.5] {
// CHECK:STDOUT:     %x.patt: %tuple.type.3 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.3 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc32_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.3]
// CHECK:STDOUT:     %C.ref.loc32_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc32_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc32_29.2: type = converted %.loc32_29.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.y: type = struct_type {.x: %C, .y: %empty_tuple.type} [template = constants.%struct_type.x.y.2]
// CHECK:STDOUT:     %.loc32_31.1: %tuple.type.1 = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:     %.loc32_31.2: type = converted %.loc32_31.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %x.param: %tuple.type.3 = value_param runtime_param0
// CHECK:STDOUT:     %x: %tuple.type.3 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %D.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.6] {
// CHECK:STDOUT:     %x.patt: %tuple.type.4 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.4 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc36_12: type = name_ref Self, @impl.4.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %ptr: type = ptr_type %D [template = constants.%ptr.4]
// CHECK:STDOUT:     %Self.ref.loc36_24: type = name_ref Self, @impl.4.%D.ref [template = constants.%D]
// CHECK:STDOUT:     %.loc36_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc36_35.2: type = converted %.loc36_35.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.y: type = struct_type {.x: %D, .y: %empty_tuple.type} [template = constants.%struct_type.x.y.3]
// CHECK:STDOUT:     %.loc36_37.1: %tuple.type.1 = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:     %.loc36_37.2: type = converted %.loc36_37.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:     %x.param: %tuple.type.4 = value_param runtime_param0
// CHECK:STDOUT:     %x: %tuple.type.4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
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
// CHECK:STDOUT:   %Self: %UseSelf.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1)](%x.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1)) -> @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1);
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
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Self.as_type.loc28_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc28_16.1 (constants.%Self.as_type.2)]
// CHECK:STDOUT:   %ptr.loc28_16.1: type = ptr_type @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.2) [symbolic = %ptr.loc28_16.1 (constants.%ptr.2)]
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1: type = struct_type {.x: @F.4.%Self.as_type.loc28_16.1 (%Self.as_type.2), .y: %empty_tuple.type} [symbolic = %struct_type.x.y.loc28_36.1 (constants.%struct_type.x.y.1)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.4.%ptr.loc28_16.1 (%ptr.2), @F.4.%struct_type.x.y.loc28_36.1 (%struct_type.x.y.1)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.4.%tuple.type (%tuple.type.2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(%x.param_patt: %tuple.type.3);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(%x.param_patt: %tuple.type.4);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.1) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.1
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%UseSelf.facet.2) {
// CHECK:STDOUT:   %Self => constants.%UseSelf.facet.2
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%Self.as_type.2
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.2
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.1
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.1) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.1
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.3
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%SelfNested.facet.2) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.2
// CHECK:STDOUT:   %Self.as_type.loc28_16.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc28_16.1 => constants.%ptr.4
// CHECK:STDOUT:   %struct_type.x.y.loc28_36.1 => constants.%struct_type.x.y.3
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
