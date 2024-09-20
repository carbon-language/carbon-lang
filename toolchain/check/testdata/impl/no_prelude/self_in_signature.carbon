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
// CHECK:STDOUT:   %.1: type = interface_type @UseSelf [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @UseSelf.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %struct.1: %C = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %struct.2: %D = struct_value () [template]
// CHECK:STDOUT:   %.9: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %Self.2: %.9 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: %Self.2, .y: %.2} [symbolic]
// CHECK:STDOUT:   %.12: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.13: type = tuple_type (%.10, %.11) [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %.9, %F.type.4 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %.16: type = ptr_type %C [template]
// CHECK:STDOUT:   %.17: type = struct_type {.x: %C, .y: %.2} [template]
// CHECK:STDOUT:   %.18: type = tuple_type (%.16, %.17) [template]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%F.5) [template]
// CHECK:STDOUT:   %.20: type = ptr_type %D [template]
// CHECK:STDOUT:   %.21: type = struct_type {.x: %D, .y: %.2} [template]
// CHECK:STDOUT:   %.22: type = tuple_type (%.20, %.21) [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.6: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%F.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, file.%UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%.9] {} {}
// CHECK:STDOUT:   impl_decl @impl.3 {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self (%Self.1) = binding_pattern self
// CHECK:STDOUT:     %x.patt: @F.1.%Self (%Self.1) = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %.1 = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %param.loc12_8: @F.1.%Self (%Self.1) = param
// CHECK:STDOUT:     %self: @F.1.%Self (%Self.1) = bind_name self, %param.loc12_8
// CHECK:STDOUT:     %Self.ref.loc12_23: %.1 = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_23.1: type = facet_type_access %Self.ref.loc12_23 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_23.2: type = converted %Self.ref.loc12_23, %.loc12_23.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %param.loc12_20: @F.1.%Self (%Self.1) = param
// CHECK:STDOUT:     %x: @F.1.%Self (%Self.1) = bind_name x, %param.loc12_20
// CHECK:STDOUT:     %Self.ref.loc12_32: %.1 = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_32.1: type = facet_type_access %Self.ref.loc12_32 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_32.2: type = converted %Self.ref.loc12_32, %.loc12_32.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %return: ref @F.1.%Self (%Self.1) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.4] {
// CHECK:STDOUT:     %x.patt: @F.4.%.3 (%.13) = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28_12: %.9 = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.1: type = facet_type_access %Self.ref.loc28_12 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.2: type = converted %Self.ref.loc28_12, %.loc28_16.1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.3: type = ptr_type %Self.2 [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:     %Self.ref.loc28_24: %.9 = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_24.1: type = facet_type_access %Self.ref.loc28_24 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_24.2: type = converted %Self.ref.loc28_24, %.loc28_24.1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_35.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc28_35.2: type = converted %.loc28_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_36: type = struct_type {.x: %Self.2, .y: %.2} [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:     %.loc28_37.1: %.12 = tuple_literal (%.loc28_16.3, %.loc28_36)
// CHECK:STDOUT:     %.loc28_37.2: type = converted %.loc28_37.1, constants.%.13 [symbolic = %.3 (constants.%.13)]
// CHECK:STDOUT:     %param: @F.4.%.3 (%.13) = param
// CHECK:STDOUT:     %x: @F.4.%.3 (%.13) = bind_name x, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28: %.14 = assoc_entity element0, %F.decl [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc28
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %param.loc20_8: %C = param
// CHECK:STDOUT:     %self: %C = bind_name self, %param.loc20_8
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %param.loc20_17: %C = param
// CHECK:STDOUT:     %x: %C = bind_name x, %param.loc20_17
// CHECK:STDOUT:     %C.ref.loc20_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %x.patt: %D = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc24_14: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %param.loc24_8: %D = param
// CHECK:STDOUT:     %self: %D = bind_name self, %param.loc24_8
// CHECK:STDOUT:     %Self.ref.loc24_23: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %param.loc24_20: %D = param
// CHECK:STDOUT:     %x: %D = bind_name x, %param.loc24_20
// CHECK:STDOUT:     %Self.ref.loc24_32: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %return: ref %D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C as %.9 {
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.5] {
// CHECK:STDOUT:     %x.patt: %.18 = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc32_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc32_13: type = ptr_type %C [template = constants.%.16]
// CHECK:STDOUT:     %C.ref.loc32_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc32_29.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc32_29.2: type = converted %.loc32_29.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc32_30: type = struct_type {.x: %C, .y: %.2} [template = constants.%.17]
// CHECK:STDOUT:     %.loc32_31.1: %.12 = tuple_literal (%.loc32_13, %.loc32_30)
// CHECK:STDOUT:     %.loc32_31.2: type = converted %.loc32_31.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %param: %.18 = param
// CHECK:STDOUT:     %x: %.18 = bind_name x, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %D as %.9 {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.6] {
// CHECK:STDOUT:     %x.patt: %.22 = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc36_12: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc36_16: type = ptr_type %D [template = constants.%.20]
// CHECK:STDOUT:     %Self.ref.loc36_24: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc36_35.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc36_35.2: type = converted %.loc36_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc36_36: type = struct_type {.x: %D, .y: %.2} [template = constants.%.21]
// CHECK:STDOUT:     %.loc36_37.1: %.12 = tuple_literal (%.loc36_16, %.loc36_36)
// CHECK:STDOUT:     %.loc36_37.2: type = converted %.loc36_37.1, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %param: %.22 = param
// CHECK:STDOUT:     %x: %.22 = bind_name x, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.23]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@UseSelf.%Self: %.1) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.patt: @F.1.%Self (%Self.1)](%x.patt: @F.1.%Self (%Self.1)) -> @F.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.patt: %C](%x.patt: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_38.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc20_38.2: init %C = class_init (), %return [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_39: init %C = converted %.loc20_38.1, %.loc20_38.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc20_39 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.patt: %D](%x.patt: %D) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_47.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc24_47.2: init %D = class_init (), %return [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc24_48: init %D = converted %.loc24_47.1, %.loc24_47.2 [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc24_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.4(@SelfNested.%Self: %.9) {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.4.%Self (%Self.2) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: type = struct_type {.x: @F.4.%Self (%Self.2), .y: %.2} [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:   %.3: type = tuple_type (@F.4.%.1 (%.10), @F.4.%.2 (%.11)) [symbolic = %.3 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.patt: @F.4.%.3 (%.13));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(%x.patt: %.18);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(%x.patt: %.22);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%D) {
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT:   %.3 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.4(constants.%D) {
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %.2 => constants.%.21
// CHECK:STDOUT:   %.3 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
