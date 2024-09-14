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
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref.loc19: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %D.ref.loc23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref.loc23: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%.9] {}
// CHECK:STDOUT:   impl_decl @impl.3 {
// CHECK:STDOUT:     %C.ref.loc31: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref.loc31: type = name_ref SelfNested, %SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 {
// CHECK:STDOUT:     %D.ref.loc35: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref.loc35: type = name_ref SelfNested, %SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %Self.ref.loc12_14: %.1 = name_ref Self, %Self [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %self.loc12_8.1: @F.1.%Self (%Self.1) = param self, runtime_param0
// CHECK:STDOUT:     %self.loc12_8.2: @F.1.%Self (%Self.1) = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %Self.ref.loc12_23: %.1 = name_ref Self, %Self [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_23.1: type = facet_type_access %Self.ref.loc12_23 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_23.2: type = converted %Self.ref.loc12_23, %.loc12_23.1 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %x.loc12_20.1: @F.1.%Self (%Self.1) = param x, runtime_param1
// CHECK:STDOUT:     %x.loc12_20.2: @F.1.%Self (%Self.1) = bind_name x, %x.loc12_20.1
// CHECK:STDOUT:     %Self.ref.loc12_32: %.1 = name_ref Self, %Self [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_32.1: type = facet_type_access %Self.ref.loc12_32 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_32.2: type = converted %Self.ref.loc12_32, %.loc12_32.1 [symbolic = @F.1.%Self (constants.%Self.1)]
// CHECK:STDOUT:     %return.var: ref @F.1.%Self (%Self.1) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_36: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12_36
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.4] {
// CHECK:STDOUT:     %Self.ref.loc28_12: %.9 = name_ref Self, %Self [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.1: type = facet_type_access %Self.ref.loc28_12 [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.2: type = converted %Self.ref.loc28_12, %.loc28_16.1 [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_16.3: type = ptr_type %Self.2 [symbolic = @F.4.%.1 (constants.%.10)]
// CHECK:STDOUT:     %Self.ref.loc28_24: %.9 = name_ref Self, %Self [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_24.1: type = facet_type_access %Self.ref.loc28_24 [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_24.2: type = converted %Self.ref.loc28_24, %.loc28_24.1 [symbolic = @F.4.%Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc28_35.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc28_35.2: type = converted %.loc28_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_36: type = struct_type {.x: %Self.2, .y: %.2} [symbolic = @F.4.%.2 (constants.%.11)]
// CHECK:STDOUT:     %.loc28_37.1: %.12 = tuple_literal (%.loc28_16.3, %.loc28_36)
// CHECK:STDOUT:     %.loc28_37.2: type = converted %.loc28_37.1, constants.%.13 [symbolic = @F.4.%.3 (constants.%.13)]
// CHECK:STDOUT:     %x.loc28_8.1: @F.4.%.3 (%.13) = param x, runtime_param0
// CHECK:STDOUT:     %x.loc28_8.2: @F.4.%.3 (%.13) = bind_name x, %x.loc28_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_39: %.14 = assoc_entity element0, %F.decl [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc28_39
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %C.ref.loc20_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc20_8.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc20_8.2: %C = bind_name self, %self.loc20_8.1
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc20_17.1: %C = param x, runtime_param1
// CHECK:STDOUT:     %x.loc20_17.2: %C = bind_name x, %x.loc20_17.1
// CHECK:STDOUT:     %C.ref.loc20_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %Self.ref.loc24_14: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %self.loc24_8.1: %D = param self, runtime_param0
// CHECK:STDOUT:     %self.loc24_8.2: %D = bind_name self, %self.loc24_8.1
// CHECK:STDOUT:     %Self.ref.loc24_23: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %x.loc24_20.1: %D = param x, runtime_param1
// CHECK:STDOUT:     %x.loc24_20.2: %D = bind_name x, %x.loc24_20.1
// CHECK:STDOUT:     %Self.ref.loc24_32: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %return.var: ref %D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C as %.9 {
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.5] {
// CHECK:STDOUT:     %C.ref.loc32_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc32_13: type = ptr_type %C [template = constants.%.16]
// CHECK:STDOUT:     %C.ref.loc32_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc32_29.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc32_29.2: type = converted %.loc32_29.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc32_30: type = struct_type {.x: %C, .y: %.2} [template = constants.%.17]
// CHECK:STDOUT:     %.loc32_31.1: %.12 = tuple_literal (%.loc32_13, %.loc32_30)
// CHECK:STDOUT:     %.loc32_31.2: type = converted %.loc32_31.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %x.loc32_8.1: %.18 = param x, runtime_param0
// CHECK:STDOUT:     %x.loc32_8.2: %.18 = bind_name x, %x.loc32_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31: <witness> = interface_witness (%F.decl) [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %D as %.9 {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.6] {
// CHECK:STDOUT:     %Self.ref.loc36_12: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc36_16: type = ptr_type %D [template = constants.%.20]
// CHECK:STDOUT:     %Self.ref.loc36_24: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc36_35.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc36_35.2: type = converted %.loc36_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc36_36: type = struct_type {.x: %D, .y: %.2} [template = constants.%.21]
// CHECK:STDOUT:     %.loc36_37.1: %.12 = tuple_literal (%.loc36_16, %.loc36_36)
// CHECK:STDOUT:     %.loc36_37.2: type = converted %.loc36_37.1, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %x.loc36_8.1: %.22 = param x, runtime_param0
// CHECK:STDOUT:     %x.loc36_8.2: %.22 = bind_name x, %x.loc36_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35: <witness> = interface_witness (%F.decl) [template = constants.%.23]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc35
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
// CHECK:STDOUT:   fn[@UseSelf.%self.loc12_8.2: @F.1.%Self (%Self.1)](@UseSelf.%x.loc12_20.2: @F.1.%Self (%Self.1)) -> @F.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@impl.1.%self.loc20_8.2: %C](@impl.1.%x.loc20_17.2: %C) -> @impl.1.%return.var: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_38.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc20_38.2: init %C = class_init (), @impl.1.%return.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_39: init %C = converted %.loc20_38.1, %.loc20_38.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc20_39 to @impl.1.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[@impl.2.%self.loc24_8.2: %D](@impl.2.%x.loc24_20.2: %D) -> @impl.2.%return.var: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_47.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc24_47.2: init %D = class_init (), @impl.2.%return.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc24_48: init %D = converted %.loc24_47.1, %.loc24_47.2 [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc24_48 to @impl.2.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.4(@SelfNested.%Self: %.9) {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.4.%Self (%Self.2) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: type = struct_type {.x: @F.4.%Self (%Self.2), .y: %.2} [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:   %.3: type = tuple_type (@F.4.%.1 (%.10), @F.4.%.2 (%.11)) [symbolic = %.3 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@SelfNested.%x.loc28_8.2: @F.4.%.3 (%.13));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(@impl.3.%x.loc32_8.2: %.18);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(@impl.4.%x.loc36_8.2: %.22);
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
