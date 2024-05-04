// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %Self.1: UseSelf = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @UseSelf, F [template]
// CHECK:STDOUT:   %.4: <associated F in UseSelf> = assoc_entity element0, @UseSelf.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%struct.2) [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %F.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %struct.4: F = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%struct.4) [template]
// CHECK:STDOUT:   %struct.5: D = struct_value () [template]
// CHECK:STDOUT:   %.9: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %Self.2: SelfNested = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: Self, .y: ()} [symbolic]
// CHECK:STDOUT:   %.12: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.13: type = tuple_type (Self*, {.x: Self, .y: ()}) [symbolic]
// CHECK:STDOUT:   %F.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %struct.6: F = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type @SelfNested, F [template]
// CHECK:STDOUT:   %.15: <associated F in SelfNested> = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %.16: type = ptr_type C [template]
// CHECK:STDOUT:   %.17: type = struct_type {.x: C, .y: ()} [template]
// CHECK:STDOUT:   %.18: type = tuple_type (C*, {.x: C, .y: ()}) [template]
// CHECK:STDOUT:   %F.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %struct.7: F = struct_value () [template]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%struct.7) [template]
// CHECK:STDOUT:   %.20: type = ptr_type D [template]
// CHECK:STDOUT:   %.21: type = struct_type {.x: D, .y: ()} [template]
// CHECK:STDOUT:   %.22: type = tuple_type (D*, {.x: D, .y: ()}) [template]
// CHECK:STDOUT:   %F.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %struct.8: F = struct_value () [template]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%struct.8) [template]
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
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref.loc15: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %D.ref.loc19: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %UseSelf.ref.loc19: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%.9] {}
// CHECK:STDOUT:   impl_decl @impl.3 {
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref.loc27: type = name_ref SelfNested, %SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 {
// CHECK:STDOUT:     %D.ref.loc31: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %SelfNested.ref.loc31: type = name_ref SelfNested, %SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: UseSelf = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc8_14: UseSelf = name_ref Self, %Self [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_14.1: type = facet_type_access %Self.ref.loc8_14 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %Self.ref.loc8_14, %.loc8_14.1 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %self.loc8_8.1: Self = param self
// CHECK:STDOUT:     %self.loc8_8.2: Self = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %Self.ref.loc8_23: UseSelf = name_ref Self, %Self [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_23.1: type = facet_type_access %Self.ref.loc8_23 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_23.2: type = converted %Self.ref.loc8_23, %.loc8_23.1 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %x.loc8_20.1: Self = param x
// CHECK:STDOUT:     %x.loc8_20.2: Self = bind_name x, %x.loc8_20.1
// CHECK:STDOUT:     %Self.ref.loc8_32: UseSelf = name_ref Self, %Self [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_32.1: type = facet_type_access %Self.ref.loc8_32 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %Self.ref.loc8_32, %.loc8_32.1 [symbolic = constants.%Self.1]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_36: <associated F in UseSelf> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8_36
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: SelfNested = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.4 [template = constants.%struct.6] {
// CHECK:STDOUT:     %Self.ref.loc24_12: SelfNested = name_ref Self, %Self [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_16.1: type = facet_type_access %Self.ref.loc24_12 [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_16.2: type = converted %Self.ref.loc24_12, %.loc24_16.1 [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_16.3: type = ptr_type Self [symbolic = constants.%.10]
// CHECK:STDOUT:     %Self.ref.loc24_24: SelfNested = name_ref Self, %Self [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_24.1: type = facet_type_access %Self.ref.loc24_24 [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_24.2: type = converted %Self.ref.loc24_24, %.loc24_24.1 [symbolic = constants.%Self.2]
// CHECK:STDOUT:     %.loc24_35.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc24_35.2: type = converted %.loc24_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc24_36: type = struct_type {.x: Self, .y: ()} [symbolic = constants.%.11]
// CHECK:STDOUT:     %.loc24_37.1: (type, type) = tuple_literal (%.loc24_16.3, %.loc24_36)
// CHECK:STDOUT:     %.loc24_37.2: type = converted %.loc24_37.1, constants.%.13 [symbolic = constants.%.13]
// CHECK:STDOUT:     %x.loc24_8.1: (Self*, {.x: Self, .y: ()}) = param x
// CHECK:STDOUT:     %x.loc24_8.2: (Self*, {.x: Self, .y: ()}) = bind_name x, %x.loc24_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_39: <associated F in SelfNested> = assoc_entity element0, %F.decl [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc24_39
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as UseSelf {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc16_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc16_8.1: C = param self
// CHECK:STDOUT:     %self.loc16_8.2: C = bind_name self, %self.loc16_8.1
// CHECK:STDOUT:     %C.ref.loc16_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc16_17.1: C = param x
// CHECK:STDOUT:     %x.loc16_17.2: C = bind_name x, %x.loc16_17.1
// CHECK:STDOUT:     %C.ref.loc16_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: D as UseSelf {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.3 [template = constants.%struct.4] {
// CHECK:STDOUT:     %Self.ref.loc20_14: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %self.loc20_8.1: D = param self
// CHECK:STDOUT:     %self.loc20_8.2: D = bind_name self, %self.loc20_8.1
// CHECK:STDOUT:     %Self.ref.loc20_23: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %x.loc20_20.1: D = param x
// CHECK:STDOUT:     %x.loc20_20.2: D = bind_name x, %x.loc20_20.1
// CHECK:STDOUT:     %Self.ref.loc20_32: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %return.var: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: C as SelfNested {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.5 [template = constants.%struct.7] {
// CHECK:STDOUT:     %C.ref.loc28_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc28_13: type = ptr_type C [template = constants.%.16]
// CHECK:STDOUT:     %C.ref.loc28_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc28_29.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc28_29.2: type = converted %.loc28_29.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_30: type = struct_type {.x: C, .y: ()} [template = constants.%.17]
// CHECK:STDOUT:     %.loc28_31.1: (type, type) = tuple_literal (%.loc28_13, %.loc28_30)
// CHECK:STDOUT:     %.loc28_31.2: type = converted %.loc28_31.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %x.loc28_8.1: (C*, {.x: C, .y: ()}) = param x
// CHECK:STDOUT:     %x.loc28_8.2: (C*, {.x: C, .y: ()}) = bind_name x, %x.loc28_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: D as SelfNested {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.6 [template = constants.%struct.8] {
// CHECK:STDOUT:     %Self.ref.loc32_12: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc32_16: type = ptr_type D [template = constants.%.20]
// CHECK:STDOUT:     %Self.ref.loc32_24: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %.loc32_35.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc32_35.2: type = converted %.loc32_35.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc32_36: type = struct_type {.x: D, .y: ()} [template = constants.%.21]
// CHECK:STDOUT:     %.loc32_37.1: (type, type) = tuple_literal (%.loc32_16, %.loc32_36)
// CHECK:STDOUT:     %.loc32_37.2: type = converted %.loc32_37.1, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %x.loc32_8.1: (D*, {.x: D, .y: ()}) = param x
// CHECK:STDOUT:     %x.loc32_8.2: (D*, {.x: D, .y: ()}) = bind_name x, %x.loc32_8.1
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
// CHECK:STDOUT: fn @F.1[@UseSelf.%self.loc8_8.2: Self](@UseSelf.%x.loc8_20.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@impl.1.%self.loc16_8.2: C](@impl.1.%x.loc16_17.2: C) -> @impl.1.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_38.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc16_38.2: init C = class_init (), @impl.1.%return.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc16_39: init C = converted %.loc16_38.1, %.loc16_38.2 [template = constants.%struct.3]
// CHECK:STDOUT:   return %.loc16_39 to @impl.1.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[@impl.2.%self.loc20_8.2: D](@impl.2.%x.loc20_20.2: D) -> @impl.2.%return.var: D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_47.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc20_47.2: init D = class_init (), @impl.2.%return.var [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc20_48: init D = converted %.loc20_47.1, %.loc20_47.2 [template = constants.%struct.5]
// CHECK:STDOUT:   return %.loc20_48 to @impl.2.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4(@SelfNested.%x.loc24_8.2: (Self*, {.x: Self, .y: ()}));
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(@impl.3.%x.loc28_8.2: (C*, {.x: C, .y: ()}));
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6(@impl.4.%x.loc32_8.2: (D*, {.x: D, .y: ()}));
// CHECK:STDOUT:
