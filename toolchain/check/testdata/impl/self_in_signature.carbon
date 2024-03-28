// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface UseSelf {
  fn F[self: Self](x: Self) -> Self;
}

class C {}

impl C as UseSelf {
  // TODO: Use `Self` below once it's supported.
  fn F[self: C](x: C) -> C { return {}; }
}

interface SelfNested {
  fn F(x: (Self*, {.x: Self, .y: i32}));
}

impl C as SelfNested {
  fn F(x: (C*, {.x: C, .y: i32}));
}

// CHECK:STDOUT: --- self_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @UseSelf [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @UseSelf, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in UseSelf> = assoc_entity element0, @UseSelf.%F [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %.6: type = ptr_type {} [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (@impl.1.%F) [template]
// CHECK:STDOUT:   %.8: C = struct_value () [template]
// CHECK:STDOUT:   %.9: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %.10: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: Self, .y: i32} [symbolic]
// CHECK:STDOUT:   %.12: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.13: type = tuple_type (Self*, {.x: Self, .y: i32}) [symbolic]
// CHECK:STDOUT:   %.14: type = assoc_entity_type @SelfNested, <function> [template]
// CHECK:STDOUT:   %.15: <associated <function> in SelfNested> = assoc_entity element0, @SelfNested.%F [template]
// CHECK:STDOUT:   %.16: type = ptr_type C [template]
// CHECK:STDOUT:   %.17: type = struct_type {.x: C, .y: i32} [template]
// CHECK:STDOUT:   %.18: type = tuple_type (C*, {.x: C, .y: i32}) [template]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (@impl.2.%F) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%.9] {}
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, %SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: UseSelf = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_14: UseSelf = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.1: type = facet_type_access %Self.ref.loc8_14 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %Self.ref.loc8_14, %.loc8_14.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc8_8.1: Self = param self
// CHECK:STDOUT:     %self.loc8_8.2: Self = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %Self.ref.loc8_23: UseSelf = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_23.1: type = facet_type_access %Self.ref.loc8_23 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_23.2: type = converted %Self.ref.loc8_23, %.loc8_23.1 [symbolic = %Self]
// CHECK:STDOUT:     %x.loc8_20.1: Self = param x
// CHECK:STDOUT:     %x.loc8_20.2: Self = bind_name x, %x.loc8_20.1
// CHECK:STDOUT:     %Self.ref.loc8_32: UseSelf = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_32.1: type = facet_type_access %Self.ref.loc8_32 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %Self.ref.loc8_32, %.loc8_32.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_36: <associated <function> in UseSelf> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8_36
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: SelfNested = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.3 [template] {
// CHECK:STDOUT:     %Self.ref.loc19_12: SelfNested = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_16.1: type = facet_type_access %Self.ref.loc19_12 [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_12: type = converted %Self.ref.loc19_12, %.loc19_16.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_16.2: type = ptr_type Self [symbolic = constants.%.10]
// CHECK:STDOUT:     %Self.ref.loc19_24: SelfNested = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_24.1: type = facet_type_access %Self.ref.loc19_24 [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_24.2: type = converted %Self.ref.loc19_24, %.loc19_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc19_37: type = struct_type {.x: Self, .y: i32} [symbolic = constants.%.11]
// CHECK:STDOUT:     %.loc19_38.1: (type, type) = tuple_literal (%.loc19_16.2, %.loc19_37)
// CHECK:STDOUT:     %.loc19_38.2: type = converted %.loc19_38.1, constants.%.13 [symbolic = constants.%.13]
// CHECK:STDOUT:     %x.loc19_8.1: (Self*, {.x: Self, .y: i32}) = param x
// CHECK:STDOUT:     %x.loc19_8.2: (Self*, {.x: Self, .y: i32}) = bind_name x, %x.loc19_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_40: <associated <function> in SelfNested> = assoc_entity element0, %F [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc19_40
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as UseSelf {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %C.ref.loc15_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc15_8.1: C = param self
// CHECK:STDOUT:     %self.loc15_8.2: C = bind_name self, %self.loc15_8.1
// CHECK:STDOUT:     %C.ref.loc15_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc15_17.1: C = param x
// CHECK:STDOUT:     %x.loc15_17.2: C = bind_name x, %x.loc15_17.1
// CHECK:STDOUT:     %C.ref.loc15_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: C as SelfNested {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.4 [template] {
// CHECK:STDOUT:     %C.ref.loc23_12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_13: type = ptr_type C [template = constants.%.16]
// CHECK:STDOUT:     %C.ref.loc23_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_31: type = struct_type {.x: C, .y: i32} [template = constants.%.17]
// CHECK:STDOUT:     %.loc23_32.1: (type, type) = tuple_literal (%.loc23_13, %.loc23_31)
// CHECK:STDOUT:     %.loc23_32.2: type = converted %.loc23_32.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %x.loc23_8.1: (C*, {.x: C, .y: i32}) = param x
// CHECK:STDOUT:     %x.loc23_8.2: (C*, {.x: C, .y: i32}) = bind_name x, %x.loc23_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@UseSelf.%self.loc8_8.2: Self](@UseSelf.%x.loc8_20.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@impl.1.%self.loc15_8.2: C](@impl.1.%x.loc15_17.2: C) -> @impl.1.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_38.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc15_38.2: init C = class_init (), @impl.1.%return.var [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_38.3: init C = converted %.loc15_38.1, %.loc15_38.2 [template = constants.%.8]
// CHECK:STDOUT:   return %.loc15_38.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(@SelfNested.%x.loc19_8.2: (Self*, {.x: Self, .y: i32}));
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4(@impl.2.%x.loc23_8.2: (C*, {.x: C, .y: i32}));
// CHECK:STDOUT:
