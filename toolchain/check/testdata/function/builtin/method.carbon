// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  fn F[self: Self](other: Self) -> Self;
}

impl i32 as I {
  fn F[self: i32](other: i32) -> i32 = "int.add";
}

var arr: [i32; 1.(I.F)(2)];

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in I> = assoc_entity element0, @I.%F [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (@impl.%F) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: <bound method> = bound_method %.5, @impl.%F [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, i32 [template]
// CHECK:STDOUT:   %.10: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %I.ref.loc15: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated <function> in I> = name_ref F, @I.%.loc8_40 [template = constants.%.3]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%F]
// CHECK:STDOUT:   %.loc15_17: <bound method> = bound_method %.loc15_16, %.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc15_24: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_23: init i32 = call %.loc15_17(%.loc15_16, %.loc15_24) [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_26: type = array_type %.loc15_23, i32 [template = constants.%.9]
// CHECK:STDOUT:   %arr.var: ref [i32; 3] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 3] = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_14: I = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.1: type = facet_type_access %Self.ref.loc8_14 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %Self.ref.loc8_14, %.loc8_14.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc8_8.1: Self = param self
// CHECK:STDOUT:     %self.loc8_8.2: Self = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %Self.ref.loc8_27: I = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_27.1: type = facet_type_access %Self.ref.loc8_27 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_27.2: type = converted %Self.ref.loc8_27, %.loc8_27.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_20.1: Self = param other
// CHECK:STDOUT:     %other.loc8_20.2: Self = bind_name other, %other.loc8_20.1
// CHECK:STDOUT:     %Self.ref.loc8_36: I = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_36.1: type = facet_type_access %Self.ref.loc8_36 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_36.2: type = converted %Self.ref.loc8_36, %.loc8_36.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_40: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8_40
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %self.loc12_8.1: i32 = param self
// CHECK:STDOUT:     %self.loc12_8.2: i32 = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %other.loc12_19.1: i32 = param other
// CHECK:STDOUT:     %other.loc12_19.2: i32 = bind_name other, %other.loc12_19.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@I.%self.loc8_8.2: Self](@I.%other.loc8_20.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@impl.%self.loc12_8.2: i32](@impl.%other.loc12_19.2: i32) -> i32 = "int.add";
// CHECK:STDOUT:
