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
  // CHECK:STDERR: self_in_signature.carbon:[[@LINE+6]]:8: ERROR: Function redeclaration differs at implicit parameter 1.
  // CHECK:STDERR:   fn F[self: C](x: C) -> C { return {}; }
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR: self_in_signature.carbon:[[@LINE-10]]:8: Previous declaration's corresponding implicit parameter here.
  // CHECK:STDERR:   fn F[self: Self](x: Self) -> Self;
  // CHECK:STDERR:        ^~~~
  fn F[self: C](x: C) -> C { return {}; }
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
// CHECK:STDOUT:   %.7: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %UseSelf.ref: type = name_ref UseSelf, %UseSelf.decl [template = constants.%.1]
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
// CHECK:STDOUT: impl @impl: C as UseSelf {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %C.ref.loc24_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc24_8.1: C = param self
// CHECK:STDOUT:     %self.loc24_8.2: C = bind_name self, %self.loc24_8.1
// CHECK:STDOUT:     %C.ref.loc24_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc24_17.1: C = param x
// CHECK:STDOUT:     %x.loc24_17.2: C = bind_name x, %x.loc24_17.1
// CHECK:STDOUT:     %C.ref.loc24_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @UseSelf, (<error>) [template = <error>]
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
// CHECK:STDOUT: fn @F.2[@impl.%self.loc24_8.2: C](@impl.%x.loc24_17.2: C) -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_38.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc24_38.2: init C = class_init (), @impl.%return.var [template = constants.%.7]
// CHECK:STDOUT:   %.loc24_38.3: init C = converted %.loc24_38.1, %.loc24_38.2 [template = constants.%.7]
// CHECK:STDOUT:   return %.loc24_38.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
