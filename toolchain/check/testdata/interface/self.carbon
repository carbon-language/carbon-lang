// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface UseSelf {
  fn F[self: Self]() -> Self;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @UseSelf [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @UseSelf, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in UseSelf> = assoc_entity element0, @UseSelf.%F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: UseSelf = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref.loc8_14: UseSelf = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.1: type = facet_type_access %Self.ref.loc8_14 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %Self.ref.loc8_14, %.loc8_14.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc8_8.1: Self = param self
// CHECK:STDOUT:     %self.loc8_8.2: Self = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %Self.ref.loc8_25: UseSelf = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_25.1: type = facet_type_access %Self.ref.loc8_25 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_25.2: type = converted %Self.ref.loc8_25, %.loc8_25.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_29: <associated <function> in UseSelf> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8_29
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@UseSelf.%self.loc8_8.2: Self]() -> Self;
// CHECK:STDOUT:
