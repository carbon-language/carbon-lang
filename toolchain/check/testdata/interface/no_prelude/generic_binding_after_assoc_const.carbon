// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  fn F(T:! type);
  let U:! type;
  // This `T` should have index 1, just like the `T` above.
  // The intervening `U:! type` should not cause this to have index 2.
  fn G(T:! type);
}

// CHECK:STDOUT: --- generic_binding_after_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in I> = assoc_entity element0, @I.%F [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.5: <associated type in I> = assoc_entity element1, @I.%U [template]
// CHECK:STDOUT:   %.6: <associated <function> in I> = assoc_entity element2, @I.%G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %T.loc8_8.1: type = param T
// CHECK:STDOUT:     %T.loc8_8.2: type = bind_symbolic_name T 1, %T.loc8_8.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %.loc9: <associated type in I> = assoc_entity element1, %U [template = constants.%.5]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %T.loc10_8.1: type = param T
// CHECK:STDOUT:     %T.loc10_8.2: type = bind_symbolic_name T 1, %T.loc10_8.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <associated <function> in I> = assoc_entity element2, %G [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   .U = %.loc9
// CHECK:STDOUT:   .G = %.loc10
// CHECK:STDOUT:   witness = (%F, %U, %G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@I.%T.loc8_8.2: type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@I.%T.loc10_8.2: type);
// CHECK:STDOUT:
