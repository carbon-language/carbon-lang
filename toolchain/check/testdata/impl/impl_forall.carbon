// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn F();
}

impl forall [T:! type] T as Simple {
  fn F() {}
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Simple [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Simple, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Simple> = assoc_entity element0, @Simple.%F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Simple.decl = interface_decl @Simple {} [template = constants.%.1]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %T.loc11_14.1: type = param T
// CHECK:STDOUT:     %T.loc11_14.2: type = bind_symbolic_name T, %T.loc11_14.1 [symbolic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.2 [symbolic = %T.loc11_14.2]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, %Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 {} [template]
// CHECK:STDOUT:   %.loc8: <associated <function> in Simple> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: T as Simple {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 {} [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
