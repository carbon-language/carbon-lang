// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I { let T:! type; }

// CHECK:STDERR: fail_impl_bad_assoc_const.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `impl of interface with associated constant`.
// CHECK:STDERR: impl bool as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
impl bool as I {}

// CHECK:STDOUT: --- fail_impl_bad_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.3: <associated type in I> = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc7: <associated type in I> = assoc_entity element0, %T [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc7
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: bool as I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
