// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  // CHECK:STDERR: fail_assoc_const_bad_default.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   let T:! type = 42;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  let T:! type = 42;
}

// CHECK:STDOUT: --- fail_assoc_const_bad_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.4: <associated type in I> = assoc_entity element0, @I.%T [template]
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
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %.loc11_18: i32 = int_literal 42 [template = constants.%.2]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc11_20: <associated type in I> = assoc_entity element0, %T [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc11_20
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
