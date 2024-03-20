// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface {
  fn F();
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE+6]]:3: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE-4]]:3: Name is previously declared here.
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  fn F();
}

// CHECK:STDOUT: --- fail_redeclare_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Interface, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Interface> = assoc_entity element0, @Interface.%F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %.loc8: <associated <function> in Interface> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:   %.loc15: <function> = fn_decl @.1 [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
