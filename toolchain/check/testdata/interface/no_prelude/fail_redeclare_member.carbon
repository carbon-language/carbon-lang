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
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Interface, F [template]
// CHECK:STDOUT:   %.4: <associated F in Interface> = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %.5: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
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
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc8: <associated F in Interface> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
