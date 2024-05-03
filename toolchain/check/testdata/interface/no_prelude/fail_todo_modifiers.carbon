// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Modifiers {
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   final fn Final() { }
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final() { }
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn Default() { }
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn Default() { }
}

// CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private interface Private {
// CHECK:STDERR: ^~~~~~~
private interface Private {
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Modifiers [template]
// CHECK:STDOUT:   %Self.1: Modifiers = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Final: type = fn_type @Final [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Final = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Modifiers, Final [template]
// CHECK:STDOUT:   %.4: <associated Final in Modifiers> = assoc_entity element0, @Modifiers.%Final.decl [template]
// CHECK:STDOUT:   %Default: type = fn_type @Default [template]
// CHECK:STDOUT:   %struct.2: Default = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Modifiers, Default [template]
// CHECK:STDOUT:   %.6: <associated Default in Modifiers> = assoc_entity element1, @Modifiers.%Default.decl [template]
// CHECK:STDOUT:   %.7: type = interface_type @Private [template]
// CHECK:STDOUT:   %Self.2: Private = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Modifiers = %Modifiers.decl
// CHECK:STDOUT:     .Private = %Private.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.decl: type = interface_decl @Modifiers [template = constants.%.1] {}
// CHECK:STDOUT:   %Private.decl: type = interface_decl @Private [template = constants.%.7] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers {
// CHECK:STDOUT:   %Self: Modifiers = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %Final.decl: Final = fn_decl @Final [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc12: <associated Final in Modifiers> = assoc_entity element0, %Final.decl [template = constants.%.4]
// CHECK:STDOUT:   %Default.decl: Default = fn_decl @Default [template = constants.%struct.2] {}
// CHECK:STDOUT:   %.loc17: <associated Default in Modifiers> = assoc_entity element1, %Default.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Final = %.loc12
// CHECK:STDOUT:   .Default = %.loc17
// CHECK:STDOUT:   witness = (%Final.decl, %Default.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Private {
// CHECK:STDOUT:   %Self: Private = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Final() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Default() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
