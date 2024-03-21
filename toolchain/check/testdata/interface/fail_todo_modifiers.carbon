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
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Modifiers, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Modifiers> = assoc_entity element0, @Modifiers.%Final [template]
// CHECK:STDOUT:   %.4: <associated <function> in Modifiers> = assoc_entity element1, @Modifiers.%Default [template]
// CHECK:STDOUT:   %.5: type = interface_type @Private [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Modifiers = %Modifiers.decl
// CHECK:STDOUT:     .Private = %Private.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.decl: type = interface_decl @Modifiers [template = constants.%.1] {}
// CHECK:STDOUT:   %Private.decl: type = interface_decl @Private [template = constants.%.5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers {
// CHECK:STDOUT:   %Self: Modifiers = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Final: <function> = fn_decl @Final [template] {}
// CHECK:STDOUT:   %.loc12: <associated <function> in Modifiers> = assoc_entity element0, %Final [template = constants.%.3]
// CHECK:STDOUT:   %Default: <function> = fn_decl @Default [template] {}
// CHECK:STDOUT:   %.loc17: <associated <function> in Modifiers> = assoc_entity element1, %Default [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Final = %.loc12
// CHECK:STDOUT:   .Default = %.loc17
// CHECK:STDOUT:   witness = (%Final, %Default)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Private {
// CHECK:STDOUT:   %Self: Private = bind_symbolic_name Self [symbolic]
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
