// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Modifiers {
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   final fn Final() { }
  // CHECK:STDERR:   ^~~~~
  final fn Final() { }
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn Default() { }
  // CHECK:STDERR:   ^~~~~~~
  default fn Default() { }
}

// CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private interface Private {
// CHECK:STDERR: ^~~~~~~
private interface Private {
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Modifiers = %Modifiers.decl, .Private = %Private.decl} [template]
// CHECK:STDOUT:   %Modifiers.decl = interface_decl @Modifiers, ()
// CHECK:STDOUT:   %Private.decl = interface_decl @Private, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers {
// CHECK:STDOUT:   %Final: <function> = fn_decl @Final [template]
// CHECK:STDOUT:   %Default: <function> = fn_decl @Default [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Final = %Final
// CHECK:STDOUT:   .Default = %Default
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Private {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
