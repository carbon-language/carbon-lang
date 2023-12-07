// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface {
}

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:21: ERROR: Redefinition of interface Interface.
// CHECK:STDERR: interface Interface {
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-6]]:1: Previous definition was here.
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~
interface Interface {
  fn F();
}

// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Interface = %Interface.decl.loc7}
// CHECK:STDOUT:   %Interface.decl.loc7 = interface_decl @Interface, ()
// CHECK:STDOUT:   %Interface.decl.loc16 = interface_decl @Interface, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
