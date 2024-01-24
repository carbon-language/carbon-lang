// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface { }

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:1: ERROR: Redefinition of interface Interface.
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-5]]:1: Previous definition was here.
// CHECK:STDERR: interface Interface { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
interface Interface {
  fn F();
}

fn Function();

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: interface Function;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: fn Function();
// CHECK:STDERR: ^~~~~~~~~~~~~~
interface Function;

class Class;

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: interface Class { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
interface Class { }

// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Interface = %Interface.decl.loc7, .Function = %Function, .Class = %Class.decl} [template]
// CHECK:STDOUT:   %Interface.decl.loc7 = interface_decl @Interface, ()
// CHECK:STDOUT:   %Interface.decl.loc15 = interface_decl @Interface, ()
// CHECK:STDOUT:   %Function: <function> = fn_decl @Function [template]
// CHECK:STDOUT:   %.decl.loc27 = interface_decl @.1, ()
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %.decl.loc37 = interface_decl @.2, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.2 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function();
// CHECK:STDOUT:
