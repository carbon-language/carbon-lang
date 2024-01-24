// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `abstract` not allowed on `interface` declaration.
// CHECK:STDERR: abstract interface Abstract {
// CHECK:STDERR: ^~~~~~~~
abstract interface Abstract {
}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `default` not allowed on `interface` declaration.
// CHECK:STDERR: default interface Default;
// CHECK:STDERR: ^~~~~~~
default interface Default;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `virtual` not allowed on `interface` declaration.
// CHECK:STDERR: virtual interface Virtual {
// CHECK:STDERR: ^~~~~~~
virtual interface Virtual {
}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `protected` not allowed on `interface` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected interface Protected;
// CHECK:STDERR: ^~~~~~~~~
protected interface Protected;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Abstract = %Abstract.decl, .Default = %Default.decl, .Virtual = %Virtual.decl, .Protected = %Protected.decl} [template]
// CHECK:STDOUT:   %Abstract.decl = interface_decl @Abstract, ()
// CHECK:STDOUT:   %Default.decl = interface_decl @Default, ()
// CHECK:STDOUT:   %Virtual.decl = interface_decl @Virtual, ()
// CHECK:STDOUT:   %Protected.decl = interface_decl @Protected, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Abstract {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Virtual {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Protected;
// CHECK:STDOUT:
