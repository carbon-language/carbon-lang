// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `abstract` not allowed on `interface` declaration.
// CHECK:STDERR: abstract interface Abstract {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract interface Abstract {
}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `default` not allowed on `interface` declaration.
// CHECK:STDERR: default interface Default;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default interface Default;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `virtual` not allowed on `interface` declaration.
// CHECK:STDERR: virtual interface Virtual {
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual interface Virtual {
}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `protected` not allowed on `interface` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected interface Protected;
// CHECK:STDERR: ^~~~~~~~~
protected interface Protected;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Abstract [template]
// CHECK:STDOUT:   %Self.1: Abstract = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @Default [template]
// CHECK:STDOUT:   %.3: type = interface_type @Virtual [template]
// CHECK:STDOUT:   %Self.2: Virtual = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @Protected [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:     .Default = %Default.decl
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:     .Protected = %Protected.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.decl: type = interface_decl @Abstract [template = constants.%.1] {}
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [template = constants.%.2] {}
// CHECK:STDOUT:   %Virtual.decl: type = interface_decl @Virtual [template = constants.%.3] {}
// CHECK:STDOUT:   %Protected.decl: type = interface_decl @Protected [template = constants.%.4] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Abstract {
// CHECK:STDOUT:   %Self: Abstract = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Virtual {
// CHECK:STDOUT:   %Self: Virtual = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Protected;
// CHECK:STDOUT:
