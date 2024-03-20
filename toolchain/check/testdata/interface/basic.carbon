// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Empty {
}

interface ForwardDeclared;

interface ForwardDeclared {
  fn F();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %.2: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @ForwardDeclared, <function> [template]
// CHECK:STDOUT:   %.4: <associated <function> in ForwardDeclared> = assoc_entity element0, @ForwardDeclared.%F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc10: type = interface_decl @ForwardDeclared [template = constants.%.2] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc12: type = interface_decl @ForwardDeclared [template = constants.%.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: Empty = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: ForwardDeclared = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %.loc13: <associated <function> in ForwardDeclared> = assoc_entity element0, %F [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc13
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
