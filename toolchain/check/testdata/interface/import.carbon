// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

interface Empty {
}

interface ForwardDeclared;

interface ForwardDeclared {
  fn F();
}

// --- b.carbon

library "b" api;

import library "a";

// TODO: When ready, consider tests of basic import functionality.

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %Empty.decl, .ForwardDeclared = %ForwardDeclared.decl.loc7} [template]
// CHECK:STDOUT:   %Empty.decl = interface_decl @Empty, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc7 = interface_decl @ForwardDeclared, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc9 = interface_decl @ForwardDeclared, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %import_ref.1, .ForwardDeclared = %import_ref.2} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+2, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
