// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class Empty {
}

class ForwardDeclared;

class ForwardDeclared {
  fn F();
}

// --- b.carbon

library "b" api;

import library "a";

// TODO: When ready, consider tests of basic import functionality.

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %ForwardDeclared: type = class_type @ForwardDeclared [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %Empty.decl, .ForwardDeclared = %ForwardDeclared.decl.loc7} [template]
// CHECK:STDOUT:   %Empty.decl = class_decl @Empty, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc7 = class_decl @ForwardDeclared, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc9 = class_decl @ForwardDeclared, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared {
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
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %lazy_import_ref.1, .ForwardDeclared = %lazy_import_ref.2} [template]
// CHECK:STDOUT:   %lazy_import_ref.1 = lazy_import_ref ir1, inst+1
// CHECK:STDOUT:   %lazy_import_ref.2 = lazy_import_ref ir1, inst+4
// CHECK:STDOUT: }
// CHECK:STDOUT:
