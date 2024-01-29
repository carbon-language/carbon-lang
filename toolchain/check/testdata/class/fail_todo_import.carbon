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

fn Run() {
  // CHECK:STDERR: b.carbon:[[@LINE+3]]:10: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   var x: Empty = {};
  // CHECK:STDERR:          ^~~~~
  var x: Empty = {};
  // CHECK:STDERR: b.carbon:[[@LINE+3]]:10: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   var y: ForwardDeclared = {};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~
  var y: ForwardDeclared = {};
  y.F();
}

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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %import_ref.1, .ForwardDeclared = %import_ref.2, .Run = %Run} [template]
// CHECK:STDOUT:   %import_ref.1: invalid = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %import_ref.2: invalid = import_ref ir1, inst+4, used
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Empty.ref: invalid = name_ref Empty, file.%import_ref.1
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10: {} = struct_literal ()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %ForwardDeclared.ref: invalid = name_ref ForwardDeclared, file.%import_ref.2
// CHECK:STDOUT:   %y.var: ref <error> = var y
// CHECK:STDOUT:   %y: ref <error> = bind_name y, %y.var
// CHECK:STDOUT:   %.loc14: {} = struct_literal ()
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   %y.ref: ref <error> = name_ref y, %y
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
