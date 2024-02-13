// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on ClassType`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on ClassType`.

// --- a.carbon

library "a" api;

class Empty {
}

class ForwardDeclared;

class ForwardDeclared {
  fn F();
}

var a_ref: Empty = {};
var b_ref: ForwardDeclared = {};
var c_ref: ForwardDeclared* = &b_ref;
// var d_ref: (ForwardDeclared,) = ({},);

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  var x: Empty = {};
  var y: ForwardDeclared = {};
  y.F();
}

var a: Empty = a_ref;
var b: ForwardDeclared = b_ref;
var c: ForwardDeclared* = c_ref;
// CHECK:STDERR: b.carbon:[[@LINE+3]]:29: ERROR: Name `d_ref` not found.
// CHECK:STDERR: var d: (ForwardDeclared,) = d_ref;
// CHECK:STDERR:                             ^~~~~
var d: (ForwardDeclared,) = d_ref;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %ForwardDeclared: type = class_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: Empty = struct_value () [template]
// CHECK:STDOUT:   %.5: ForwardDeclared = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type ForwardDeclared [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %Empty.decl, .ForwardDeclared = %ForwardDeclared.decl.loc7, .a_ref = %a_ref, .b_ref = %b_ref, .c_ref = %c_ref} [template]
// CHECK:STDOUT:   %Empty.decl = class_decl @Empty, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc7 = class_decl @ForwardDeclared, ()
// CHECK:STDOUT:   %ForwardDeclared.decl.loc9 = class_decl @ForwardDeclared, ()
// CHECK:STDOUT:   %Empty.ref: type = name_ref Empty, constants.%Empty [template = constants.%Empty]
// CHECK:STDOUT:   %a_ref.var: ref Empty = var a_ref
// CHECK:STDOUT:   %a_ref: ref Empty = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, constants.%ForwardDeclared [template = constants.%ForwardDeclared]
// CHECK:STDOUT:   %b_ref.var: ref ForwardDeclared = var b_ref
// CHECK:STDOUT:   %b_ref: ref ForwardDeclared = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %ForwardDeclared.ref.loc15: type = name_ref ForwardDeclared, constants.%ForwardDeclared [template = constants.%ForwardDeclared]
// CHECK:STDOUT:   %.loc15: type = ptr_type ForwardDeclared [template = constants.%.6]
// CHECK:STDOUT:   %c_ref.var: ref ForwardDeclared* = var c_ref
// CHECK:STDOUT:   %c_ref: ref ForwardDeclared* = bind_name c_ref, %c_ref.var
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_21.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc13_21.2: init Empty = class_init (), file.%a_ref.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_21.3: init Empty = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc13_21.3
// CHECK:STDOUT:   %.loc14_31.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc14_31.2: init ForwardDeclared = class_init (), file.%b_ref.var [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_31.3: init ForwardDeclared = converted %.loc14_31.1, %.loc14_31.2 [template = constants.%.5]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc14_31.3
// CHECK:STDOUT:   %b_ref.ref: ref ForwardDeclared = name_ref b_ref, file.%b_ref
// CHECK:STDOUT:   %.loc15: ForwardDeclared* = addr_of %b_ref.ref
// CHECK:STDOUT:   assign file.%c_ref.var, %.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %import_ref.1, .ForwardDeclared = %import_ref.2, .a_ref = %import_ref.3, .b_ref = %import_ref.4, .c_ref = %import_ref.5, .Run = %Run, .a = %a, .b = %b, .c = %c, .d = %d} [template]
// CHECK:STDOUT:   %import_ref.1: <error> = import_ref ir1, inst+1, used [template = <error>]
// CHECK:STDOUT:   %import_ref.2: <error> = import_ref ir1, inst+4, used [template = <error>]
// CHECK:STDOUT:   %import_ref.3: ref <error> = import_ref ir1, inst+12, used
// CHECK:STDOUT:   %import_ref.4: ref <error> = import_ref ir1, inst+20, used
// CHECK:STDOUT:   %import_ref.5: ref <error> = import_ref ir1, inst+30, used
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT:   %Empty.ref: <error> = name_ref Empty, %import_ref.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: <error> = name_ref ForwardDeclared, %import_ref.2 [template = <error>]
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   %b: ref <error> = bind_name b, %b.var
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: <error> = name_ref ForwardDeclared, %import_ref.2 [template = <error>]
// CHECK:STDOUT:   %.loc14: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT:   %ForwardDeclared.ref.loc18: <error> = name_ref ForwardDeclared, %import_ref.2 [template = <error>]
// CHECK:STDOUT:   %.loc18: <error> = tuple_literal (%ForwardDeclared.ref.loc18)
// CHECK:STDOUT:   %d.var: ref <error> = var d
// CHECK:STDOUT:   %d: ref <error> = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Empty.ref: <error> = name_ref Empty, file.%import_ref.1 [template = <error>]
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7: {} = struct_literal ()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %ForwardDeclared.ref: <error> = name_ref ForwardDeclared, file.%import_ref.2 [template = <error>]
// CHECK:STDOUT:   %y.var: ref <error> = var y
// CHECK:STDOUT:   %y: ref <error> = bind_name y, %y.var
// CHECK:STDOUT:   %.loc8: {} = struct_literal ()
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   %y.ref: ref <error> = name_ref y, %y
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, file.%import_ref.3
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %b_ref.ref: ref <error> = name_ref b_ref, file.%import_ref.4
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %c_ref.ref: ref <error> = name_ref c_ref, file.%import_ref.5
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %d_ref.ref: <error> = name_ref d_ref, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
