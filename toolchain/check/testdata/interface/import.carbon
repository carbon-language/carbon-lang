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

var f_ref: {.f: ForwardDeclared};

// --- b.carbon

library "b" api;

import library "a";

fn UseEmpty(e: Empty) {}
fn UseForwardDeclared(f: ForwardDeclared) {}

var f: ForwardDeclared* = &f_ref.f;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %.2: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.3: type = struct_type {.f: ForwardDeclared} [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.f: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %Empty.decl, .ForwardDeclared = %ForwardDeclared.decl.loc7, .f_ref = %f_ref} [template]
// CHECK:STDOUT:   %Empty.decl = interface_decl @Empty, () [template = constants.%.1]
// CHECK:STDOUT:   %ForwardDeclared.decl.loc7 = interface_decl @ForwardDeclared, () [template = constants.%.2]
// CHECK:STDOUT:   %ForwardDeclared.decl.loc9 = interface_decl @ForwardDeclared, () [template = constants.%.2]
// CHECK:STDOUT:   %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl.loc7 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13: type = struct_type {.f: ForwardDeclared} [template = constants.%.3]
// CHECK:STDOUT:   %f_ref.var: ref {.f: ForwardDeclared} = var f_ref
// CHECK:STDOUT:   %f_ref: ref {.f: ForwardDeclared} = bind_name f_ref, %f_ref.var
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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.4: type = ptr_type ForwardDeclared [template]
// CHECK:STDOUT:   %.5: type = struct_type {.f: ForwardDeclared} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.f: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Empty = %import_ref.1, .ForwardDeclared = %import_ref.2, .f_ref = %import_ref.3, .UseEmpty = %UseEmpty, .UseForwardDeclared = %UseForwardDeclared, .f = %f} [template]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+3, used [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3: ref {.f: ForwardDeclared} = import_ref ir1, inst+16, used
// CHECK:STDOUT:   %UseEmpty: <function> = fn_decl @UseEmpty [template]
// CHECK:STDOUT:   %UseForwardDeclared: <function> = fn_decl @UseForwardDeclared [template]
// CHECK:STDOUT:   %ForwardDeclared.ref: type = name_ref ForwardDeclared, %import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9: type = ptr_type ForwardDeclared [template = constants.%.4]
// CHECK:STDOUT:   %f.var: ref ForwardDeclared* = var f
// CHECK:STDOUT:   %f: ref ForwardDeclared* = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+6, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %import_ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e: Empty) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f: ForwardDeclared) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref {.f: ForwardDeclared} = name_ref f_ref, file.%import_ref.3
// CHECK:STDOUT:   %.loc9_33: ref ForwardDeclared = struct_access %f_ref.ref, element0
// CHECK:STDOUT:   %.loc9_27: ForwardDeclared* = addr_of %.loc9_33
// CHECK:STDOUT:   assign file.%f.var, %.loc9_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
