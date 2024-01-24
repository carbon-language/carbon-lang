// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- global.carbon

package Global api;

var x: i32 = 0;

fn Main() {
  var x: i32 = 1;

  var y: i32 = package.x;
}

// --- inside_fn.carbon

package InsideFn api;

var x: i32 = 0;

fn Main() {
  var x: i32 = 1;

  var y: i32 = package.x;
}

// --- namespace.carbon

package Namespace api;

namespace NS;

class NS.C {
  fn Foo() {}
};

fn Main() {
  package.NS.C.Foo();
}

// CHECK:STDOUT: --- global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .Main = %Main} [template]
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, %.loc4
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %x.var, %.loc7
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %x.ref: ref i32 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc9: i32 = bind_value %x.ref
// CHECK:STDOUT:   assign %y.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inside_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .Main = %Main} [template]
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, %.loc4
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %x.var, %.loc7
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %x.ref: ref i32 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc9: i32 = bind_value %x.ref
// CHECK:STDOUT:   assign %y.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4, .Main = %Main} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {.C = %C.decl} [template]
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Foo = %Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%.loc4 [template = file.%.loc4]
// CHECK:STDOUT:   %C.ref: type = name_ref C, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %Foo.ref: <function> = name_ref Foo, @C.%Foo [template = @C.%Foo]
// CHECK:STDOUT:   %.loc11: init () = call %Foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
