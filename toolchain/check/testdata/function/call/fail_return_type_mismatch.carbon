// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo() -> f64 { return 1.0; }

fn Run() {
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   var x: i32 = Foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  var x: i32 = Foo();
}

// CHECK:STDOUT: --- fail_return_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: f64 = real_literal 10e-1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %Foo, .Run = %Run} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: f64 = real_literal 10e-1 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %Foo.ref: <function> = name_ref Foo, file.%Foo [template = file.%Foo]
// CHECK:STDOUT:   %.loc13: init f64 = call %Foo.ref()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
