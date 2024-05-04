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
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Foo = struct_value () [template]
// CHECK:STDOUT:   %.2: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.2: Run = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct.1] {
// CHECK:STDOUT:     @Foo.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: f64 = float_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %Foo.ref: Foo = name_ref Foo, file.%Foo.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Foo.call: init f64 = call %Foo.ref()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
