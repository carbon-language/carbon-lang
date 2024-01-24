// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32, b: i32) {}

fn Main() {
  var x: (i32,) = (1,);
  // Generates multiple IR instructions for the first parameter.
  Foo(x[0], 6);
}

// CHECK:STDOUT: --- more_param_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.7: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %Foo, .Main = %Main} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_15.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc10_15.2: type = converted %.loc10_15.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref (i32,) = var x
// CHECK:STDOUT:   %x: ref (i32,) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_22.1: (i32,) = tuple_literal (%.loc10_20)
// CHECK:STDOUT:   %.loc10_22.2: init (i32,) = tuple_init (%.loc10_20) to %x.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_22.3: init (i32,) = converted %.loc10_22.1, %.loc10_22.2 [template = constants.%.4]
// CHECK:STDOUT:   assign %x.var, %.loc10_22.3
// CHECK:STDOUT:   %Foo.ref: <function> = name_ref Foo, file.%Foo [template = file.%Foo]
// CHECK:STDOUT:   %x.ref: ref (i32,) = name_ref x, %x
// CHECK:STDOUT:   %.loc12_9: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_10.1: ref i32 = tuple_index %x.ref, %.loc12_9
// CHECK:STDOUT:   %.loc12_13: i32 = int_literal 6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_10.2: i32 = bind_value %.loc12_10.1
// CHECK:STDOUT:   %.loc12_6: init () = call %Foo.ref(%.loc12_10.2, %.loc12_13)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
